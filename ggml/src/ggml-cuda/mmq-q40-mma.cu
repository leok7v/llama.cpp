#include "mmq-q40-mma.cuh"
#include "mmq.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include <cuda.h>
#include <cstdint>
#include <mutex>
#include <unordered_map>

// Exact Q4_0 x Q8_1 GEMM on the INT8 tensor cores of sm_120/121 for a batch of tokens. Relative to MMQ:
//   * the arithmetic: int32 dot of one 32-value block (mma.sync.m16n8k32 s8), then fp32 t = fl(dot*dA), sum = fl(t*dB + sum)
//     in K order - MMQ's rounding sequence, bit for bit. The int32 accumulator starts at 0x4B400000 (12582912.0f as bits), so
//     the MMA output IS the fp32 bit pattern of 12582912 + 16*dot (|16*dot| < 2^22) and fma(D, dA, -12582912*dA) = fl(16*dot*dA)
//     with no int-to-float conversion; the 1/16 is carried by dB (exact), the nibbles are expanded to 16*(n-8) with one LOP3.
//   * the operands: the raw Q4_0 rows and the block_q8_1_mmq DS4 rows quantize_mmq_q8_1 writes, streamed by TMA (two
//     cp.async.bulk.tensor per 128-wide K stage, one mbarrier per buffer); no shared-memory pre-pass, no block barrier in the
//     main loop - the last warp done with a buffer re-arms it. The weight box is BM rows x 80 bytes at byte column
//     72*ks - 8*(ks&1), so the 72-byte chunk is 16-byte aligned and lands at shared offset 8*(ks&1); TMA zero-fills the tail.
//   * the output tile goes through shared memory and leaves as 16-byte coalesced stores.
// Dense 2-D only; ne00 % 128 == 0, ne01 % 128 == 0, ne11 >= 64; otherwise the caller runs MMQ.

namespace q4k {
constexpr int BK = 128;                            // K per stage (4 Q4_0 blocks)
constexpr int A_ROW = 80;                          // 72 raw bytes (4 blocks) + 8: 20-word stride is bank-conflict-free, 16-byte TMA box
constexpr int B_ROW = 144;                         // one block_q8_1_mmq per token per stage
constexpr int   MAGIC_I = 0x4B400000;
constexpr float MAGIC_F = 12582912.0f;
static_assert(sizeof(block_q4_0) == 18 && sizeof(block_q8_1_mmq) == B_ROW, "layouts");
}

static __device__ __forceinline__ void mbar_init(uint64_t * bar, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"((unsigned) __cvta_generic_to_shared(bar)), "r"(count));
}
static __device__ __forceinline__ void mbar_expect_tx(uint64_t * bar, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"((unsigned) __cvta_generic_to_shared(bar)), "r"(bytes) : "memory");
}
static __device__ __forceinline__ void mbar_wait(uint64_t * bar, int parity) {
    asm volatile("{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT_%=;\n}\n"
                 :: "r"((unsigned) __cvta_generic_to_shared(bar)), "r"(parity) : "memory");
}
static __device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }

static __device__ __forceinline__ void ldmatrix_x4(unsigned (&r)[4], const void * smem) {
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}

// D = A(16x32 s8) * B(32x8 s8) + C, C = the magic bias in all four slots (separate from D: no per-MMA register moves)
static __device__ __forceinline__ void mma_s8_bias(int (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1, const int (&c)[4]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}

static __device__ __forceinline__ void tma_load_2d(void * smem_dst, const CUtensorMap * map, int c0, int c1, uint64_t * bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];\n"
                 :: "r"((unsigned) __cvta_generic_to_shared(smem_dst)), "l"((uint64_t) map),
                    "r"((unsigned) __cvta_generic_to_shared(bar)), "r"(c0), "r"(c1) : "memory");
}

// v8: no block-wide barrier anywhere in the main loop. One thread issues two TMA loads per stage (the weight box: BM rows x
// 80 bytes at byte column 72*ks - 8*(ks&1), so the 72-byte chunk lands at shared offset 8*(ks&1) and the box stays 16-byte
// aligned, the tail zero-filled by TMA; the activation box: BN token rows x 144 bytes). Each warp waits on the stage's
// mbarrier for itself, computes, and takes a ticket; the warp that takes the last ticket re-arms the buffer with the stage
// NSTAGE ahead. Scales are converted per warp just in time: dA straight from the raw half (the 1/16 moved onto dB), dB for
// the warp's WN tokens into a per-warp slice (double-buffered per K block, one __syncwarp). Epilogue and rounding sequence
// identical to the other kernels; output staged through shared memory as 16-byte stores.
template <int BM, int BN, int WN, int NSTAGE, int NW, bool PIPE = false, bool SCL = false, bool BPF = false, bool RESIDENT = false, int MINB = 1, int UNR = 4>
__global__ void __launch_bounds__(32*NW, MINB) q4_0_q8_1_gemm_tma(
        const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB,
        float * __restrict__ dst, const int M, const int N, const int K, const int64_t stride_dst) {
    using namespace q4k;
    constexpr int NT = 32*NW;
    constexpr int WARPS_J = BN / WN, WARPS_I = NW / WARPS_J;
    constexpr int WM = BM / WARPS_I;
    constexpr int MT_I = WM / 16, MT_J = WN / 8;
    static_assert(WARPS_J * WARPS_I == NW && WM % 16 == 0 && MT_J % 2 == 0 && WN <= 64, "bad warp tiling");
    constexpr int A_BYTES = BM * A_ROW, B_BYTES = BN * B_ROW;
    constexpr int STAGE_BYTES = A_BYTES + B_BYTES;                  // 128-byte multiples: TMA destinations must be 128-aligned
    static_assert(STAGE_BYTES % 128 == 0 && A_BYTES % 128 == 0, "TMA alignment");
    // per-warp scale slices, double-buffered per K block: dB/16 for the warp's WN tokens (SCL: stored in fragment order so one
    // float4 serves a tile pair), and with SCL the (dA, -M*dA) pairs for the warp's WM rows
    constexpr int SLICE_FLOATS = 2 * (WN + (SCL ? 2*WM : 0));
    extern __shared__ __align__(128) char smem[];
    float    * dbslices = (float *) (smem + NSTAGE*STAGE_BYTES);
    uint64_t * full     = (uint64_t *) (dbslices + NW*SLICE_FLOATS);
    int      * ticket   = (int *) (full + NSTAGE);

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wi = warp / WARPS_J, wj = warp % WARPS_J;
    const int ntx = (N + BN - 1) / BN;
    const int it = blockIdx.x / ntx, jt = blockIdx.x - it*ntx;
    const int i_base = it*BM, j_base = jt*BN;
    const int nsteps = K / BK;
    GGML_UNUSED(M);

    auto issue = [&](const int ks) {                                // one thread: both boxes of stage ks into buffer ks % NSTAGE
        const int buf = ks % NSTAGE;
        char * stage = smem + buf*STAGE_BYTES;
        mbar_expect_tx(full + buf, STAGE_BYTES);
        tma_load_2d(stage,           &tmA, 72*ks - 8*(ks & 1), i_base, full + buf);
        tma_load_2d(stage + A_BYTES, &tmB, 0, ks*N + j_base,       full + buf);
    };
    if (tid == 0) {
        for (int b = 0; b < NSTAGE; ++b) { mbar_init(full + b, 1); ticket[b] = 0; }
        fence_proxy_async();
    }
    __syncthreads();                                                // the only block-wide barrier before the write-back
    if (tid == 0) {
        for (int s = 0; s < (RESIDENT ? 1 : NSTAGE) && s < nsteps; ++s) { issue(s); }
    }

    int cbias[4];
#pragma unroll
    for (int l = 0; l < 4; ++l) { cbias[l] = MAGIC_I; asm volatile("" : "+r"(cbias[l])); }
    float sum[MT_I][MT_J][4];
#pragma unroll
    for (int n = 0; n < MT_I; ++n) {
#pragma unroll
        for (int j = 0; j < MT_J; ++j) {
#pragma unroll
            for (int l = 0; l < 4; ++l) { sum[n][j][l] = 0.0f; }
        }
    }
    float * dbw = dbslices + warp*SLICE_FLOATS;
    const int r = lane >> 2, m = lane & 3;

    for (int ks = 0; ks < nsteps; ++ks) {
        const int buf = RESIDENT ? 0 : ks % NSTAGE;
        if (!RESIDENT || ks == 0) { mbar_wait(full + buf, (ks / NSTAGE) & 1); }
        const char * As = smem + buf*STAGE_BYTES;
        const char * Bs = As + A_BYTES;
        const int ashift = RESIDENT ? 0 : 8*(ks & 1);               // the 72-byte chunk's offset inside the 80-byte row
        // just-in-time dB/16 for this warp's WN tokens (1 or 2 per lane) of K block b into slice (b & 1); with SCL in fragment
        // order (tile pair p, lane column m: the four values a lane needs sit together) plus the warp's (dA, -M*dA) row pairs
        auto jit_db = [&](const int b) {
            float * dbb = dbw + (b & 1)*(SLICE_FLOATS/2);
            const int tok = wj*WN + lane;
            if constexpr (SCL) {
                // token t of the warp tile -> tile pair p = t/16, within the pair: tile (t/8)&1, column pair m = (t%8)/2, half t&1
                auto slot = [](int t) { return (t/16)*16 + ((t%8)/2)*4 + ((t/8)&1)*2 + (t&1); };
                dbb[slot(lane)] = __low2float(*(const half2 *) (Bs + tok*B_ROW + 4*b)) * 0.0625f;
                if (WN > 32) { dbb[slot(lane + 32)] = __low2float(*(const half2 *) (Bs + (tok + 32)*B_ROW + 4*b)) * 0.0625f; }
                float * daw = dbb + WN;                             // [row slot][da, nmda], rows (r, r+8) of a tile adjacent
                const int row = wi*WM + lane;                       // WM == 32: one row per lane
                const float d = __half2float(*(const half *) (As + row*A_ROW + ashift + 18*b));
                const int rs = (lane & ~15) + ((lane & 7) << 1) + ((lane >> 3) & 1);
                *(float2 *) (daw + 2*rs) = make_float2(d, -MAGIC_F * d);
            } else {
                dbb[lane] = __low2float(*(const half2 *) (Bs + tok*B_ROW + 4*b)) * 0.0625f;
                if (WN > 32) { dbb[lane + 32] = __low2float(*(const half2 *) (Bs + (tok + 32)*B_ROW + 4*b)) * 0.0625f; }
            }
        };
        // A fragments and the block's fp16 scales (not divided by 16: dB carries the 1/16) for K block b
        auto load_a = [&](const int b, unsigned (&a)[MT_I][4], float (&da)[MT_I][2], float (&nmda)[MT_I][2]) {
#pragma unroll
            for (int n = 0; n < MT_I; ++n) {
                const int row0 = wi*WM + n*16 + r;
                const char * R0 = As + row0*A_ROW + ashift;
                const char * R1 = As + (row0 + 8)*A_ROW + ashift;
                const unsigned * W0 = (const unsigned *) R0;
                const unsigned * W1 = (const unsigned *) R1;
                unsigned q0, q1;
                if ((b & 1) == 0) {
                    const int w = (18*b)/4 + m;
                    q0 = __byte_perm(W0[w], W0[w + 1], 0x5432);
                    q1 = __byte_perm(W1[w], W1[w + 1], 0x5432);
                } else {
                    const int w = (18*b + 2)/4 + m;
                    q0 = W0[w];
                    q1 = W1[w];
                }
                a[n][0] = ((q0 << 4) & 0xF0F0F0F0u) ^ 0x80808080u;
                a[n][1] = ((q1 << 4) & 0xF0F0F0F0u) ^ 0x80808080u;
                a[n][2] = ( q0       & 0xF0F0F0F0u) ^ 0x80808080u;
                a[n][3] = ( q1       & 0xF0F0F0F0u) ^ 0x80808080u;
                if constexpr (SCL) {
                    const float * daw = dbw + (b & 1)*(SLICE_FLOATS/2) + WN;
                    const float4 v = *(const float4 *) (daw + 2*((n*16) + 2*r));   // (da r, nmda r, da r+8, nmda r+8)
                    da[n][0] = v.x; nmda[n][0] = v.y; da[n][1] = v.z; nmda[n][1] = v.w;
                } else {
                    da[n][0] = __half2float(*(const half *) (R0 + 18*b));
                    da[n][1] = __half2float(*(const half *) (R1 + 18*b));
                    nmda[n][0] = -MAGIC_F * da[n][0];
                    nmda[n][1] = -MAGIC_F * da[n][1];
                }
            }
        };
        auto load_b = [&](const int b, const int jj, unsigned (&bf)[4], float2 (&db)[2]) {
            const int jrow = wj*WN + jj*8 + (lane & 7) + ((lane >> 4) << 3);
            const int koff = 16 + 32*b + ((lane >> 3) & 1)*16;
            ldmatrix_x4(bf, Bs + jrow*B_ROW + koff);
            const float * dbb = dbw + (b & 1)*(SLICE_FLOATS/2);
            if constexpr (SCL) {
                const float4 v = *(const float4 *) (dbb + (jj/2)*16 + m*4);       // tokens (jj*8+2m, +1, jj*8+8+2m, +1)
                db[0] = make_float2(v.x, v.y);
                db[1] = make_float2(v.z, v.w);
            } else {
                db[0] = *(const float2 *) (dbb + jj*8 + 2*m);
                db[1] = *(const float2 *) (dbb + jj*8 + 8 + 2*m);
            }
        };
        auto mma_epilogue = [&](const unsigned (&a)[MT_I][4], const float (&da)[MT_I][2], const float (&nmda)[MT_I][2],
                                const unsigned (&bf)[4], const float2 (&db)[2], const int jj) {
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const int j = jj + t;
#pragma unroll
                for (int n = 0; n < MT_I; ++n) {
                    int d[4];
                    mma_s8_bias(d, a[n], bf[2*t], bf[2*t + 1], cbias);
                    // t = fl(16*dot*dA) = 16*fl(dot*dA) exactly; sum = fl(t*(dB/16) + sum) = fl(fl(dot*dA)*dB + sum): MMQ's sequence
                    const float t0 = fmaf(__int_as_float(d[0]), da[n][0], nmda[n][0]);
                    const float t1 = fmaf(__int_as_float(d[1]), da[n][0], nmda[n][0]);
                    const float t2 = fmaf(__int_as_float(d[2]), da[n][1], nmda[n][1]);
                    const float t3 = fmaf(__int_as_float(d[3]), da[n][1], nmda[n][1]);
                    sum[n][j][0] = fmaf(t0, db[t].x, sum[n][j][0]);
                    sum[n][j][1] = fmaf(t1, db[t].y, sum[n][j][1]);
                    sum[n][j][2] = fmaf(t2, db[t].x, sum[n][j][2]);
                    sum[n][j][3] = fmaf(t3, db[t].y, sum[n][j][3]);
                }
            }
        };
        if constexpr (!PIPE && !BPF) {
#pragma unroll UNR
            for (int b = 0; b < BK/32; ++b) {
                jit_db(b);
                __syncwarp();
                unsigned a[MT_I][4];
                float da[MT_I][2], nmda[MT_I][2];
                load_a(b, a, da, nmda);
#pragma unroll
                for (int jj = 0; jj < MT_J; jj += 2) {
                    unsigned bf[4]; float2 db[2];
                    load_b(b, jj, bf, db);
                    mma_epilogue(a, da, nmda, bf, db, jj);
                }
            }
        } else if constexpr (BPF) {
            // only the B fragments one tile pair ahead (4 registers), A and the scales as in the plain loop
#pragma unroll
            for (int b = 0; b < BK/32; ++b) {
                jit_db(b);
                __syncwarp();
                unsigned a[MT_I][4];
                float da[MT_I][2], nmda[MT_I][2];
                load_a(b, a, da, nmda);
                unsigned bf[2][4]; float2 db[2][2];
                load_b(b, 0, bf[0], db[0]);
#pragma unroll
                for (int jj = 0; jj < MT_J; jj += 2) {
                    const int pc = (jj/2) & 1;
                    if (jj + 2 < MT_J) { load_b(b, jj + 2, bf[pc ^ 1], db[pc ^ 1]); }
                    mma_epilogue(a, da, nmda, bf[pc], db[pc], jj);
                }
            }
        } else {
            // operands one step ahead: block b+1's dB slice and A fragments during block b, the next tile pair's B fragments
            // during the current pair's MMAs. Same instructions in the same K order: the rounding sequence is unchanged.
            unsigned a[2][MT_I][4];
            float da[2][MT_I][2], nmda[2][MT_I][2];
            unsigned bf[2][4];
            float2 db[2][2];
            jit_db(0);
            __syncwarp();
            load_a(0, a[0], da[0], nmda[0]);
            load_b(0, 0, bf[0], db[0]);
#pragma unroll
            for (int b = 0; b < BK/32; ++b) {
                const int cur = b & 1;
                if (b + 1 < BK/32) { jit_db(b + 1); __syncwarp(); load_a(b + 1, a[cur ^ 1], da[cur ^ 1], nmda[cur ^ 1]); }
#pragma unroll
                for (int jj = 0; jj < MT_J; jj += 2) {
                    const int pc = (jj/2) & 1;
                    if (jj + 2 < MT_J)   { load_b(b, jj + 2, bf[pc ^ 1], db[pc ^ 1]); }
                    else if (b + 1 < BK/32) { load_b(b + 1, 0, bf[pc ^ 1], db[pc ^ 1]); }
                    mma_epilogue(a[cur], da[cur], nmda[cur], bf[pc], db[pc], jj);
                }
            }
        }
        // this warp is done with the buffer; the last warp to say so re-arms it with stage ks + NSTAGE
        __syncwarp();
        __threadfence_block();                                      // our reads of the buffer precede the ticket (release)
        if (!RESIDENT && lane == 0) {
            const int t = atomicAdd(ticket + buf, 1);
            if (t == NW - 1) {
                ticket[buf] = 0;
                fence_proxy_async();                                // our generic-proxy reads before the async-proxy writes
                if (ks + NSTAGE < nsteps) { issue(ks + NSTAGE); }
            }
        }
    }

    // ---- write back through shared memory, 16-byte coalesced stores, one token's BM rows contiguous
    constexpr int CS = BM + 4;
    constexpr int BNH = (BN*CS*4 <= NSTAGE*STAGE_BYTES) ? BN : WN;
    static_assert(BNH*CS*4 <= NSTAGE*STAGE_BYTES, "staging slab must fit the ring");
    float * C = (float *) smem;
#pragma unroll
    for (int h = 0; h < BN; h += BNH) {
        __syncthreads();
        if (wj*WN >= h && wj*WN < h + BNH) {
#pragma unroll
            for (int n = 0; n < MT_I; ++n) {
#pragma unroll
                for (int j = 0; j < MT_J; ++j) {
#pragma unroll
                    for (int l = 0; l < 4; ++l) {
                        C[(wj*WN - h + j*8 + 2*m + (l & 1))*CS + wi*WM + n*16 + r + 8*(l >> 1)] = sum[n][j][l];
                    }
                }
            }
        }
        __syncthreads();
        for (int idx = tid; idx < BNH*(BM/4); idx += NT) {
            const int col = idx / (BM/4), q = idx - col*(BM/4);
            if (j_base + h + col < N) {
                *(float4 *) (dst + (size_t) (j_base + h + col)*stride_dst + i_base + 4*q) = *(const float4 *) (C + col*CS + 4*q);
            }
        }
    }
}


// ---------------------------------------------------------------- host side
namespace {
// the configuration measured best on sm_121: 128 x 128 tile, 8 warps of 32 x 64, 3 stages, scale
// slices in fragment order, B fragments one tile pair ahead
constexpr int Q40MMA_BM = 128, Q40MMA_BN = 128, Q40MMA_WN = 64, Q40MMA_NSTAGE = 3, Q40MMA_NW = 8, Q40MMA_WM = 32;
constexpr bool Q40MMA_PIPE = false, Q40MMA_SCL = true, Q40MMA_BPF = true, Q40MMA_RES = false;
constexpr int Q40MMA_SMEM = Q40MMA_NSTAGE*(Q40MMA_BM*q4k::A_ROW + Q40MMA_BN*q4k::B_ROW) + Q40MMA_NW*2*(Q40MMA_WN + 2*Q40MMA_WM)*4 + Q40MMA_NSTAGE*12 + 64;

typedef CUresult (*encode_tiled_t)(CUtensorMap *, CUtensorMapDataType, cuuint32_t, void *, const cuuint64_t *, const cuuint64_t *,
                                   const cuuint32_t *, const cuuint32_t *, CUtensorMapInterleave, CUtensorMapSwizzle,
                                   CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

static encode_tiled_t q40mma_encode_tiled() {
    static encode_tiled_t fn = nullptr;
    static bool tried = false;
    if (!tried) {
        tried = true;
#if CUDART_VERSION >= 12050
        void * p = nullptr;
        cudaDriverEntryPointQueryResult q = cudaDriverEntryPointSymbolNotFound;
        if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q) == cudaSuccess &&
            q == cudaDriverEntryPointSuccess) {
            fn = (encode_tiled_t) p;
        } else {
            GGML_LOG_WARN("%s: cuTensorMapEncodeTiled unavailable, the Q4_0 MMA kernel is off and MMQ runs\n", __func__);
        }
#endif // CUDART_VERSION >= 12050
    }
    return fn;
}

// 2-D byte tensor: `rows` rows of `inner` valid bytes at pitch `pitch`; box of box_rows x box_inner bytes; out of bounds reads zero
static bool q40mma_map_2d(CUtensorMap & map, const void * base, uint64_t inner, uint64_t rows, uint64_t pitch, uint32_t box_inner, uint32_t box_rows) {
    const cuuint64_t dims[2] = { inner, rows };
    const cuuint64_t strides[1] = { pitch };
    const cuuint32_t box[2] = { box_inner, box_rows };
    const cuuint32_t estr[2] = { 1, 1 };
    const CUresult res = q40mma_encode_tiled()(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void *>(base), dims, strides, box, estr,
                                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return res == CUDA_SUCCESS;
}

// the weight maps, one per (tensor address, shape): weights do not move during a run, the activation buffer does
struct q40mma_weight_key {
    const void * x; int64_t ne00, ne01, nb01;
    bool operator==(const q40mma_weight_key & o) const { return x == o.x && ne00 == o.ne00 && ne01 == o.ne01 && nb01 == o.nb01; }
};
struct q40mma_weight_key_hash {
    size_t operator()(const q40mma_weight_key & k) const { return std::hash<const void *>()(k.x) ^ (size_t) k.ne00 * 31u ^ (size_t) k.ne01 * 131u ^ (size_t) k.nb01 * 1031u; }
};
static std::mutex q40mma_maps_mutex;
static std::unordered_map<q40mma_weight_key, CUtensorMap, q40mma_weight_key_hash> q40mma_maps;
} // namespace

bool ggml_cuda_mmq_q40_mma_supported(const int cc, const size_t smpbo, const int64_t ne00, const int64_t ne01, const int64_t ne11) {
    if (!blackwell_mma_available(cc) || smpbo < (size_t) Q40MMA_SMEM) {
        return false;
    }
    if (ne00 % q4k::BK != 0 || ne01 % Q40MMA_BM != 0 || ne11 < 64) {
        return false;
    }
    return q40mma_encode_tiled() != nullptr;
}

void ggml_cuda_mmq_q40_mma(const char * x, const char * y, float * dst,
                           const int64_t ne00, const int64_t ne01, const int64_t ne11, const int64_t stride_dst, const int64_t nb01, cudaStream_t stream) {
    GGML_ASSERT(ne00 % q4k::BK == 0 && ne01 % Q40MMA_BM == 0 && ne11 >= 1);
    GGML_ASSERT(((uintptr_t) x) % 16 == 0 && ((uintptr_t) y) % 16 == 0 && nb01 % 16 == 0);
    CUtensorMap tmA, tmB;
    {
        const q40mma_weight_key key = { x, ne00, ne01, nb01 };
        std::lock_guard<std::mutex> lock(q40mma_maps_mutex);
        auto it = q40mma_maps.find(key);
        if (it == q40mma_maps.end()) {
            CUtensorMap m;
            const bool ok = q40mma_map_2d(m, x, (uint64_t) (ne00/QK4_0)*sizeof(block_q4_0), (uint64_t) ne01, (uint64_t) nb01, q4k::A_ROW, Q40MMA_BM);
            GGML_ASSERT(ok && "cuTensorMapEncodeTiled failed for the Q4_0 weight");
            it = q40mma_maps.emplace(key, m).first;
        }
        tmA = it->second;
    }
    const bool okB = q40mma_map_2d(tmB, y, sizeof(block_q8_1_mmq), (uint64_t) (ne00/q4k::BK)*ne11, sizeof(block_q8_1_mmq), q4k::B_ROW, Q40MMA_BN);
    GGML_ASSERT(okB && "cuTensorMapEncodeTiled failed for the Q8_1 operand");

    CUDA_SET_SHARED_MEMORY_LIMIT((q4_0_q8_1_gemm_tma<Q40MMA_BM, Q40MMA_BN, Q40MMA_WN, Q40MMA_NSTAGE, Q40MMA_NW, Q40MMA_PIPE, Q40MMA_SCL, Q40MMA_BPF, Q40MMA_RES>), Q40MMA_SMEM);
    const int grid = (int) ((ne01/Q40MMA_BM) * ((ne11 + Q40MMA_BN - 1)/Q40MMA_BN));
    q4_0_q8_1_gemm_tma<Q40MMA_BM, Q40MMA_BN, Q40MMA_WN, Q40MMA_NSTAGE, Q40MMA_NW, Q40MMA_PIPE, Q40MMA_SCL, Q40MMA_BPF, Q40MMA_RES><<<grid, 32*Q40MMA_NW, Q40MMA_SMEM, stream>>>
        (tmA, tmB, dst, (int) ne01, (int) ne11, (int) ne00, stride_dst);
    CUDA_CHECK(cudaGetLastError());
}
#else // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
bool ggml_cuda_mmq_q40_mma_supported(int, size_t, int64_t, int64_t, int64_t) {
    return false;
}

void ggml_cuda_mmq_q40_mma(const char *, const char *, float *, int64_t, int64_t, int64_t, int64_t, int64_t, const float *, int64_t, cudaStream_t) {
    GGML_ABORT("fatal error");
}
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
