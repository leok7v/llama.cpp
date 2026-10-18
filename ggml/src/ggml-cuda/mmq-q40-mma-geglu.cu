#include "mmq-q40-mma-geglu.cuh"
#include "mmq.cuh"
#include "unary.cuh"   // ggml_cuda_op_gelu_single, ggml_cuda_op_silu_single

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include <cuda.h>
#include <cstdint>
#include <mutex>
#include <unordered_map>

// Fused gate/up GLU GEMM: the gate and up projections of a GLU block for one 64-row x 128-token tile, both accumulated with the
// exact Q4_0 x Q8_1 MMA arithmetic of mmq-q40-mma.cu (int32 accumulator seeded with 0x4B400000, so the MMA output is the fp32 bit
// pattern of 12582912 + 16*dot and the per-block scaling is two FFMAs - MMQ's rounding sequence), then op(gate)*up (op = gelu or
// silu, the expression of unary_gated_op_kernel) quantized straight into the down projection's block_q8_1_mmq operand with
// quantize_mmq_q8_1's own code over the staged fp32 tiles (four consecutive rows per lane, eight lanes per 32-row block, xor
// shuffles at 4/2/1). One TMA stage carries the gate box, the up box and the activation box, one mbarrier per buffer, no block
// barrier in the main loop; 8 warps as 4 x 2, warp tile 32 rows x 32 tokens for gate AND up sharing the B fragments and dB.

namespace q4g {
constexpr int BK = 128;                            // K per stage (4 Q4_0 blocks)
constexpr int A_ROW = 80;                          // 72 raw bytes (4 blocks) + 8: 20-word stride is bank-conflict-free, 16-byte TMA box
constexpr int B_ROW = 144;                         // one block_q8_1_mmq per token per stage
constexpr int   MAGIC_I = 0x4B400000;
constexpr float MAGIC_F = 12582912.0f;
constexpr int NW = 8, NT = 32*NW, NSTAGE = 3;
constexpr int WM = 32, WN = 32;                     // warp tile: 32 rows x 32 tokens, for gate and for up
constexpr int MT_I = WM / 16, MT_J = WN / 8;        // 2 x 4 MMA tiles per matrix
// per-warp scale slices, double-buffered per K block, in fragment order: dB/16 for the warp's WN tokens, then the
// (dA, -M*dA) pairs for the warp's WM gate rows, then the same for its up rows
constexpr int SLICE_FLOATS = 2 * (WN + 2*WM + 2*WM);
// the CTA tile: BM rows x BN tokens, BM*BN == 8192 (8 warps of 32 x 32). Both orientations were measured (geglu_test.cu):
// 64 x 128 streams two 5 KB strided weight boxes and one 18 KB contiguous activation box per stage - the stage shape whose
// loads mmq-q40-mma.cu hides - where 128 x 64 exposed 15% of its time to the two 10 KB strided boxes.
template <int BM, int BN> struct cfg {
    static constexpr int WARPS_I = BM / WM, WARPS_J = BN / WN;
    static constexpr int A_BYTES = BM * A_ROW, B_BYTES = BN * B_ROW;
    static constexpr int STAGE_BYTES = 2*A_BYTES + B_BYTES;         // gate box, up box, activation box
    static constexpr int CS = BM + 4;                               // staging row stride in floats: conflict-free fragment stores
    static constexpr int SMEM = NSTAGE*STAGE_BYTES + NW*SLICE_FLOATS*4 + NSTAGE*8 + NSTAGE*4 + 64;
    static_assert(WARPS_I*WARPS_J == NW && BM % 32 == 0 && QK8_1_MMQ % BM == 0, "8 warps of 32 x 32; a row tile divides a K-block of the operand");
    static_assert(STAGE_BYTES % 128 == 0 && A_BYTES % 128 == 0 && B_BYTES % 128 == 0, "TMA destinations must be 128-byte aligned");
    static_assert(2*BN*CS*4 <= NSTAGE*STAGE_BYTES, "the two staged fp32 tiles must fit the ring");
    static_assert((BN*(BM/4)) % NT == 0 && (BM/4) % 8 == 0 && 32 % (BM/4) == 0, "the quantize pass: whole tokens per warp, 8 lanes per 32-row block");
};
constexpr int BM = 64, BN = 128;                    // the production tile
static_assert(sizeof(block_q4_0) == 18 && sizeof(block_q8_1_mmq) == B_ROW, "layouts");

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

// the gated unit's op on the gate value, the single-value functions unary_gated_op_kernel<op_gelu> / <op_silu> call
constexpr int OP_GELU = 0, OP_SILU = 1;
template <int OP> static __device__ __forceinline__ float glu_single(const float x) {
    static_assert(OP == OP_GELU || OP == OP_SILU, "OP: 0 gelu, 1 silu");
    if constexpr (OP == OP_GELU) {
        return ggml_cuda_op_gelu_single(x);
    } else {
        return ggml_cuda_op_silu_single(x);
    }
}
} // namespace q4g

// DIAG (timing only, the operand written is meaningless): 1 = stage 0 loaded once, the compute loop runs over it nsteps times
// with constant addresses (no loads, no waits); 2 = no loads either, but the compute cycles through the buffers and the 8-byte
// shift as the real kernel does (the address arithmetic alone); 3 = real loads and waits, compute on buffer 0 with constant
// addresses (the load exposure alone). Measured and rejected (geglu/README.md v4-v6): an L2 prefetch of the weight boxes
// stages ahead (+20%: a strided box costs the TMA unit the same line requests whether prefetched or loaded), 256-byte L2
// promotion (no change), the stage loop unrolled by two for an immediate shift (+2%).
// OP: the gated unit's op on the gate value - q4g::OP_GELU (GEGLU) or q4g::OP_SILU (SWIGLU); the epilogue computes op(gate) * up
template <int BM, int BN, int DIAG, int OP = q4g::OP_GELU>
__global__ void __launch_bounds__(q4g::NT, 1) q4_0_q8_1_geglu_tma(
        const __grid_constant__ CUtensorMap tmG, const __grid_constant__ CUtensorMap tmU, const __grid_constant__ CUtensorMap tmB,
        block_q8_1_mmq * __restrict__ out, const int N, const int K) {
    using namespace q4g;
    using C = cfg<BM, BN>;
    constexpr int WARPS_J = C::WARPS_J, A_BYTES = C::A_BYTES, STAGE_BYTES = C::STAGE_BYTES, CS = C::CS;
    constexpr bool LOADS = DIAG == 0 || DIAG == 3, REALADDR = DIAG == 0 || DIAG == 2;
    extern __shared__ __align__(128) char smem[];
    float    * slices = (float *) (smem + NSTAGE*STAGE_BYTES);
    uint64_t * full   = (uint64_t *) (slices + NW*SLICE_FLOATS);
    int      * ticket = (int *) (full + NSTAGE);

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wi = warp / WARPS_J, wj = warp % WARPS_J;
    const int ntx = (N + BN - 1) / BN;
    const int it = blockIdx.x / ntx, jt = blockIdx.x - it*ntx;
    const int i_base = it*BM, j_base = jt*BN;
    const int nsteps = K / BK;

    auto issue = [&](const int ks) {                                // one thread: the three boxes of stage ks into buffer ks % NSTAGE
        const int buf = ks % NSTAGE;
        char * stage = smem + buf*STAGE_BYTES;
        mbar_expect_tx(full + buf, STAGE_BYTES);
        tma_load_2d(stage,             &tmG, 72*ks - 8*(ks & 1), i_base,        full + buf);
        tma_load_2d(stage + A_BYTES,   &tmU, 72*ks - 8*(ks & 1), i_base,        full + buf);
        tma_load_2d(stage + 2*A_BYTES, &tmB, 0,                  ks*N + j_base, full + buf);
    };
    if (tid == 0) {
        for (int b = 0; b < NSTAGE; ++b) { mbar_init(full + b, 1); ticket[b] = 0; }
        fence_proxy_async();
    }
    __syncthreads();                                                // the only block-wide barrier before the epilogue
    if (tid == 0) {
        for (int s = 0; s < (LOADS ? NSTAGE : 1) && s < nsteps; ++s) { issue(s); }
    }

    int cbias[4];
#pragma unroll
    for (int l = 0; l < 4; ++l) { cbias[l] = MAGIC_I; asm volatile("" : "+r"(cbias[l])); }
    float sg[MT_I][MT_J][4], su[MT_I][MT_J][4];                     // gate and up accumulators for the same output coordinates
#pragma unroll
    for (int n = 0; n < MT_I; ++n) {
#pragma unroll
        for (int j = 0; j < MT_J; ++j) {
#pragma unroll
            for (int l = 0; l < 4; ++l) { sg[n][j][l] = 0.0f; su[n][j][l] = 0.0f; }
        }
    }
    float * slw = slices + warp*SLICE_FLOATS;
    const int r = lane >> 2, m = lane & 3;

    auto stage = [&](const int ks, const int ashift_in) {
        const int buf = ks % NSTAGE;                                // the stage's buffer: waited on, ticketed, re-armed
        if (LOADS || ks == 0) { mbar_wait(full + buf, (ks / NSTAGE) & 1); }
        const char * Ag = smem + (REALADDR ? buf : 0)*STAGE_BYTES;  // the buffer the compute reads
        const char * Au = Ag + A_BYTES;
        const char * Bs = Ag + 2*A_BYTES;
        const int ashift = REALADDR ? ashift_in : 0;                // the 72-byte chunk's offset inside the 80-byte row
        // just-in-time scales of K block b into slice (b & 1): dB/16 for this warp's WN tokens in fragment order (token t of the
        // warp tile -> tile pair t/16, tile (t/8)&1, column pair (t%8)/2, half t&1: the four values a lane needs sit together),
        // and (dA, -M*dA) for the warp's 32 gate rows and 32 up rows with rows (r, r+8) of a 16-row group adjacent
        auto jit = [&](const int b) {
            float * sl = slw + (b & 1)*(SLICE_FLOATS/2);
            const int tok  = wj*WN + lane;
            const int slot = (lane/16)*16 + ((lane%8)/2)*4 + ((lane/8)&1)*2 + (lane&1);
            sl[slot] = __low2float(*(const half2 *) (Bs + tok*B_ROW + 4*b)) * 0.0625f;
            const int row = wi*WM + lane;                           // WM == 32: one row per lane
            const int rs  = (lane & ~15) + ((lane & 7) << 1) + ((lane >> 3) & 1);
            const float dg = __half2float(*(const half *) (Ag + row*A_ROW + ashift + 18*b));
            const float du = __half2float(*(const half *) (Au + row*A_ROW + ashift + 18*b));
            *(float2 *) (sl + WN + 2*rs)        = make_float2(dg, -MAGIC_F * dg);
            *(float2 *) (sl + WN + 2*WM + 2*rs) = make_float2(du, -MAGIC_F * du);
        };
        // A fragments (nibbles expanded to 16*(n-8): the 1/16 is carried by dB) and the (dA, -M*dA) pairs of K block b
        auto load_a = [&](const int b, const char * As, const float * daw, unsigned (&a)[MT_I][4], float (&da)[MT_I][2], float (&nmda)[MT_I][2]) {
#pragma unroll
            for (int n = 0; n < MT_I; ++n) {
                const int row0 = wi*WM + n*16 + r;
                const unsigned * W0 = (const unsigned *) (As + row0*A_ROW + ashift);
                const unsigned * W1 = (const unsigned *) (As + (row0 + 8)*A_ROW + ashift);
                unsigned q0, q1;
                if ((b & 1) == 0) {                                 // qs start at byte 18b+2: straddles two words
                    const int w = (18*b)/4 + m;
                    q0 = __byte_perm(W0[w], W0[w + 1], 0x5432);
                    q1 = __byte_perm(W1[w], W1[w + 1], 0x5432);
                } else {                                            // 18b+2 is word aligned
                    const int w = (18*b + 2)/4 + m;
                    q0 = W0[w];
                    q1 = W1[w];
                }
                a[n][0] = ((q0 << 4) & 0xF0F0F0F0u) ^ 0x80808080u;
                a[n][1] = ((q1 << 4) & 0xF0F0F0F0u) ^ 0x80808080u;
                a[n][2] = ( q0       & 0xF0F0F0F0u) ^ 0x80808080u;
                a[n][3] = ( q1       & 0xF0F0F0F0u) ^ 0x80808080u;
                const float4 v = *(const float4 *) (daw + 2*((n*16) + 2*r));   // (da r, nmda r, da r+8, nmda r+8)
                da[n][0] = v.x; nmda[n][0] = v.y; da[n][1] = v.z; nmda[n][1] = v.w;
            }
        };
        auto load_b = [&](const int b, const int jj, unsigned (&bf)[4], float2 (&db)[2]) {
            const int jrow = wj*WN + jj*8 + (lane & 7) + ((lane >> 4) << 3);
            const int koff = 16 + 32*b + ((lane >> 3) & 1)*16;
            ldmatrix_x4(bf, Bs + jrow*B_ROW + koff);
            const float4 v = *(const float4 *) (slw + (b & 1)*(SLICE_FLOATS/2) + (jj/2)*16 + m*4);   // tokens (jj*8+2m, +1, jj*8+8+2m, +1)
            db[0] = make_float2(v.x, v.y);
            db[1] = make_float2(v.z, v.w);
        };
        // t = fl(16*dot*dA) = 16*fl(dot*dA) exactly; sum = fl(t*(dB/16) + sum) = fl(fl(dot*dA)*dB + sum): MMQ's sequence
        auto mma_pair = [&](const unsigned (&a)[MT_I][4], const float (&da)[MT_I][2], const float (&nmda)[MT_I][2],
                            const unsigned (&bf)[4], const float2 (&db)[2], const int jj, float (&sum)[MT_I][MT_J][4]) {
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const int j = jj + t;
#pragma unroll
                for (int n = 0; n < MT_I; ++n) {
                    int d[4];
                    mma_s8_bias(d, a[n], bf[2*t], bf[2*t + 1], cbias);
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
#pragma unroll
        for (int b = 0; b < BK/32; ++b) {
            jit(b);
            __syncwarp();
            const float * sl = slw + (b & 1)*(SLICE_FLOATS/2);
            unsigned ag[MT_I][4], au[MT_I][4];
            float dag[MT_I][2], nmdag[MT_I][2], dau[MT_I][2], nmdau[MT_I][2];
            load_a(b, Ag, sl + WN,        ag, dag, nmdag);
            load_a(b, Au, sl + WN + 2*WM, au, dau, nmdau);
            unsigned bf[2][4]; float2 db[2][2];                     // B fragments one tile pair ahead, shared by gate and up
            load_b(b, 0, bf[0], db[0]);
#pragma unroll
            for (int jj = 0; jj < MT_J; jj += 2) {
                const int pc = (jj/2) & 1;
                if (jj + 2 < MT_J) { load_b(b, jj + 2, bf[pc ^ 1], db[pc ^ 1]); }
                mma_pair(ag, dag, nmdag, bf[pc], db[pc], jj, sg);
                mma_pair(au, dau, nmdau, bf[pc], db[pc], jj, su);
            }
        }
        // this warp is done with the buffer; the last warp to say so re-arms it with stage ks + NSTAGE
        __syncwarp();
        __threadfence_block();                                      // our reads of the buffer precede the ticket (release)
        if (LOADS && lane == 0) {
            const int t = atomicAdd(ticket + buf, 1);
            if (t == NW - 1) {
                ticket[buf] = 0;
                fence_proxy_async();                                // our generic-proxy reads before the async-proxy writes
                if (ks + NSTAGE < nsteps) { issue(ks + NSTAGE); }
            }
        }
    };
    for (int ks = 0; ks < nsteps; ++ks) { stage(ks, 8*(ks & 1)); }

    // ---- epilogue: the fp32 gate and up tiles staged in the ring as [token][row], then quantize_mmq_q8_1_fused<DS4, GEGLU | SWIGLU>'s
    // own code over them: a thread takes four consecutive rows of one token, eight lanes one 32-row Q8_1 block, the amax and
    // the block sum reduced over the eight lanes by xor shuffles at offsets 4, 2, 1 - so the operand is that kernel's, bit for bit
    __syncthreads();                                                // every warp has consumed every stage: the ring is free
    float * Cg = (float *) smem;
    float * Cu = Cg + BN*CS;
#pragma unroll
    for (int n = 0; n < MT_I; ++n) {
#pragma unroll
        for (int j = 0; j < MT_J; ++j) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int idx = (wj*WN + j*8 + 2*m + (l & 1))*CS + wi*WM + n*16 + r + 8*(l >> 1);
                Cg[idx] = sg[n][j][l];
                Cu[idx] = su[n][j][l];
            }
        }
    }
    __syncthreads();
    constexpr int F4 = BM/4;                                        // float4 per token in this tile (32 or 16)
    const int kb  = i_base / QK8_1_MMQ;                             // this tile's K-block of the down operand ...
    const int sub = (i_base % QK8_1_MMQ) / 32;                      // ... and its first 32-row sub-block inside it
#pragma unroll 1
    for (int pass = 0; pass < BN*F4/NT; ++pass) {                   // a warp covers 32/F4 whole tokens; float4 q = rows 4q .. 4q+3
        const int tok = pass*(NT/F4) + tid/F4, q = tid % F4;
        float4 xi = *(const float4 *) (Cg + tok*CS + 4*q);
        const float4 gi = *(const float4 *) (Cu + tok*CS + 4*q);
        xi.x = glu_single<OP>(xi.x) * gi.x;                         // unary_gated_op_kernel<op_gelu | op_silu>: op(x) * g
        xi.y = glu_single<OP>(xi.y) * gi.y;
        xi.z = glu_single<OP>(xi.z) * gi.z;
        xi.w = glu_single<OP>(xi.w) * gi.w;
        float amax = fabsf(xi.x);
        amax = fmaxf(amax, fabsf(xi.y));
        amax = fmaxf(amax, fabsf(xi.z));
        amax = fmaxf(amax, fabsf(xi.w));
#pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
        }
        float sum = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
        }
        const float d_inv = 127.0f / amax;
        char4 qc;
        qc.x = roundf(xi.x*d_inv);
        qc.y = roundf(xi.y*d_inv);
        qc.z = roundf(xi.z*d_inv);
        qc.w = roundf(xi.w*d_inv);
        const float d = 1.0f / d_inv;
        if (j_base + tok < N) {
            block_q8_1_mmq & y = out[(size_t) kb*N + j_base + tok];
            ((char4 *) y.qs)[sub*8 + q] = qc;
            if (q % 8 == 0) {
                y.ds4[sub + q/8] = make_half2(d, sum);
            }
        }
    }
}

// ---------------------------------------------------------------- host side
namespace {
typedef CUresult (*q4g_encode_tiled_t)(CUtensorMap *, CUtensorMapDataType, cuuint32_t, void *, const cuuint64_t *, const cuuint64_t *,
                                       const cuuint32_t *, const cuuint32_t *, CUtensorMapInterleave, CUtensorMapSwizzle,
                                       CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

static q4g_encode_tiled_t q4g_encode_tiled() {
    static q4g_encode_tiled_t fn = nullptr;
    static bool tried = false;
    if (!tried) {
        tried = true;
#if CUDART_VERSION >= 12050
        void * p = nullptr;
        cudaDriverEntryPointQueryResult q = cudaDriverEntryPointSymbolNotFound;
        if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q) == cudaSuccess &&
            q == cudaDriverEntryPointSuccess) {
            fn = (q4g_encode_tiled_t) p;
        } else {
            GGML_LOG_WARN("%s: cuTensorMapEncodeTiled unavailable, the fused gate/up GLU GEMM is off and the separate GEMMs run\n", __func__);
        }
#endif // CUDART_VERSION >= 12050
    }
    return fn;
}

// 2-D byte tensor: `rows` rows of `inner` valid bytes at pitch `pitch`; box of box_rows x box_inner bytes; out of bounds reads zero
static bool q4g_map_2d(CUtensorMap & map, const void * base, uint64_t inner, uint64_t rows, uint64_t pitch, uint32_t box_inner, uint32_t box_rows) {
    const cuuint64_t dims[2] = { inner, rows };
    const cuuint64_t strides[1] = { pitch };
    const cuuint32_t box[2] = { box_inner, box_rows };
    const cuuint32_t estr[2] = { 1, 1 };
    const CUresult res = q4g_encode_tiled()(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void *>(base), dims, strides, box, estr,
                                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return res == CUDA_SUCCESS;
}

// the weight maps, one per (tensor address, shape): weights do not move during a run, the activation buffer does
struct q4g_weight_key {
    const void * x; int64_t K, M, nb01;
    bool operator==(const q4g_weight_key & o) const { return x == o.x && K == o.K && M == o.M && nb01 == o.nb01; }
};
struct q4g_weight_key_hash {
    size_t operator()(const q4g_weight_key & k) const { return std::hash<const void *>()(k.x) ^ (size_t) k.K * 31u ^ (size_t) k.M * 131u ^ (size_t) k.nb01 * 1031u; }
};
static std::mutex q4g_maps_mutex;
static std::unordered_map<q4g_weight_key, CUtensorMap, q4g_weight_key_hash> q4g_maps;

static CUtensorMap q4g_weight_map(const char * x, const int64_t K, const int64_t M, const int64_t nb01) {
    const q4g_weight_key key = { x, K, M, nb01 };
    std::lock_guard<std::mutex> lock(q4g_maps_mutex);
    auto it = q4g_maps.find(key);
    if (it == q4g_maps.end()) {
        CUtensorMap m;
        const bool ok = q4g_map_2d(m, x, (uint64_t) (K/QK4_0)*sizeof(block_q4_0), (uint64_t) M, (uint64_t) nb01, q4g::A_ROW, q4g::BM);
        GGML_ASSERT(ok && "cuTensorMapEncodeTiled failed for a Q4_0 weight of the gate/up pair");
        it = q4g_maps.emplace(key, m).first;
    }
    return it->second;
}

// the kernel's OP for a ggml GLU op: the two gated units whose single-value functions the epilogue carries; -1 for any other
static int q4g_op_index(const ggml_glu_op op) {
    switch (op) {
        case GGML_GLU_OP_GEGLU:  return q4g::OP_GELU;
        case GGML_GLU_OP_SWIGLU: return q4g::OP_SILU;
        default:                 return -1;
    }
}

template <int OP>
static void q4g_launch(const CUtensorMap & tmG, const CUtensorMap & tmU, const CUtensorMap & tmB, void * operand_out,
                       const int K, const int M, const int N, cudaStream_t stream) {
    constexpr int SMEM = q4g::cfg<q4g::BM, q4g::BN>::SMEM;
    CUDA_SET_SHARED_MEMORY_LIMIT((q4_0_q8_1_geglu_tma<q4g::BM, q4g::BN, 0, OP>), SMEM);
    const int grid = (M/q4g::BM) * ((N + q4g::BN - 1)/q4g::BN);
    q4_0_q8_1_geglu_tma<q4g::BM, q4g::BN, 0, OP><<<grid, q4g::NT, SMEM, stream>>>(tmG, tmU, tmB, (block_q8_1_mmq *) operand_out, N, K);
}
} // namespace

bool ggml_cuda_mmq_q40_mma_glu_supported(const ggml_glu_op op, const int cc, const size_t smpbo, const int64_t K, const int64_t M, const int64_t N) {
    if (q4g_op_index(op) < 0) {
        return false;
    }
    if (!blackwell_mma_available(cc) || smpbo < (size_t) q4g::cfg<q4g::BM, q4g::BN>::SMEM) {
        return false;
    }
    if (K % (2*q4g::BK) != 0 || M % MATRIX_ROW_PADDING != 0 || N < 64) {   // K % 256: the Q4_0 row pitch 72*K/128 must be a multiple of 16 for TMA
        return false;
    }
    return q4g_encode_tiled() != nullptr;
}

bool ggml_cuda_mmq_q40_mma_geglu_supported(const int cc, const size_t smpbo, const int64_t K, const int64_t M, const int64_t N) {
    return ggml_cuda_mmq_q40_mma_glu_supported(GGML_GLU_OP_GEGLU, cc, smpbo, K, M, N);
}

void ggml_cuda_mmq_q40_mma_geglu(const char * x_gate, const char * x_up, const char * y_q8_1_mmq, void * operand_out,
                                 const int64_t K, const int64_t M, const int64_t N, const int64_t nb01_gate, const int64_t nb01_up, cudaStream_t stream) {
    ggml_cuda_mmq_q40_mma_glu(GGML_GLU_OP_GEGLU, x_gate, x_up, y_q8_1_mmq, operand_out, K, M, N, nb01_gate, nb01_up, stream);
}

void ggml_cuda_mmq_q40_mma_glu(const ggml_glu_op op, const char * x_gate, const char * x_up, const char * y_q8_1_mmq, void * operand_out,
                               const int64_t K, const int64_t M, const int64_t N, const int64_t nb01_gate, const int64_t nb01_up, cudaStream_t stream) {
    const int opi = q4g_op_index(op);
    GGML_ASSERT(opi >= 0 && "ggml_cuda_mmq_q40_mma_glu: only GEGLU and SWIGLU");
    GGML_ASSERT(K % q4g::BK == 0 && M % q4g::BM == 0 && N >= 1);
    GGML_ASSERT(((uintptr_t) x_gate) % 16 == 0 && ((uintptr_t) x_up) % 16 == 0 && ((uintptr_t) y_q8_1_mmq) % 16 == 0 &&
                ((uintptr_t) operand_out) % 16 == 0 && nb01_gate % 16 == 0 && nb01_up % 16 == 0);
    const CUtensorMap tmG = q4g_weight_map(x_gate, K, M, nb01_gate);
    const CUtensorMap tmU = q4g_weight_map(x_up,   K, M, nb01_up);
    CUtensorMap tmB;
    const bool okB = q4g_map_2d(tmB, y_q8_1_mmq, sizeof(block_q8_1_mmq), (uint64_t) (K/q4g::BK)*N, sizeof(block_q8_1_mmq), q4g::B_ROW, q4g::BN);
    GGML_ASSERT(okB && "cuTensorMapEncodeTiled failed for the Q8_1 operand");

    if (opi == q4g::OP_SILU) {
        q4g_launch<q4g::OP_SILU>(tmG, tmU, tmB, operand_out, (int) K, (int) M, (int) N, stream);
    } else {
        q4g_launch<q4g::OP_GELU>(tmG, tmU, tmB, operand_out, (int) K, (int) M, (int) N, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}
#else // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
bool ggml_cuda_mmq_q40_mma_glu_supported(ggml_glu_op, int, size_t, int64_t, int64_t, int64_t) {
    return false;
}

void ggml_cuda_mmq_q40_mma_glu(ggml_glu_op, const char *, const char *, const char *, void *, int64_t, int64_t, int64_t, int64_t, int64_t, cudaStream_t) {
    GGML_ABORT("fatal error");
}

bool ggml_cuda_mmq_q40_mma_geglu_supported(int, size_t, int64_t, int64_t, int64_t) {
    return false;
}

void ggml_cuda_mmq_q40_mma_geglu(const char *, const char *, const char *, void *, int64_t, int64_t, int64_t, int64_t, int64_t, cudaStream_t) {
    GGML_ABORT("fatal error");
}
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
