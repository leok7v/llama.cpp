#include "mmq-q80-mma.cuh"
#include "mmq.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include <cuda.h>
#include <cstdint>
#include <mutex>
#include <unordered_map>

// Exact Q8_0 x Q8_1 GEMM on the INT8 tensor cores of sm_120/121 for a batch of tokens - the sibling of mmq-q40-mma.cu for Q8_0
// weights. Same scheme: raw block_q8_0 rows streamed by TMA (a 144-byte box per 128-wide K stage), the D4 fp32 scale read directly,
// mma.sync.m16n8k32 s8 into an int32 accumulator seeded with 0x4B400000, the epilogue fma(fl(C*dA), dB, sum) in K order - MMQ's
// rounding sequence, bit for bit. Three weight buffers and two activation buffers per CTA. Dense 2-D only; otherwise MMQ runs.

namespace q8k {
constexpr int BK = 128;                            // K per stage (4 Q8_0 blocks)
constexpr int A_RAW = 4 * 34;                      // 136 raw bytes per row per stage
constexpr int A_ROW = 144;                         // + 8: 36-word stride (conflict-free), 16-byte TMA box
constexpr int B_ROW = 144;                         // one block_q8_1_mmq per token per stage
constexpr int   MAGIC_I = 0x4B400000;
constexpr float MAGIC_F = 12582912.0f;
static_assert(sizeof(block_q8_0) == 34 && sizeof(block_q8_1_mmq) == B_ROW, "layouts");

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

// the shared-memory plan of one configuration: NSA weight buffers, NSB activation buffers, per-warp scale slices in fragment
// order (dB for the warp's WN tokens, then (dA, -M*dA) pairs for its WM rows, double-buffered per K block), NB mbarriers + tickets
template <int BM, int BN, int WN, int NSA, int NSB, int NW> struct cfg {
    static constexpr int WARPS_J = BN / WN, WARPS_I = NW / WARPS_J;
    static constexpr int WM = BM / WARPS_I;
    static constexpr int MT_I = WM / 16, MT_J = WN / 8;
    static constexpr int NB = NSA > NSB ? NSA : NSB;
    static constexpr int A_BYTES = BM * A_ROW, B_BYTES = BN * B_ROW;
    static constexpr int RING_BYTES = NSA*A_BYTES + NSB*B_BYTES;
    static constexpr int SLICE_FLOATS = 2 * (WN + 2*WM);
    static constexpr int SMEM = RING_BYTES + NW*SLICE_FLOATS*4 + NB*8 + NB*4 + 64;
    static_assert(WARPS_J*WARPS_I == NW && WM == 32 && MT_J % 2 == 0 && WN <= 64, "warp tiling: WM 32 (one scale row per lane), WN 32 or 64");
    static_assert(A_BYTES % 128 == 0 && B_BYTES % 128 == 0, "TMA destinations must be 128-byte aligned");
    static_assert(NSA >= 1 && NSB >= 1 && NSA >= NSB, "the weight ring is the deeper one");
};
} // namespace q8k

// One thread issues the TMA boxes; each warp waits on the stage's mbarrier for itself, computes its 32 x WN tile over the four
// K blocks of the stage with just-in-time scale slices, and takes a ticket; the warp that takes the last ticket re-arms the
// stage's two buffers with the stages NSA / NSB ahead. BPF: the B fragments one tile pair ahead (4 registers). RESIDENT: the
// timing diagnostic - stage 0 loaded once and computed nsteps times with constant addresses (results meaningless).
template <int BM, int BN, int WN, int NSA, int NSB, int NW, bool BPF = true, bool RESIDENT = false, int MINB = 1>
__global__ void __launch_bounds__(32*NW, MINB) q8_0_q8_1_gemm_tma(
        const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB,
        float * __restrict__ dst, const int M, const int N, const int K, const int64_t stride_dst,
        const float * __restrict__ res, const int64_t stride_res) {
    using namespace q8k;
    using C = cfg<BM, BN, WN, NSA, NSB, NW>;
    constexpr int NT = 32*NW;
    constexpr int WARPS_J = C::WARPS_J, WM = C::WM, MT_I = C::MT_I, MT_J = C::MT_J, NB = C::NB;
    constexpr int A_BYTES = C::A_BYTES, B_BYTES = C::B_BYTES, SLICE_FLOATS = C::SLICE_FLOATS;
    extern __shared__ __align__(128) char smem[];
    char     * Abuf   = smem;
    char     * Bbuf   = smem + NSA*A_BYTES;
    float    * slices = (float *) (smem + C::RING_BYTES);
    uint64_t * full   = (uint64_t *) (slices + NW*SLICE_FLOATS);
    int      * ticket = (int *) (full + NB);

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wi = warp / WARPS_J, wj = warp % WARPS_J;
    const int ntx = (N + BN - 1) / BN;
    const int it = blockIdx.x / ntx, jt = blockIdx.x - it*ntx;
    const int i_base = it*BM, j_base = jt*BN;
    const int nsteps = K / BK;
    GGML_UNUSED(M);

    // the weight box of stage ks into weight buffer ks % NSA (one arrival on the stage's barrier), the activation box into
    // activation buffer ks % NSB (the other arrival); both are 16-byte aligned on the global side
    auto issue_a = [&](const int ks) {
        uint64_t * bar = full + ks % NB;
        mbar_expect_tx(bar, A_BYTES);
        tma_load_2d(Abuf + (ks % NSA)*A_BYTES, &tmA, A_RAW*ks - 8*(ks & 1), i_base, bar);
    };
    auto issue_b = [&](const int ks) {
        uint64_t * bar = full + ks % NB;
        mbar_expect_tx(bar, B_BYTES);
        tma_load_2d(Bbuf + (ks % NSB)*B_BYTES, &tmB, 0, ks*N + j_base, bar);
    };
    if (tid == 0) {
        for (int b = 0; b < NB; ++b) { mbar_init(full + b, 2); ticket[b] = 0; }
        fence_proxy_async();
    }
    __syncthreads();                                                // the only block-wide barrier before the write-back
    if (tid == 0) {
        for (int s = 0; s < (RESIDENT ? 1 : NSA) && s < nsteps; ++s) { issue_a(s); }
        for (int s = 0; s < (RESIDENT ? 1 : NSB) && s < nsteps; ++s) { issue_b(s); }
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
    float * slw = slices + warp*SLICE_FLOATS;
    const int r = lane >> 2, m = lane & 3;

    for (int ks = 0; ks < nsteps; ++ks) {
        const int slot = ks % NB;
        if (!RESIDENT || ks == 0) { mbar_wait(full + slot, (ks / NB) & 1); }
        const char * As = Abuf + (RESIDENT ? 0 : ks % NSA)*A_BYTES;
        const char * Bs = Bbuf + (RESIDENT ? 0 : ks % NSB)*B_BYTES;
        const int ashift = RESIDENT ? 0 : 8*(ks & 1);               // the 136-byte chunk's offset inside the 144-byte row
        // just-in-time scales of K block b into slice (b & 1): dB for this warp's WN tokens in fragment order (token t of the
        // warp tile -> tile pair t/16, tile (t/8)&1, column pair (t%8)/2, half t&1: the four values a lane needs sit together),
        // and the (dA, -M*dA) pairs of the warp's 32 rows with rows (r, r+8) of a 16-row group adjacent
        auto jit = [&](const int b) {
            float * sl = slw + (b & 1)*(SLICE_FLOATS/2);
            const int tok = wj*WN + lane;
            auto fslot = [](int t) { return (t/16)*16 + ((t%8)/2)*4 + ((t/8)&1)*2 + (t&1); };
            sl[fslot(lane)] = *(const float *) (Bs + tok*B_ROW + 4*b);
            if (WN > 32) { sl[fslot(lane + 32)] = *(const float *) (Bs + (tok + 32)*B_ROW + 4*b); }
            const int row = wi*WM + lane;                           // WM == 32: one row per lane
            const float d = __half2float(*(const half *) (As + row*A_ROW + ashift + 34*b));
            const int rs = (lane & ~15) + ((lane & 7) << 1) + ((lane >> 3) & 1);
            *(float2 *) (sl + WN + 2*rs) = make_float2(d, -MAGIC_F * d);
        };
        // A fragments of K block b: lane (r, m) holds k = 4m..4m+3 (a[0] row r, a[1] row r+8) and k = 16+4m.. (a[2], a[3]);
        // the block's qs start at byte 2 + 34*b: a half-word straddle on even b (two words, one byte permute), aligned on odd b
        auto load_a = [&](const int b, unsigned (&a)[MT_I][4], float (&da)[MT_I][2], float (&nmda)[MT_I][2]) {
#pragma unroll
            for (int n = 0; n < MT_I; ++n) {
                const int row0 = wi*WM + n*16 + r;
                const unsigned * W0 = (const unsigned *) (As + row0*A_ROW + ashift);
                const unsigned * W1 = (const unsigned *) (As + (row0 + 8)*A_ROW + ashift);
                const int w = (34*b + 2)/4 + m;
                if ((b & 1) == 0) {
                    a[n][0] = __byte_perm(W0[w],     W0[w + 1], 0x5432);
                    a[n][1] = __byte_perm(W1[w],     W1[w + 1], 0x5432);
                    a[n][2] = __byte_perm(W0[w + 4], W0[w + 5], 0x5432);
                    a[n][3] = __byte_perm(W1[w + 4], W1[w + 5], 0x5432);
                } else {
                    a[n][0] = W0[w];
                    a[n][1] = W1[w];
                    a[n][2] = W0[w + 4];
                    a[n][3] = W1[w + 4];
                }
                const float * daw = slw + (b & 1)*(SLICE_FLOATS/2) + WN;
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
        // t = fl(dot*dA) exactly (the magic bias cancels in the fma); sum = fl(t*dB + sum): MMQ's sequence
        auto mma_epilogue = [&](const unsigned (&a)[MT_I][4], const float (&da)[MT_I][2], const float (&nmda)[MT_I][2],
                                const unsigned (&bf)[4], const float2 (&db)[2], const int jj) {
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
            unsigned a[MT_I][4];
            float da[MT_I][2], nmda[MT_I][2];
            load_a(b, a, da, nmda);
            if constexpr (BPF) {
                unsigned bf[2][4]; float2 db[2][2];
                load_b(b, 0, bf[0], db[0]);
#pragma unroll
                for (int jj = 0; jj < MT_J; jj += 2) {
                    const int pc = (jj/2) & 1;
                    if (jj + 2 < MT_J) { load_b(b, jj + 2, bf[pc ^ 1], db[pc ^ 1]); }
                    mma_epilogue(a, da, nmda, bf[pc], db[pc], jj);
                }
            } else {
#pragma unroll
                for (int jj = 0; jj < MT_J; jj += 2) {
                    unsigned bf[4]; float2 db[2];
                    load_b(b, jj, bf, db);
                    mma_epilogue(a, da, nmda, bf, db, jj);
                }
            }
        }
        // this warp is done with the stage; the last warp to say so re-arms its two buffers with the stages NSA / NSB ahead
        __syncwarp();
        __threadfence_block();                                      // our reads of the buffers precede the ticket (release)
        if (!RESIDENT && lane == 0) {
            const int t = atomicAdd(ticket + slot, 1);
            if (t == NW - 1) {
                ticket[slot] = 0;
                fence_proxy_async();                                // our generic-proxy reads before the async-proxy writes
                if (ks + NSA < nsteps) { issue_a(ks + NSA); }
                if (ks + NSB < nsteps) { issue_b(ks + NSB); }
            }
        }
    }

    // ---- write back through shared memory, 16-byte coalesced stores, one token's BM rows contiguous
    constexpr int CS = BM + 4;
    constexpr int BNH = (BN*CS*4 <= C::RING_BYTES) ? BN : WN;
    static_assert(BNH*CS*4 <= C::RING_BYTES, "staging slab must fit the ring");
    float * Cst = (float *) smem;
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
                        Cst[(wj*WN - h + j*8 + 2*m + (l & 1))*CS + wi*WM + n*16 + r + 8*(l >> 1)] = sum[n][j][l];
                    }
                }
            }
        }
        __syncthreads();
        for (int idx = tid; idx < BNH*(BM/4); idx += NT) {
            const int col = idx / (BM/4), q = idx - col*(BM/4);
            if (j_base + h + col < N) {
                float4 v = *(const float4 *) (Cst + col*CS + 4*q);
                if (res) {   // the residual ADD that follows the matmul, folded in: the same fp32 sum the separate kernel makes
                    const float4 r = *(const float4 *) (res + (size_t) (j_base + h + col)*stride_res + i_base + 4*q);
                    v.x += r.x; v.y += r.y; v.z += r.z; v.w += r.w;
                }
                *(float4 *) (dst + (size_t) (j_base + h + col)*stride_dst + i_base + 4*q) = v;
            }
        }
    }
}


// ---------------------------------------------------------------- host side
namespace {
// the production configuration: 128 x 128 tile, 8 warps of 32 x 64, 3 weight buffers, 2 activation
// buffers, B fragments one tile pair ahead
constexpr int Q80MMA_BM = 128, Q80MMA_BN = 128, Q80MMA_WN = 64, Q80MMA_NSA = 3, Q80MMA_NSB = 2, Q80MMA_NW = 8;
using Q80MMA_CFG = q8k::cfg<Q80MMA_BM, Q80MMA_BN, Q80MMA_WN, Q80MMA_NSA, Q80MMA_NSB, Q80MMA_NW>;

typedef CUresult (*q80_encode_tiled_t)(CUtensorMap *, CUtensorMapDataType, cuuint32_t, void *, const cuuint64_t *, const cuuint64_t *,
                                       const cuuint32_t *, const cuuint32_t *, CUtensorMapInterleave, CUtensorMapSwizzle,
                                       CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

static q80_encode_tiled_t q80mma_encode_tiled() {
    static q80_encode_tiled_t fn = nullptr;
    static bool tried = false;
    if (!tried) {
        tried = true;
#if CUDART_VERSION >= 12050
        void * p = nullptr;
        cudaDriverEntryPointQueryResult q = cudaDriverEntryPointSymbolNotFound;
        if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q) == cudaSuccess &&
            q == cudaDriverEntryPointSuccess) {
            fn = (q80_encode_tiled_t) p;
        } else {
            GGML_LOG_WARN("%s: cuTensorMapEncodeTiled unavailable, the Q8_0 MMA kernel is off and MMQ runs\n", __func__);
        }
#endif // CUDART_VERSION >= 12050
    }
    return fn;
}

// 2-D byte tensor: `rows` rows of `inner` valid bytes at pitch `pitch`; box of box_rows x box_inner bytes; out of bounds reads zero
static bool q80mma_map_2d(CUtensorMap & map, const void * base, uint64_t inner, uint64_t rows, uint64_t pitch, uint32_t box_inner, uint32_t box_rows) {
    const cuuint64_t dims[2] = { inner, rows };
    const cuuint64_t strides[1] = { pitch };
    const cuuint32_t box[2] = { box_inner, box_rows };
    const cuuint32_t estr[2] = { 1, 1 };
    const CUresult res = q80mma_encode_tiled()(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void *>(base), dims, strides, box, estr,
                                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return res == CUDA_SUCCESS;
}

// the weight maps, one per (tensor address, shape): weights do not move during a run, the activation buffer does
struct q80mma_weight_key {
    const void * x; int64_t ne00, ne01, nb01;
    bool operator==(const q80mma_weight_key & o) const { return x == o.x && ne00 == o.ne00 && ne01 == o.ne01 && nb01 == o.nb01; }
};
struct q80mma_weight_key_hash {
    size_t operator()(const q80mma_weight_key & k) const { return std::hash<const void *>()(k.x) ^ (size_t) k.ne00 * 31u ^ (size_t) k.ne01 * 131u ^ (size_t) k.nb01 * 1031u; }
};
static std::mutex q80mma_maps_mutex;
static std::unordered_map<q80mma_weight_key, CUtensorMap, q80mma_weight_key_hash> q80mma_maps;
} // namespace

bool ggml_cuda_mmq_q80_mma_supported(const int cc, const size_t smpbo, const int64_t ne00, const int64_t ne01, const int64_t ne11) {
    if (!blackwell_mma_available(cc) || smpbo < (size_t) Q80MMA_CFG::SMEM) {
        return false;
    }
    if (ne00 % (2*q8k::BK) != 0 || ne01 % Q80MMA_BM != 0 || ne11 < 64) {   // ne00 % 256: the Q8_0 row pitch 34*ne00/32 must be a multiple of 16 for TMA
        return false;
    }
    return q80mma_encode_tiled() != nullptr;
}

void ggml_cuda_mmq_q80_mma(const char * x, const char * y, float * dst,
                           const int64_t ne00, const int64_t ne01, const int64_t ne11, const int64_t stride_dst, const int64_t nb01,
                           const float * res, const int64_t stride_res, cudaStream_t stream) {
    GGML_ASSERT(ne00 % (2*q8k::BK) == 0 && ne01 % Q80MMA_BM == 0 && ne11 >= 1);
    GGML_ASSERT(((uintptr_t) x) % 16 == 0 && ((uintptr_t) y) % 16 == 0 && nb01 % 16 == 0);
    CUtensorMap tmA, tmB;
    {
        const q80mma_weight_key key = { x, ne00, ne01, nb01 };
        std::lock_guard<std::mutex> lock(q80mma_maps_mutex);
        auto it = q80mma_maps.find(key);
        if (it == q80mma_maps.end()) {
            CUtensorMap mp;
            const bool ok = q80mma_map_2d(mp, x, (uint64_t) (ne00/QK8_0)*sizeof(block_q8_0), (uint64_t) ne01, (uint64_t) nb01, q8k::A_ROW, Q80MMA_BM);
            GGML_ASSERT(ok && "cuTensorMapEncodeTiled failed for the Q8_0 weight");
            it = q80mma_maps.emplace(key, mp).first;
        }
        tmA = it->second;
    }
    const bool okB = q80mma_map_2d(tmB, y, sizeof(block_q8_1_mmq), (uint64_t) (ne00/q8k::BK)*ne11, sizeof(block_q8_1_mmq), q8k::B_ROW, Q80MMA_BN);
    GGML_ASSERT(okB && "cuTensorMapEncodeTiled failed for the Q8_1 operand");

    CUDA_SET_SHARED_MEMORY_LIMIT((q8_0_q8_1_gemm_tma<Q80MMA_BM, Q80MMA_BN, Q80MMA_WN, Q80MMA_NSA, Q80MMA_NSB, Q80MMA_NW>), Q80MMA_CFG::SMEM);
    const int grid = (int) ((ne01/Q80MMA_BM) * ((ne11 + Q80MMA_BN - 1)/Q80MMA_BN));
    q8_0_q8_1_gemm_tma<Q80MMA_BM, Q80MMA_BN, Q80MMA_WN, Q80MMA_NSA, Q80MMA_NSB, Q80MMA_NW><<<grid, 32*Q80MMA_NW, Q80MMA_CFG::SMEM, stream>>>
        (tmA, tmB, dst, (int) ne01, (int) ne11, (int) ne00, stride_dst, res, stride_res);
    CUDA_CHECK(cudaGetLastError());
}
#else // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
bool ggml_cuda_mmq_q80_mma_supported(int, size_t, int64_t, int64_t, int64_t) {
    return false;
}

void ggml_cuda_mmq_q80_mma(const char *, const char *, float *, int64_t, int64_t, int64_t, int64_t, int64_t, const float *, int64_t, cudaStream_t) {
    GGML_ABORT("fatal error");
}
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
