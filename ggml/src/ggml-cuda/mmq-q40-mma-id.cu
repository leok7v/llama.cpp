#include "mmq-q40-mma-id.cuh"
#include "mmq.cuh"
#include "mmid.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include <cuda.h>
#include <cstdint>
#include <mutex>
#include <unordered_map>

// The exact Q4_0 x Q8_1 GEMM of mmq-q40-mma.cu (INT8 tensor cores of sm_120/121, TMA-fed, MMQ's rounding sequence bit for bit)
// for the routed experts of mul_mat_id. What differs from the dense kernel:
//   * the weight map is 3-D, dims {(K/32)*18 bytes, M rows, ne02 experts}, strides {nb01, nb02}, box {80, BM, 1}: one descriptor per
//     weight tensor, the expert is the third TMA coordinate; TMA zero-fills past M and past K, and a zero block has a zero scale,
//     so a padded row or K tail contributes nothing (the scale makes the tail safe, not the quant: a zero nibble expands to -128);
//   * the tile: blockIdx.x is the row tile (fastest, so consecutive CTAs stream one expert's rows and share its activation rows),
//     blockIdx.y a linear column tile of the grouped schedule (tile_bounds, mmid.cu) decoded to (expert, column tile) by the binary
//     search of mmq_tile_at; the activation box starts at compact row expert_bounds[e] + jt*BN of every k-block; rows past the
//     expert's are computed and masked at the write-back;
//   * the tile is 256 rows x 32 tokens (an expert sees 16-32 tokens of a 512-token batch; the MMA work scales with the token tile,
//     the weight stream does not), 8 warps as 8 x 1 of 32 rows each, 3 stages, the weight box 20 KB; the whole K per CTA, so no
//     stream-k and no fixup: every output is written once, in the single-pass rounding sequence;
//   * the write-back goes through ids_dst to the token's row of dst.
// The grid is nty x T_max, T_max the host's upper bound on the tile count (N/BN + the non-empty experts); the real count T lives on
// the device, and a CTA past it returns before its first TMA. Measured on a 20-SM sm_121 part against the grouped MMQ schedule at
// the same shapes: the 256-row tile and the 128-row tile with two CTAs per SM tie; a persistent CTA over a tile sequence does not
// gain; the row tile fastest in the grid is worth 4-5%.

namespace q4i {
constexpr int BK = 128;                            // K per stage (4 Q4_0 blocks)
constexpr int A_ROW = 80;                          // 72 raw bytes (4 blocks) + 8: 20-word stride is bank-conflict-free, 16-byte TMA box
constexpr int B_ROW = 144;                         // one block_q8_1_mmq per token per stage
constexpr int   MAGIC_I = 0x4B400000;
constexpr float MAGIC_F = 12582912.0f;
constexpr int NW = 8, NT = 32*NW;
constexpr int BN = 32;                             // the token tile
static_assert(sizeof(block_q4_0) == 18 && sizeof(block_q8_1_mmq) == B_ROW, "layouts");

template <int BM, int NSTAGE> struct cfg {
    static constexpr int WN = BN, WM = BM / NW;    // 8 warps as 8 x 1
    static constexpr int MT_I = WM / 16, MT_J = WN / 8;
    static constexpr int A_BYTES = BM * A_ROW, B_BYTES = BN * B_ROW;
    static constexpr int STAGE_BYTES = A_BYTES + B_BYTES;
    static constexpr int SLICE_FLOATS = 2 * WN;    // dB/16 for the warp's WN tokens, double-buffered per K block
    static constexpr int CS = BM + 4;              // staging row stride in floats
    static constexpr int SMEM = NSTAGE*STAGE_BYTES + NW*SLICE_FLOATS*4 + NSTAGE*8 + NSTAGE*4 + 64;
    static_assert(WM % 16 == 0 && MT_J % 2 == 0, "bad warp tiling");
    static_assert(STAGE_BYTES % 128 == 0 && A_BYTES % 128 == 0, "TMA destinations must be 128-byte aligned");
    static_assert(BN*CS*4 <= NSTAGE*STAGE_BYTES, "the staging slab must fit the ring");
    static_assert(BM % 4 == 0, "16-byte stores");
};
} // namespace q4i

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
static __device__ __forceinline__ void tma_load_3d(void * smem_dst, const CUtensorMap * map, int c0, int c1, int c2, uint64_t * bar) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];\n"
                 :: "r"((unsigned) __cvta_generic_to_shared(smem_dst)), "l"((uint64_t) map),
                    "r"((unsigned) __cvta_generic_to_shared(bar)), "r"(c0), "r"(c1), "r"(c2) : "memory");
}

template <int BM, int NSTAGE, int MINB>
__global__ void __launch_bounds__(q4i::NT, MINB) q4_0_q8_1_gemm_tma_id(
        const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB,
        float * __restrict__ dst, const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ expert_bounds,
        const int32_t * __restrict__ tile_bounds, const int n_experts, const int M, const int N, const int K, const int64_t stride_dst) {
    using namespace q4i;
    using C = cfg<BM, NSTAGE>;
    constexpr int WM = C::WM, MT_I = C::MT_I, MT_J = C::MT_J;
    constexpr int A_BYTES = C::A_BYTES, STAGE_BYTES = C::STAGE_BYTES, SLICE_FLOATS = C::SLICE_FLOATS, CS = C::CS;
    extern __shared__ __align__(128) char smem[];
    float    * dbslices = (float *) (smem + NSTAGE*STAGE_BYTES);
    uint64_t * full     = (uint64_t *) (dbslices + NW*SLICE_FLOATS);
    int      * ticket   = (int *) (full + NSTAGE);

    // the tile of this CTA: blockIdx.x the row tile (fastest, so consecutive CTAs stream one expert's rows and share its
    // activation rows), blockIdx.y the linear column tile; past the real tile count there is nothing to do (uniform across the
    // CTA, before any barrier)
    const int T   = tile_bounds[n_experts];
    const int lin = blockIdx.y;
    if (lin >= T) {
        return;
    }
    int lo = 0;
    int hi = n_experts;                             // the largest e with tile_bounds[e] <= lin; ties go to the largest e,
    while (hi - lo > 1) {                           // which skips an empty expert whose bound equals the next one's
        const int mid = (lo + hi) / 2;
        if (tile_bounds[mid] <= lin) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    const int e       = lo;
    const int jt      = lin - tile_bounds[e];
    const int col_low = expert_bounds[e];
    const int ntok    = min(BN, expert_bounds[e + 1] - col_low - jt*BN);
    const int j_base  = col_low + jt*BN;            // the first compact row of this tile
    const int i_base  = blockIdx.x * BM;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wi = warp;                            // 8 warps as 8 x 1: warp i owns rows [wi*WM, wi*WM + WM) of the tile
    const int nsteps = K / BK;

    auto issue = [&](const int ks) {                // one thread: both boxes of stage ks into buffer ks % NSTAGE
        const int buf = ks % NSTAGE;
        char * stage = smem + buf*STAGE_BYTES;
        mbar_expect_tx(full + buf, STAGE_BYTES);
        tma_load_3d(stage,           &tmA, 72*ks - 8*(ks & 1), i_base, e, full + buf);
        tma_load_2d(stage + A_BYTES, &tmB, 0, ks*N + j_base,       full + buf);
    };
    if (tid == 0) {
        for (int b = 0; b < NSTAGE; ++b) { mbar_init(full + b, 1); ticket[b] = 0; }
        fence_proxy_async();
    }
    __syncthreads();                                // the only block-wide barrier before the write-back
    if (tid == 0) {
        for (int s = 0; s < NSTAGE && s < nsteps; ++s) { issue(s); }
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
        const int buf = ks % NSTAGE;
        mbar_wait(full + buf, (ks / NSTAGE) & 1);
        const char * As = smem + buf*STAGE_BYTES;
        const char * Bs = As + A_BYTES;
        const int ashift = 8*(ks & 1);              // the 72-byte chunk's offset inside the 80-byte row
        // just-in-time dB/16 for the tile's WN tokens (one per lane) of K block b into slice (b & 1)
        auto jit_db = [&](const int b) {
            float * dbb = dbw + (b & 1)*(SLICE_FLOATS/2);
            dbb[lane] = __low2float(*(const half2 *) (Bs + lane*B_ROW + 4*b)) * 0.0625f;
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
                da[n][0] = __half2float(*(const half *) (R0 + 18*b));
                da[n][1] = __half2float(*(const half *) (R1 + 18*b));
                nmda[n][0] = -MAGIC_F * da[n][0];
                nmda[n][1] = -MAGIC_F * da[n][1];
            }
        };
        auto load_b = [&](const int b, const int jj, unsigned (&bf)[4], float2 (&db)[2]) {
            const int jrow = jj*8 + (lane & 7) + ((lane >> 4) << 3);
            const int koff = 16 + 32*b + ((lane >> 3) & 1)*16;
            ldmatrix_x4(bf, Bs + jrow*B_ROW + koff);
            const float * dbb = dbw + (b & 1)*(SLICE_FLOATS/2);
            db[0] = *(const float2 *) (dbb + jj*8 + 2*m);
            db[1] = *(const float2 *) (dbb + jj*8 + 8 + 2*m);
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
        // the B fragments one tile pair ahead (the dense kernel's BPF form), A and the scales per K block
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
        // this warp is done with the buffer; the last warp to say so re-arms it with stage ks + NSTAGE
        __syncwarp();
        __threadfence_block();                      // our reads of the buffer precede the ticket (release)
        if (lane == 0) {
            const int t = atomicAdd(ticket + buf, 1);
            if (t == NW - 1) {
                ticket[buf] = 0;
                fence_proxy_async();                // our generic-proxy reads before the async-proxy writes
                if (ks + NSTAGE < nsteps) { issue(ks + NSTAGE); }
            }
        }
    }

    // ---- write back through shared memory: the tile as [token][row], then 16-byte stores into the token's dst row via ids_dst;
    // columns past the expert's tokens and rows past M (the last row tile of an M that is not a multiple of BM) are not written
    __syncthreads();                                // every warp has consumed every stage: the ring is free
    float * Cs = (float *) smem;
#pragma unroll
    for (int n = 0; n < MT_I; ++n) {
#pragma unroll
        for (int j = 0; j < MT_J; ++j) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                Cs[(j*8 + 2*m + (l & 1))*CS + wi*WM + n*16 + r + 8*(l >> 1)] = sum[n][j][l];
            }
        }
    }
    __syncthreads();
    const int rows = min(BM, M - i_base);
    for (int idx = tid; idx < BN*(BM/4); idx += NT) {
        const int col = idx / (BM/4), q = idx - col*(BM/4);
        if (col < ntok && 4*q < rows) {
            const float4 v = *(const float4 *) (Cs + col*CS + 4*q);
            *(float4 *) (dst + (size_t) ids_dst[j_base + col]*stride_dst + i_base + 4*q) = v;
        }
    }
}

// ---------------------------------------------------------------- host side
namespace {
typedef CUresult (*q4i_encode_tiled_t)(CUtensorMap *, CUtensorMapDataType, cuuint32_t, void *, const cuuint64_t *, const cuuint64_t *,
                                       const cuuint32_t *, const cuuint32_t *, CUtensorMapInterleave, CUtensorMapSwizzle,
                                       CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

static q4i_encode_tiled_t q4i_encode_tiled() {
    static q4i_encode_tiled_t fn = nullptr;
    static bool tried = false;
    if (!tried) {
        tried = true;
#if CUDART_VERSION >= 12050
        void * p = nullptr;
        cudaDriverEntryPointQueryResult q = cudaDriverEntryPointSymbolNotFound;
        if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q) == cudaSuccess &&
            q == cudaDriverEntryPointSuccess) {
            fn = (q4i_encode_tiled_t) p;
        } else {
            GGML_LOG_WARN("%s: cuTensorMapEncodeTiled unavailable, the Q4_0 MMA kernel for mul_mat_id is off and MMQ runs\n", __func__);
        }
#endif // CUDART_VERSION >= 12050
    }
    return fn;
}

// 2-D byte tensor: `rows` rows of `inner` valid bytes at pitch `pitch`; box of box_rows x box_inner bytes; out of bounds reads zero
static bool q4i_map_2d(CUtensorMap & map, const void * base, uint64_t inner, uint64_t rows, uint64_t pitch, uint32_t box_inner, uint32_t box_rows) {
    const cuuint64_t dims[2] = { inner, rows };
    const cuuint64_t strides[1] = { pitch };
    const cuuint32_t box[2] = { box_inner, box_rows };
    const cuuint32_t estr[2] = { 1, 1 };
    const CUresult res = q4i_encode_tiled()(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void *>(base), dims, strides, box, estr,
                                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return res == CUDA_SUCCESS;
}

// 3-D byte tensor: `planes` experts of `rows` rows of `inner` valid bytes, row pitch nb01, expert pitch nb02; box box_rows x box_inner
// bytes of one expert; out of bounds reads zero
static bool q4i_map_3d(CUtensorMap & map, const void * base, uint64_t inner, uint64_t rows, uint64_t planes, uint64_t nb01, uint64_t nb02,
                       uint32_t box_inner, uint32_t box_rows) {
    const cuuint64_t dims[3] = { inner, rows, planes };
    const cuuint64_t strides[2] = { nb01, nb02 };
    const cuuint32_t box[3] = { box_inner, box_rows, 1 };
    const cuuint32_t estr[3] = { 1, 1, 1 };
    const CUresult res = q4i_encode_tiled()(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, const_cast<void *>(base), dims, strides, box, estr,
                                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return res == CUDA_SUCCESS;
}

// the weight maps, one per (tensor address, shape, row tile): weights do not move during a run, the activation buffer does
struct q4i_weight_key {
    const void * x; int64_t ne00, ne01, ne02, nb01, nb02; int bm;
    bool operator==(const q4i_weight_key & o) const {
        return x == o.x && ne00 == o.ne00 && ne01 == o.ne01 && ne02 == o.ne02 && nb01 == o.nb01 && nb02 == o.nb02 && bm == o.bm;
    }
};
struct q4i_weight_key_hash {
    size_t operator()(const q4i_weight_key & k) const {
        return std::hash<const void *>()(k.x) ^ (size_t) k.ne00 * 31u ^ (size_t) k.ne01 * 131u ^ (size_t) k.ne02 * 331u ^
               (size_t) k.nb01 * 1031u ^ (size_t) k.nb02 * 2027u ^ (size_t) k.bm * 4099u;
    }
};
static std::mutex q4i_maps_mutex;
static std::unordered_map<q4i_weight_key, CUtensorMap, q4i_weight_key_hash> q4i_maps;

static CUtensorMap q4i_weight_map(const char * x, const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb01, const int64_t nb02, const int bm) {
    const q4i_weight_key key = { x, ne00, ne01, ne02, nb01, nb02, bm };
    std::lock_guard<std::mutex> lock(q4i_maps_mutex);
    auto it = q4i_maps.find(key);
    if (it == q4i_maps.end()) {
        CUtensorMap m;
        const bool ok = q4i_map_3d(m, x, (uint64_t) (ne00/QK4_0)*sizeof(block_q4_0), (uint64_t) ne01, (uint64_t) ne02, (uint64_t) nb01, (uint64_t) nb02,
                                   q4i::A_ROW, (uint32_t) bm);
        GGML_ASSERT(ok && "cuTensorMapEncodeTiled failed for the Q4_0 expert weights");
        it = q4i_maps.emplace(key, m).first;
    }
    return it->second;
}

constexpr int Q4I_BM = 256, Q4I_NSTAGE = 3, Q4I_MINB = 1;
constexpr int Q4I_SMEM = q4i::cfg<Q4I_BM, Q4I_NSTAGE>::SMEM;
} // namespace

bool ggml_cuda_mmq_q40_mma_id_supported(const int cc, const size_t smpbo, const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb01, const int64_t nb02) {
    if (!blackwell_mma_available(cc) || smpbo < (size_t) Q4I_SMEM) {
        return false;
    }
    // K % 128: whole 4-block stages; the pitches: TMA strides are 16-byte multiples (nb01 = K/32*18 needs K % 256);
    // ne02 <= 1024: the tile-count scan of mmid.cu; ne01 % 4: 16-byte output stores
    if (ne00 % q4i::BK != 0 || nb01 % 16 != 0 || nb02 % 16 != 0 || ne01 % 4 != 0 || ne02 < 1 || ne02 > 1024) {
        return false;
    }
    return q4i_encode_tiled() != nullptr;
}

void ggml_cuda_mmq_q40_mma_id(ggml_backend_cuda_context & ctx, const char * x, const char * y, float * dst,
                              const int32_t * ids_dst, const int32_t * expert_bounds,
                              const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t N, const int64_t stride_dst, const int64_t nb01, const int64_t nb02,
                              cudaStream_t stream) {
    GGML_ASSERT(ne00 % q4i::BK == 0 && ne01 % 4 == 0 && ne02 >= 1 && N >= 1);
    GGML_ASSERT(((uintptr_t) x) % 16 == 0 && ((uintptr_t) y) % 16 == 0 && nb01 % 16 == 0 && nb02 % 16 == 0);

    // the grouped schedule over BN-wide column tiles: tile_bounds[e] the first tile of expert e, [ne02] the count (device side)
    ggml_cuda_pool_alloc<int32_t> tile_bounds(ctx.pool(), ne02 + 1);
    ggml_cuda_launch_mm_tile_bounds(expert_bounds, tile_bounds.get(), (int) ne02, q4i::BN, stream);
    CUDA_CHECK(cudaGetLastError());
    // the host's upper bound on the count: ceil(tokens_e / BN) <= tokens_e / BN + 1 for each non-empty expert
    const int T_max = (int) (N / q4i::BN + std::min<int64_t>(ne02, N));

    const CUtensorMap tmA = q4i_weight_map(x, ne00, ne01, ne02, nb01, nb02, Q4I_BM);
    CUtensorMap tmB;
    const bool okB = q4i_map_2d(tmB, y, sizeof(block_q8_1_mmq), (uint64_t) (ne00/q4i::BK)*N, sizeof(block_q8_1_mmq), q4i::B_ROW, q4i::BN);
    GGML_ASSERT(okB && "cuTensorMapEncodeTiled failed for the Q8_1 operand");

    CUDA_SET_SHARED_MEMORY_LIMIT((q4_0_q8_1_gemm_tma_id<Q4I_BM, Q4I_NSTAGE, Q4I_MINB>), Q4I_SMEM);
    const dim3 grid((int) ((ne01 + Q4I_BM - 1)/Q4I_BM), T_max, 1);
    q4_0_q8_1_gemm_tma_id<Q4I_BM, Q4I_NSTAGE, Q4I_MINB><<<grid, q4i::NT, Q4I_SMEM, stream>>>
        (tmA, tmB, dst, ids_dst, expert_bounds, tile_bounds.get(), (int) ne02, (int) ne01, (int) N, (int) ne00, stride_dst);
    CUDA_CHECK(cudaGetLastError());
}
#else // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
bool ggml_cuda_mmq_q40_mma_id_supported(int, size_t, int64_t, int64_t, int64_t, int64_t, int64_t) {
    return false;
}

void ggml_cuda_mmq_q40_mma_id(ggml_backend_cuda_context &, const char *, const char *, float *, const int32_t *, const int32_t *,
                              int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, cudaStream_t) {
    GGML_ABORT("fatal error");
}
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
