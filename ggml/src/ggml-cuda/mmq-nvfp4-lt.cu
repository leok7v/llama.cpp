// mmq-nvfp4-lt.cu - NVFP4 weights x F32 activations through cuBLASLt's block-scaled FP4 tensor-core GEMM (Blackwell).
//
// Why: llama.cpp's native W4A4 MMQ reaches about 47 TFLOPS on an sm_121 part's gemma-4-12B projections, cuBLASLt's cutlass
// kernels 87-108 on the same shapes once the best heuristic candidate is picked per shape (a standalone microbench
// checked the numerics against a CPU reference at 1e-7 relative error before this file existed). What this does:
//   - the weight tensor (GGUF block_nvfp4: 64 values, four UE4M3 sub-block scales, nibbles j | j+8 << 4 inside a
//     sub-block) is repacked ONCE per tensor into cuBLASLt's operand form: a row-major M x K E2M1 matrix with consecutive
//     pairs per byte, plus the scale factors in the tensor-core tile layout (512-byte atoms of 128 rows x 4 scale
//     columns, byte (r%32)*16 + ((r/32)%4)*4 + c%4 inside an atom, atoms over K inner, rows outer). The GGUF blocks stay
//     where they are: the mmvq decode path still reads them;
//   - the activations are quantized per call in one read of x: each 16-block against its own scale rounded to UE4M3's
//     three mantissa bits (unbounded exponent), then a tiny second pass picks the power-of-two tensor scale g that puts
//     the largest block scale at UE4M3's top (448) and encodes every block scale as d0 / g, an exact exponent shift in
//     the normal range; E2M1 values; the same tile layout;
//   - alpha = g * (the weight's global `.scale` when the graph fused the multiply, else 1) is a device pointer, so no host
//     synchronization; the F32 result lands straight in dst; the scratch (quantized activations, scales, workspace, scalars)
//     is plain cudaMalloc memory grown on demand, because cuBLASLt refused the ggml pool's mapped pointers for the scalars;
//   - the first call for a shape times every heuristic candidate and keeps the fastest (the default pick is 2.4x off on
//     the down projection), and remembers the choice in ~/.ggml-cuda-nvfp4-lt.tune so later processes pick the same
//     kernel: the candidates sum in different orders, and a timing-dependent choice made greedy decoding differ run to run;
//   - a GEGLU whose only consumer is such a matmul (gemma's FFN) is folded into the quantize kernel: gelu(gate) * up is
//     computed on the way to FP4 and the F32 GLU output is never written (ggml_cuda_mul_mat_nvfp4_lt_glu);
//   - consecutive matmuls of one graph evaluation that read the same src1 tensor (q, k, v; gate, up) quantize it once: the
//     dispatcher notes the node being computed (ggml_cuda_nvfp4_lt_note_node) and a later node with the same src1 object
//     reuses the FP4 operand in the scratch, recomputing only alpha for its own weight scale;
//   - a pre-norm (RMS_NORM + weight MUL) whose every consumer is such a matmul is written straight to FP4 by one kernel
//     (ggml_cuda_rms_norm_nvfp4_lt): the F32 normed activation never exists in memory either.
#include "mmq-nvfp4-lt.cuh"
#include "unary.cuh"   // ggml_cuda_op_gelu_single: the same tanh gelu the GLU kernel applies

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 12080
#define GGML_CUDA_NVFP4_LT 1
#endif

#ifdef GGML_CUDA_NVFP4_LT
#include <cublasLt.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <unordered_map>
#include <cstdio>
#include <string>
#include <cstring>
#include <cinttypes>

namespace {

constexpr int64_t LT_MIN_TOKENS = 16;   // thinner batches stay on MMQ

__host__ __device__ constexpr int64_t pad128(int64_t x) { return (x + 127) / 128 * 128; }
__host__ __device__ constexpr int64_t pad4(int64_t x)   { return (x + 3) / 4 * 4; }

__device__ __forceinline__ size_t sf_offset(int r, int c, int kblk_pad) {
    const int inner = (r % 32) * 16 + ((r / 32) % 4) * 4 + (c % 4);
    const int atom  = (r / 128) * (kblk_pad / 4) + c / 4;
    return (size_t) atom * 512 + inner;
}

static __global__ void k_repack(const block_nvfp4 * __restrict__ W, int M, int K, uint8_t * __restrict__ A, uint8_t * __restrict__ SF, int kblk_pad) {
    const int nsb = K / 16;
    const size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t) M * nsb) {
        return;
    }
    const int r = (int) (idx / nsb), c = (int) (idx % nsb), s = c % 4;
    const block_nvfp4 & b = W[(size_t) r * (K / 64) + c / 4];
    uint8_t v[16];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        v[j]     = b.qs[s * 8 + j] & 0xF;
        v[j + 8] = b.qs[s * 8 + j] >> 4;
    }
    uint64_t packed = 0;
#pragma unroll
    for (int p = 0; p < 8; ++p) {
        packed |= (uint64_t) (v[2 * p] | (v[2 * p + 1] << 4)) << (8 * p);
    }
    *reinterpret_cast<uint64_t *>(A + (size_t) r * (K / 2) + (size_t) c * 8) = packed;
    SF[sf_offset(r, c, kblk_pad)] = b.d[s];
}

// pass 1: one thread per (row, 16-block). d0 = |block| max / 6 rounded to 3 mantissa bits with an unbounded exponent, so
// that the later division by a power-of-two tensor scale is exact; E2M1 values are quantized against d0 right here.
static __device__ __forceinline__ float round_to_ue4m3_mantissa(float d) {
    int e;
    float m = frexpf(d, &e);                 // m in [0.5, 1)
    m = rintf(m * 16.0f) / 16.0f;            // 3 bits after the leading one
    return ldexpf(m, e);                     // m == 1.0 after a carry is still exact
}
// pass 1, a warp per 128 consecutive values of one row: lane l holds values 4l..4l+3 (one float4 per lane, 512 B per warp
// load), a 16-value block is 4 lanes and its maximum a two-step shuffle; d0 = block max / 6 rounded to UE4M3's 3 mantissa bits
// with an unbounded exponent (so the later division by a power-of-two tensor scale is exact); E2M1 values quantized against d0.
// GEGLU: X is the gate and G the up projection, the value quantized is gelu(x) * g (ggml's tanh gelu, bit-identical to the GLU op).
// K must be a multiple of 128 (every gemma width is); the caller checks.
template <int GLU>   // 0: x; 1: gelu(x) * g (gemma's GEGLU); 2: silu(x) * g (Qwen's SWIGLU)
static __global__ void k_quant_pass1(const float * __restrict__ X, const float * __restrict__ G, int N, int K, uint8_t * __restrict__ B,
                                     float * __restrict__ d0s, float * __restrict__ bmax) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int chunks_per_row = K / 128;
    const int gwarp = blockIdx.x * warps_per_block + warp;
    const int n = gwarp / chunks_per_row, ch = gwarp % chunks_per_row;
    float d0 = 0.0f;
    if (n < N) {   // no early return: the warp shuffles below need every lane
        const size_t off = (size_t) n * K + (size_t) ch * 128;
        float4 v = reinterpret_cast<const float4 *>(X + off)[lane];
        if constexpr (GLU == 1) {
            const float4 g = reinterpret_cast<const float4 *>(G + off)[lane];
            v.x = ggml_cuda_op_gelu_single(v.x) * g.x;
            v.y = ggml_cuda_op_gelu_single(v.y) * g.y;
            v.z = ggml_cuda_op_gelu_single(v.z) * g.z;
            v.w = ggml_cuda_op_gelu_single(v.w) * g.w;
        } else if constexpr (GLU == 2) {
            const float4 g = reinterpret_cast<const float4 *>(G + off)[lane];
            v.x = ggml_cuda_op_silu_single(v.x) * g.x;   // unary_gated_op_kernel<op_silu>: op(x) * g
            v.y = ggml_cuda_op_silu_single(v.y) * g.y;
            v.z = ggml_cuda_op_silu_single(v.z) * g.z;
            v.w = ggml_cuda_op_silu_single(v.w) * g.w;
        }
        float bam = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
        bam = fmaxf(bam, __shfl_xor_sync(0xffffffff, bam, 1));
        bam = fmaxf(bam, __shfl_xor_sync(0xffffffff, bam, 2));
        d0 = bam > 0.0f ? round_to_ue4m3_mantissa(bam / 6.0f) : 0.0f;
        const float inv = d0 > 0.0f ? 1.0f / d0 : 0.0f;
        const __nv_fp4x2_e2m1 q0(make_float2(v.x * inv, v.y * inv));
        const __nv_fp4x2_e2m1 q1(make_float2(v.z * inv, v.w * inv));
        reinterpret_cast<uint16_t *>(B + (size_t) n * (K / 2) + (size_t) ch * 64)[lane] = (uint16_t) ((uint8_t) q0.__x | ((uint16_t) (uint8_t) q1.__x << 8));
        if ((lane & 3) == 0) {
            d0s[(size_t) n * (K / 16) + (size_t) ch * 8 + (lane >> 2)] = d0;
        }
    }
    // block max -> bmax[block]
    float wmax = d0;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        wmax = fmaxf(wmax, __shfl_xor_sync(0xffffffff, wmax, o));
    }
    __shared__ float sm[32];
    if (lane == 0) {
        sm[warp] = wmax;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        wmax = threadIdx.x < (unsigned) warps_per_block ? sm[threadIdx.x] : 0.0f;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            wmax = fmaxf(wmax, __shfl_xor_sync(0xffffffff, wmax, o));
        }
        if (threadIdx.x == 0) {
            bmax[blockIdx.x] = wmax;
        }
    }
}
// pass 1 for a pre-norm: one block per row; thread t owns 16-block t. Sum of squares over the row (block reduction), the
// same rsqrt(mean + eps) scale and per-channel weight the fused RMS_NORM + MUL kernel applies, then the block's d0 and E2M1
// values as in k_quant_pass1. The F32 normed row is never written. blockDim = K / 16 rounded up to a warp (K <= 16384).
static __global__ void k_norm_quant_pass1(const float * __restrict__ X, int64_t s_row, const float * __restrict__ W, int K, float eps,
                                          uint8_t * __restrict__ B, float * __restrict__ d0s, float * __restrict__ bmax) {
    const int row = blockIdx.x, t = threadIdx.x, nsb = K / 16;
    const bool valid = t < nsb;
    float xr[16];
    float ss = 0.0f;
    if (valid) {
        const float4 * x4 = reinterpret_cast<const float4 *>(X + (size_t) row * s_row + (size_t) t * 16);
#pragma unroll
        for (int q4 = 0; q4 < 4; ++q4) {
            const float4 v = x4[q4];
            xr[4 * q4] = v.x; xr[4 * q4 + 1] = v.y; xr[4 * q4 + 2] = v.z; xr[4 * q4 + 3] = v.w;
            ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    }
    // block sum: warp shuffles, then the warp partials through shared memory
    __shared__ float sm[32];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        ss += __shfl_xor_sync(0xffffffff, ss, o);
    }
    if ((t & 31) == 0) {
        sm[t >> 5] = ss;
    }
    __syncthreads();
    if (t < 32) {
        ss = t < (int) (blockDim.x >> 5) ? sm[t] : 0.0f;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            ss += __shfl_xor_sync(0xffffffff, ss, o);
        }
        if (t == 0) {
            sm[0] = ss;
        }
    }
    __syncthreads();
    const float scale = rsqrtf(sm[0] / K + eps);
    float d0 = 0.0f;
    if (valid) {
        const float4 * w4 = reinterpret_cast<const float4 *>(W + (size_t) t * 16);
        float bam = 0.0f;
#pragma unroll
        for (int q4 = 0; q4 < 4; ++q4) {
            const float4 w = w4[q4];
            xr[4 * q4]     = scale * xr[4 * q4]     * w.x;
            xr[4 * q4 + 1] = scale * xr[4 * q4 + 1] * w.y;
            xr[4 * q4 + 2] = scale * xr[4 * q4 + 2] * w.z;
            xr[4 * q4 + 3] = scale * xr[4 * q4 + 3] * w.w;
        }
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            bam = fmaxf(bam, fabsf(xr[k]));
        }
        d0 = bam > 0.0f ? round_to_ue4m3_mantissa(bam / 6.0f) : 0.0f;
        const float inv = d0 > 0.0f ? 1.0f / d0 : 0.0f;
        uint64_t packed = 0;
#pragma unroll
        for (int p = 0; p < 8; ++p) {
            const __nv_fp4x2_e2m1 q(make_float2(xr[2 * p] * inv, xr[2 * p + 1] * inv));
            packed |= (uint64_t) (uint8_t) q.__x << (8 * p);
        }
        *reinterpret_cast<uint64_t *>(B + (size_t) row * (K / 2) + (size_t) t * 8) = packed;
        d0s[(size_t) row * nsb + t] = d0;
    }
    // row max -> bmax[row]
    __syncthreads();
    float m = d0;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    }
    if ((t & 31) == 0) {
        sm[t >> 5] = m;
    }
    __syncthreads();
    if (t < 32) {
        m = t < (int) (blockDim.x >> 5) ? sm[t] : 0.0f;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
        }
        if (t == 0) {
            bmax[row] = m;
        }
    }
}

// one block: the largest block scale of the whole activation, into dmax_bits (as float bits)
static __global__ void k_quant_fold(const float * __restrict__ bmax, int nblocks, unsigned * __restrict__ dmax_bits) {
    float m = 0.0f;
    for (int i = threadIdx.x; i < nblocks; i += blockDim.x) m = fmaxf(m, bmax[i]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    __shared__ float sm[32];
    if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x < 32) {
        m = threadIdx.x < (blockDim.x >> 5) ? sm[threadIdx.x] : 0.0f;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
        if (threadIdx.x == 0) *dmax_bits = __float_as_uint(m);
    }
}
// pass 2: g = the power of two with max d0 / g <= 448 (UE4M3's top), every block scale becomes the UE4M3 code of d0 / g
// (exact in the normal range: a pure exponent shift of a 3-mantissa-bit number), alpha = g * (weight global scale or 1).
static __global__ void k_quant_pass2(const float * __restrict__ d0s, int N, int K, const unsigned * __restrict__ dmax_bits,
                                     const float * __restrict__ w_gs, uint8_t * __restrict__ SF, int kblk_pad, float * __restrict__ alpha) {
    const float dmax = __uint_as_float(*dmax_bits);
    int e = 0;
    if (dmax > 0.0f) { frexpf(dmax / 448.0f, &e); }   // dmax/448 = m * 2^e, m in [0.5,1): the smallest power of two >= dmax/448 is 2^e
    const float g = ldexpf(1.0f, e);
    const int nsb = K / 16;
    const size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        *alpha = g * (w_gs ? *w_gs : 1.0f);
    }
    GGML_UNUSED(alpha);
    if (idx >= (size_t) N * nsb) {
        return;
    }
    const int n = (int) (idx / nsb), c = (int) (idx % nsb);
    const float d0 = d0s[idx];
    const __nv_fp8_e4m3 sc8(d0 / g);
    uint8_t code = sc8.__x & 0x7F;
    if (d0 > 0.0f && code == 0) {
        code = 1;
    }
    SF[sf_offset(n, c, kblk_pad)] = code;
}

// alpha = g * (weight scale or 1) for a matmul that reuses the FP4 operand quantized for the previous one
static __global__ void k_alpha(const unsigned * __restrict__ dmax_bits, const float * __restrict__ w_gs, float * __restrict__ alpha) {
    const float dmax = __uint_as_float(*dmax_bits);
    int e = 0;
    if (dmax > 0.0f) { frexpf(dmax / 448.0f, &e); }
    *alpha = ldexpf(1.0f, e) * (w_gs ? *w_gs : 1.0f);
}

struct lt_repacked {
    uint8_t * A  = nullptr;
    uint8_t * SF = nullptr;
    int64_t   M  = 0;
    int64_t   K  = 0;
};

struct lt_plan {
    cublasLtMatmulDesc_t            op = nullptr;
    cublasLtMatrixLayout_t          Al = nullptr, Bl = nullptr, Dl = nullptr;
    cublasLtMatmulPreference_t      pref = nullptr;
    cublasLtMatmulHeuristicResult_t heur[16];
    int                             n_heur = 0;
    int                             best   = 0;
    bool                            tuned  = false;
    bool                            ok     = false;
};

struct lt_scratch {
    void * ptr = nullptr;
    size_t cap = 0;
};

// the candidate chosen per shape is remembered across processes in a small text file: the candidates differ in their fp32
// summation order, so a choice that depends on a timing race made greedy decoding differ between runs.
static const char * lt_tune_path() {
    static std::string p;
    if (p.empty()) {
        const char * home = getenv("HOME");
        p = std::string(home && *home ? home : "/tmp") + "/.ggml-cuda-nvfp4-lt.tune";
    }
    return p.c_str();
}
static int lt_tune_lookup(int64_t M, int64_t N, int64_t K, int cc) {
    FILE * f = fopen(lt_tune_path(), "r");
    if (!f) {
        return -1;
    }
    long long m, n, k; int c, idx, found = -1;
    while (fscanf(f, "%lld %lld %lld %d %d", &m, &n, &k, &c, &idx) == 5) {
        if (m == M && n == N && k == K && c == cc) {
            found = idx;
        }
    }
    fclose(f);
    return found;
}
static void lt_tune_store(int64_t M, int64_t N, int64_t K, int cc, int idx) {
    FILE * f = fopen(lt_tune_path(), "a");
    if (f) {
        fprintf(f, "%lld %lld %lld %d %d\n", (long long) M, (long long) N, (long long) K, cc, idx);
        fclose(f);
    }
}

struct lt_state {
    cublasLtHandle_t                                 handle = nullptr;
    std::unordered_map<const void *, lt_repacked>    weights;
    std::unordered_map<uint64_t, lt_plan>            plans;
    lt_scratch                                       B, SFB, ws, d0, bmax;   // grown on demand; stream order makes reuse safe
    float *                                          scal = nullptr;   // [0] alpha, [1] beta = 0, [2] max block scale bits
    bool                                             announced = false;
    bool                                             warned    = false;
    // the graph node being computed (ggml_cuda_nvfp4_lt_note_node) and the operand the scratch currently holds: a later node of
    // the SAME graph evaluation whose src1 is the same tensor object reuses it (q, k, v read one normed input; gate and up another)
    const ggml_cgraph *                              cur_graph = nullptr;
    int                                              cur_node  = -1;
    const ggml_cgraph *                              q_graph   = nullptr;
    int                                              q_node    = -1;
    const ggml_tensor *                              q_src1    = nullptr;
    int64_t                                          q_N = 0, q_K = 0;
};

// cuBLASLt validates its pointers; the ggml pool's virtual-memory mappings failed that check for the device-side
// alpha/beta (CUBLAS_STATUS_INVALID_VALUE), so this path keeps plain cudaMalloc scratch per device. cudaFree
// synchronizes the device, so growing a buffer cannot race the matmul still reading the old one.
static void * lt_scratch_get(lt_scratch & sc, size_t need) {
    if (sc.cap < need) {
        if (sc.ptr) {
            CUDA_CHECK(cudaFree(sc.ptr));
        }
        CUDA_CHECK(cudaMalloc(&sc.ptr, need));
        sc.cap = need;
    }
    return sc.ptr;
}

static lt_state & lt_state_for(int device) {
    static lt_state s[GGML_CUDA_MAX_DEVICES];
    return s[device];
}

constexpr size_t LT_WORKSPACE = 32u << 20;

static bool lt_make_plan(lt_state & st, lt_plan & P, int64_t M, int64_t N, int64_t K, const void * sfa, const void * sfb) {
    const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;   // narrow types want TN: both operands K-contiguous
    const cublasLtMatmulMatrixScale_t sm = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    const cublasLtPointerMode_t pm = CUBLASLT_POINTER_MODE_DEVICE;
    bool ok = cublasLtMatmulDescCreate(&P.op, CUBLAS_COMPUTE_32F, CUDA_R_32F) == CUBLAS_STATUS_SUCCESS;
    ok = ok && cublasLtMatmulDescSetAttribute(P.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)) == CUBLAS_STATUS_SUCCESS;
    ok = ok && cublasLtMatmulDescSetAttribute(P.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)) == CUBLAS_STATUS_SUCCESS;
    ok = ok && cublasLtMatmulDescSetAttribute(P.op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof(sm)) == CUBLAS_STATUS_SUCCESS;
    ok = ok && cublasLtMatmulDescSetAttribute(P.op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof(sm)) == CUBLAS_STATUS_SUCCESS;
    ok = ok && cublasLtMatmulDescSetAttribute(P.op, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pm, sizeof(pm)) == CUBLAS_STATUS_SUCCESS;
    // the heuristic query validates the descriptor, and a block-scaled one is invalid without its scale pointers
    ok = ok && cublasLtMatmulDescSetAttribute(P.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sfa, sizeof(sfa)) == CUBLAS_STATUS_SUCCESS;
    ok = ok && cublasLtMatmulDescSetAttribute(P.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sfb, sizeof(sfb)) == CUBLAS_STATUS_SUCCESS;
    ok = ok && cublasLtMatrixLayoutCreate(&P.Al, CUDA_R_4F_E2M1, K, M, K) == CUBLAS_STATUS_SUCCESS;
    ok = ok && cublasLtMatrixLayoutCreate(&P.Bl, CUDA_R_4F_E2M1, K, N, K) == CUBLAS_STATUS_SUCCESS;
    ok = ok && cublasLtMatrixLayoutCreate(&P.Dl, CUDA_R_32F, M, N, M) == CUBLAS_STATUS_SUCCESS;
    ok = ok && cublasLtMatmulPreferenceCreate(&P.pref) == CUBLAS_STATUS_SUCCESS;
    ok = ok && cublasLtMatmulPreferenceSetAttribute(P.pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &LT_WORKSPACE, sizeof(LT_WORKSPACE)) == CUBLAS_STATUS_SUCCESS;
    if (ok) {
        const cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(st.handle, P.op, P.Al, P.Bl, P.Dl, P.Dl, P.pref, 16, P.heur, &P.n_heur);
        ok = hs == CUBLAS_STATUS_SUCCESS && P.n_heur > 0;
    }
    P.ok = ok;
    return ok;
}

static cublasStatus_t lt_matmul(lt_state & st, lt_plan & P, int which, const float * scal, const uint8_t * A, const uint8_t * B, float * D,
                         char * ws, size_t ws_size, cudaStream_t stream) {
    return cublasLtMatmul(st.handle, P.op, scal, A, P.Al, B, P.Bl, scal + 1, D, P.Dl, D, P.Dl, &P.heur[which].algo, ws, ws_size, stream);
}

} // namespace

// src1 gives the activation SHAPE (and, with act1 == nullptr, the data); with act1 set the activation is gelu(act0) * act1,
// the GEGLU of two matmul outputs of src1's shape that never needs to exist in memory.
static bool lt_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                       ggml_tensor * dst, const ggml_tensor * x_scale, const float * act0, const float * act1, const int glu = 0) {
    if (src0->type != GGML_TYPE_NVFP4 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    if (!blackwell_mma_available(cc)) {
        return false;
    }
    GGML_TENSOR_BINARY_OP_LOCALS
    if (ne02 != 1 || ne03 != 1 || ne12 != 1 || ne13 != 1 || ne11 < LT_MIN_TOKENS || ne00 != ne10 || ne00 % 128 != 0) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }
    // the repack cache keys on the weight's address, so only tensors that live in a weights buffer qualify
    if (ggml_backend_buffer_get_usage(src0->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        return false;
    }
    if (x_scale && (x_scale->type != GGML_TYPE_F32 || ggml_nelements(x_scale) != 1)) {
        return false;
    }
    const int64_t M = ne01, K = ne00, N = ne11;
    const int64_t kblk = K / 16, kblk_pad = pad4(kblk);

    lt_state & st = lt_state_for(id);
    if (!st.handle && cublasLtCreate(&st.handle) != CUBLAS_STATUS_SUCCESS) {
        return false;
    }
    const uint64_t key = ((uint64_t) (M & 0xFFFFF) << 44) | ((uint64_t) (N & 0xFFFFFF) << 20) | (uint64_t) (K & 0xFFFFF);
    lt_plan & P = st.plans[key];
    if (P.pref && !P.ok) {
        return false;   // this shape was tried and refused
    }
    cudaStream_t stream = ctx.stream();

    // the weight, repacked once
    lt_repacked & w = st.weights[src0->data];
    if (!w.A) {
        const size_t sf_size = (size_t) pad128(M) * kblk_pad;
        CUDA_CHECK(cudaMalloc(&w.A, (size_t) M * K / 2));
        CUDA_CHECK(cudaMalloc(&w.SF, sf_size));
        CUDA_CHECK(cudaMemsetAsync(w.SF, 0, sf_size, stream));
        const int64_t n = M * kblk;
        k_repack<<<(unsigned) ((n + 255) / 256), 256, 0, stream>>>((const block_nvfp4 *) src0->data, (int) M, (int) K, w.A, w.SF, (int) kblk_pad);
        CUDA_CHECK(cudaGetLastError());
        w.M = M;
        w.K = K;
    } else if (w.M != M || w.K != K) {
        return false;   // the same address with another shape is not the weight this cache knows
    }

    // the activations, quantized for this call
    const size_t sfb_size = (size_t) pad128(N) * kblk_pad;
    const float * w_gs = x_scale ? (const float *) x_scale->data : nullptr;
    // the same src1 tensor object, quantized for an earlier node of this very graph evaluation: the FP4 operand is still in the
    // scratch (only this path writes it), so only alpha changes. Never across evaluations (the node index restarts from zero).
    const bool reuse = !act1 && st.cur_graph && st.q_graph == st.cur_graph && st.q_node >= 0 && st.q_node < st.cur_node &&
                       st.q_src1 == src1 && st.q_N == N && st.q_K == K && st.B.ptr && st.SFB.ptr && st.scal;
    uint8_t * B   = (uint8_t *) lt_scratch_get(st.B, (size_t) N * K / 2);
    uint8_t * SFB = (uint8_t *) lt_scratch_get(st.SFB, sfb_size);
    char *    ws  = (char *) lt_scratch_get(st.ws, LT_WORKSPACE);
    if (!st.scal) {
        CUDA_CHECK(cudaMalloc(&st.scal, 4 * sizeof(float)));
    }
    float * scal = st.scal;
    if (reuse) {
        k_alpha<<<1, 1, 0, stream>>>(reinterpret_cast<const unsigned *>(scal) + 2, w_gs, scal);
    } else {
        float *   d0  = (float *) lt_scratch_get(st.d0, (size_t) N * kblk * sizeof(float));
        CUDA_CHECK(cudaMemsetAsync(SFB, 0, sfb_size, stream));
        CUDA_CHECK(cudaMemsetAsync(scal, 0, 4 * sizeof(float), stream));
        const int64_t nq = N * kblk;
        const unsigned nblocks = (unsigned) ((N * (K / 128) + 7) / 8);   // a warp per 128 values, 8 warps per block
        float * bmax = (float *) lt_scratch_get(st.bmax, (size_t) nblocks * sizeof(float));
        if (act1 && glu == 2) {
            k_quant_pass1<2><<<nblocks, 256, 0, stream>>>(act0, act1, (int) N, (int) K, B, d0, bmax);
        } else if (act1) {
            k_quant_pass1<1><<<nblocks, 256, 0, stream>>>(act0, act1, (int) N, (int) K, B, d0, bmax);
        } else {
            k_quant_pass1<0><<<nblocks, 256, 0, stream>>>(act0, nullptr, (int) N, (int) K, B, d0, bmax);
        }
        k_quant_fold<<<1, 1024, 0, stream>>>(bmax, (int) nblocks, reinterpret_cast<unsigned *>(scal) + 2);
        k_quant_pass2<<<(unsigned) ((nq + 255) / 256), 256, 0, stream>>>(d0, (int) N, (int) K, reinterpret_cast<const unsigned *>(scal) + 2,
            w_gs, SFB, (int) kblk_pad, scal);
        // remember what the scratch holds (a GEGLU operand is unique to its matmul: not reusable)
        st.q_graph = act1 ? nullptr : st.cur_graph;
        st.q_node  = act1 ? -1 : st.cur_node;
        st.q_src1  = act1 ? nullptr : src1;
        st.q_N = N;
        st.q_K = K;
    }
    CUDA_CHECK(cudaGetLastError());

    const void * sfa = w.SF;
    const void * sfb = SFB;
    if (!P.pref && !lt_make_plan(st, P, M, N, K, sfa, sfb)) {
        if (!st.warned) {
            GGML_LOG_WARN("%s: cuBLASLt has no block-scaled FP4 algorithm for M=%" PRId64 " N=%" PRId64 " K=%" PRId64 ", MMQ runs it\n", __func__, M, N, K);
            st.warned = true;
        }
        return false;
    }
    if (!st.announced) {
        GGML_LOG_INFO("%s: NVFP4 matmuls with %" PRId64 "+ tokens run on cuBLASLt's block-scaled FP4 tensor-core path (cc %d)\n", __func__, LT_MIN_TOKENS, cc);
        st.announced = true;
    }
    if (cublasLtMatmulDescSetAttribute(P.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sfa, sizeof(sfa)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(P.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sfb, sizeof(sfb)) != CUBLAS_STATUS_SUCCESS) {
        return false;
    }
    float * D = (float *) dst->data;

    // first call for this shape: time every heuristic candidate on the real operands and keep the fastest (the operands
    // are consumed identically by each, so D holds the right result whichever ran last). Skipped under graph capture.
    if (!P.tuned) {
        const int remembered = lt_tune_lookup(M, N, K, cc);
        if (remembered >= 0 && remembered < P.n_heur) {
            P.best  = remembered;
            P.tuned = true;
        }
    }
    if (!P.tuned) {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &cap));
        if (cap == cudaStreamCaptureStatusNone) {
            cudaEvent_t e0, e1;
            CUDA_CHECK(cudaEventCreate(&e0));
            CUDA_CHECK(cudaEventCreate(&e1));
            float best_ms = 1e30f;
            for (int a = 0; a < P.n_heur; ++a) {
                if (lt_matmul(st, P, a, scal, w.A, B, D, ws, LT_WORKSPACE, stream) != CUBLAS_STATUS_SUCCESS) {
                    continue;
                }
                CUDA_CHECK(cudaEventRecord(e0, stream));
                for (int i = 0; i < 4; ++i) {
                    lt_matmul(st, P, a, scal, w.A, B, D, ws, LT_WORKSPACE, stream);
                }
                CUDA_CHECK(cudaEventRecord(e1, stream));
                CUDA_CHECK(cudaEventSynchronize(e1));
                float ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
                if (ms < best_ms) {
                    best_ms = ms;
                    P.best = a;
                }
            }
            CUDA_CHECK(cudaEventDestroy(e0));
            CUDA_CHECK(cudaEventDestroy(e1));
            P.tuned = true;
            lt_tune_store(M, N, K, cc, P.best);
        }
    }
    const cublasStatus_t s = lt_matmul(st, P, P.best, scal, w.A, B, D, ws, LT_WORKSPACE, stream);
    if (s != CUBLAS_STATUS_SUCCESS) {
        if (!st.warned) {
            GGML_LOG_WARN("%s: cublasLtMatmul failed (%d) for M=%" PRId64 " N=%" PRId64 " K=%" PRId64 ", MMQ runs it\n", __func__, (int) s, M, N, K);
            st.warned = true;
        }
        return false;
    }
    return true;
}

bool ggml_cuda_mul_mat_nvfp4_lt(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                ggml_tensor * dst, const ggml_tensor * x_scale) {
    return lt_mul_mat(ctx, src0, src1, dst, x_scale, (const float *) src1->data, nullptr);
}

void ggml_cuda_nvfp4_lt_note_node(const ggml_cgraph * cgraph, int node_idx) {
    lt_state & st = lt_state_for(ggml_cuda_get_device());
    st.cur_graph = cgraph;
    st.cur_node  = node_idx;
}

int ggml_cuda_rms_norm_nvfp4_lt(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
    if (node_idx + 2 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * norm = cgraph->nodes[node_idx];
    const ggml_tensor * mul  = cgraph->nodes[node_idx + 1];
    if (norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || norm->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 ||
        (norm->flags & GGML_TENSOR_FLAG_OUTPUT) || (mul->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_node_get_use_count(cgraph, node_idx) != 1) {
        return 0;
    }
    const ggml_tensor * x = norm->src[0];
    const ggml_tensor * w = mul->src[0] == norm ? mul->src[1] : (mul->src[1] == norm ? mul->src[0] : nullptr);
    if (!w || x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w) || !ggml_is_contiguous(mul) ||
        x->ne[1] < LT_MIN_TOKENS || x->ne[2] != 1 || x->ne[3] != 1 || x->ne[0] % 128 != 0 || x->ne[0] > 16384 ||
        w->ne[0] != x->ne[0] || ggml_nelements(w) != x->ne[0] || x->nb[0] != sizeof(float) || x->nb[1] % sizeof(float) != 0 ||
        !ggml_are_same_shape(norm, mul) || !ggml_are_same_shape(x, norm)) {
        return 0;
    }
    const int id = ggml_cuda_get_device();
    if (!blackwell_mma_available(ggml_cuda_info().devices[id].cc)) {
        return 0;
    }
    // every consumer of the normed tensor must be an NVFP4 matmul this path takes, else the F32 output is needed after all
    const int uses = ggml_node_get_use_count(cgraph, node_idx + 1);
    int found = 0;
    for (int j = node_idx + 2; j < cgraph->n_nodes && found < uses; ++j) {
        const ggml_tensor * c = cgraph->nodes[j];
        bool consumes = false;
        for (int k = 0; k < GGML_MAX_SRC; ++k) {
            consumes = consumes || c->src[k] == mul;
        }
        if (!consumes) {
            continue;
        }
        if (c->op != GGML_OP_MUL_MAT || c->src[1] != mul || c->src[0]->type != GGML_TYPE_NVFP4 || c->src[0]->ne[2] != 1 ||
            c->src[0]->ne[3] != 1 || c->src[0]->ne[0] != x->ne[0] || !ggml_is_contiguous(c->src[0]) ||
            ggml_backend_buffer_get_usage(c->src[0]->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            return 0;
        }
        found++;
    }
    if (found != uses || uses == 0) {
        return 0;
    }
    lt_state & st = lt_state_for(id);
    if (!st.handle && cublasLtCreate(&st.handle) != CUBLAS_STATUS_SUCCESS) {
        return 0;
    }
    const int64_t N = x->ne[1], K = x->ne[0], kblk = K / 16, kblk_pad = pad4(kblk);
    const size_t sfb_size = (size_t) pad128(N) * kblk_pad;
    cudaStream_t stream = ctx.stream();
    uint8_t * B   = (uint8_t *) lt_scratch_get(st.B, (size_t) N * K / 2);
    uint8_t * SFB = (uint8_t *) lt_scratch_get(st.SFB, sfb_size);
    float *   d0  = (float *) lt_scratch_get(st.d0, (size_t) N * kblk * sizeof(float));
    float *   bmax = (float *) lt_scratch_get(st.bmax, (size_t) N * sizeof(float));
    if (!st.scal) {
        CUDA_CHECK(cudaMalloc(&st.scal, 4 * sizeof(float)));
    }
    float * scal = st.scal;
    float eps = 0.0f;
    memcpy(&eps, norm->op_params, sizeof(float));
    CUDA_CHECK(cudaMemsetAsync(SFB, 0, sfb_size, stream));
    CUDA_CHECK(cudaMemsetAsync(scal, 0, 4 * sizeof(float), stream));
    const unsigned nthreads = (unsigned) ((kblk + 31) / 32 * 32);
    k_norm_quant_pass1<<<(unsigned) N, nthreads, 0, stream>>>((const float *) x->data, (int64_t) (x->nb[1] / sizeof(float)), (const float *) w->data,
                                                              (int) K, eps, B, d0, bmax);
    k_quant_fold<<<1, 1024, 0, stream>>>(bmax, (int) N, reinterpret_cast<unsigned *>(scal) + 2);
    k_quant_pass2<<<(unsigned) ((N * kblk + 255) / 256), 256, 0, stream>>>(d0, (int) N, (int) K, reinterpret_cast<const unsigned *>(scal) + 2,
        nullptr, SFB, (int) kblk_pad, scal);
    CUDA_CHECK(cudaGetLastError());
    // park it: the consumers (later nodes of this evaluation with src1 == mul) reuse the operand and set their own alpha
    st.q_graph = cgraph;
    st.q_node  = node_idx + 1;
    st.q_src1  = mul;
    st.q_N = N;
    st.q_K = K;
    return 1;
}

int ggml_cuda_mul_mat_nvfp4_lt_glu(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
    if (node_idx + 1 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * glu = cgraph->nodes[node_idx];
    const ggml_tensor * mm  = cgraph->nodes[node_idx + 1];
    const ggml_glu_op glu_op = glu->op == GGML_OP_GLU ? ggml_get_glu_op(glu) : GGML_GLU_OP_COUNT;
    if (glu->op != GGML_OP_GLU || (glu_op != GGML_GLU_OP_GEGLU && glu_op != GGML_GLU_OP_SWIGLU) || glu->src[1] == nullptr ||
        ((const int32_t *) glu->op_params)[1] != 0 /* swapped */ || glu->type != GGML_TYPE_F32) {
        return 0;
    }
    const ggml_tensor * gate = glu->src[0];
    const ggml_tensor * up   = glu->src[1];
    if (gate->type != GGML_TYPE_F32 || up->type != GGML_TYPE_F32 || !ggml_is_contiguous(gate) || !ggml_is_contiguous(up) ||
        !ggml_is_contiguous(glu) || !ggml_are_same_shape(gate, glu) || !ggml_are_same_shape(up, glu)) {
        return 0;
    }
    if (mm->op != GGML_OP_MUL_MAT || mm->src[1] != glu || mm->src[0]->type != GGML_TYPE_NVFP4 ||
        (glu->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_node_get_use_count(cgraph, node_idx) != 1) {
        return 0;
    }
    // the weight's global scale, when the graph applies it right after the matmul (the MUL_MAT + MUL pattern of the MMQ fusion)
    ggml_tensor *       out = cgraph->nodes[node_idx + 1];
    const ggml_tensor * sc  = nullptr;
    int extra = 1;
    if (node_idx + 2 < cgraph->n_nodes) {
        const ggml_tensor * mul = cgraph->nodes[node_idx + 2];
        if (mul->op == GGML_OP_MUL && mul->type == GGML_TYPE_F32 && (mul->src[0] == mm || mul->src[1] == mm) &&
            ggml_are_same_shape(mm, mul) && !(mm->flags & GGML_TENSOR_FLAG_OUTPUT) && ggml_node_get_use_count(cgraph, node_idx + 1) == 1) {
            const ggml_tensor * other = mul->src[0] == mm ? mul->src[1] : mul->src[0];
            if (other->type == GGML_TYPE_F32 && ggml_nelements(other) == 1) {
                sc    = other;
                out   = cgraph->nodes[node_idx + 2];
                extra = 2;
            }
        }
    }
    if (!lt_mul_mat(ctx, mm->src[0], glu, out, sc, (const float *) gate->data, (const float *) up->data, glu_op == GGML_GLU_OP_SWIGLU ? 2 : 1)) {
        return 0;
    }
    return extra;
}

#else // GGML_CUDA_NVFP4_LT

bool ggml_cuda_mul_mat_nvfp4_lt(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                ggml_tensor * dst, const ggml_tensor * x_scale) {
    GGML_UNUSED_VARS(ctx, src0, src1, dst, x_scale);
    return false;   // no cuBLASLt here (HIP, MUSA, or a CUDA toolkit without FP4 types)
}

int ggml_cuda_mul_mat_nvfp4_lt_glu(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
    GGML_UNUSED_VARS(ctx, cgraph, node_idx);
    return 0;
}

void ggml_cuda_nvfp4_lt_note_node(const ggml_cgraph * cgraph, int node_idx) {
    GGML_UNUSED_VARS(cgraph, node_idx);
}

int ggml_cuda_rms_norm_nvfp4_lt(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
    GGML_UNUSED_VARS(ctx, cgraph, node_idx);
    return 0;
}

#endif // GGML_CUDA_NVFP4_LT
