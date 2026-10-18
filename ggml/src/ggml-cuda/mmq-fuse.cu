// MMQ operand fusions. The quantized src1 operand (block_q8_1_mmq) lives in persistent scratch keyed on the src1 tensor within
// one graph evaluation, so matmuls sharing an input quantize it once; a GEGLU/SwiGLU or an RMS_NORM+MUL whose consumers are all
// MMQ matmuls is quantized straight into that operand with quantize_mmq_q8_1's arithmetic, the F32 intermediate never written;
// gate, up, GLU and down of an FFN block run as one fused GEMM (mmq-q40-mma-geglu.cu) plus one matmul when the shapes allow.
// Two slots: PLAIN holds the operand of an existing F32 tensor (evictable); HOOK holds one whose F32 source does not exist, which
// only its own consumers may take - a consumer that cannot aborts. Nothing fires below FUSE_MIN_TOKENS or on concurrent streams.
#include "mmq-fuse.cuh"
#include "mmq.cuh"
#include "mmq-q40-mma-geglu.cuh"   // gate + up + GEGLU/SWIGLU + Q8_1 quantize as one GEMM
#include "mmvq.cuh"
#include "quantize.cuh"
#include "unary.cuh"   // ggml_cuda_op_gelu_single, ggml_cuda_op_silu_single

#include <algorithm>
#include <cstring>

namespace {

constexpr int FUSE_MIN_TOKENS = 64;   // under that the plain path is cheap and the hooks stay out of the way


// ---- quantize_mmq_q8_1, with the value produced on the way in. MODE 0: x. MODE 1 (GEGLU): gelu(x) * g.
// MODE 2 (norm): rowscale[row] * x * w. MODE 3 (SWIGLU): silu(x) * g. MODE 4 (gated per-head norm): silu(g) * (rowscale[row*H + i/hd]
// * x * w[i % hd]) - rms_norm_f32<do_multiply> over each hd-wide head, then unary_gated_op_kernel<op_silu>(z, normed), in that order.
// Grid as quantize_mmq_q8_1_cuda launches it: (ne1 rows, ne0/(4*blockDim), 1).
constexpr int QMODE_GEGLU = 1, QMODE_NORM = 2, QMODE_SWIGLU = 3, QMODE_NORM_GATED = 4;   // MODE 0 (plain) has no instantiation here: quantize_mmq_q8_1_cuda runs it
template <mmq_q8_1_ds_layout ds_layout, int MODE>
static __global__ void quantize_mmq_q8_1_fused(
        const float * __restrict__ x, const float * __restrict__ g, const float * __restrict__ rowscale, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t ne0, const int ne1, const float * __restrict__ w, const int64_t hd) {

    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;
    GGML_UNUSED(g);
    GGML_UNUSED(rowscale);
    GGML_UNUSED(w);
    GGML_UNUSED(hd);

    const int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i00 = i0;
    const int64_t base_idx = (int64_t) blockIdx.x * s01;

    const float4 * x4 = (const float4 *) x;
    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    const int64_t k_block = i0 / QK8_1_MMQ;
    const int64_t iqs     = i0 % QK8_1_MMQ;

    float4 xi = i0 < ne00 ? x4[(base_idx + i00)/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if constexpr (MODE == QMODE_GEGLU) {
        const float4 gi = i0 < ne00 ? ((const float4 *) g)[(base_idx + i00)/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        xi.x = ggml_cuda_op_gelu_single(xi.x) * gi.x;   // unary_gated_op_kernel<op_gelu>: op(x) * g
        xi.y = ggml_cuda_op_gelu_single(xi.y) * gi.y;
        xi.z = ggml_cuda_op_gelu_single(xi.z) * gi.z;
        xi.w = ggml_cuda_op_gelu_single(xi.w) * gi.w;
    } else if constexpr (MODE == QMODE_SWIGLU) {
        const float4 gi = i0 < ne00 ? ((const float4 *) g)[(base_idx + i00)/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        xi.x = ggml_cuda_op_silu_single(xi.x) * gi.x;   // unary_gated_op_kernel<op_silu>: op(x) * g, op_silu(x) = x / (1 + expf(-x))
        xi.y = ggml_cuda_op_silu_single(xi.y) * gi.y;
        xi.z = ggml_cuda_op_silu_single(xi.z) * gi.z;
        xi.w = ggml_cuda_op_silu_single(xi.w) * gi.w;
    } else if constexpr (MODE == QMODE_NORM) {
        const float  sc = rowscale[blockIdx.x];
        const float4 wi = i0 < ne00 ? ((const float4 *) g)[i00/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        xi.x = sc * xi.x;   // rms_norm_f32<do_multiply>: r = scale*x; r *= m
        xi.y = sc * xi.y;
        xi.z = sc * xi.z;
        xi.w = sc * xi.w;
        xi.x *= wi.x;
        xi.y *= wi.y;
        xi.z *= wi.z;
        xi.w *= wi.w;
    } else if constexpr (MODE == QMODE_NORM_GATED) {
        const float  sc = rowscale[blockIdx.x * (ne00 / hd) + i0 / hd];
        const float4 wi = i0 < ne00 ? ((const float4 *) w)[(i0 % hd) / 4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const float4 gi = i0 < ne00 ? ((const float4 *) g)[(base_idx + i00)/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        xi.x = sc * xi.x;   // rms_norm_f32<do_multiply> on the head: r = scale*x; r *= m
        xi.y = sc * xi.y;
        xi.z = sc * xi.z;
        xi.w = sc * xi.w;
        xi.x *= wi.x;
        xi.y *= wi.y;
        xi.z *= wi.z;
        xi.w *= wi.w;
        xi.x = ggml_cuda_op_silu_single(gi.x) * xi.x;   // unary_gated_op_kernel<op_silu>: op(z) * normed
        xi.y = ggml_cuda_op_silu_single(gi.y) * xi.y;
        xi.z = ggml_cuda_op_silu_single(gi.z) * xi.z;
        xi.w = ggml_cuda_op_silu_single(gi.w) * xi.w;
    }
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

#pragma unroll
    for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    float sum = 0.0f;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
        for (int offset = vals_per_sum/8; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
        }
    }

    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x*d_inv);
    q.y = roundf(xi.y*d_inv);
    q.z = roundf(xi.z*d_inv);
    q.w = roundf(xi.w*d_inv);
    const float d = 1.0f / d_inv;

    const int64_t ib = k_block*ne1 + blockIdx.x;
    char4 * yqs4 = (char4 *) y[ib].qs;
    yqs4[iqs/4] = q;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs % 16 == 0 && iqs < 96) {
            y[ib].d2s6[2 + iqs/16] = sum;
            if (iqs % 64 == 0) {
                y[ib].d2s6[iqs/64] = d;
            }
        }
    } else if (iqs % 32 == 0) {
        if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
            y[ib].ds4[iqs/32] = make_half2(d, sum);
        } else {
            y[ib].d4[iqs/32]  = d;
        }
    }
}

// the RMS norm's per-row scale, rsqrt(mean(x^2) + eps), accumulated and reduced exactly as rms_norm_f32<block_size, vpt, vec>
// does it (one float4 per thread per step, block_reduce), so the scale is the unfused kernel's bit for bit
template <int block_size, int vpt>
static __global__ void k_rms_rowscale(const float * __restrict__ x, const int64_t stride_row, const int ncols, const float eps,
                                      float * __restrict__ rowscale) {
    __shared__ float s_sum[32];
    const int      ncols4 = ncols / 4;
    const int      tid    = threadIdx.x;
    const float4 * x4     = (const float4 *) (x + (int64_t) blockIdx.x * stride_row);

    float4 xs[vpt];
    float  tmp = 0.0f;

#pragma unroll
    for (int k = 0; k < vpt; k++) {
        const int i = tid + k*block_size;
        if (i < ncols4) {
            xs[k] = x4[i];
            tmp += xs[k].x*xs[k].x + xs[k].y*xs[k].y + xs[k].z*xs[k].z + xs[k].w*xs[k].w;
        }
    }
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    if (tid == 0) {
        rowscale[blockIdx.x] = rsqrtf(tmp / ncols + eps);
    }
}

// rms_norm_f32_dispatch's ladder for the vectorised kernel; false when it would take a kernel this file does not mirror
static bool rms_rowscale_cuda(const float * x, int64_t stride_row, int ncols, int nrows, float eps, float * rowscale, cudaStream_t stream) {
    const int n4 = ncols / 4;
    if (n4 <= 32) {
        k_rms_rowscale<  32, 1><<<nrows,   32, 0, stream>>>(x, stride_row, ncols, eps, rowscale);
    } else if (n4 <= 128) {
        k_rms_rowscale< 128, 1><<<nrows,  128, 0, stream>>>(x, stride_row, ncols, eps, rowscale);
    } else if (n4 <= 256) {
        k_rms_rowscale< 256, 1><<<nrows,  256, 0, stream>>>(x, stride_row, ncols, eps, rowscale);
    } else if (n4 <= 512) {
        k_rms_rowscale< 512, 1><<<nrows,  512, 0, stream>>>(x, stride_row, ncols, eps, rowscale);
    } else if (n4 <= 1024) {
        k_rms_rowscale<1024, 1><<<nrows, 1024, 0, stream>>>(x, stride_row, ncols, eps, rowscale);
    } else if (n4 <= 4096) {
        k_rms_rowscale<1024, 4><<<nrows, 1024, 0, stream>>>(x, stride_row, ncols, eps, rowscale);
    } else {
        return false;
    }
    return true;
}

template <int MODE>
static void quantize_fused_cuda(const float * x, const float * g, const float * rowscale, void * vy, ggml_type type_x,
                                int64_t ne00, int64_t s01, int64_t ne0, int ne1, cudaStream_t stream,
                                const float * w = nullptr, int64_t hd = 0) {
    const int64_t block_num_y = (ne0 + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(ne1, block_num_y, 1);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    switch (mmq_get_q8_1_ds_layout(type_x)) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            quantize_mmq_q8_1_fused<MMQ_Q8_1_DS_LAYOUT_D4, MODE><<<num_blocks, block_size, 0, stream>>>(x, g, rowscale, vy, ne00, s01, ne0, ne1, w, hd);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            quantize_mmq_q8_1_fused<MMQ_Q8_1_DS_LAYOUT_DS4, MODE><<<num_blocks, block_size, 0, stream>>>(x, g, rowscale, vy, ne00, s01, ne0, ne1, w, hd);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            quantize_mmq_q8_1_fused<MMQ_Q8_1_DS_LAYOUT_D2S6, MODE><<<num_blocks, block_size, 0, stream>>>(x, g, rowscale, vy, ne00, s01, ne0, ne1, w, hd);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

struct operand_slot {
    char * buf = nullptr; size_t cap = 0;
    bool valid = false;
    const ggml_cgraph * graph = nullptr;        // the evaluation it belongs to
    const ggml_tensor * src1  = nullptr;        // the F32 tensor it stands for
    mmq_q8_1_ds_layout layout = MMQ_Q8_1_DS_LAYOUT_D4;
    int64_t ne10_padded = 0, ne11 = 0, ne12 = 0, ne13 = 0;
    int remaining = 0;                          // HOOK slot: consumers still to come
};
// hook slots: operands whose F32 source is never written. HOOK_IN holds a pre-norm's or a GEGLU's operand for the matmuls that
// consume it; HOOK_GLU_OUT holds the down-proj operand the fused gate/up GEMM writes while HOOK_IN still holds its input.
enum { HOOK_IN = 0, HOOK_GLU_OUT = 1, N_HOOKS = 2 };
// the gated per-head norm pattern, armed at its RMS_NORM node and fired at its SILU node (the gate's own matmul sits between
// them in the graph order): RMS_NORM(x) -> MUL(w) -> MUL(SILU(z)) [-> RESHAPE] -> one MMQ matmul
struct gated_norm_pending {
    bool armed = false;
    const ggml_cgraph * graph = nullptr;
    const ggml_tensor * silu = nullptr, * mul2 = nullptr, * operand = nullptr;   // operand: the reshape, or mul2 itself
    const ggml_tensor * x = nullptr, * w = nullptr, * z = nullptr;
    int64_t hd = 0, K = 0, T = 0;
    int n_extra = 0;   // nodes the fire step consumes after the SILU: mul2 (+ the reshape)
    ggml_type type_x = GGML_TYPE_COUNT;
};
struct fuse_state {
    operand_slot plain, hooks[N_HOOKS];
    gated_norm_pending gated;
    float * rowscale = nullptr; size_t rowscale_cap = 0;             // the pre-norm hook's, used within one call
    float * rowscale_g = nullptr; size_t rowscale_g_cap = 0;         // the gated norm's, alive from arm to fire
    const ggml_cgraph * cur_graph = nullptr; int cur_node = -1;
    bool announced = false;
};
static fuse_state & fuse_state_for(int device) {
    static fuse_state s[GGML_CUDA_MAX_DEVICES];
    return s[device];
}
static void slot_reserve(operand_slot & sl, size_t need) {
    if (sl.cap < need) {
        if (sl.buf) {
            CUDA_CHECK(cudaFree(sl.buf));   // synchronous: nothing in flight reads the old buffer afterwards
        }
        CUDA_CHECK(cudaMalloc(&sl.buf, need));
        sl.cap = need;
    }
}
// the operand's byte size for a layout, with room for the largest tile padding mul_mat_q may ask for (J <= 128 blocks + a chunk)
static size_t operand_bytes(int64_t ne10_padded, int64_t ne11, int64_t ne12, int64_t ne13) {
    return (size_t) ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1) / QK8_1 + 256 * sizeof(block_q8_1_mmq) + 4096;
}
static bool slot_matches(const operand_slot & sl, const ggml_cgraph * graph, const ggml_tensor * src1, ggml_type type_x,
                         int64_t ne10_padded, int64_t ne11, int64_t ne12, int64_t ne13, size_t nbytes) {
    return sl.valid && sl.graph == graph && sl.src1 == src1 && sl.layout == mmq_get_q8_1_ds_layout(type_x) &&
           sl.ne10_padded == ne10_padded && sl.ne11 == ne11 && sl.ne12 == ne12 && sl.ne13 == ne13 && sl.cap >= nbytes;
}
static void slot_park(operand_slot & sl, const ggml_cgraph * graph, const ggml_tensor * src1, ggml_type type_x,
                      int64_t ne10_padded, int64_t ne11, int64_t ne12, int64_t ne13, int remaining) {
    sl.valid = true; sl.graph = graph; sl.src1 = src1; sl.layout = mmq_get_q8_1_ds_layout(type_x);
    sl.ne10_padded = ne10_padded; sl.ne11 = ne11; sl.ne12 = ne12; sl.ne13 = ne13; sl.remaining = remaining;
}
// the hooks stand down while the dispatcher runs nodes on concurrent streams (GGML_CUDA_GRAPH_OPT): one scratch, one stream; and
// while a hook operand other than the one about to be consumed still waits for consumers (never evict what was never written)
static bool hooks_allowed(ggml_backend_cuda_context & ctx, const fuse_state & st, const ggml_tensor * consuming) {
    if (!ctx.stream_context().concurrent_events.empty()) {
        return false;
    }
    for (int h = 0; h < N_HOOKS; ++h) {
        if (st.hooks[h].valid && st.hooks[h].remaining > 0 && st.hooks[h].src1 != consuming) {
            return false;
        }
    }
    return true;
}
static operand_slot * find_hook(fuse_state & st, const ggml_tensor * src1) {
    for (int h = 0; h < N_HOOKS; ++h) {
        if (st.hooks[h].valid && st.hooks[h].src1 == src1) {
            return &st.hooks[h];
        }
    }
    return nullptr;
}

// does this matmul take the MMQ path with a quantized src1 operand this file can provide?
static bool mmq_consumer_ok(const ggml_tensor * mm, const ggml_tensor * operand, int & cc) {
    if (mm->op != GGML_OP_MUL_MAT || mm->src[1] != operand || mm->src[2] != nullptr) {
        return false;
    }
    const ggml_tensor * src0 = mm->src[0];
    if (!ggml_is_quantized(src0->type) || src0->type == GGML_TYPE_NVFP4 || src0->type == GGML_TYPE_MXFP4 || mm->type != GGML_TYPE_F32) {
        return false;   // the FP4 types have their own routes and their own operand format
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || operand->ne[2] != 1 || operand->ne[3] != 1 || operand->ne[1] < FUSE_MIN_TOKENS ||
        src0->ne[0] != operand->ne[0] || src0->ne[0] % 4 != 0 || !ggml_is_contiguous(src0) || !ggml_is_contiguous(operand)) {
        // (this master has no split buffers: a quantized src0 with a token batch is the dispatcher's MMQ case or nothing)
        return false;
    }
    const int id = ggml_cuda_get_device();
    cc = ggml_cuda_info().devices[id].cc;
    const int64_t ne11 = operand->ne[1];
    if (ggml_cuda_should_use_mmvq(src0->type, cc, ne11) || !ggml_cuda_should_use_mmq(src0->type, cc, ne11, /*n_experts =*/ 0)) {
        return false;
    }
    return true;
}

static void announce(fuse_state & st, int cc) {
    if (!st.announced) {
        GGML_LOG_INFO("mmq-fuse: MMQ operand fusions active - gelu/silu(gate)*up and pre-norms quantize straight into the Q8_1 operand, shared inputs once (cc %d)\n", cc);
        st.announced = true;
    }
}

// the two gated units the fused quantizer and the fused gate/up GEMM carry (split form: gate and up as separate tensors);
// SWIGLU_OAI, GEGLU_ERF, GEGLU_QUICK and REGLU are not mirrored and keep the plain path
static bool glu_op_fusable(const ggml_glu_op op) {
    return op == GGML_GLU_OP_GEGLU || op == GGML_GLU_OP_SWIGLU;
}

} // namespace

void ggml_cuda_mmq_fuse_begin_graph(const ggml_cgraph * cgraph) {
    fuse_state & st = fuse_state_for(ggml_cuda_get_device());
    for (int h = 0; h < N_HOOKS; ++h) {
        if (st.hooks[h].valid && st.hooks[h].remaining > 0) {
            GGML_ABORT("mmq-fuse: a fused operand parked for %d more consumer(s) of %s was never taken - the graph changed under it\n",
                       st.hooks[h].remaining, st.hooks[h].src1 ? st.hooks[h].src1->name : "?");
        }
        st.hooks[h].valid = false;
    }
    if (st.gated.armed) {
        GGML_ABORT("mmq-fuse: a gated norm armed at %s was never fired - the graph changed under it\n", st.gated.x ? st.gated.x->name : "?");
    }
    st.plain.valid = false;
    st.cur_graph = cgraph;
    st.cur_node  = -1;
}

void ggml_cuda_mmq_fuse_note_node(const ggml_cgraph * cgraph, int node_idx) {
    fuse_state & st = fuse_state_for(ggml_cuda_get_device());
    if (cgraph != st.cur_graph) {
        ggml_cuda_mmq_fuse_begin_graph(cgraph);
    }
    st.cur_node = node_idx;
}

const char * ggml_cuda_mmq_operand(ggml_backend_cuda_context & ctx, ggml_type type_x, const ggml_tensor * src1,
                                   int64_t ne10, int64_t ne10_padded, int64_t ne11, int64_t ne12, int64_t ne13, size_t nbytes,
                                   cudaStream_t stream) {
    fuse_state & st = fuse_state_for(ggml_cuda_get_device());
    // an operand a hook parked for this very tensor: its F32 source was never written, so it is this or nothing
    if (operand_slot * hk = find_hook(st, src1)) {
        if (!slot_matches(*hk, st.cur_graph, src1, type_x, ne10_padded, ne11, ne12, ne13, nbytes)) {
            GGML_ABORT("mmq-fuse: the fused operand of %s does not fit its consumer (layout %d/%d, padded K %lld/%lld, N %lld/%lld, bytes %zu/%zu)\n",
                       src1->name, (int) hk->layout, (int) mmq_get_q8_1_ds_layout(type_x), (long long) hk->ne10_padded,
                       (long long) ne10_padded, (long long) hk->ne11, (long long) ne11, hk->cap, nbytes);
        }
        if (--hk->remaining <= 0) {
            hk->valid = false;
        }
        return hk->buf;
    }
    const bool reuse_ok = ctx.stream_context().concurrent_events.empty();
    if (reuse_ok && slot_matches(st.plain, st.cur_graph, src1, type_x, ne10_padded, ne11, ne12, ne13, nbytes)) {
        return st.plain.buf;
    }
    slot_reserve(st.plain, std::max(nbytes, operand_bytes(ne10_padded, ne11, ne12, ne13)));
    const size_t ts = ggml_type_size(src1->type);
    quantize_mmq_q8_1_cuda((const float *) src1->data, nullptr, st.plain.buf, type_x, ne10, src1->nb[1] / ts, src1->nb[2] / ts,
                           src1->nb[3] / ts, ne10_padded, ne11, ne12, ne13, stream);
    CUDA_CHECK(cudaGetLastError());
    slot_park(st.plain, st.cur_graph, src1, type_x, ne10_padded, ne11, ne12, ne13, 0);
    st.plain.valid = reuse_ok;
    return st.plain.buf;
}

int ggml_cuda_mmq_fuse_glu(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
    if (node_idx + 1 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * glu = cgraph->nodes[node_idx];
    ggml_tensor *       mm  = cgraph->nodes[node_idx + 1];
    if (glu->op != GGML_OP_GLU || !glu_op_fusable(ggml_get_glu_op(glu)) || glu->src[1] == nullptr ||
        ((const int32_t *) glu->op_params)[1] != 0 || glu->type != GGML_TYPE_F32) {
        return 0;
    }
    const ggml_glu_op glu_op = ggml_get_glu_op(glu);
    const ggml_tensor * gate = glu->src[0];
    const ggml_tensor * up   = glu->src[1];
    if (gate->type != GGML_TYPE_F32 || up->type != GGML_TYPE_F32 || !ggml_is_contiguous(gate) || !ggml_is_contiguous(up) ||
        !ggml_are_same_shape(gate, glu) || !ggml_are_same_shape(up, glu) || gate->nb[1] != up->nb[1] || glu->ne[0] % 4 != 0 ||
        ((uintptr_t) gate->data & 15) != 0 || ((uintptr_t) up->data & 15) != 0) {
        return 0;
    }
    if ((glu->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_node_get_use_count(cgraph, node_idx) != 1) {
        return 0;
    }
    int cc = 0;
    if (!mmq_consumer_ok(mm, glu, cc)) {
        return 0;
    }
    fuse_state & st = fuse_state_for(ggml_cuda_get_device());
    if (!hooks_allowed(ctx, st, nullptr)) {
        return 0;
    }
    announce(st, cc);
    const ggml_type type_x = mm->src[0]->type;
    const int64_t ne10 = glu->ne[0], ne11 = glu->ne[1];
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    operand_slot & hk = st.hooks[HOOK_IN];
    slot_reserve(hk, operand_bytes(ne10_padded, ne11, 1, 1));
    if (glu_op == GGML_GLU_OP_SWIGLU) {
        quantize_fused_cuda<QMODE_SWIGLU>((const float *) gate->data, (const float *) up->data, nullptr, hk.buf, type_x, ne10,
                                          gate->nb[1] / sizeof(float), ne10_padded, (int) ne11, ctx.stream());
    } else {
        quantize_fused_cuda<QMODE_GEGLU>((const float *) gate->data, (const float *) up->data, nullptr, hk.buf, type_x, ne10,
                                         gate->nb[1] / sizeof(float), ne10_padded, (int) ne11, ctx.stream());
    }
    CUDA_CHECK(cudaGetLastError());
    slot_park(hk, cgraph, glu, type_x, ne10_padded, ne11, 1, 1, /*remaining =*/ 1);
    // the consumer runs now, through the normal MMQ entry, and takes the operand parked for it
    st.cur_node = node_idx + 1;
    ggml_cuda_mul_mat_q(ctx, mm->src[0], glu, nullptr, mm);
    GGML_ASSERT(!hk.valid);
    return 1;
}

int ggml_cuda_mmq_fuse_norm(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
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
        x->ne[1] < FUSE_MIN_TOKENS || x->ne[2] != 1 || x->ne[3] != 1 || x->ne[0] % 4 != 0 || w->ne[0] != x->ne[0] ||
        ggml_nelements(w) != x->ne[0] || x->nb[0] != sizeof(float) || x->nb[1] % (4 * sizeof(float)) != 0 ||
        ((uintptr_t) x->data & 15) != 0 || ((uintptr_t) w->data & 15) != 0 ||
        !ggml_are_same_shape(norm, mul) || !ggml_are_same_shape(x, norm)) {
        return 0;
    }
    // every consumer of the normed tensor must be a matmul this path serves, all of one operand layout
    const int uses = ggml_node_get_use_count(cgraph, node_idx + 1);
    int found = 0, cc = 0;
    ggml_type type_x = GGML_TYPE_COUNT;
    for (int j = node_idx + 2; j < cgraph->n_nodes && found < uses; ++j) {
        const ggml_tensor * c = cgraph->nodes[j];
        bool consumes = false;
        for (int k = 0; k < GGML_MAX_SRC; ++k) {
            consumes = consumes || c->src[k] == mul;
        }
        if (!consumes) {
            continue;
        }
        if (!mmq_consumer_ok(c, mul, cc)) {
            return 0;
        }
        if (type_x == GGML_TYPE_COUNT) {
            type_x = c->src[0]->type;
        } else if (mmq_get_q8_1_ds_layout(type_x) != mmq_get_q8_1_ds_layout(c->src[0]->type)) {
            return 0;
        }
        found++;
    }
    if (found != uses || uses == 0) {
        return 0;
    }
    fuse_state & st = fuse_state_for(ggml_cuda_get_device());
    if (!hooks_allowed(ctx, st, nullptr)) {
        return 0;
    }
    const int64_t K = x->ne[0], N = x->ne[1];
    if (K / 4 > 4096) {
        return 0;   // rms_norm_f32 takes its scalar kernel there; not mirrored
    }
    announce(st, cc);
    const int64_t ne10_padded = GGML_PAD(K, MATRIX_ROW_PADDING);
    cudaStream_t stream = ctx.stream();
    if (st.rowscale_cap < (size_t) N) {
        if (st.rowscale) {
            CUDA_CHECK(cudaFree(st.rowscale));
        }
        CUDA_CHECK(cudaMalloc(&st.rowscale, (size_t) N * sizeof(float)));
        st.rowscale_cap = (size_t) N;
    }
    float eps = 0.0f;
    memcpy(&eps, norm->op_params, sizeof(float));
    if (!rms_rowscale_cuda((const float *) x->data, (int64_t) (x->nb[1] / sizeof(float)), (int) K, (int) N, eps, st.rowscale, stream)) {
        return 0;
    }
    operand_slot & hk = st.hooks[HOOK_IN];
    slot_reserve(hk, operand_bytes(ne10_padded, N, 1, 1));
    quantize_fused_cuda<QMODE_NORM>((const float *) x->data, (const float *) w->data, st.rowscale, hk.buf, type_x, K,
                                    x->nb[1] / sizeof(float), ne10_padded, (int) N, stream);
    CUDA_CHECK(cudaGetLastError());
    slot_park(hk, cgraph, mul, type_x, ne10_padded, N, 1, 1, /*remaining =*/ uses);   // its consumers, later nodes, take it
    return 1;
}

// gate matmul, up matmul, GLU (GEGLU or SWIGLU, split form), down matmul - the first three as one GEMM whose epilogue writes
// op(gate)*up quantized straight into the down-proj's operand (mmq-q40-mma-geglu.cu), then the down matmul on it. The gate and
// up F32 outputs never exist.
int ggml_cuda_mmq_fuse_gate_up_glu(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
    if (node_idx + 3 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * gate = cgraph->nodes[node_idx];
    const ggml_tensor * up   = cgraph->nodes[node_idx + 1];
    const ggml_tensor * glu  = cgraph->nodes[node_idx + 2];
    ggml_tensor *       down = cgraph->nodes[node_idx + 3];
    if (gate->op != GGML_OP_MUL_MAT || up->op != GGML_OP_MUL_MAT || glu->op != GGML_OP_GLU || down->op != GGML_OP_MUL_MAT) {
        return 0;
    }
    const ggml_glu_op glu_op = ggml_get_glu_op(glu);
    const ggml_tensor * x = gate->src[1];
    if (up->src[1] != x || glu->src[0] != gate || glu->src[1] != up || down->src[1] != glu || !glu_op_fusable(glu_op) ||
        ((const int32_t *) glu->op_params)[1] != 0 || glu->type != GGML_TYPE_F32 || gate->type != GGML_TYPE_F32 || up->type != GGML_TYPE_F32) {
        return 0;
    }
    if (gate->src[0]->type != GGML_TYPE_Q4_0 || up->src[0]->type != GGML_TYPE_Q4_0 || !ggml_are_same_shape(gate, up) ||
        !ggml_are_same_shape(gate, glu) || !ggml_are_same_shape(gate->src[0], up->src[0]) ||
        (gate->flags & GGML_TENSOR_FLAG_OUTPUT) || (up->flags & GGML_TENSOR_FLAG_OUTPUT) || (glu->flags & GGML_TENSOR_FLAG_OUTPUT) ||
        ggml_node_get_use_count(cgraph, node_idx) != 1 || ggml_node_get_use_count(cgraph, node_idx + 1) != 1 ||
        ggml_node_get_use_count(cgraph, node_idx + 2) != 1) {
        return 0;
    }
    int cc = 0, cc2 = 0, cc3 = 0;
    if (!mmq_consumer_ok(gate, x, cc) || !mmq_consumer_ok(up, x, cc2) || !mmq_consumer_ok(down, glu, cc3)) {
        return 0;
    }
    const int64_t K = x->ne[0], M = gate->ne[0], N = x->ne[1];
    const size_t  smpbo = ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo;
    if (!ggml_cuda_mmq_q40_mma_glu_supported(glu_op, cc, smpbo, K, M, N)) {
        return 0;
    }
    fuse_state & st = fuse_state_for(ggml_cuda_get_device());
    if (!hooks_allowed(ctx, st, x)) {
        return 0;
    }
    announce(st, cc);
    cudaStream_t stream = ctx.stream();
    // the shared input operand: parked by the pre-norm hook, or quantized here once (both matmuls' shares are taken)
    const int64_t K_padded = GGML_PAD(K, MATRIX_ROW_PADDING);
    const size_t  in_bytes = operand_bytes(K_padded, N, 1, 1);
    const char * y = ggml_cuda_mmq_operand(ctx, GGML_TYPE_Q4_0, x, K, K_padded, N, 1, 1, in_bytes, stream);
    const char * y2 = ggml_cuda_mmq_operand(ctx, GGML_TYPE_Q4_0, x, K, K_padded, N, 1, 1, in_bytes, stream);
    GGML_ASSERT(y == y2);
    // the down operand, written by the fused GEMM into its own slot (the input may still live in HOOK_IN)
    const int64_t M_padded = GGML_PAD(M, MATRIX_ROW_PADDING);
    operand_slot & out = st.hooks[HOOK_GLU_OUT];
    slot_reserve(out, operand_bytes(M_padded, N, 1, 1));
    ggml_cuda_mmq_q40_mma_glu(glu_op, (const char *) gate->src[0]->data, (const char *) up->src[0]->data, y, out.buf, K, M, N,
                              gate->src[0]->nb[1], up->src[0]->nb[1], stream);
    CUDA_CHECK(cudaGetLastError());
    slot_park(out, cgraph, glu, down->src[0]->type, M_padded, N, 1, 1, /*remaining =*/ 1);
    st.cur_node = node_idx + 3;
    ggml_cuda_mul_mat_q(ctx, down->src[0], glu, nullptr, down);
    GGML_ASSERT(!out.valid);
    return 3;
}

// RMS_NORM over hd-wide heads -> MUL(w) -> MUL(SILU(z)) [-> RESHAPE] -> one MMQ matmul (a gated output norm before a projection).
// Armed here: the per-head row scales are computed, the norm and its weight MUL are skipped; the quantize happens at the SILU.
int ggml_cuda_mmq_fuse_norm_gated_arm(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
    if (node_idx + 5 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * norm  = cgraph->nodes[node_idx];
    const ggml_tensor * mul_w = cgraph->nodes[node_idx + 1];
    if (norm->op != GGML_OP_RMS_NORM || mul_w->op != GGML_OP_MUL || norm->type != GGML_TYPE_F32 || mul_w->type != GGML_TYPE_F32 ||
        (norm->flags & GGML_TENSOR_FLAG_OUTPUT) || (mul_w->flags & GGML_TENSOR_FLAG_OUTPUT) ||
        ggml_node_get_use_count(cgraph, node_idx) != 1 || ggml_node_get_use_count(cgraph, node_idx + 1) != 1) {
        return 0;
    }
    const ggml_tensor * x = norm->src[0];
    const ggml_tensor * w = mul_w->src[0] == norm ? mul_w->src[1] : (mul_w->src[1] == norm ? mul_w->src[0] : nullptr);
    const int64_t hd = x->ne[0], H = x->ne[1], T = x->ne[2];
    if (!w || x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || !ggml_is_contiguous(w) ||
        x->ne[3] != 1 || hd % 4 != 0 || hd / 4 > 4096 || H < 1 || T < FUSE_MIN_TOKENS || ggml_nelements(w) != hd ||
        ((uintptr_t) x->data & 15) != 0 || ((uintptr_t) w->data & 15) != 0 || !ggml_are_same_shape(norm, mul_w) || !ggml_are_same_shape(x, norm)) {
        return 0;
    }
    // the gate: a SILU immediately before the MUL that joins it with the normed tensor; between here and there only nodes that
    // touch neither (the gate's own matmul)
    int j = node_idx + 2;
    for (; j + 1 < cgraph->n_nodes; ++j) {
        const ggml_tensor * c = cgraph->nodes[j];
        if (c->op == GGML_OP_MUL && (c->src[0] == mul_w || c->src[1] == mul_w)) {
            break;
        }
        for (int k = 0; k < GGML_MAX_SRC; ++k) {
            if (c->src[k] == mul_w || c->src[k] == norm) {
                return 0;
            }
        }
    }
    if (j + 1 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * mul2 = cgraph->nodes[j];
    const ggml_tensor * silu = mul2->src[0] == mul_w ? mul2->src[1] : mul2->src[0];
    if (silu != cgraph->nodes[j - 1] || silu->op != GGML_OP_UNARY || ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU ||
        silu->type != GGML_TYPE_F32 || mul2->type != GGML_TYPE_F32 || (silu->flags & GGML_TENSOR_FLAG_OUTPUT) ||
        (mul2->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_node_get_use_count(cgraph, j - 1) != 1 || ggml_node_get_use_count(cgraph, j) != 1 ||
        !ggml_is_contiguous(mul2) || !ggml_are_same_shape(mul2, mul_w)) {
        return 0;
    }
    const ggml_tensor * z = silu->src[0];
    if (z->type != GGML_TYPE_F32 || !ggml_is_contiguous(z) || ggml_nelements(z) != ggml_nelements(x) || ((uintptr_t) z->data & 15) != 0 ||
        z->ne[0] * z->ne[1] * z->ne[2] * z->ne[3] != hd * H * T) {
        return 0;
    }
    // the consumer: the next node, or a RESHAPE of mul2 and then the next node
    const ggml_tensor * operand = mul2;
    int n_extra = 1;
    const ggml_tensor * mm = cgraph->nodes[j + 1];
    if (mm->op == GGML_OP_RESHAPE && mm->src[0] == mul2) {
        if (j + 2 >= cgraph->n_nodes || ggml_node_get_use_count(cgraph, j + 1) != 1) {
            return 0;
        }
        operand = mm;
        n_extra = 2;
        mm      = cgraph->nodes[j + 2];
    }
    int cc = 0;
    if (operand->ne[0] != hd * H || operand->ne[1] != T || !mmq_consumer_ok(mm, operand, cc)) {
        return 0;
    }
    fuse_state & st = fuse_state_for(ggml_cuda_get_device());
    // nothing is parked here (only row scales are written), so a pre-norm operand still waiting for the gate's own matmul
    // is no obstacle; concurrent streams are (one scratch, one stream)
    if (st.gated.armed || !ctx.stream_context().concurrent_events.empty()) {
        return 0;
    }
    announce(st, cc);
    cudaStream_t stream = ctx.stream();
    const int64_t N = H * T;
    if (st.rowscale_g_cap < (size_t) N) {
        if (st.rowscale_g) {
            CUDA_CHECK(cudaFree(st.rowscale_g));
        }
        CUDA_CHECK(cudaMalloc(&st.rowscale_g, (size_t) N * sizeof(float)));
        st.rowscale_g_cap = (size_t) N;
    }
    float eps = 0.0f;
    memcpy(&eps, norm->op_params, sizeof(float));
    if (!rms_rowscale_cuda((const float *) x->data, hd, (int) hd, (int) N, eps, st.rowscale_g, stream)) {
        return 0;
    }
    CUDA_CHECK(cudaGetLastError());
    st.gated.armed = true; st.gated.graph = cgraph; st.gated.silu = silu; st.gated.mul2 = mul2; st.gated.operand = operand;
    st.gated.x = x; st.gated.w = w; st.gated.z = z; st.gated.hd = hd; st.gated.K = hd * H; st.gated.T = T;
    st.gated.n_extra = n_extra; st.gated.type_x = mm->src[0]->type;
    return 1;   // the weight MUL goes with the norm
}

// the SILU node of an armed pattern: silu(z) * normed quantized straight into the consumer's operand; the MUL (and the reshape) go with it
int ggml_cuda_mmq_fuse_norm_gated_fire(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
    fuse_state & st = fuse_state_for(ggml_cuda_get_device());
    gated_norm_pending & g = st.gated;
    if (!g.armed || g.graph != cgraph || cgraph->nodes[node_idx] != g.silu) {
        return 0;
    }
    if (node_idx + g.n_extra >= cgraph->n_nodes || cgraph->nodes[node_idx + 1] != g.mul2) {
        GGML_ABORT("mmq-fuse: the gated norm armed at %s does not find its MUL after the SILU\n", g.x->name);
    }
    operand_slot & hk = st.hooks[HOOK_IN];
    if (hk.valid && hk.remaining > 0) {
        GGML_ABORT("mmq-fuse: the gated norm armed at %s fires while an operand for %s is still parked\n", g.x->name, hk.src1 ? hk.src1->name : "?");
    }
    if (!ctx.stream_context().concurrent_events.empty()) {
        GGML_ABORT("mmq-fuse: the gated norm armed at %s fires under concurrent streams\n", g.x->name);
    }
    const int64_t ne10_padded = GGML_PAD(g.K, MATRIX_ROW_PADDING);
    slot_reserve(hk, operand_bytes(ne10_padded, g.T, 1, 1));
    quantize_fused_cuda<QMODE_NORM_GATED>((const float *) g.x->data, (const float *) g.z->data, st.rowscale_g, hk.buf, g.type_x, g.K,
                                          g.K, ne10_padded, (int) g.T, ctx.stream(), (const float *) g.w->data, g.hd);
    CUDA_CHECK(cudaGetLastError());
    slot_park(hk, cgraph, g.operand, g.type_x, ne10_padded, g.T, 1, 1, /*remaining =*/ 1);   // the matmul, next, takes it
    const int n_extra = g.n_extra;
    g.armed = false;
    return n_extra;
}

