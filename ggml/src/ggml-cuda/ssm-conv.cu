#include "common.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * src0_ptr, const float * src1_ptr,
                                    const float * bias_ptr,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

    ggml_cuda_pdl_sync();
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const float * __restrict__ bias,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const float * bias, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, threads, kNC>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,
                                                                        src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 7:  launch_kernel(std::integral_constant<int, 7 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 7, 9, 15 right now.");
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;

    // bias always comes with silu.
    GGML_ASSERT(!fuse_bias || fuse_silu);

    // The bias (when fused) is the non-conv operand of the ADD node.
    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
    float *       dst_d  = (float *) out->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    if (fuse_bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(ggml_nelements(bias) == nr);
    }

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    }
}

// Token-major input: x [d_inner, n_t, n_s] as a projection writes it (channels fastest), the carried state s [d_conv-1, d_inner, n_s]
// (taps fastest) read only at the first d_conv-1 window positions - no transposed concat on the way in. One thread per channel; a block
// stages its 32-token window plus the d_conv-1 preceding positions in shared memory, coalesced across channels on both sides.
template <bool apply_silu, int split_d_inner, int d_conv, int split_n_t>
static __global__ void ssm_conv_tm_f32(const float * __restrict__ x, const float * __restrict__ w, const float * __restrict__ s,
                                       const int x_nb1, const int x_nb2, const int w_nb1, const int s_nb1, const int s_nb2,
                                       float * __restrict__ dst, const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;   // sequence
    const int bidy = blockIdx.y;   // channel block
    const int bidz = blockIdx.z;   // token block
    const int ch   = bidy * split_d_inner + tid;

    const int64_t t0        = (int64_t) bidz * split_n_t;
    const int64_t local_n_t = min((int64_t) split_n_t, n_t - t0);

    constexpr int window = split_n_t + d_conv - 1;
    __shared__ float smem[window * split_d_inner];

    const char  * xb = (const char *) x + bidx * x_nb2;
    const float * sb = (const float *) ((const char *) s + bidx * s_nb2 + ch * s_nb1);

    for (int j = 0; j < window; ++j) {
        const int64_t t = t0 + j - (d_conv - 1);
        float v = 0.0f;
        if (t < 0) {
            v = sb[t + d_conv - 1];
        } else if (t < n_t) {
            v = *(const float *) (xb + t * x_nb1 + ch * sizeof(float));
        }
        smem[j * split_d_inner + tid] = v;
    }
    __syncthreads();

    float wr[d_conv];
    const float * wb = (const float *) ((const char *) w + ch * w_nb1);
#pragma unroll
    for (int j = 0; j < d_conv; ++j) {
        wr[j] = wb[j];
    }

    char * db = (char *) dst + bidx * dst_nb2 + ch * sizeof(float);
    for (int64_t i = 0; i < local_n_t; ++i) {
        float sumf = 0.0f;
#pragma unroll
        for (int j = 0; j < d_conv; ++j) {
            sumf += smem[(i + j) * split_d_inner + tid] * wr[j];
        }
        *(float *) (db + (t0 + i) * dst_nb1) = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu>
static void ssm_conv_tm_f32_cuda(const float * x, const float * w, const float * s,
                                 const int x_nb1, const int x_nb2, const int w_nb1, const int s_nb1, const int s_nb2,
                                 float * dst, const int dst_nb1, const int dst_nb2,
                                 const int64_t nc, const int64_t nr, const int64_t n_t, const int64_t n_s, cudaStream_t stream) {
    constexpr int threads   = 128;
    constexpr int split_n_t = 32;
    GGML_ASSERT(nr % threads == 0);
    const dim3 blocks(n_s, nr / threads, (n_t + split_n_t - 1) / split_n_t);
    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        ssm_conv_tm_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, 0, stream>>>(
            x, w, s, x_nb1, x_nb2, w_nb1, s_nb1, s_nb2, dst, dst_nb1, dst_nb2, n_t);
    };
    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 7:  launch_kernel(std::integral_constant<int, 7 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 7, 9, 15 right now.");
    }
}

void ggml_cuda_op_ssm_conv_tm(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * silu_dst) {
    const ggml_tensor * x = dst->src[0];   // [d_inner, n_t, n_s]
    const ggml_tensor * c = dst->src[1];   // [d_conv, d_inner]
    const ggml_tensor * s = dst->src[2];   // [d_conv-1, d_inner, n_s]
    ggml_tensor * out = silu_dst ? silu_dst : dst;

    const int64_t nc  = c->ne[0];
    const int64_t nr  = x->ne[0];
    const int64_t n_t = x->ne[1];
    const int64_t n_s = x->ne[2];

    GGML_ASSERT(x->type == GGML_TYPE_F32 && c->type == GGML_TYPE_F32 && s->type == GGML_TYPE_F32 && out->type == GGML_TYPE_F32);
    GGML_ASSERT(x->nb[0] == sizeof(float) && c->nb[0] == sizeof(float) && s->nb[0] == sizeof(float));
    GGML_ASSERT(s->ne[0] == nc - 1 && s->ne[1] == nr && s->ne[2] == n_s);
    GGML_ASSERT(out->ne[0] == nr && out->ne[1] == n_t && out->ne[2] == n_s && out->nb[0] == sizeof(float));

    cudaStream_t stream = ctx.stream();
    if (silu_dst) {
        ssm_conv_tm_f32_cuda<true>((const float *) x->data, (const float *) c->data, (const float *) s->data, x->nb[1], x->nb[2], c->nb[1],
                                   s->nb[1], s->nb[2], (float *) out->data, out->nb[1], out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_tm_f32_cuda<false>((const float *) x->data, (const float *) c->data, (const float *) s->data, x->nb[1], x->nb[2], c->nb[1],
                                    s->nb[1], s->nb[2], (float *) out->data, out->nb[1], out->nb[2], nc, nr, n_t, n_s, stream);
    }
}

