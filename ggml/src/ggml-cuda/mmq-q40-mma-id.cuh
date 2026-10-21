#pragma once

#include "common.cuh"

// The exact Q4_0 x Q8_1 TMA GEMM of mmq-q40-mma.cu for mul_mat_id: the routed experts' projections over the compact, expert-sorted
// block_q8_1_mmq operand that ggml_cuda_mul_mat_q's ids branch builds. One 3-D tensor map per weight tensor (the expert is the
// third coordinate), one CTA per (row tile, non-empty column tile) of the grouped schedule, the whole K per CTA: no stream-k and no
// fixup, every output written once in MMQ's single-pass rounding sequence, bit for bit. The ids branch calls it after the
// quantization when the shape qualifies; MMQ runs otherwise.
bool ggml_cuda_mmq_q40_mma_id_supported(int cc, size_t smpbo, int64_t ne00, int64_t ne01, int64_t ne02, int64_t nb01, int64_t nb02);

// x: ne02 experts of ne01 rows of ne00/32 block_q4_0 (row pitch nb01, expert pitch nb02 bytes); y: (ne00/128) x N block_q8_1_mmq,
// k-block major, the N compact rows sorted by expert; expert_bounds[e], expert_bounds[e+1]: the compact rows of expert e;
// ids_dst[row]: the dst row of a compact row, stride_dst floats apart.
void ggml_cuda_mmq_q40_mma_id(ggml_backend_cuda_context & ctx, const char * x, const char * y, float * dst,
                              const int32_t * ids_dst, const int32_t * expert_bounds,
                              int64_t ne00, int64_t ne01, int64_t ne02, int64_t N, int64_t stride_dst, int64_t nb01, int64_t nb02,
                              cudaStream_t stream);
