#pragma once

#include "common.cuh"

// Q4_0 weights x Q8_1 activations (MMQ's block_q8_1_mmq DS4 operand, as quantize_mmq_q8_1 writes it) on the INT8 tensor cores
// of sm_120/121: the same integer dot products and the same fp32 rounding sequence as MMQ's single-pass path, bit for bit.
// Dense 2-D only; the Q4_0 GEMM case of ggml_cuda_mul_mat_q calls it after the src1 quantization when the shape qualifies,
// MMQ runs otherwise.
bool ggml_cuda_mmq_q40_mma_supported(int cc, size_t smpbo, int64_t ne00, int64_t ne01, int64_t ne11);

// x: M rows of K/32 block_q4_0 (row pitch nb01 bytes); y: (K/128) x N block_q8_1_mmq, k-block major; dst: N x M, token stride
// stride_dst floats. res (nullable): an N x M residual, token stride stride_res floats, added in the epilogue.
void ggml_cuda_mmq_q40_mma(const char * x, const char * y, float * dst,
                           int64_t ne00, int64_t ne01, int64_t ne11, int64_t stride_dst, int64_t nb01,
                           const float * res, int64_t stride_res, cudaStream_t stream);
