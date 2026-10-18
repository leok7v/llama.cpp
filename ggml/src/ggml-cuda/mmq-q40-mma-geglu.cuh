#pragma once

#include "common.cuh"

// gate GEMM + up GEMM + op(gate)*up + quantize_mmq_q8_1 (DS4) as ONE kernel, op = GEGLU or SWIGLU: the two Q4_0 x Q8_1 products
// for one output tile accumulated side by side on the INT8 tensor cores (the exact arithmetic of mmq-q40-mma.cu), the tile leaving
// as the down projection's block_q8_1_mmq operand - the two F32 outputs are never written and never read back. The operand is
// bit-identical to [mmq-q40-mma.cu outputs -> quantize_mmq_q8_1_fused<DS4, op>] because the epilogue runs that kernel's own code.
// Dense 2-D only; the GLU hook calls it when the shape qualifies, the separate GEMMs run otherwise.
bool ggml_cuda_mmq_q40_mma_glu_supported(ggml_glu_op op, int cc, size_t smpbo, int64_t K, int64_t M, int64_t N);

void ggml_cuda_mmq_q40_mma_glu(ggml_glu_op op, const char * x_gate, const char * x_up, const char * y_q8_1_mmq, void * operand_out,
                               int64_t K, int64_t M, int64_t N, int64_t nb01_gate, int64_t nb01_up, cudaStream_t stream);

// the GEGLU form under its original names: the two above with op = GGML_GLU_OP_GEGLU
bool ggml_cuda_mmq_q40_mma_geglu_supported(int cc, size_t smpbo, int64_t K, int64_t M, int64_t N);

void ggml_cuda_mmq_q40_mma_geglu(const char * x_gate, const char * x_up, const char * y_q8_1_mmq, void * operand_out,
                                 int64_t K, int64_t M, int64_t N, int64_t nb01_gate, int64_t nb01_up, cudaStream_t stream);
