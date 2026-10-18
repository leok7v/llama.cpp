#pragma once

#include "common.cuh"

// The MMQ path's quantized src1 operand (block_q8_1_mmq, the layout quantize_mmq_q8_1 writes), provided from persistent
// scratch instead of a per-call pool buffer, so that consecutive matmuls of one graph evaluation reading the same src1 tensor
// quantize it once, and so that a GEGLU or a pre-norm can write the operand directly (the F32 intermediate never exists).
// The fused kernels reproduce quantize_mmq_q8_1's arithmetic and layout value for value: the GEGLU and shared operands are
// bit-identical to the unfused path's; the pre-norm's row scale mirrors rms_norm_f32 but lands a last bit apart on some rows
// (fp32 summation order under -use_fast_math), see mmq-fuse.cu.
// Returns the operand for (src1, this call's layout); quantizes into the scratch unless a parked operand matches.
const char * ggml_cuda_mmq_operand(ggml_backend_cuda_context & ctx, ggml_type type_x, const ggml_tensor * src1,
                                   int64_t ne10, int64_t ne10_padded, int64_t ne11, int64_t ne12, int64_t ne13, size_t nbytes,
                                   cudaStream_t stream);

// Dispatcher hooks. GEGLU whose only consumer is a quantized matmul on the MMQ path: gelu(gate) * up written straight to the
// operand, then that matmul run; returns the extra nodes consumed (1) or 0. RMS_NORM + weight MUL whose every consumer is such a
// matmul: the normed activation written straight to the operand and parked for them; returns 1 or 0.
int  ggml_cuda_mmq_fuse_glu (ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx);
int  ggml_cuda_mmq_fuse_norm(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx);
// the start of a graph evaluation (nothing parked survives it) and the node being computed
void ggml_cuda_mmq_fuse_begin_graph(const ggml_cgraph * cgraph);
void ggml_cuda_mmq_fuse_note_node(const ggml_cgraph * cgraph, int node_idx);
