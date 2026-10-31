#pragma once

#include "common.cuh"

// dst = src0 (NVFP4 weights) x src1 (F32 activations), optionally times the one-element F32 x_scale the graph fused in,
// through cuBLASLt's block-scaled FP4 tensor-core GEMM. Returns false, having launched nothing, when the shape, the
// device or the library cannot take it - the caller then continues on the MMQ path.
// res (nullable): a same-shape F32 residual the dispatcher found as the ADD after the matmul (or after its scale MUL); folded in
// as beta * C, and dst is then the ADD's output.
bool ggml_cuda_mul_mat_nvfp4_lt(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                ggml_tensor * dst, const ggml_tensor * x_scale, const ggml_tensor * res = nullptr);

// GEGLU or SWIGLU node at cgraph->nodes[node_idx] whose only consumer is an NVFP4 matmul (optionally followed by the one-element
// MUL of the weight's global scale): the gated gelu and the activation quantize run as one kernel, the GLU output is
// never written, and the matmul takes the cuBLASLt path. Returns the number of extra nodes consumed (1 or 2), 0 when
// the pattern or the shape does not qualify (nothing launched).
int ggml_cuda_mul_mat_nvfp4_lt_glu(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx);

// The dispatcher calls this before computing cgraph->nodes[node_idx]; it lets consecutive matmuls of one evaluation that read
// the same src1 tensor share one FP4 quantization of it.
void ggml_cuda_nvfp4_lt_note_node(const ggml_cgraph * cgraph, int node_idx);

// RMS_NORM at cgraph->nodes[node_idx] followed by its weight MUL, whose every consumer is an NVFP4 matmul this path takes:
// the normed activation is written straight to FP4 and parked for those matmuls; the F32 tensor is never written. Returns 1
// (the MUL consumed) or 0 (nothing launched).
int ggml_cuda_rms_norm_nvfp4_lt(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx);

// The dense FFN at cgraph->nodes[node_idx], its first NVFP4 matmul: the gate and up matmuls of one input with their one-element
// scale MULs, the split SWIGLU or GEGLU, the down matmul with its scale MUL and the residual ADD, in consecutive nodes. The gate and
// up GEMMs run per column chunk whose two F32 slabs stay in L2, the GLU of each chunk is quantized from there, and the down GEMM
// reads the operand the whole quantize would write. Returns the extra nodes consumed (3 to 7) or 0 (nothing of the graph's tensors
// written); a refusal's reason is logged once per distinct reason.
int ggml_cuda_mul_mat_nvfp4_lt_ffn(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx);

// RMS_NORM over 128-wide heads at cgraph->nodes[node_idx], then its weight MUL, the gate's SILU, the MUL that joins them and an
// optional RESHAPE, in consecutive nodes, read by one NVFP4 matmul this path takes: silu(z) * normed is written straight to FP4
// at the norm and parked for the matmul. Returns the extra nodes consumed (3 or 4) or 0 (nothing launched); a refusal's reason
// is logged once per distinct reason.
int ggml_cuda_gated_norm_nvfp4_lt(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx);
