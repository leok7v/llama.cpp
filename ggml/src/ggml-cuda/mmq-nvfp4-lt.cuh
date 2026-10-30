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

// The dense FFN at cgraph->nodes[node_idx], its first NVFP4 matmul: gate and up (either order, each with its optional scale MUL),
// their GLU, the NVFP4 down matmul, its optional scale MUL and residual ADD, as one group. The gate and up GEMMs write BF16 into
// scratch, the GLU quantize reads BF16, and the F32 gate, up and GLU tensors of the graph are never written. _match returns the
// index of the group's last node (-1 when the pattern or the shape does not qualify) and launches nothing; the dispatcher runs its
// memory-range check over [node_idx, last] and then calls the run, which returns false, having launched nothing, when the BF16 plan
// does not exist for the shape.
int  ggml_cuda_ffn_nvfp4_lt_match(const ggml_cgraph * cgraph, int node_idx);
bool ggml_cuda_ffn_nvfp4_lt(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx, int last);

// A gated per-head output norm: RMS_NORM over hd-wide heads at cgraph->nodes[node_idx], its weight MUL, later SILU(z) and the MUL
// that joins them [and a RESHAPE], feeding one NVFP4 matmul. _arm at the norm computes the per-head row scales and returns 1 (the
// weight MUL consumed); _fire at the SILU quantizes silu(z) * normed straight to FP4, parks it for the matmul, and returns the nodes
// consumed (the MUL, and the reshape). Both return 0 when the pattern does not qualify (nothing launched).
int ggml_cuda_gated_norm_nvfp4_lt_arm(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx);
int ggml_cuda_gated_norm_nvfp4_lt_fire(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx);

// The dispatcher calls this before computing cgraph->nodes[node_idx]; it lets consecutive matmuls of one evaluation that read
// the same src1 tensor share one FP4 quantization of it.
void ggml_cuda_nvfp4_lt_note_node(const ggml_cgraph * cgraph, int node_idx);

// RMS_NORM at cgraph->nodes[node_idx] followed by its weight MUL, whose every consumer is an NVFP4 matmul this path takes:
// the normed activation is written straight to FP4 and parked for those matmuls; the F32 tensor is never written. Returns 1
// (the MUL consumed) or 0 (nothing launched).
int ggml_cuda_rms_norm_nvfp4_lt(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx);
