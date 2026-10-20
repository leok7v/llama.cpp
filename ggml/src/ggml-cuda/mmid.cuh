#pragma once

// The grouped tile schedule of mul_mat_q over experts: tile_bounds[e] = the number of J-wide column tiles of the experts
// before e (the exclusive prefix sum of ceil(tokens_e / J) over expert_bounds), tile_bounds[n_experts] = the total. A block
// of the MMQ kernel maps a linear tile index to (expert, column tile) through it and never visits an empty tile.
void ggml_cuda_launch_mm_tile_bounds(
        const int32_t * __restrict__ expert_bounds, int32_t * __restrict__ tile_bounds, int n_experts, int J, cudaStream_t stream);

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, bool write_inverse, cudaStream_t stream);
