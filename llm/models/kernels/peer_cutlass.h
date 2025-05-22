/**
 * Header file for PEER CUTLASS operator
 */
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace peer {

class PEEROperatorEnhanced {
public:
    PEEROperatorEnhanced(
        int num_experts,
        int num_heads,
        int top_k,
        int query_dim,
        int expert_hidden_size,
        int input_dim,
        int output_dim,
        size_t hbm_cache_mb = 16384,
        bool use_managed = false
    );
    
    ~PEEROperatorEnhanced();
    
    void allocate_weights();
    
    // Copy weights from PyTorch tensors to internal buffers
    void copy_weights_from_torch(const __half* torch_u_weights, const __half* torch_v_weights);
    
    void forward(
        const __half* input,
        const __half* query_weight,
        const __half* query_bias,
        const __half* sub_keys1,
        const __half* sub_keys2,
        __half* output,
        int batch_size,
        int seq_len,
        cudaStream_t stream = 0
    );
    
    void print_cache_stats();
    
    // Getters for validation
    int num_experts() const;
    int num_heads() const;
    int input_dim() const;
    int output_dim() const;
};

}  // namespace peer