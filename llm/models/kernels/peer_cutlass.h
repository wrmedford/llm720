/**
 * Header file for PEER CUTLASS operator
 */
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <memory>

namespace peer {

// Forward declaration of implementation
struct PEEROperatorEnhancedImpl;

class PEEROperatorEnhanced {
private:
    // PIMPL idiom - hide all implementation details
    std::unique_ptr<PEEROperatorEnhancedImpl> pImpl;
    
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
    
    // Direct pointer mode: Use PyTorch tensors directly without copying
    void set_weight_pointers(const __half* torch_u_weights, const __half* torch_v_weights);
    
    void forward(
        const __half* input,
        const __half* query_weight,
        const __half* query_bias,
        const __half* sub_keys1,
        const __half* sub_keys2,
        __half* output,
        const __half* ln_scale,  // Added for LayerNorm scale
        const __half* ln_bias,   // Added for LayerNorm bias
        int batch_size,
        int seq_len,
        float dropout_rate = 0.0f,
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
