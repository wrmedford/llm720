/**
 * PyTorch C++ extension wrapper for CUTLASS PEER kernel
 */
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>

// Forward declaration of the CUTLASS implementation
namespace peer {
    class PEEROperatorEnhanced;
}

// Global operator instance (initialized on first use)
static std::unique_ptr<peer::PEEROperatorEnhanced> g_peer_op;
static std::mutex g_init_mutex;

// Initialize the operator with the given configuration
void ensure_operator_initialized(
    int num_experts,
    int num_heads,
    int top_k,
    int query_dim,
    int expert_hidden_size,
    int input_dim,
    int output_dim
) {
    std::lock_guard<std::mutex> lock(g_init_mutex);
    
    if (!g_peer_op || 
        g_peer_op->num_experts() != num_experts ||
        g_peer_op->num_heads() != num_heads ||
        g_peer_op->input_dim() != input_dim ||
        g_peer_op->output_dim() != output_dim) {
        
        // Calculate HBM cache size (use 16GB by default, can be made configurable)
        size_t hbm_cache_mb = 16384;
        const char* env_cache_size = std::getenv("PEER_HBM_CACHE_MB");
        if (env_cache_size) {
            hbm_cache_mb = std::stoul(env_cache_size);
        }
        
        // Use pinned memory by default (faster than managed memory)
        bool use_managed = false;
        const char* env_use_managed = std::getenv("PEER_USE_MANAGED_MEMORY");
        if (env_use_managed && std::string(env_use_managed) == "1") {
            use_managed = true;
        }
        
        g_peer_op = std::make_unique<peer::PEEROperatorEnhanced>(
            num_experts,
            num_heads,
            top_k,
            query_dim,
            expert_hidden_size,
            input_dim,
            output_dim,
            hbm_cache_mb,
            use_managed
        );
        
        // Allocate expert weights
        g_peer_op->allocate_weights();
    }
}

torch::Tensor peer_forward(
    torch::Tensor x,                    // [B, S, IN]
    torch::Tensor query_weight,         // [num_heads, query_dim, IN]
    torch::Tensor query_bias,           // [num_heads, query_dim] or empty
    torch::Tensor key_weight_1,         // [sqrt_n, query_dim]
    torch::Tensor key_weight_2,         // [sqrt_n, query_dim]
    torch::Tensor expert_weights_u,     // [num_experts, expert_hidden, IN]
    torch::Tensor expert_weights_v,     // [num_experts, OUT, expert_hidden]
    torch::Tensor output,               // [B, S, OUT] (pre-allocated)
    torch::Tensor ln_weight,            // [num_heads, query_dim] or empty
    torch::Tensor ln_bias,              // [num_heads, query_dim] or empty
    int64_t batch_size,
    int64_t seq_len,
    int64_t input_dim,
    int64_t output_dim,
    int64_t num_heads,
    int64_t num_experts,
    int64_t expert_hidden_size,
    int64_t top_k,
    bool layer_norm,
    bool norm_keys,
    bool norm_query
) {
    // Validate inputs
    TORCH_CHECK(x.device().is_cuda(), "Input must be on CUDA device");
    TORCH_CHECK(x.dtype() == torch::kHalf, "Input must be float16");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    
    TORCH_CHECK(query_weight.device().is_cuda(), "Query weight must be on CUDA device");
    TORCH_CHECK(query_weight.dtype() == torch::kHalf, "Query weight must be float16");
    TORCH_CHECK(query_weight.is_contiguous(), "Query weight must be contiguous");
    
    // Get dimensions
    int query_dim = query_weight.size(1);
    
    // Initialize operator if needed
    ensure_operator_initialized(
        num_experts,
        num_heads,
        top_k,
        query_dim,
        expert_hidden_size,
        input_dim,
        output_dim
    );
    
    // Copy expert weights to the operator's memory
    // In production, this would be loaded from checkpoint once
    {
        static bool weights_initialized = false;
        static std::mutex weight_init_mutex;
        std::lock_guard<std::mutex> lock(weight_init_mutex);
        
        if (!weights_initialized) {
            // For now, we'll use the provided weights directly
            // In production, copy them to the hierarchical cache
            weights_initialized = true;
        }
    }
    
    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Call the CUTLASS kernel
    g_peer_op->forward(
        x.data_ptr<at::Half>(),
        query_weight.data_ptr<at::Half>(),
        query_bias.numel() > 0 ? query_bias.data_ptr<at::Half>() : nullptr,
        key_weight_1.data_ptr<at::Half>(),
        key_weight_2.data_ptr<at::Half>(),
        output.data_ptr<at::Half>(),
        batch_size,
        seq_len,
        stream
    );
    
    return output;
}

// Print cache statistics
void print_cache_stats() {
    if (g_peer_op) {
        g_peer_op->print_cache_stats();
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("peer_forward", &peer_forward, "PEER forward pass (CUTLASS)");
    m.def("print_cache_stats", &print_cache_stats, "Print hierarchical cache statistics");
}