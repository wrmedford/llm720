/**
 * PyTorch C++ extension wrapper for CUTLASS PEER kernel with FP8 support
 */
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>

// Include the CUTLASS implementation header
#include "peer_cutlass.h"

// Global operator instance (initialized on first use)
static std::unique_ptr<peer::PEEROperatorEnhanced> g_peer_op;
static std::unique_ptr<peer::PEEROperatorEnhanced> g_peer_op_fp8; // Separate instance for FP8
static std::mutex g_init_mutex;

// Initialize the operator with the given configuration
void ensure_operator_initialized(
    int num_experts,
    int num_heads,
    int top_k,
    int query_dim,
    int expert_hidden_size,
    int input_dim,
    int output_dim,
    bool use_fp8 = false)
{
    std::lock_guard<std::mutex> lock(g_init_mutex);

    auto &op_ptr = use_fp8 ? g_peer_op_fp8 : g_peer_op;

    if (!op_ptr ||
        op_ptr->num_experts() != num_experts ||
        op_ptr->num_heads() != num_heads ||
        op_ptr->input_dim() != input_dim ||
        op_ptr->output_dim() != output_dim ||
        op_ptr->use_fp8() != use_fp8)
    {

        // Calculate HBM cache size (use 16GB by default, can be made configurable)
        size_t hbm_cache_mb = 16384;
        const char *env_cache_size = std::getenv("PEER_HBM_CACHE_MB");
        if (env_cache_size)
        {
            hbm_cache_mb = std::stoul(env_cache_size);
        }

        // Use pinned memory by default (faster than managed memory)
        bool use_managed = false;
        const char *env_use_managed = std::getenv("PEER_USE_MANAGED_MEMORY");
        if (env_use_managed && std::string(env_use_managed) == "1")
        {
            use_managed = true;
        }

        op_ptr = std::make_unique<peer::PEEROperatorEnhanced>(
            num_experts,
            num_heads,
            top_k,
            query_dim,
            expert_hidden_size,
            input_dim,
            output_dim,
            hbm_cache_mb,
            use_managed,
            use_fp8 // Pass FP8 flag
        );

        // Allocate expert weights
        op_ptr->allocate_weights();
    }
}

torch::Tensor peer_forward_impl(
    torch::Tensor x,                // [B, S, IN]
    torch::Tensor query_weight,     // [num_heads, query_dim, IN]
    torch::Tensor query_bias,       // [num_heads, query_dim] or empty
    torch::Tensor key_weight_1,     // [sqrt_n, query_dim]
    torch::Tensor key_weight_2,     // [sqrt_n, query_dim]
    torch::Tensor expert_weights_u, // [num_experts, expert_hidden, IN]
    torch::Tensor expert_weights_v, // [num_experts, OUT, expert_hidden]
    torch::Tensor output,           // [B, S, OUT] (pre-allocated)
    torch::Tensor ln_weight,        // [num_heads, query_dim] or empty
    torch::Tensor ln_bias,          // [num_heads, query_dim] or empty
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
    bool norm_query,
    double dropout_rate,
    bool use_fp8)
{
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
        output_dim,
        use_fp8);

    auto &op_ptr = use_fp8 ? g_peer_op_fp8 : g_peer_op;

    // Handle expert weights synchronization
    {
        static bool initialized = false;
        static bool initialized_fp8 = false;
        static bool use_direct_mode = false;
        static std::mutex init_mutex;

        bool &init_flag = use_fp8 ? initialized_fp8 : initialized;

        if (!init_flag)
        {
            std::lock_guard<std::mutex> lock(init_mutex);
            if (!init_flag)
            {
                // Check if we should use direct pointer mode (no copies)
                const char *direct_mode_env = std::getenv("PEER_DIRECT_WEIGHT_ACCESS");
                use_direct_mode = (direct_mode_env && std::string(direct_mode_env) == "1");

                if (!use_direct_mode)
                {
                    // Traditional mode: allocate internal buffers
                    op_ptr->allocate_weights();
                }

                init_flag = true;
            }
        }

        if (use_direct_mode)
        {
            // Direct mode: Use PyTorch tensors directly (zero-copy)
            op_ptr->set_weight_pointers(
                reinterpret_cast<const __half *>(expert_weights_u.data_ptr<at::Half>()),
                reinterpret_cast<const __half *>(expert_weights_v.data_ptr<at::Half>()));
        }
        else
        {
            // Traditional mode: Copy weights on every forward pass
            op_ptr->copy_weights_from_torch(
                reinterpret_cast<const __half *>(expert_weights_u.data_ptr<at::Half>()),
                reinterpret_cast<const __half *>(expert_weights_v.data_ptr<at::Half>()));
        }
    }

    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Call the CUTLASS kernel
    op_ptr->forward(
        reinterpret_cast<const __half *>(x.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(query_weight.data_ptr<at::Half>()),
        query_bias.numel() > 0 ? reinterpret_cast<const __half *>(query_bias.data_ptr<at::Half>()) : nullptr,
        reinterpret_cast<const __half *>(key_weight_1.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(key_weight_2.data_ptr<at::Half>()),
        reinterpret_cast<__half *>(output.data_ptr<at::Half>()),
        ln_weight.numel() > 0 ? reinterpret_cast<const __half *>(ln_weight.data_ptr<at::Half>()) : nullptr,
        ln_bias.numel() > 0 ? reinterpret_cast<const __half *>(ln_bias.data_ptr<at::Half>()) : nullptr,
        batch_size,
        seq_len,
        static_cast<float>(dropout_rate),
        stream);

    return output;
}

// Wrapper functions for different precision modes
torch::Tensor peer_forward(
    torch::Tensor x, torch::Tensor query_weight, torch::Tensor query_bias,
    torch::Tensor key_weight_1, torch::Tensor key_weight_2,
    torch::Tensor expert_weights_u, torch::Tensor expert_weights_v,
    torch::Tensor output, torch::Tensor ln_weight, torch::Tensor ln_bias,
    int64_t batch_size, int64_t seq_len, int64_t input_dim, int64_t output_dim,
    int64_t num_heads, int64_t num_experts, int64_t expert_hidden_size, int64_t top_k,
    bool layer_norm, bool norm_keys, bool norm_query, double dropout_rate)
{
    return peer_forward_impl(
        x, query_weight, query_bias, key_weight_1, key_weight_2,
        expert_weights_u, expert_weights_v, output, ln_weight, ln_bias,
        batch_size, seq_len, input_dim, output_dim, num_heads, num_experts,
        expert_hidden_size, top_k, layer_norm, norm_keys, norm_query, dropout_rate,
        false // use_fp8 = false
    );
}

torch::Tensor peer_forward_fp8(
    torch::Tensor x, torch::Tensor query_weight, torch::Tensor query_bias,
    torch::Tensor key_weight_1, torch::Tensor key_weight_2,
    torch::Tensor expert_weights_u, torch::Tensor expert_weights_v,
    torch::Tensor output, torch::Tensor ln_weight, torch::Tensor ln_bias,
    int64_t batch_size, int64_t seq_len, int64_t input_dim, int64_t output_dim,
    int64_t num_heads, int64_t num_experts, int64_t expert_hidden_size, int64_t top_k,
    bool layer_norm, bool norm_keys, bool norm_query, double dropout_rate)
{
    return peer_forward_impl(
        x, query_weight, query_bias, key_weight_1, key_weight_2,
        expert_weights_u, expert_weights_v, output, ln_weight, ln_bias,
        batch_size, seq_len, input_dim, output_dim, num_heads, num_experts,
        expert_hidden_size, top_k, layer_norm, norm_keys, norm_query, dropout_rate,
        true // use_fp8 = true
    );
}

// Print cache statistics
void print_cache_stats()
{
    if (g_peer_op)
    {
        std::cout << "FP16 cache statistics:" << std::endl;
        g_peer_op->print_cache_stats();
    }
    if (g_peer_op_fp8)
    {
        std::cout << "FP8 cache statistics:" << std::endl;
        g_peer_op_fp8->print_cache_stats();
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("peer_forward", &peer_forward, "PEER forward pass (CUTLASS)");
    m.def("peer_forward_fp8", &peer_forward_fp8, "PEER forward pass with FP8 (CUTLASS)");
    m.def("print_cache_stats", &print_cache_stats, "Print hierarchical cache statistics");
}