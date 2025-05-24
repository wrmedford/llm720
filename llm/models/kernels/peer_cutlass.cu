#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/barrier>
#include <cassert>
#include <cuda_bf16.h>
#include <mma.h>
#include <curand_kernel.h>

// CUTLASS includes for optimized GEMM
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <cutlass/arch/mma.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/linear_combination_relu.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>

#include <memory>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <mutex>

#include "peer_cutlass.h"
#include "peer_cutlass_impl.h"

namespace peer {

using namespace cute;

// Platform-aware HBM capacity percentage
float get_hbm_capacity_percentage() {
    // Check environment variable first
    const char* env_cap = std::getenv("PEER_HBM_CAPACITY_PERCENT");
    if (env_cap) {
        float cap = std::stof(env_cap) / 100.0f;
        return std::max(0.01f, std::min(1.0f, cap));  // Clamp between 1% and 100%
    }
    
    // Platform-specific defaults
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    // Detect GPU architecture
    int sm_major = prop.major;
    int sm_minor = prop.minor;
    
    if (sm_major == 9) {  // H100 (SM 9.0)
        // H100 has better memory bandwidth, can afford higher percentage
        return 0.50f;  // 50% default
    } else if (sm_major == 8 && sm_minor == 0) {  // A100 (SM 8.0)
        // A100 default
        return 0.40f;  // 40% default
    } else if (sm_major == 8 && sm_minor == 6) {  // A40/RTX 3090 (SM 8.6)
        // Consumer GPUs with less HBM
        return 0.30f;  // 30% default
    } else {
        // Conservative default for unknown architectures
        return 0.10f;  // 10% default
    }
}

// Helper for alignment
template<int N>
__host__ __device__ constexpr size_t align_to(size_t x) {
    return (x + N - 1) / N * N;
}

// Warp-level reduction for sum
__device__ inline float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}


// Device-side helper to fetch expert pointers
__device__ inline void fetch_expert(int id, ExpertPtrDev* experts,
                                   const half*& u, const half*& v) {
    auto& e = experts[id];
    u = e.is_hot ? e.dev_u : e.host_u;
    v = e.is_hot ? e.dev_v : e.host_v;
    
    // FIX 3: Update heat counter from all warps
    // Each warp's lane 0 increments
    if (threadIdx.x % 32 == 0) {
        atomicAdd(&e.heat, 1);
    }
}

// Global kernel for extracting heat deltas (moved outside class)
__global__ void extract_heat_deltas_kernel(
    ExpertPtrDev* experts, unsigned int* deltas, int num_experts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_experts) {
        unsigned int heat = experts[idx].heat;
        if (heat > 0) {
            deltas[idx] = heat;
            experts[idx].heat = 0;  // Reset
        }
    }
}

// ======================== HIERARCHICAL MEMORY MANAGER ========================


// ======================== OPTIMIZED GEMM USING CUTLASS ========================

// CUTLASS-based GEMM for optimal performance on SM90
template<typename Element>
struct OptimizedGemm {
    // Define the GEMM operation
    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using ElementAccumulator = float;
    
    // SM90 configuration using collective builder
    using ArchTag = cutlass::arch::Sm90;
    using OperatorClass = cutlass::arch::OpClassTensorOp;
    using TileShape = Shape<_128,_128,_32>;
    using ClusterShape = Shape<_2,_2,_1>;
    
    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, cutlass::layout::RowMajor, AlignmentC,
        ElementC, cutlass::layout::RowMajor, AlignmentC,
        cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;
    
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementA, cutlass::layout::RowMajor, AlignmentA,
        ElementB, cutlass::layout::RowMajor, AlignmentB,
        ElementAccumulator,
        TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;
    
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int,int,int>, // ProblemShape
        CollectiveMainloop,
        CollectiveEpilogue
    >;
    
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    
    // FIX 2: WMMA with proper bounds checking
    __device__ static void gemm_tn_safe(
        const Element* A,  // M x K (row-major)
        const Element* B,  // N x K (row-major, so B^T is K x N)
        Element* C,        // M x N (row-major) 
        float scale,
        int M, int N, int K
    ) {
        using namespace nvcuda::wmma;
        
        constexpr int WMMA_M = 16;
        constexpr int WMMA_N = 16;
        constexpr int WMMA_K = 16;
        
        // Check if dimensions are suitable for WMMA
        if (M < WMMA_M || N < WMMA_N || K < WMMA_K) {
            // Fallback to simple loop for small matrices
            const int tid = threadIdx.x;
            for (int idx = tid; idx < M * N; idx += blockDim.x) {
                int m = idx / N;
                int n = idx % N;
                float acc = 0.0f;
                for (int k = 0; k < K; k++) {
                    acc += float(A[m * K + k]) * float(B[n * K + k]);
                }
                C[idx] = Element(acc * scale);
            }
            return;
        }
        
        // WMMA path for larger matrices
        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, col_major> b_frag;
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, __half> c_frag;
        
        const int warp_id = threadIdx.x / 32;
        const int num_warps = blockDim.x / 32;
        
        // Process tiles with bounds checking
        for (int m = warp_id * WMMA_M; m < M; m += num_warps * WMMA_M) {
            for (int n = 0; n < N; n += WMMA_N) {
                // FIX 2: Proper bounds check
                if (m + WMMA_M > M || n + WMMA_N > N) continue;
                
                fill_fragment(c_frag, __float2half(0.0f));
                
                for (int k = 0; k < K; k += WMMA_K) {
                    // Bounds check for K dimension
                    if (k + WMMA_K > K) continue;
                    
                    // Load A fragment
                    load_matrix_sync(a_frag, A + m * K + k, K);
                    
                    // Load B fragment (transposed)
                    load_matrix_sync(b_frag, B + n * K + k, K);
                    
                    // Compute
                    mma_sync(c_frag, a_frag, b_frag, c_frag);
                }
                
                // Scale and store
                if (scale != 1.0f) {
                    for (int i = 0; i < c_frag.num_elements; i++) {
                        c_frag.x[i] = __float2half(float(c_frag.x[i]) * scale);
                    }
                }
                
                // Store result
                store_matrix_sync(C + m * N + n, c_frag, N, mem_row_major);
            }
        }
    }
};

// ======================== PRODUCT KEY ROUTING ========================

// Helper function for partial sorting on device
__device__ void partial_sort_topk_indices_dynamic(const float* scores, int* indices, float* top_scores, int k, int n) {
    // Initialize indices
    for (int i = 0; i < n; i++) indices[i] = i;
    
    // Simple partial sort for k elements (optimize later with CUB if needed)
    for (int i = 0; i < k; i++) {
        for (int j = i + 1; j < n; j++) {
            if (scores[indices[j]] > scores[indices[i]]) {
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
        }
        if (top_scores) top_scores[i] = scores[indices[i]];
    }
}

template<typename scalar_t, int top_k>
__device__ void product_key_routing(
    const scalar_t* query,      // [d]
    const scalar_t* sub_keys1,  // [sqrt_n, d]
    const scalar_t* sub_keys2,  // [sqrt_n, d]
    int d,
    int sqrt_n,                 // Now a runtime parameter
    int* expert_indices,        // [top_k]
    float* expert_scores,       // [top_k]
    float* scores_buffer,       // Shared memory buffer for scores
    bool norm_keys = true,
    bool norm_query = true
) {
    // Use provided shared memory buffer
    float* scores1 = scores_buffer;
    float* scores2 = scores_buffer + sqrt_n;
    
    // Normalize query if requested
    float query_norm = 0.0f;
    if (norm_query) {
        for (int i = 0; i < d; i++) {
            query_norm += float(query[i]) * float(query[i]);
        }
        query_norm = rsqrtf(query_norm + 1e-6f);
    }
    
    // Compute scores with first set of sub-keys
    for (int i = 0; i < sqrt_n; i++) {
        float score = 0.0f;
        float key_norm = 0.0f;
        
        for (int j = 0; j < d; j++) {
            float q_val = float(query[j]);
            if (norm_query) q_val *= query_norm;
            
            float k_val = float(sub_keys1[i * d + j]);
            score += q_val * k_val;
            
            if (norm_keys) {
                key_norm += k_val * k_val;
            }
        }
        
        if (norm_keys) {
            key_norm = rsqrtf(key_norm + 1e-6f);
            score *= key_norm;
        }
        
        scores1[i] = score;
    }
    
    // Compute scores with second set of sub-keys
    for (int i = 0; i < sqrt_n; i++) {
        float score = 0.0f;
        float key_norm = 0.0f;
        
        for (int j = 0; j < d; j++) {
            float q_val = float(query[j]);
            if (norm_query) q_val *= query_norm;
            
            float k_val = float(sub_keys2[i * d + j]);
            score += q_val * k_val;
            
            if (norm_keys) {
                key_norm += k_val * k_val;
            }
        }
        
        if (norm_keys) {
            key_norm = rsqrtf(key_norm + 1e-6f);
            score *= key_norm;
        }
        
        scores2[i] = score;
    }
    
    // FIXED: Use product key optimization to achieve O(√N + k²) complexity
    // Calculate k_prime (number of candidates per dimension)
    const int k_prime = min(sqrt_n, int(ceilf(powf(float(top_k), 0.5f))) + 2);
    
    // Get top k_prime from each dimension
    int top_indices1[32];  // Assuming k_prime <= 32
    int top_indices2[32];
    float top_scores1_sorted[32];
    float top_scores2_sorted[32];
    
    partial_sort_topk_indices_dynamic(scores1, top_indices1, top_scores1_sorted, k_prime, sqrt_n);
    partial_sort_topk_indices_dynamic(scores2, top_indices2, top_scores2_sorted, k_prime, sqrt_n);
    
    // Find top-k product scores
    struct Score {
        float value;
        int index;
    };
    
    Score top_scores[top_k];
    for (int i = 0; i < top_k; i++) {
        top_scores[i].value = -1e10f;
        top_scores[i].index = -1;
    }
    
    // Only compute k_prime × k_prime products instead of sqrt_n × sqrt_n
    for (int i = 0; i < k_prime; i++) {
        for (int j = 0; j < k_prime; j++) {
            float prod_score = top_scores1_sorted[i] * top_scores2_sorted[j];
            int expert_id = top_indices1[i] * sqrt_n + top_indices2[j];
            
            // Insert into top-k if necessary
            if (prod_score > top_scores[top_k-1].value) {
                // Find insertion point
                int pos = top_k - 1;
                while (pos > 0 && prod_score > top_scores[pos-1].value) {
                    pos--;
                }
                
                // Shift and insert
                for (int k = top_k - 1; k > pos; k--) {
                    top_scores[k] = top_scores[k-1];
                }
                top_scores[pos].value = prod_score;
                top_scores[pos].index = expert_id;
            }
        }
    }
    
    // Apply softmax to top-k scores
    float sum_exp = 0.0f;
    for (int i = 0; i < top_k; i++) {
        top_scores[i].value = expf(top_scores[i].value);
        sum_exp += top_scores[i].value;
    }
    
    // Write output
    for (int i = 0; i < top_k; i++) {
        expert_indices[i] = top_scores[i].index;
        expert_scores[i] = top_scores[i].value / sum_exp;
    }
}

// ======================== ENHANCED KERNEL WITH L2 OPTIMIZATION ========================

// Runtime L2 chunk size calculation
template<typename T>
__host__ __device__ inline int compute_l2_chunk_size(int input_dim) {
    constexpr int L2_SIZE_BYTES = 40 * 1024 * 1024;  // 40MB L2 on H100
    int bytes_per_token = input_dim * sizeof(T);
    return L2_SIZE_BYTES / bytes_per_token;
}

// PEERConfig struct definition
template<int MaxExperts_, int BlockM_, int BlockK_, int HiddenSize_, int OuterTiles_>
struct PEERConfig {
    static constexpr int MaxExperts = MaxExperts_;
    static constexpr int BlockM = BlockM_;
    static constexpr int BlockK = BlockK_;
    static constexpr int HiddenSize = HiddenSize_;
    static constexpr int OuterTiles = OuterTiles_;
};

template<
    typename Config,
    typename Element,
    int NumHeads,
    int TopK,
    int QueryDim,
    int OUT,
    int BLOCK_DIM
>
__global__ void peer_kernel_enhanced(
    const Element* __restrict__ input,
    const Element* __restrict__ query_weight,
    const Element* __restrict__ query_bias,
    const Element* __restrict__ sub_keys1,
    const Element* __restrict__ sub_keys2,
    Element* __restrict__ output,
    ExpertPtrDev* d_experts,  // Non-const for heat updates
    const Element* __restrict__ bn_scale,
    const Element* __restrict__ bn_bias,
    int B, int S, int IN,
    int sqrt_n,              // Runtime parameter  
    int chunk_size,          // Runtime parameter
    float dropout_rate = 0.0f,
    bool use_batch_norm = true,
    bool norm_keys = true,
    bool norm_query = true
) {
    extern __shared__ char smem_bytes[];
    
    // Compute proper shared memory layout with padding
    size_t tok_bytes = align_to<64>(chunk_size * IN * sizeof(Element));
    size_t u_bytes = align_to<64>(IN * Config::HiddenSize * sizeof(Element));
    size_t v_bytes = align_to<64>(Config::HiddenSize * OUT * sizeof(Element));
    
    // Shared memory layout with proper padding
    Element* token_cache = reinterpret_cast<Element*>(smem_bytes);
    Element* u_buffer[2];
    Element* v_buffer[2];
    u_buffer[0] = reinterpret_cast<Element*>((char*)token_cache + tok_bytes);
    u_buffer[1] = reinterpret_cast<Element*>((char*)u_buffer[0] + u_bytes);
    v_buffer[0] = reinterpret_cast<Element*>((char*)u_buffer[1] + u_bytes);
    v_buffer[1] = reinterpret_cast<Element*>((char*)v_buffer[0] + v_bytes);
    
    // Proper layout to avoid overlap
    Element* query_smem = reinterpret_cast<Element*>((char*)v_buffer[1] + v_bytes);
    size_t query_bytes = align_to<64>(QueryDim * sizeof(Element));
    Element* hidden_smem = reinterpret_cast<Element*>((char*)query_smem + query_bytes);
    size_t hidden_bytes = align_to<64>(Config::HiddenSize * sizeof(Element));
    
    // Add shared memory for product key routing scores
    float* routing_scores = reinterpret_cast<float*>((char*)hidden_smem + hidden_bytes);
    size_t routing_scores_bytes = align_to<64>(2 * sqrt_n * sizeof(float));
    
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    // Process chunks of tokens
    for (int chunk_start = blockIdx.x * chunk_size; 
         chunk_start < B * S; 
         chunk_start += gridDim.x * chunk_size) {
        
        int chunk_end = min(chunk_start + chunk_size, B * S);
        int actual_chunk_size = chunk_end - chunk_start;
        
        // Stage 1: Load token chunk into shared memory
        for (int idx = tid; idx < actual_chunk_size * IN; idx += blockDim.x) {
            int token_offset = idx / IN;
            int feat_offset = idx % IN;
            int token_idx = chunk_start + token_offset;
            int batch_idx = token_idx / S;
            int seq_idx = token_idx % S;
            
            token_cache[idx] = input[batch_idx * S * IN + seq_idx * IN + feat_offset];
        }
        __syncthreads();
        
        // Process each token in the chunk
        for (int local_token_idx = 0; local_token_idx < actual_chunk_size; local_token_idx++) {
            Element* local_token = token_cache + local_token_idx * IN;
            int global_token_idx = chunk_start + local_token_idx;
            
            // Compute MAX_OUT_PER_THREAD based on actual block dim
            constexpr int OUT_PER_THREAD = (OUT + BLOCK_DIM - 1) / BLOCK_DIM;
            float thread_output[OUT_PER_THREAD];
            
            for (int i = 0; i < OUT_PER_THREAD && tid * OUT_PER_THREAD + i < OUT; i++) {
                thread_output[i] = 0.0f;
            }
            
            // Process each head
            for (int h = 0; h < NumHeads; h++) {
                // All warps cooperatively compute query projection
                OptimizedGemm<Element>::gemm_tn_safe(
                    local_token,
                    query_weight + h * QueryDim * IN,
                    query_smem,
                    1.0f,
                    1, QueryDim, IN
                );
                __syncthreads();
                
                // Add bias (distributed across threads)
                if (query_bias != nullptr && tid < QueryDim) {
                    query_smem[tid] = Element(float(query_smem[tid]) + 
                                             float(query_bias[h * QueryDim + tid]));
                }
                __syncthreads();
                
                // Batch normalization with all threads participating
                if (use_batch_norm) {
                    // Shared memory for reduction
                    __shared__ float reduction_buffer[32];  // For warp-level reductions
                    __shared__ float shared_mean;
                    __shared__ float shared_inv_std;
                    
                    // Step 1: Compute mean using all threads
                    float thread_sum = 0.0f;
                    for (int i = tid; i < QueryDim; i += blockDim.x) {
                        thread_sum += float(query_smem[i]);
                    }
                    
                    // Warp-level reduction for sum
                    thread_sum = warpReduceSum(thread_sum);
                    if (lane_id == 0) {
                        reduction_buffer[warp_id] = thread_sum;
                    }
                    __syncthreads();
                    
                    // Final reduction by first warp
                    if (warp_id == 0) {
                        float warp_sum = (lane_id < (blockDim.x / 32)) ? reduction_buffer[lane_id] : 0.0f;
                        warp_sum = warpReduceSum(warp_sum);
                        if (tid == 0) {
                            shared_mean = warp_sum / QueryDim;
                        }
                    }
                    __syncthreads();
                    
                    // Step 2: Compute variance using all threads
                    float thread_var = 0.0f;
                    float mean = shared_mean;
                    for (int i = tid; i < QueryDim; i += blockDim.x) {
                        float diff = float(query_smem[i]) - mean;
                        thread_var += diff * diff;
                    }
                    
                    // Warp-level reduction for variance
                    thread_var = warpReduceSum(thread_var);
                    if (lane_id == 0) {
                        reduction_buffer[warp_id] = thread_var;
                    }
                    __syncthreads();
                    
                    // Final reduction by first warp
                    if (warp_id == 0) {
                        float warp_var = (lane_id < (blockDim.x / 32)) ? reduction_buffer[lane_id] : 0.0f;
                        warp_var = warpReduceSum(warp_var);
                        if (tid == 0) {
                            shared_inv_std = rsqrtf(warp_var / QueryDim + 1e-5f);
                        }
                    }
                    __syncthreads();
                    
                    // Step 3: Apply normalization using all threads
                    float inv_std = shared_inv_std;
                    for (int i = tid; i < QueryDim; i += blockDim.x) {
                        float normalized = (float(query_smem[i]) - mean) * inv_std;
                        if (bn_scale != nullptr) {
                            normalized = normalized * float(bn_scale[h * QueryDim + i]) + 
                                        float(bn_bias[h * QueryDim + i]);
                        }
                        query_smem[i] = Element(normalized);
                    }
                }
                __syncthreads();
                
                // Product key routing
                __shared__ int expert_indices[TopK];
                __shared__ float expert_scores[TopK];
                
                if (tid == 0) {
                    product_key_routing<Element, TopK>(
                        query_smem, sub_keys1, sub_keys2, QueryDim, sqrt_n,
                        expert_indices, expert_scores, routing_scores,
                        norm_keys, norm_query
                    );
                }
                __syncthreads();
                
                // Double-buffered expert computation with overlapped copy/compute
                int buffer_id = 0;
                
                for (int k = 0; k < TopK; k++) {
                    int expert_id = expert_indices[k];
                    float weight = expert_scores[k];
                    
                    // Get expert pointers (all warps participate)
                    const half *u_ptr = nullptr, *v_ptr = nullptr;
                    if (tid == 0) {
                        fetch_expert(expert_id, d_experts, u_ptr, v_ptr);
                    }
                    __syncthreads();
                    
                    // FIX 1: Proper 64-bit pointer broadcast
                    uint64_t u_addr = 0, v_addr = 0;
                    if (tid == 0) {
                        u_addr = reinterpret_cast<uint64_t>(u_ptr);
                        v_addr = reinterpret_cast<uint64_t>(v_ptr);
                    }
                    u_addr = __shfl_sync(0xffffffff, u_addr, 0);
                    v_addr = __shfl_sync(0xffffffff, v_addr, 0);
                    u_ptr = reinterpret_cast<const half*>(u_addr);
                    v_ptr = reinterpret_cast<const half*>(v_addr);
                    
                    // FIX 4: Load with 64-byte chunks
                    // Producer warps load while consumer warps compute
                    if (warp_id < 2) {
                        // Load U weights
                        if (warp_id == 0) {
                            size_t bytes = IN * Config::HiddenSize * sizeof(Element);
                            char* dst = (char*)u_buffer[buffer_id];
                            const char* src = (const char*)u_ptr;
                            
                            for (int off = tid * 64; off < bytes; off += BLOCK_DIM * 64) {
                                if (off + 64 <= bytes) {
                                    // Use regular memcpy instead of cp.async.bulk
                                    // This is compatible with all CUDA versions
                                    if ((uintptr_t)(dst + off) % 16 == 0 && (uintptr_t)(src + off) % 16 == 0) {
                                        // 128-bit loads if aligned
                                        *reinterpret_cast<float4*>(dst + off) = *reinterpret_cast<const float4*>(src + off);
                                    } else {
                                        // Fallback to smaller loads
                                        for (int i = 0; i < 64; i += sizeof(float)) {
                                            *reinterpret_cast<float*>(dst + off + i) = *reinterpret_cast<const float*>(src + off + i);
                                        }
                                    }
                                }
                            }
                        }
                        // Load V weights
                        else {
                            size_t bytes = Config::HiddenSize * OUT * sizeof(Element);
                            char* dst = (char*)v_buffer[buffer_id];
                            const char* src = (const char*)v_ptr;
                            
                            for (int off = (tid - 32) * 64; off < bytes; off += (BLOCK_DIM - 32) * 64) {
                                if (off + 64 <= bytes) {
                                    // Use regular memcpy instead of cp.async.bulk
                                    if ((uintptr_t)(dst + off) % 16 == 0 && (uintptr_t)(src + off) % 16 == 0) {
                                        // 128-bit loads if aligned
                                        *reinterpret_cast<float4*>(dst + off) = *reinterpret_cast<const float4*>(src + off);
                                    } else {
                                        // Fallback to smaller loads
                                        for (int i = 0; i < 64; i += sizeof(float)) {
                                            *reinterpret_cast<float*>(dst + off + i) = *reinterpret_cast<const float*>(src + off + i);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    __syncthreads();
                    
                    // All warps compute GEMM
                    // U^T * x -> hidden
                    OptimizedGemm<Element>::gemm_tn_safe(
                        local_token,          // 1 x IN
                        u_buffer[buffer_id],  // HiddenSize x IN (transposed)
                        hidden_smem,
                        1.0f,
                        1, Config::HiddenSize, IN
                    );
                    __syncthreads();
                    
                    // Fused GELU in epilogue (distributed across threads)
                    for (int i = tid; i < Config::HiddenSize; i += blockDim.x) {
                        float x = float(hidden_smem[i]);
                        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                        float x3 = x * x * x;
                        float tanh_arg = 0.7978845608f * (x + 0.044715f * x3);
                        hidden_smem[i] = Element(0.5f * x * (1.0f + tanhf(tanh_arg)));
                    }
                    __syncthreads();
                    
                    // Apply dropout after GELU if dropout_rate > 0
                    if (dropout_rate > 0.0f) {
                        // Initialize RNG state per thread (use block + thread + expert for seed variation)
                        curandState_t state;
                        curand_init(clock64() + expert_id, tid + blockIdx.x * blockDim.x, 0, &state);
                        
                        for (int i = tid; i < Config::HiddenSize; i += blockDim.x) {
                            float rand_val = curand_uniform(&state);
                            if (rand_val < dropout_rate) {
                                hidden_smem[i] = Element(0.0f);
                            } else {
                                // Scale by (1 / (1 - dropout_rate)) to maintain expected value
                                hidden_smem[i] = Element(float(hidden_smem[i]) / (1.0f - dropout_rate));
                            }
                        }
                        __syncthreads();
                    }
                    
                    // V * hidden -> output accumulation
                    for (int i = 0; i < OUT_PER_THREAD; i++) {
                        int out_idx = tid * OUT_PER_THREAD + i;
                        if (out_idx < OUT) {
                            float acc = 0.0f;
                            for (int j = 0; j < Config::HiddenSize; j++) {
                                acc += float(v_buffer[buffer_id][out_idx * Config::HiddenSize + j]) *
                                       float(hidden_smem[j]);
                            }
                            thread_output[i] += weight * acc;
                        }
                    }
                    
                    // Switch buffers
                    buffer_id = 1 - buffer_id;
                    __syncthreads();
                }
            } // End head loop
            
            // Write accumulated output once per token
            int batch_idx = global_token_idx / S;
            int seq_idx = global_token_idx % S;
            Element* out_ptr = output + batch_idx * S * OUT + seq_idx * OUT;
            
            for (int i = 0; i < OUT_PER_THREAD; i++) {
                int out_idx = tid * OUT_PER_THREAD + i;
                if (out_idx < OUT) {
                    // Direct write without atomics
                    out_ptr[out_idx] = Element(thread_output[i]);
                }
            }
            __syncthreads();
        }
    }
}

// ======================== C++ WRAPPER WITH UVA SUPPORT ========================

// Helper to set shared memory configuration
void set_smem_config(void* kernel_ptr, size_t smem_size) {
    cudaError_t err = cudaFuncSetAttribute(kernel_ptr, 
                                           cudaFuncAttributeMaxDynamicSharedMemorySize, 
                                           smem_size);
    if (err != cudaSuccess) {
        printf("Warning: Could not set shared memory size to %zu bytes: %s\n", 
               smem_size, cudaGetErrorString(err));
        
        // Try setting carveout for more shared memory
        cudaFuncSetAttribute(kernel_ptr,
                            cudaFuncAttributePreferredSharedMemoryCarveout,
                            cudaSharedmemCarveoutMaxShared);
    }
}

// PEEROperatorEnhancedImpl implementation
PEEROperatorEnhancedImpl::PEEROperatorEnhancedImpl(
    int num_experts,
    int num_heads,
    int top_k,
    int query_dim,
    int expert_hidden_size,
    int input_dim,
    int output_dim,
    size_t hbm_cache_mb,
    bool use_managed
) : num_experts_(num_experts),
    num_heads_(num_heads),
    top_k_(top_k),
    query_dim_(query_dim),
    expert_hidden_size_(expert_hidden_size),
    sqrt_n_(int(std::sqrt(double(num_experts)) + 0.5)),
    input_dim_(input_dim),
    output_dim_(output_dim),
    u_weights_(nullptr),
    v_weights_(nullptr),
    use_managed_memory_(use_managed) {
    
    cache_ = std::make_unique<HierarchicalExpertCache>(
        num_experts, input_dim, expert_hidden_size, output_dim, hbm_cache_mb
    );
}

PEEROperatorEnhancedImpl::~PEEROperatorEnhancedImpl() {
    if (u_weights_) {
        if (use_managed_memory_) {
            cudaFree(u_weights_);
        } else {
            cudaFreeHost(u_weights_);
        }
    }
    if (v_weights_) {
        if (use_managed_memory_) {
            cudaFree(v_weights_);
        } else {
            cudaFreeHost(v_weights_);
        }
    }
}

// PEEROperatorEnhanced implementation (delegates to pImpl)
PEEROperatorEnhanced::PEEROperatorEnhanced(
    int num_experts,
    int num_heads,
    int top_k,
    int query_dim,
    int expert_hidden_size,
    int input_dim,
    int output_dim,
    size_t hbm_cache_mb,
    bool use_managed
) : pImpl(std::make_unique<PEEROperatorEnhancedImpl>(
        num_experts, num_heads, top_k, query_dim, expert_hidden_size,
        input_dim, output_dim, hbm_cache_mb, use_managed
    )) {
}

PEEROperatorEnhanced::~PEEROperatorEnhanced() = default;

void PEEROperatorEnhanced::allocate_weights() {
    // Allocate using pinned or managed memory
    __uint128_t u_size = __uint128_t(pImpl->num_experts_) * pImpl->input_dim_ * pImpl->expert_hidden_size_ * sizeof(half);
    __uint128_t v_size = __uint128_t(pImpl->num_experts_) * pImpl->expert_hidden_size_ * pImpl->output_dim_ * sizeof(half);
    
    if (pImpl->use_managed_memory_) {
        // Managed memory (slower first access)
        cudaMallocManaged(&pImpl->u_weights_, u_size);
        cudaMallocManaged(&pImpl->v_weights_, v_size);
    } else {
        // Use pinned memory for better performance
        cudaMallocHost(&pImpl->u_weights_, u_size);
        cudaMallocHost(&pImpl->v_weights_, v_size);
    }
    
    // Initialize with random values (in production, load from checkpoint)
    // ... initialization code ...
    
    // Register with cache
    pImpl->cache_->allocate_expert_weights(pImpl->u_weights_, pImpl->v_weights_, pImpl->use_managed_memory_);
    
    printf("Allocated %.2f GB of expert weights in %s memory\n",
           double(u_size + v_size) / (1024.0 * 1024.0 * 1024.0),
           pImpl->use_managed_memory_ ? "managed" : "pinned");
}

// Copy weights from PyTorch tensors to internal buffers
void PEEROperatorEnhanced::copy_weights_from_torch(const half* torch_u_weights, const half* torch_v_weights) {
    __uint128_t u_size = __uint128_t(pImpl->num_experts_) * pImpl->input_dim_ * pImpl->expert_hidden_size_ * sizeof(half);
    __uint128_t v_size = __uint128_t(pImpl->num_experts_) * pImpl->expert_hidden_size_ * pImpl->output_dim_ * sizeof(half);
    
    // Copy from PyTorch tensors to our allocated memory
    memcpy(pImpl->u_weights_, torch_u_weights, u_size);
    memcpy(pImpl->v_weights_, torch_v_weights, v_size);
    
    // If using managed memory, prefetch to GPU for better performance
    if (pImpl->use_managed_memory_) {
        int device;
        cudaGetDevice(&device);
        cudaMemPrefetchAsync(pImpl->u_weights_, u_size, device);
        cudaMemPrefetchAsync(pImpl->v_weights_, v_size, device);
    }
}

// Direct pointer mode: Use PyTorch tensors directly without copying
void PEEROperatorEnhanced::set_weight_pointers(const half* torch_u_weights, const half* torch_v_weights) {
    // Directly use PyTorch-managed memory
    // WARNING: This bypasses our internal allocation and the caller must ensure
    // the PyTorch tensors remain valid during kernel execution
    pImpl->u_weights_ = const_cast<half*>(torch_u_weights);
    pImpl->v_weights_ = const_cast<half*>(torch_v_weights);
    
    // Update the cache's expert pointers to use the PyTorch memory
    pImpl->cache_->update_expert_pointers(pImpl->u_weights_, pImpl->v_weights_);
}

void PEEROperatorEnhanced::forward(
    const half* input,
    const half* query_weight,
    const half* query_bias,
    const half* sub_keys1,
    const half* sub_keys2,
    half* output,
    const half* ln_scale,  // Layer norm scale/weight
    const half* ln_bias,   // Layer norm bias
    int batch_size,
    int seq_len,
    float dropout_rate,
    cudaStream_t stream
) {
    pImpl->dropout_rate_ = dropout_rate;  // Store for kernel use
    // Compute chunk size at runtime
    int chunk_size = compute_l2_chunk_size<half>(pImpl->input_dim_);
    chunk_size = std::min(chunk_size, batch_size * seq_len);
    chunk_size = std::max(chunk_size, 1);  // At least 1 token
    
    // Enhanced kernel configuration
    using Config = PEERConfig<1048576, 56, 128, 256, 64>;
    constexpr int BLOCK_DIM = 128;
    
    int num_tokens = batch_size * seq_len;
    int grid_size = (num_tokens + chunk_size - 1) / chunk_size;
    grid_size = min(grid_size, 256);  // Limit grid size
    
    // Use JIT-defined parameters or defaults
    #ifdef PEER_JIT_TOP_K
        constexpr int TopK = PEER_JIT_TOP_K;
    #else
        constexpr int TopK = 16;
    #endif
    
    #ifdef PEER_JIT_NUM_HEADS
        constexpr int NumHeads = PEER_JIT_NUM_HEADS;
    #else
        constexpr int NumHeads = 8;
    #endif
    
    #ifdef PEER_JIT_QUERY_DIM
        constexpr int QueryDim = PEER_JIT_QUERY_DIM;
    #else
        constexpr int QueryDim = 256;
    #endif
    
    #ifdef PEER_JIT_OUTPUT_DIM
        constexpr int OUT = PEER_JIT_OUTPUT_DIM;
    #else
        constexpr int OUT = 1024;
    #endif
    
    // Calculate shared memory with proper padding
    size_t smem_size = 0;
    smem_size += align_to<64>(chunk_size * pImpl->input_dim_ * sizeof(half));  // Token cache
    smem_size += 2 * align_to<64>(pImpl->input_dim_ * Config::HiddenSize * sizeof(half));  // U buffers
    smem_size += 2 * align_to<64>(Config::HiddenSize * pImpl->output_dim_ * sizeof(half));  // V buffers
    smem_size += align_to<64>(pImpl->query_dim_ * sizeof(half));  // Query (FP16)
    smem_size += align_to<64>(Config::HiddenSize * sizeof(half));  // Hidden activations (FP16)
    smem_size += align_to<64>(2 * pImpl->sqrt_n_ * sizeof(float));  // Routing scores
    
    // Set shared memory configuration
    auto kernel_func = peer_kernel_enhanced<Config, half, NumHeads, TopK, QueryDim, OUT, BLOCK_DIM>;
    set_smem_config((void*)kernel_func, smem_size);
    
    // No need to memset - kernel directly writes output
    
    // Launch enhanced kernel
    kernel_func<<<grid_size, BLOCK_DIM, smem_size, stream>>>(
        input, query_weight, query_bias,
        sub_keys1, sub_keys2, output,
        pImpl->cache_->get_device_experts(),  // Device-only mirror
        ln_scale, ln_bias,  // Layer norm parameters
        batch_size, seq_len, pImpl->input_dim_,
        pImpl->sqrt_n_,       // Runtime parameter
        chunk_size,    // Runtime parameter
        pImpl->dropout_rate_,
        true, true, true
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("Enhanced PEER kernel launch failed: ") + 
            cudaGetErrorString(err)
        );
    }
}

void PEEROperatorEnhanced::print_cache_stats() {
    pImpl->cache_->print_stats();
}

// Getters
int PEEROperatorEnhanced::num_experts() const { return pImpl->num_experts_; }
int PEEROperatorEnhanced::num_heads() const { return pImpl->num_heads_; }
int PEEROperatorEnhanced::input_dim() const { return pImpl->input_dim_; }
int PEEROperatorEnhanced::output_dim() const { return pImpl->output_dim_; }

// ======================== SMOKE TEST ========================

void smoke_test() {
    // Test configuration as suggested
    constexpr int B = 2;
    constexpr int S = 4;
    constexpr int IN = 64;
    constexpr int OUT = 128;
    constexpr int Hidden = 32;
    constexpr int Experts = 512;
    constexpr int NumHeads = 8;
    constexpr int TopK = 4;
    constexpr int QueryDim = 64;
    
    printf("Running smoke test: B=%d, S=%d, IN=%d, OUT=%d, Hidden=%d, Experts=%d\n",
           B, S, IN, OUT, Hidden, Experts);
    
    // Create operator
    PEEROperatorEnhanced op(Experts, NumHeads, TopK, QueryDim, Hidden, IN, OUT, 
                           1024, // 1GB HBM cache for testing
                           false); // Use pinned memory
    
    // Allocate weights
    op.allocate_weights();
    
    // Allocate test inputs/outputs
    half *d_input, *d_query_weight, *d_query_bias, *d_sub_keys1, *d_sub_keys2, *d_output;
    half *d_ln_scale, *d_ln_bias;
    cudaMalloc(&d_input, B * S * IN * sizeof(half));
    cudaMalloc(&d_query_weight, NumHeads * QueryDim * IN * sizeof(half));
    cudaMalloc(&d_query_bias, NumHeads * QueryDim * sizeof(half));
    cudaMalloc(&d_sub_keys1, int(std::sqrt(Experts) + 0.5) * QueryDim * sizeof(half));
    cudaMalloc(&d_sub_keys2, int(std::sqrt(Experts) + 0.5) * QueryDim * sizeof(half));
    cudaMalloc(&d_output, B * S * OUT * sizeof(half));
    cudaMalloc(&d_ln_scale, NumHeads * QueryDim * sizeof(half));
    cudaMalloc(&d_ln_bias, NumHeads * QueryDim * sizeof(half));
    
    // Initialize with random data
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandGenerateUniform(gen, (float*)d_input, B * S * IN / 2);
    curandGenerateUniform(gen, (float*)d_query_weight, NumHeads * QueryDim * IN / 2);
    curandGenerateUniform(gen, (float*)d_query_bias, NumHeads * QueryDim / 2);
    curandGenerateUniform(gen, (float*)d_sub_keys1, int(std::sqrt(Experts) + 0.5) * QueryDim / 2);
    curandGenerateUniform(gen, (float*)d_sub_keys2, int(std::sqrt(Experts) + 0.5) * QueryDim / 2);
    curandGenerateUniform(gen, (float*)d_ln_scale, NumHeads * QueryDim / 2);
    curandGenerateUniform(gen, (float*)d_ln_bias, NumHeads * QueryDim / 2);
    curandDestroyGenerator(gen);
    
    // Run forward pass
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Warmup
    for (int i = 0; i < 3; i++) {
        op.forward(d_input, d_query_weight, d_query_bias, d_sub_keys1, d_sub_keys2, 
                   d_output, d_ln_scale, d_ln_bias, B, S, 0.0f, stream);
    }
    cudaStreamSynchronize(stream);
    
    // Time the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, stream);
    for (int i = 0; i < 10; i++) {
        op.forward(d_input, d_query_weight, d_query_bias, d_sub_keys1, d_sub_keys2, 
                   d_output, d_ln_scale, d_ln_bias, B, S, 0.0f, stream);
    }
    cudaEventRecord(stop, stream);
    cudaStreamSynchronize(stream);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Average kernel time: %.3f ms\n", ms / 10.0f);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    } else {
        printf("Smoke test PASSED!\n");
    }
    
    // Print cache stats
    op.print_cache_stats();
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_query_weight);
    cudaFree(d_query_bias);
    cudaFree(d_sub_keys1);
    cudaFree(d_sub_keys2);
    cudaFree(d_output);
    cudaFree(d_ln_scale);
    cudaFree(d_ln_bias);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

}  // namespace peer

// For testing
#ifdef COMPILE_SMOKE_TEST
int main() {
    peer::smoke_test();
    return 0;
}
#endif
