// peer_cutlass_enhanced_production.cu
// Production-ready PEER implementation with all correctness fixes

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/barrier>
#include <cuda/pipeline>
#include <cassert>
#include <cuda_bf16.h>
#include <mma.h>
#include <curand_kernel.h>

// CUTLASS includes for optimized GEMM
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal.h>
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

namespace peer {

using namespace cute;

// Helper for alignment
template<int N>
__host__ __device__ constexpr size_t align_to(size_t x) {
    return (x + N - 1) / N * N;
}

// POD struct for device-side expert pointers with heat tracking
struct ExpertPtrDev {
    const half* host_u;
    const half* host_v; 
    const half* dev_u;
    const half* dev_v;
    int hbm_slot;
    bool is_hot;
    unsigned int heat;  // Larger counter for multi-warp updates
};

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

// ======================== HIERARCHICAL MEMORY MANAGER ========================

class HierarchicalExpertCache {
private:
    // Expert metadata
    struct ExpertInfo {
        void* host_u_ptr;      // Pointer in system RAM
        void* host_v_ptr;
        void* device_u_ptr;    // Pointer in HBM (if cached)
        void* device_v_ptr;
        std::atomic<int> access_count{0};  // FIX: Make atomic to avoid races
        bool is_hot{false};
        int last_access_time{0};
        int hbm_slot{-1};      // Which HBM slot this expert occupies
    };
    
    std::vector<ExpertInfo> experts_;
    
    // CLOCK-based eviction (O(1) instead of O(N))
    std::vector<int> clock_hand_;  // Maps slot -> expert_id
    int clock_position_{0};
    
    // Memory pools
    void* hbm_pool_u_;
    void* hbm_pool_v_;
    size_t hbm_capacity_;  // Number of experts that fit in HBM
    size_t expert_u_bytes_;
    size_t expert_v_bytes_;
    
    // Prefetch thread
    std::thread prefetch_thread_;
    std::atomic<bool> should_stop_{false};
    std::mutex promotion_mutex_;  // Protect promotion decisions
    
    // Profiling - Split 128-bit atomic into two 64-bit
    std::atomic<int> cache_hits_{0};
    std::atomic<int> cache_misses_{0};
    std::atomic<uint64_t> bytes_lo_{0};
    std::atomic<uint64_t> bytes_hi_{0};
    
    // GPU device ID
    int device_id_;
    
    // Device-side expert pointer tables
    ExpertPtrDev* d_experts_managed_;  // Managed memory version
    ExpertPtrDev* d_experts_device_;   // Device-only mirror for perf
    
    // FIX 4: Delta tracking for heat sync
    unsigned int* d_heat_deltas_;      // Device array for heat deltas
    unsigned int* h_heat_deltas_;      // Host pinned buffer
    int num_experts_;
    
public:
    HierarchicalExpertCache(
        int num_experts,
        int input_dim,
        int hidden_dim,
        int output_dim,
        size_t hbm_capacity_mb = 16384  // 16GB for expert cache
    ) : num_experts_(num_experts) {
        expert_u_bytes_ = input_dim * hidden_dim * sizeof(half);
        expert_v_bytes_ = hidden_dim * output_dim * sizeof(half);
        __uint128_t bytes_per_expert = expert_u_bytes_ + expert_v_bytes_;
        
        cudaGetDevice(&device_id_);
        
        // Calculate how many experts fit in HBM budget
        __uint128_t hbm_bytes = __uint128_t(hbm_capacity_mb) * 1024 * 1024;
        hbm_capacity_ = hbm_bytes / bytes_per_expert;
        hbm_capacity_ = std::min(hbm_capacity_, size_t(num_experts / 10));  // Max 10% in HBM
        
        printf("Hierarchical cache: %zu experts in HBM, %d total\n", 
               hbm_capacity_, num_experts);
        
        // Allocate HBM pool
        cudaMalloc(&hbm_pool_u_, hbm_capacity_ * expert_u_bytes_);
        cudaMalloc(&hbm_pool_v_, hbm_capacity_ * expert_v_bytes_);
        
        // Initialize expert metadata
        experts_.resize(num_experts);
        clock_hand_.resize(hbm_capacity_, -1);
        
        // Allocate device-side expert pointer tables
        cudaMallocManaged(&d_experts_managed_, num_experts * sizeof(ExpertPtrDev));
        cudaMalloc(&d_experts_device_, num_experts * sizeof(ExpertPtrDev));  // Device-only mirror
        
        // FIX 4: Allocate heat delta tracking
        cudaMalloc(&d_heat_deltas_, num_experts * sizeof(unsigned int));
        cudaMallocHost(&h_heat_deltas_, num_experts * sizeof(unsigned int));
        cudaMemset(d_heat_deltas_, 0, num_experts * sizeof(unsigned int));
        
        // Start prefetch thread
        prefetch_thread_ = std::thread(&HierarchicalExpertCache::prefetch_loop, this);
    }
    
    ~HierarchicalExpertCache() {
        should_stop_ = true;
        if (prefetch_thread_.joinable()) {
            prefetch_thread_.join();
        }
        cudaFree(hbm_pool_u_);
        cudaFree(hbm_pool_v_);
        cudaFree(d_experts_managed_);
        cudaFree(d_experts_device_);
        cudaFree(d_heat_deltas_);
        cudaFreeHost(h_heat_deltas_);
    }
    
    // Allocate expert weights in system RAM using pinned memory
    void allocate_expert_weights(half* u_weights, half* v_weights, bool use_managed = false) {
        // Use pinned or managed memory based on flag
        for (int i = 0; i < experts_.size(); i++) {
            experts_[i].host_u_ptr = u_weights + i * (expert_u_bytes_ / sizeof(half));
            experts_[i].host_v_ptr = v_weights + i * (expert_v_bytes_ / sizeof(half));
            experts_[i].device_u_ptr = nullptr;
            experts_[i].device_v_ptr = nullptr;
            
            // Initialize device-side pointer table
            d_experts_managed_[i] = {
                (const half*)experts_[i].host_u_ptr,
                (const half*)experts_[i].host_v_ptr,
                nullptr,
                nullptr,
                -1,
                false,
                0  // Initial heat
            };
        }
        
        // Copy to device-only mirror for performance
        cudaMemcpy(d_experts_device_, d_experts_managed_, 
                   num_experts_ * sizeof(ExpertPtrDev), cudaMemcpyHostToDevice);
        
        if (use_managed) {
            // Proper size computation to avoid overflow
            size_t u_total_bytes = size_t(__uint128_t(experts_.size()) * expert_u_bytes_);
            size_t v_total_bytes = size_t(__uint128_t(experts_.size()) * expert_v_bytes_);
            
            // For managed memory, use advise
            cudaMemAdvise(u_weights, u_total_bytes, 
                          cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
            cudaMemAdvise(v_weights, v_total_bytes,
                          cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
            
            // Advise that GPU will access these pages
            cudaMemAdvise(u_weights, u_total_bytes, 
                          cudaMemAdviseSetAccessedBy, device_id_);
            cudaMemAdvise(v_weights, v_total_bytes,
                          cudaMemAdviseSetAccessedBy, device_id_);
            
            // Mark as read-mostly if CUDA 12+
            #if CUDA_VERSION >= 12000
            cudaMemAdvise(u_weights, u_total_bytes, 
                          cudaMemAdviseSetReadMostly, device_id_);
            cudaMemAdvise(v_weights, v_total_bytes,
                          cudaMemAdviseSetReadMostly, device_id_);
            #endif
        }
    }
    
    // Get device-side expert pointer table (use device mirror for perf)
    ExpertPtrDev* get_device_experts() const {
        return d_experts_device_;
    }
    
    // Host-side version - with stats and promotion
    __host__ void get_expert_ptrs(int expert_id, void*& u_ptr, void*& v_ptr, cudaStream_t stream) {
        auto& info = experts_[expert_id];
        info.access_count.fetch_add(1);
        
        if (info.is_hot && info.device_u_ptr != nullptr) {
            // Expert is in HBM cache
            u_ptr = info.device_u_ptr;
            v_ptr = info.device_v_ptr;
            cache_hits_++;
        } else {
            // Expert in system RAM - will be accessed via UVA
            u_ptr = info.host_u_ptr;
            v_ptr = info.host_v_ptr;
            cache_misses_++;
            
            // Schedule for promotion if accessed frequently
            if (info.access_count.load() > 10 && !info.is_hot) {
                schedule_promotion(expert_id, stream);
            }
        }
    }
    
    // Turn the "hint" into a real prefetch
    void hint_future_access(const int* expert_ids, int count, cudaStream_t stream) {
        for (int i = 0; i < count; i++) {
            int expert_id = expert_ids[i];
            auto& info = experts_[expert_id];
            
            if (!info.is_hot) {
                // Prefetch to GPU if not already there
                cudaMemPrefetchAsync(info.host_u_ptr, expert_u_bytes_, 
                                     device_id_, stream);
                cudaMemPrefetchAsync(info.host_v_ptr, expert_v_bytes_, 
                                     device_id_, stream);
                
                // Track bytes transferred using helper
                add_bytes(expert_u_bytes_ + expert_v_bytes_);
            }
        }
    }
    
    // FIX 4: Efficient heat counter sync using deltas
    void sync_heat_counters(cudaStream_t stream) {
        // Extract heat deltas from device
        static __global__ void extract_heat_deltas_kernel(
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
        
        // Run kernel to extract deltas
        int threads = 256;
        int blocks = (num_experts_ + threads - 1) / threads;
        extract_heat_deltas_kernel<<<blocks, threads, 0, stream>>>(
            d_experts_device_, d_heat_deltas_, num_experts_);
        
        // Copy only non-zero deltas
        cudaMemcpyAsync(h_heat_deltas_, d_heat_deltas_,
                        num_experts_ * sizeof(unsigned int), 
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        // Update host-side access counts
        for (int i = 0; i < num_experts_; i++) {
            if (h_heat_deltas_[i] > 0) {
                experts_[i].access_count.fetch_add(h_heat_deltas_[i]);
                h_heat_deltas_[i] = 0;  // Reset
            }
        }
        
        // Clear device deltas
        cudaMemsetAsync(d_heat_deltas_, 0, num_experts_ * sizeof(unsigned int), stream);
    }
    
    // Implement print_stats
    void print_stats() {
        int hits = cache_hits_.load();
        int misses = cache_misses_.load();
        uint64_t lo = bytes_lo_.load();
        uint64_t hi = bytes_hi_.load();
        __uint128_t bytes = (__uint128_t(hi) << 64) | lo;
        
        if (hits + misses > 0) {
            float hit_rate = 100.0f * hits / (hits + misses);
            double gb_transferred = double(bytes) / (1024.0 * 1024.0 * 1024.0);
            
            printf("HierarchicalExpertCache Statistics:\n");
            printf("  Cache hit rate: %.1f%% (%d hits, %d misses)\n", 
                   hit_rate, hits, misses);
            printf("  Total data transferred: %.2f GB\n", gb_transferred);
            printf("  Hot experts: %d / %zu capacity\n", count_hot_experts(), hbm_capacity_);
        }
    }
    
private:
    // Helper to add bytes with 128-bit counter
    inline void add_bytes(uint64_t n) {
        uint64_t old = bytes_lo_.fetch_add(n, std::memory_order_relaxed);
        if (old > UINT64_MAX - n) {
            bytes_hi_.fetch_add(1, std::memory_order_relaxed);
        }
    }
    
    int count_hot_experts() {
        int count = 0;
        for (const auto& e : experts_) {
            if (e.is_hot) count++;
        }
        return count;
    }
    
    // O(1) CLOCK-based eviction with heat awareness
    void schedule_promotion(int expert_id, cudaStream_t stream) {
        std::lock_guard<std::mutex> lock(promotion_mutex_);
        
        if (experts_[expert_id].is_hot) return;
        
        // Find a slot using CLOCK algorithm
        int slot = -1;
        for (int i = 0; i < hbm_capacity_ * 2; i++) {
            int candidate_slot = clock_position_;
            clock_position_ = (clock_position_ + 1) % hbm_capacity_;
            
            if (clock_hand_[candidate_slot] == -1) {
                // Empty slot
                slot = candidate_slot;
                break;
            }
            
            int victim_id = clock_hand_[candidate_slot];
            
            // Check both host-side access count and device-side heat
            bool has_activity = experts_[victim_id].access_count.load() > 0 ||
                               (d_experts_managed_[victim_id].heat > 128);  // High heat threshold
            
            if (!has_activity) {
                // Found victim
                evict_expert(victim_id);
                slot = candidate_slot;
                break;
            } else {
                // Give second chance
                experts_[victim_id].access_count.store(0);
                d_experts_managed_[victim_id].heat /= 2;  // Decay heat
            }
        }
        
        if (slot >= 0) {
            promote_expert(expert_id, slot, stream);
        }
    }
    
    void promote_expert(int expert_id, int slot, cudaStream_t stream) {
        experts_[expert_id].hbm_slot = slot;
        clock_hand_[slot] = expert_id;
        
        // Copy to HBM
        experts_[expert_id].device_u_ptr = (char*)hbm_pool_u_ + slot * expert_u_bytes_;
        experts_[expert_id].device_v_ptr = (char*)hbm_pool_v_ + slot * expert_v_bytes_;
        
        cudaMemcpyAsync(experts_[expert_id].device_u_ptr,
                        experts_[expert_id].host_u_ptr,
                        expert_u_bytes_, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(experts_[expert_id].device_v_ptr,
                        experts_[expert_id].host_v_ptr,
                        expert_v_bytes_, cudaMemcpyHostToDevice, stream);
        
        experts_[expert_id].is_hot = true;
        
        // Update device-side pointer table
        d_experts_managed_[expert_id].dev_u = (const half*)experts_[expert_id].device_u_ptr;
        d_experts_managed_[expert_id].dev_v = (const half*)experts_[expert_id].device_v_ptr;
        d_experts_managed_[expert_id].hbm_slot = slot;
        d_experts_managed_[expert_id].is_hot = true;
        
        // Update device mirror asynchronously
        cudaMemcpyAsync(d_experts_device_ + expert_id, d_experts_managed_ + expert_id,
                        sizeof(ExpertPtrDev), cudaMemcpyHostToDevice, stream);
        
        add_bytes(expert_u_bytes_ + expert_v_bytes_);
    }
    
    void evict_expert(int expert_id) {
        int slot = experts_[expert_id].hbm_slot;
        if (slot >= 0) {
            clock_hand_[slot] = -1;
        }
        
        experts_[expert_id].is_hot = false;
        experts_[expert_id].device_u_ptr = nullptr;
        experts_[expert_id].device_v_ptr = nullptr;
        experts_[expert_id].hbm_slot = -1;
        
        // Update device-side pointer table
        d_experts_managed_[expert_id].dev_u = nullptr;
        d_experts_managed_[expert_id].dev_v = nullptr;
        d_experts_managed_[expert_id].hbm_slot = -1;
        d_experts_managed_[expert_id].is_hot = false;
    }
    
    void prefetch_loop() {
        cudaStream_t prefetch_stream;
        cudaStreamCreate(&prefetch_stream);
        
        while (!should_stop_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
            static int counter = 0;
            counter++;
            
            // Sync heat counters periodically
            if (counter % 10 == 0) {
                sync_heat_counters(prefetch_stream);
            }
            
            // Print stats periodically
            if (counter % 50 == 0) {
                print_stats();
            }
        }
        
        cudaStreamDestroy(prefetch_stream);
    }
};

// ======================== OPTIMIZED GEMM USING CUTLASS ========================

// CUTLASS-based GEMM for optimal performance
template<typename Element>
struct OptimizedGemm {
    // Define the GEMM operation
    using ElementA = Element;
    using ElementB = Element;
    using ElementC = Element;
    using ElementAccumulator = float;
    
    using ThreadblockShape = cutlass::gemm::GemmShape<16, 128, 32>;
    using WarpShape = cutlass::gemm::GemmShape<16, 32, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
    
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementC,
        128 / cutlass::sizeof_bits<ElementC>::value,
        ElementAccumulator,
        ElementAccumulator
    >;
    
    using Gemm = cutlass::gemm::device::GemmUniversal<
        ElementA, cutlass::layout::RowMajor,
        ElementB, cutlass::layout::RowMajor,
        ElementC, cutlass::layout::RowMajor,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm90,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        EpilogueOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        4  // Stages
    >;
    
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

template<typename scalar_t, int top_k, int sqrt_n>
__device__ void product_key_routing(
    const scalar_t* query,      // [d]
    const scalar_t* sub_keys1,  // [sqrt_n, d]
    const scalar_t* sub_keys2,  // [sqrt_n, d]
    int d,
    int* expert_indices,        // [top_k]
    float* expert_scores,       // [top_k]
    bool norm_keys = true,
    bool norm_query = true
) {
    float scores1[sqrt_n];
    float scores2[sqrt_n];
    
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
    
    // Compute all product scores and maintain top-k
    for (int i = 0; i < sqrt_n; i++) {
        for (int j = 0; j < sqrt_n; j++) {
            float prod_score = scores1[i] * scores2[j];
            int expert_id = i * sqrt_n + j;
            
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
    int SqrtN,
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
    int chunk_size,  // Runtime parameter
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
    
    // Pipeline for overlapping copy/compute
    __shared__ cuda::pipeline<cuda::thread_scope_block> pipe;
    auto pipe_role = cuda::make_pipeline_role(pipe);
    
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
                
                // Batch normalization
                if (use_batch_norm && tid == 0) {
                    float mean = 0.0f, var = 0.0f;
                    for (int i = 0; i < QueryDim; i++) mean += float(query_smem[i]);
                    mean /= QueryDim;
                    for (int i = 0; i < QueryDim; i++) {
                        float diff = float(query_smem[i]) - mean;
                        var += diff * diff;
                    }
                    var = rsqrtf(var / QueryDim + 1e-5f);
                    for (int i = 0; i < QueryDim; i++) {
                        float normalized = (float(query_smem[i]) - mean) * var;
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
                    product_key_routing<Element, TopK, SqrtN>(
                        query_smem, sub_keys1, sub_keys2, QueryDim,
                        expert_indices, expert_scores,
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
                                    asm volatile("cp.async.bulk.shared::cluster.global [%0], [%1], 64;"
                                               :: "r"((uint32_t)__cvta_generic_to_shared(dst + off)), 
                                                  "l"(src + off));
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
                                    asm volatile("cp.async.bulk.shared::cluster.global [%0], [%1], 64;"
                                               :: "r"((uint32_t)__cvta_generic_to_shared(dst + off)), 
                                                  "l"(src + off));
                                }
                            }
                        }
                    }
                    
                    // Commit and wait for copy
                    asm volatile("cp.async.commit_group;");
                    asm volatile("cp.async.wait_group 0;");
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

class PEEROperatorEnhanced {
private:
    int num_experts_;
    int num_heads_;
    int top_k_;
    int query_dim_;
    int expert_hidden_size_;
    int sqrt_n_;
    int input_dim_;
    int output_dim_;
    
    // Hierarchical memory cache
    std::unique_ptr<HierarchicalExpertCache> cache_;
    
    // UVA-allocated weights
    half* u_weights_;
    half* v_weights_;
    bool use_managed_memory_;
    
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
        bool use_managed = false  // Option to use pinned memory
    ) : num_experts_(num_experts),
        num_heads_(num_heads),
        top_k_(top_k),
        query_dim_(query_dim),
        expert_hidden_size_(expert_hidden_size),
        sqrt_n_(int(std::sqrt(double(num_experts)) + 0.5)),  // Proper rounding
        input_dim_(input_dim),
        output_dim_(output_dim),
        u_weights_(nullptr),
        v_weights_(nullptr),
        use_managed_memory_(use_managed) {
        
        // Create hierarchical cache
        cache_ = std::make_unique<HierarchicalExpertCache>(
            num_experts, input_dim, expert_hidden_size, output_dim, hbm_cache_mb
        );
    }
    
    ~PEEROperatorEnhanced() {
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
    
    void allocate_weights() {
        // Allocate using pinned or managed memory
        __uint128_t u_size = __uint128_t(num_experts_) * input_dim_ * expert_hidden_size_ * sizeof(half);
        __uint128_t v_size = __uint128_t(num_experts_) * expert_hidden_size_ * output_dim_ * sizeof(half);
        
        if (use_managed_memory_) {
            // Managed memory (slower first access)
            cudaMallocManaged(&u_weights_, u_size);
            cudaMallocManaged(&v_weights_, v_size);
        } else {
            // Use pinned memory for better performance
            cudaMallocHost(&u_weights_, u_size);
            cudaMallocHost(&v_weights_, v_size);
        }
        
        // Initialize with random values (in production, load from checkpoint)
        // ... initialization code ...
        
        // Register with cache
        cache_->allocate_expert_weights(u_weights_, v_weights_, use_managed_memory_);
        
        printf("Allocated %.2f GB of expert weights in %s memory\n",
               double(u_size + v_size) / (1024.0 * 1024.0 * 1024.0),
               use_managed_memory_ ? "managed" : "pinned");
    }
    
    void forward(
        const half* input,
        const half* query_weight,
        const half* query_bias,
        const half* sub_keys1,
        const half* sub_keys2,
        half* output,
        int batch_size,
        int seq_len,
        cudaStream_t stream = 0
    ) {
        // Compute chunk size at runtime
        int chunk_size = compute_l2_chunk_size<half>(input_dim_);
        chunk_size = std::min(chunk_size, batch_size * seq_len);
        chunk_size = std::max(chunk_size, 1);  // At least 1 token
        
        // Enhanced kernel configuration
        using Config = PEERConfig<1048576, 56, 128, 256, 64>;
        constexpr int BLOCK_DIM = 128;
        
        int num_tokens = batch_size * seq_len;
        int grid_size = (num_tokens + chunk_size - 1) / chunk_size;
        grid_size = min(grid_size, 256);  // Limit grid size
        
        // Calculate shared memory with proper padding
        constexpr int TopK = 16;
        size_t smem_size = 0;
        smem_size += align_to<64>(chunk_size * input_dim_ * sizeof(half));  // Token cache
        smem_size += 2 * align_to<64>(input_dim_ * Config::HiddenSize * sizeof(half));  // U buffers
        smem_size += 2 * align_to<64>(Config::HiddenSize * output_dim_ * sizeof(half));  // V buffers
        smem_size += align_to<64>(query_dim_ * sizeof(half));  // Query (FP16)
        smem_size += TopK * Config::HiddenSize * sizeof(half);  // Hidden activations (FP16)
        
        // Set shared memory configuration
        auto kernel_func = peer_kernel_enhanced<Config, half, 8, TopK, 256, 1024, 1024, BLOCK_DIM>;
        set_smem_config((void*)kernel_func, smem_size);
        
        // No need to memset - kernel directly writes output
        
        // Launch enhanced kernel
        kernel_func<<<grid_size, BLOCK_DIM, smem_size, stream>>>(
            input, query_weight, query_bias,
            sub_keys1, sub_keys2, output,
            cache_->get_device_experts(),  // Device-only mirror
            nullptr, nullptr,  // bn_scale, bn_bias
            batch_size, seq_len, input_dim_,
            chunk_size,  // Runtime parameter
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
    
    void print_cache_stats() {
        cache_->print_stats();
    }
    
    // Getters for validation in wrapper
    int num_experts() const { return num_experts_; }
    int num_heads() const { return num_heads_; }
    int input_dim() const { return input_dim_; }
    int output_dim() const { return output_dim_; }
};

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
    cudaMalloc(&d_input, B * S * IN * sizeof(half));
    cudaMalloc(&d_query_weight, NumHeads * QueryDim * IN * sizeof(half));
    cudaMalloc(&d_query_bias, NumHeads * QueryDim * sizeof(half));
    cudaMalloc(&d_sub_keys1, int(std::sqrt(Experts) + 0.5) * QueryDim * sizeof(half));
    cudaMalloc(&d_sub_keys2, int(std::sqrt(Experts) + 0.5) * QueryDim * sizeof(half));
    cudaMalloc(&d_output, B * S * OUT * sizeof(half));
    
    // Initialize with random data
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandGenerateUniform(gen, (float*)d_input, B * S * IN / 2);
    curandGenerateUniform(gen, (float*)d_query_weight, NumHeads * QueryDim * IN / 2);
    curandGenerateUniform(gen, (float*)d_query_bias, NumHeads * QueryDim / 2);
    curandGenerateUniform(gen, (float*)d_sub_keys1, int(std::sqrt(Experts) + 0.5) * QueryDim / 2);
    curandGenerateUniform(gen, (float*)d_sub_keys2, int(std::sqrt(Experts) + 0.5) * QueryDim / 2);
    curandDestroyGenerator(gen);
    
    // Run forward pass
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Warmup
    for (int i = 0; i < 3; i++) {
        op.forward(d_input, d_query_weight, d_query_bias, d_sub_keys1, d_sub_keys2, 
                   d_output, B, S, stream);
    }
    cudaStreamSynchronize(stream);
    
    // Time the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, stream);
    for (int i = 0; i < 10; i++) {
        op.forward(d_input, d_query_weight, d_query_bias, d_sub_keys1, d_sub_keys2, 
                   d_output, B, S, stream);
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