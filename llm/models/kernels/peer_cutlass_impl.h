/**
 * Internal implementation header for PEER CUTLASS operator
 * This file contains implementation details that should not be exposed in the public API
 */
#pragma once

#include "peer_cutlass.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <chrono>

namespace peer {

// Forward declarations
struct ExpertPtrDev;

// Platform-aware HBM capacity percentage function
float get_hbm_capacity_percentage();

// Global kernel for extracting heat deltas
__global__ void extract_heat_deltas_kernel(
    ExpertPtrDev* experts, unsigned int* deltas, int num_experts);

// ======================== HIERARCHICAL MEMORY MANAGER ========================

class HierarchicalExpertCache {
private:
    // Expert metadata
    struct ExpertInfo {
        void* host_u_ptr;      // Pointer in system RAM
        void* host_v_ptr;
        void* dev_u_ptr;       // Pointer in HBM (if cached)
        void* dev_v_ptr;
        int hbm_slot;          // -1 if not in HBM
        std::atomic<uint64_t> cpu_heat;  // Heat counter on CPU
        bool is_hot;           // Quick check flag
        std::chrono::steady_clock::time_point last_access;
        
        // Need explicit constructors because of atomic member
        ExpertInfo() : host_u_ptr(nullptr), host_v_ptr(nullptr), 
                      dev_u_ptr(nullptr), dev_v_ptr(nullptr),
                      hbm_slot(-1), cpu_heat(0), is_hot(false) {}
        
        // Copy constructor (needed for vector resize)
        ExpertInfo(const ExpertInfo& other) 
            : host_u_ptr(other.host_u_ptr), host_v_ptr(other.host_v_ptr),
              dev_u_ptr(other.dev_u_ptr), dev_v_ptr(other.dev_v_ptr),
              hbm_slot(other.hbm_slot), cpu_heat(other.cpu_heat.load()),
              is_hot(other.is_hot), last_access(other.last_access) {}
    };
    
    // Cache configuration
    int num_experts_;
    int input_dim_;
    int hidden_dim_;       // expert_hidden_size in PEEROperatorEnhanced
    int output_dim_;
    size_t expert_u_bytes_; // Size of one expert's U weights
    size_t expert_v_bytes_; // Size of one expert's V weights
    size_t hbm_capacity_;   // Total HBM to use for cache
    int max_hot_experts_;   // Maximum experts in HBM
    
    // Expert storage
    std::vector<ExpertInfo> experts_;
    __half* hbm_pool_;      // Pre-allocated HBM pool
    std::vector<bool> hbm_slots_free_;
    std::vector<int> clock_hand_;  // Maps slot -> expert_id
    std::mutex cache_mutex_;
    
    // Device-side expert pointers
    ExpertPtrDev* d_experts_host_;    // Host-pinned mirror
    ExpertPtrDev* d_experts_device_;  // Device copy
    
    // Heat tracking
    unsigned int* d_heat_deltas_;     // Device buffer for heat deltas
    unsigned int* h_heat_deltas_;     // Host buffer for heat deltas
    
    // Statistics
    std::atomic<uint64_t> total_accesses_;
    std::atomic<uint64_t> hbm_hits_;
    std::atomic<uint64_t> bytes_transferred_;
    std::atomic<uint64_t> total_evictions_;
    
    // Helper for tracking bytes
    void add_bytes(size_t bytes);
    
public:
    HierarchicalExpertCache(
        int num_experts,
        int input_dim,
        int hidden_dim,  // expert_hidden_size in PEEROperatorEnhanced
        int output_dim,
        size_t hbm_cache_mb = 16384
    );
    
    ~HierarchicalExpertCache();
    
    void allocate_expert_weights(void* host_u_weights, void* host_v_weights, bool use_managed = false);
    void update_expert_pointers(void* u_weights, void* v_weights);
    
    ExpertPtrDev* get_device_experts() { return d_experts_device_; }
    
    void prefetch_experts(const std::vector<int>& expert_ids, cudaStream_t stream);
    void sync_heat_counters(cudaStream_t stream);
    void evict_cold_experts();
    void print_stats();
};

// POD struct for device-side expert pointers with heat tracking
struct ExpertPtrDev {
    const __half* host_u;
    const __half* host_v; 
    const __half* dev_u;
    const __half* dev_v;
    int hbm_slot;
    bool is_hot;
    unsigned int heat;  // Larger counter for multi-warp updates
};

// PIMPL implementation structure
struct PEEROperatorEnhancedImpl {
    int num_experts_;
    int num_heads_;
    int top_k_;
    int query_dim_;
    int expert_hidden_size_;
    int sqrt_n_;
    int input_dim_;
    int output_dim_;
    float dropout_rate_;
    
    std::unique_ptr<HierarchicalExpertCache> cache_;
    
    // UVA-allocated weights
    __half* u_weights_;
    __half* v_weights_;
    bool use_managed_memory_;
    
    PEEROperatorEnhancedImpl(
        int num_experts,
        int num_heads,
        int top_k,
        int query_dim,
        int expert_hidden_size,
        int input_dim,
        int output_dim,
        size_t hbm_cache_mb,
        bool use_managed
    );
    
    ~PEEROperatorEnhancedImpl();
};

// Implementation of HierarchicalExpertCache methods (needed for PIMPL)

HierarchicalExpertCache::HierarchicalExpertCache(
    int num_experts,
    int input_dim,
    int hidden_dim,
    int output_dim,
    size_t hbm_cache_mb
) : num_experts_(num_experts),
    input_dim_(input_dim),
    hidden_dim_(hidden_dim),
    output_dim_(output_dim) {
    
    // Calculate sizes
    expert_u_bytes_ = input_dim_ * hidden_dim_ * sizeof(__half);
    expert_v_bytes_ = hidden_dim_ * output_dim_ * sizeof(__half);
    
    // Determine HBM capacity
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    
    size_t total_mem = prop.totalGlobalMem;
    float capacity_pct = get_hbm_capacity_percentage();
    hbm_capacity_ = std::min(size_t(hbm_cache_mb * 1024ULL * 1024ULL), 
                            size_t(total_mem * capacity_pct));
    
    // Calculate max experts that fit
    size_t per_expert_size = expert_u_bytes_ + expert_v_bytes_;
    max_hot_experts_ = hbm_capacity_ / per_expert_size;
    max_hot_experts_ = std::min(max_hot_experts_, num_experts / 4);  // Cap at 25%
    
    // Allocate HBM pool
    cudaMalloc(&hbm_pool_, max_hot_experts_ * per_expert_size);
    
    // Initialize metadata
    experts_.resize(num_experts);
    hbm_slots_free_.resize(max_hot_experts_, true);
    clock_hand_.resize(max_hot_experts_, -1);
    
    // Allocate device-side structures
    cudaMallocHost(&d_experts_host_, num_experts * sizeof(ExpertPtrDev));
    cudaMalloc(&d_experts_device_, num_experts * sizeof(ExpertPtrDev));
    cudaMalloc(&d_heat_deltas_, num_experts * sizeof(unsigned int));
    cudaMallocHost(&h_heat_deltas_, num_experts * sizeof(unsigned int));
    
    // Initialize
    memset(d_experts_host_, 0, num_experts * sizeof(ExpertPtrDev));
    cudaMemset(d_heat_deltas_, 0, num_experts * sizeof(unsigned int));
    
    printf("HierarchicalExpertCache: %d experts, %.2f GB HBM capacity, max %d hot experts\n",
           num_experts, hbm_capacity_ / (1024.0 * 1024.0 * 1024.0), max_hot_experts_);
}

HierarchicalExpertCache::~HierarchicalExpertCache() {
    if (hbm_pool_) cudaFree(hbm_pool_);
    if (d_experts_host_) cudaFreeHost(d_experts_host_);
    if (d_experts_device_) cudaFree(d_experts_device_);
    if (d_heat_deltas_) cudaFree(d_heat_deltas_);
    if (h_heat_deltas_) cudaFreeHost(h_heat_deltas_);
}

void HierarchicalExpertCache::allocate_expert_weights(void* host_u_weights, void* host_v_weights, bool use_managed) {
    // Set up expert pointers
    for (int i = 0; i < num_experts_; i++) {
        experts_[i].host_u_ptr = (char*)host_u_weights + i * expert_u_bytes_;
        experts_[i].host_v_ptr = (char*)host_v_weights + i * expert_v_bytes_;
        experts_[i].dev_u_ptr = nullptr;
        experts_[i].dev_v_ptr = nullptr;
        experts_[i].hbm_slot = -1;
        experts_[i].is_hot = false;
        experts_[i].cpu_heat.store(0);
        
        // Initialize device-side pointers
        d_experts_host_[i].host_u = reinterpret_cast<const __half*>(experts_[i].host_u_ptr);
        d_experts_host_[i].host_v = reinterpret_cast<const __half*>(experts_[i].host_v_ptr);
        d_experts_host_[i].dev_u = nullptr;
        d_experts_host_[i].dev_v = nullptr;
        d_experts_host_[i].hbm_slot = -1;
        d_experts_host_[i].is_hot = false;
        d_experts_host_[i].heat = 0;
    }
    
    // Copy to device
    cudaMemcpy(d_experts_device_, d_experts_host_, 
               num_experts_ * sizeof(ExpertPtrDev), cudaMemcpyHostToDevice);
}

void HierarchicalExpertCache::update_expert_pointers(void* u_weights, void* v_weights) {
    // Update host pointers when using external memory
    for (int i = 0; i < num_experts_; i++) {
        experts_[i].host_u_ptr = (char*)u_weights + i * expert_u_bytes_;
        experts_[i].host_v_ptr = (char*)v_weights + i * expert_v_bytes_;
        
        d_experts_host_[i].host_u = reinterpret_cast<const __half*>(experts_[i].host_u_ptr);
        d_experts_host_[i].host_v = reinterpret_cast<const __half*>(experts_[i].host_v_ptr);
    }
    
    // Update device copy
    cudaMemcpy(d_experts_device_, d_experts_host_, 
               num_experts_ * sizeof(ExpertPtrDev), cudaMemcpyHostToDevice);
}

void HierarchicalExpertCache::prefetch_experts(const std::vector<int>& expert_ids, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    for (int expert_id : expert_ids) {
        if (expert_id < 0 || expert_id >= num_experts_) continue;
        
        auto& expert = experts_[expert_id];
        
        // Already in cache
        if (expert.is_hot) {
            expert.cpu_heat.fetch_add(1);
            continue;
        }
        
        // Find a free slot or evict
        int slot = -1;
        for (int i = 0; i < max_hot_experts_; i++) {
            if (hbm_slots_free_[i]) {
                slot = i;
                break;
            }
        }
        
        if (slot == -1) {
            // Simple eviction - find coldest expert
            uint64_t min_heat = UINT64_MAX;
            int evict_slot = 0;
            
            for (int i = 0; i < max_hot_experts_; i++) {
                int cached_expert_id = clock_hand_[i];
                if (cached_expert_id >= 0) {
                    uint64_t heat = experts_[cached_expert_id].cpu_heat.load();
                    if (heat < min_heat) {
                        min_heat = heat;
                        evict_slot = i;
                    }
                }
            }
            
            // Evict
            int evicted_id = clock_hand_[evict_slot];
            if (evicted_id >= 0) {
                experts_[evicted_id].is_hot = false;
                experts_[evicted_id].hbm_slot = -1;
                experts_[evicted_id].dev_u_ptr = nullptr;
                experts_[evicted_id].dev_v_ptr = nullptr;
                
                d_experts_host_[evicted_id].is_hot = false;
                d_experts_host_[evicted_id].dev_u = nullptr;
                d_experts_host_[evicted_id].dev_v = nullptr;
                
                total_evictions_.fetch_add(1);
            }
            
            slot = evict_slot;
        }
        
        // Allocate in HBM
        char* slot_base = (char*)hbm_pool_ + slot * (expert_u_bytes_ + expert_v_bytes_);
        expert.dev_u_ptr = slot_base;
        expert.dev_v_ptr = slot_base + expert_u_bytes_;
        expert.hbm_slot = slot;
        expert.is_hot = true;
        
        // Copy weights asynchronously
        cudaMemcpyAsync(expert.dev_u_ptr, expert.host_u_ptr, 
                       expert_u_bytes_, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(expert.dev_v_ptr, expert.host_v_ptr, 
                       expert_v_bytes_, cudaMemcpyHostToDevice, stream);
        
        // Update device pointers
        d_experts_host_[expert_id].dev_u = reinterpret_cast<const __half*>(expert.dev_u_ptr);
        d_experts_host_[expert_id].dev_v = reinterpret_cast<const __half*>(expert.dev_v_ptr);
        d_experts_host_[expert_id].hbm_slot = slot;
        d_experts_host_[expert_id].is_hot = true;
        
        // Update tracking
        hbm_slots_free_[slot] = false;
        clock_hand_[slot] = expert_id;
        bytes_transferred_.fetch_add(expert_u_bytes_ + expert_v_bytes_);
    }
    
    // Copy updated pointers to device
    cudaMemcpyAsync(d_experts_device_, d_experts_host_,
                   num_experts_ * sizeof(ExpertPtrDev), 
                   cudaMemcpyHostToDevice, stream);
}

void HierarchicalExpertCache::sync_heat_counters(cudaStream_t stream) {
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
    
    // Update CPU-side counters
    for (int i = 0; i < num_experts_; i++) {
        if (h_heat_deltas_[i] > 0) {
            experts_[i].cpu_heat.fetch_add(h_heat_deltas_[i]);
            h_heat_deltas_[i] = 0;  // Reset
        }
    }
}

void HierarchicalExpertCache::print_stats() {
    printf("\n=== Hierarchical Expert Cache Stats ===\n");
    printf("Total accesses: %lu\n", total_accesses_.load());
    printf("HBM hits: %lu (%.2f%%)\n", hbm_hits_.load(), 
           100.0 * hbm_hits_.load() / std::max(1UL, total_accesses_.load()));
    printf("Bytes transferred: %.2f GB\n", 
           bytes_transferred_.load() / (1024.0 * 1024.0 * 1024.0));
    printf("Total evictions: %lu\n", total_evictions_.load());
    
    // Count hot experts
    int hot_count = 0;
    for (const auto& expert : experts_) {
        if (expert.is_hot) hot_count++;
    }
    printf("Hot experts: %d / %d\n", hot_count, num_experts_);
    printf("=====================================\n");
}

void HierarchicalExpertCache::add_bytes(size_t bytes) {
    bytes_transferred_.fetch_add(bytes);
}

} // namespace peer