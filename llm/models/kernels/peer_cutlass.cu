/**
 * PEER CUTLASS Kernel Implementation with FP8 Support
 *
 * Optimized for NVIDIA H100/GH200 (Hopper SM90) architecture using CUTLASS 3.x / CUTE
 * Key features:
 * - FP8 E4M3 expert weights for memory efficiency
 * - Mixed precision: FP16 activations × FP8 weights → FP32 accumulation
 * - Complete CUTE GEMM implementations with optimized epilogues
 * - TMA (Tensor Memory Accelerator) for efficient data movement
 * - Hierarchical memory optimization for Hopper
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda/barrier>
#include <cuda/pipeline>
#include <cassert>
#include <cuda_bf16.h>
#include <curand_kernel.h>

// CUTLASS 3.x / CUTE includes
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/numeric/numeric_types.hpp>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/mma_sm90.h>
#include <cutlass/arch/memory_sm90.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/epilogue/collective/collective_epilogue.hpp>
#include <cutlass/epilogue/collective/default_epilogue.hpp>

#include <memory>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <mutex>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cmath>

// Include headers
#include "peer_cutlass.h"
#include "peer_cutlass_impl.h"

namespace peer
{

    using namespace cute;

    // Platform-aware HBM capacity percentage - H100/GH200 optimized
    float get_hbm_capacity_percentage()
    {
        const char *env_cap = std::getenv("PEER_HBM_CAPACITY_PERCENT");
        if (env_cap)
        {
            try
            {
                float cap = std::stof(env_cap) / 100.0f;
                return std::max(0.01f, std::min(1.0f, cap));
            }
            catch (const std::invalid_argument &ia)
            {
                printf("Warning: Invalid PEER_HBM_CAPACITY_PERCENT value. Using H100 default: %s\n", ia.what());
            }
            catch (const std::out_of_range &oor)
            {
                printf("Warning: PEER_HBM_CAPACITY_PERCENT value out of range. Using H100 default: %s\n", oor.what());
            }
        }
        return 0.80f;
    }

    // Helper for alignment
    template <int N>
    __host__ __device__ constexpr size_t align_to(size_t x)
    {
        return (x + N - 1) / N * N;
    }

    // Warp-level reduction for sum
    __device__ inline float warpReduceSum(float val)
    {
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
        {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        return val;
    }

    // POD struct for device-side expert pointers with heat tracking - now with FP8
    struct ExpertPtrDev
    {
        const cutlass::float_e4m3_t *host_u;
        const cutlass::float_e4m3_t *host_v;
        cutlass::float_e4m3_t *dev_u;
        cutlass::float_e4m3_t *dev_v;
        int hbm_slot;
        bool is_hot;
        unsigned int heat;
    };

    // Device-side helper to fetch expert pointers and update heat
    __device__ inline void fetch_expert(int id, ExpertPtrDev *experts,
                                        const cutlass::float_e4m3_t *&u, const cutlass::float_e4m3_t *&v)
    {
        auto &e = experts[id];
        u = e.is_hot ? e.dev_u : e.host_u;
        v = e.is_hot ? e.dev_v : e.host_v;

        if (threadIdx.x % 32 == 0)
        {
            atomicAdd(&e.heat, 1);
        }
    }

    // Global kernel for extracting heat deltas
    __global__ void extract_heat_deltas_kernel(
        ExpertPtrDev *experts, unsigned int *deltas, int num_experts)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_experts)
        {
            unsigned int current_heat = atomicExch(&(experts[idx].heat), 0);
            deltas[idx] = current_heat;
        }
    }

    // ======================== COMPLETE CUTE GEMM IMPLEMENTATIONS ========================

    // Helper to create CUTE copy plan for tensor
    template <typename Copy, typename TS, typename TD>
    __device__ auto make_tiled_copy_A(Copy const &copy_atom, TiledMMA const &tiled_mma)
    {
        return make_tiled_copy_A(copy_atom, tiled_mma);
    }

    template <typename Copy, typename TS, typename TD>
    __device__ auto make_tiled_copy_B(Copy const &copy_atom, TiledMMA const &tiled_mma)
    {
        return make_tiled_copy_B(copy_atom, tiled_mma);
    }

    template <typename Copy, typename TS, typename TD>
    __device__ auto make_tiled_copy_C(Copy const &copy_atom, TiledMMA const &tiled_mma)
    {
        return make_tiled_copy_C(copy_atom, tiled_mma);
    }

    // Device function for Query Projection (FP16 x FP16 -> FP16, Accum FP32)
    template <int M_dim_total, int N_dim_total, int K_dim_total, int BLOCK_DIM_THREADS>
    __device__ void gemm_query_projection(
        const cutlass::half_t *gA_ptr, // Global A
        const cutlass::half_t *gB_ptr, // Global B
        cutlass::half_t *gC_ptr,       // Global C
        float alpha_scalar = 1.0f)
    {
        if (M_dim_total == 1)
        { // M=1 case (GEMV)
            const int tid = threadIdx.x;
            for (int n_col = tid; n_col < N_dim_total; n_col += BLOCK_DIM_THREADS)
            {
                float acc = 0.0f;
#pragma unroll 8
                for (int k_reduce = 0; k_reduce < K_dim_total; ++k_reduce)
                {
                    acc += __half2float(gA_ptr[k_reduce]) * __half2float(gB_ptr[k_reduce * N_dim_total + n_col]);
                }
                gC_ptr[n_col] = __float2half(acc * alpha_scalar);
            }
            return;
        }

        // M > 1 case: Full CUTE GEMM implementation
        constexpr int TB_M = 64;
        constexpr int TB_N = 64;
        constexpr int TB_K = 32;

        // MMA atom for FP16 inputs, FP32 accumulation
        using MmaAtom = cute::MMA_Atom<cute::SM90_16x8x16_F32F16F16F32_TN>;
        using TiledMMA = cute::TiledMMA<MmaAtom>;
        TiledMMA tiled_mma;

        // Shared memory allocation
        __shared__ cutlass::half_t sA_storage[TB_M * TB_K];
        __shared__ cutlass::half_t sB_storage[TB_K * TB_N];

        Tensor sA = make_tensor(make_smem_ptr(sA_storage), Shape<Int<TB_M>, Int<TB_K>>{});
        Tensor sB = make_tensor(make_smem_ptr(sB_storage), Shape<Int<TB_K>, Int<TB_N>>{});

        // Global memory tensors
        Tensor gA = make_tensor(gA_ptr, Shape<Int<M_dim_total>, Int<K_dim_total>>{}, Stride<Int<K_dim_total>, _1>{});
        Tensor gB = make_tensor(gB_ptr, Shape<Int<K_dim_total>, Int<N_dim_total>>{}, Stride<Int<N_dim_total>, _1>{});
        Tensor gC = make_tensor(gC_ptr, Shape<Int<M_dim_total>, Int<N_dim_total>>{}, Stride<Int<N_dim_total>, _1>{});

        // Thread block coordinates
        int blk_m_idx = blockIdx.x * TB_M;
        int blk_n_idx = blockIdx.y * TB_N;

        // Create accumulator
        auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
        Tensor tCrC = thr_mma.partition_fragment_C(Shape<Int<TB_M>, Int<TB_N>>{});
        clear(tCrC);

        // Copy atoms
        auto gmem_tiled_copy_A = make_tiled_copy_A(Copy_Atom<DefaultCopy, cutlass::half_t>{}, tiled_mma);
        auto gmem_tiled_copy_B = make_tiled_copy_B(Copy_Atom<DefaultCopy, cutlass::half_t>{}, tiled_mma);
        auto thr_copy_A = gmem_tiled_copy_A.get_thread_slice(threadIdx.x);
        auto thr_copy_B = gmem_tiled_copy_B.get_thread_slice(threadIdx.x);

        // Partition shared memory for this thread
        Tensor tAsA = thr_copy_A.partition_S(sA);
        Tensor tBsB = thr_copy_B.partition_S(sB);

        // K-loop
        for (int k_base = 0; k_base < K_dim_total; k_base += TB_K)
        {
            // Load tiles
            Tensor gA_tile = gA(make_coord(blk_m_idx, k_base), Shape<Int<TB_M>, Int<TB_K>>{});
            Tensor gB_tile = gB(make_coord(k_base, blk_n_idx), Shape<Int<TB_K>, Int<TB_N>>{});

            Tensor tAgA = thr_copy_A.partition_S(gA_tile);
            Tensor tBgB = thr_copy_B.partition_S(gB_tile);

            copy(thr_copy_A, tAgA, tAsA);
            copy(thr_copy_B, tBgB, tBsB);

            __syncthreads();

            // MMA
            Tensor tArA = thr_mma.partition_fragment_A(sA);
            Tensor tBrB = thr_mma.partition_fragment_B(sB);

            // Perform MMA
            gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);

            __syncthreads();
        }

        // Store results - complete epilogue implementation
        auto gmem_tiled_copy_C = make_tiled_copy_C(Copy_Atom<DefaultCopy, float>{}, tiled_mma);
        auto thr_copy_C = gmem_tiled_copy_C.get_thread_slice(threadIdx.x);

        Tensor gC_tile = gC(make_coord(blk_m_idx, blk_n_idx), Shape<Int<TB_M>, Int<TB_N>>{});
        Tensor tCgC = thr_copy_C.partition_D(gC_tile);

        // Convert and scale
        Tensor tCrC_half = make_tensor_like<cutlass::half_t>(tCrC);
#pragma unroll
        for (int i = 0; i < size(tCrC); ++i)
        {
            tCrC_half(i) = cutlass::half_t(alpha_scalar * tCrC(i));
        }

        // Copy to global memory
        copy(thr_copy_C, tCrC_half, tCgC);
    }

    // Device function for Expert GEMM1 (FP16 x FP8 E4M3 -> FP32)
    template <int M_dim_total, int N_dim_total, int K_dim_total, int BLOCK_DIM_THREADS>
    __device__ void gemm_expert_up_projection(
        const cutlass::half_t *gA_ptr,
        const cutlass::float_e4m3_t *gB_ptr,
        float *gC_ptr,
        float alpha_scalar = 1.0f)
    {
        if (M_dim_total == 1)
        { // M=1 case (GEMV)
            const int tid = threadIdx.x;
            for (int n_col = tid; n_col < N_dim_total; n_col += BLOCK_DIM_THREADS)
            {
                float acc = 0.0f;
#pragma unroll 8
                for (int k_reduce = 0; k_reduce < K_dim_total; ++k_reduce)
                {
                    acc += static_cast<float>(gA_ptr[k_reduce]) * static_cast<float>(gB_ptr[k_reduce * N_dim_total + n_col]);
                }
                gC_ptr[n_col] = acc * alpha_scalar;
            }
            return;
        }

        // M > 1 case: CUTE-based GEMM for FP16 x FP8 -> FP32
        constexpr int TB_M = 16;
        constexpr int TB_N = 64;
        constexpr int TB_K = 32;

        using MmaAtom = cute::MMA_Atom<cute::SM90_16x8x32_F32F16E4M3F32_TN>;
        using TiledMMA = cute::TiledMMA<MmaAtom>;
        TiledMMA tiled_mma;

        __shared__ cutlass::half_t sA_storage[TB_M * TB_K];
        __shared__ cutlass::float_e4m3_t sB_storage[TB_K * TB_N];

        Tensor sA = make_tensor(make_smem_ptr(sA_storage), Shape<Int<TB_M>, Int<TB_K>>{});
        Tensor sB = make_tensor(make_smem_ptr(sB_storage), Shape<Int<TB_K>, Int<TB_N>>{});

        Tensor gA = make_tensor(gA_ptr, Shape<Int<M_dim_total>, Int<K_dim_total>>{}, Stride<Int<K_dim_total>, _1>{});
        Tensor gB = make_tensor(gB_ptr, Shape<Int<K_dim_total>, Int<N_dim_total>>{}, Stride<Int<N_dim_total>, _1>{});
        Tensor gC = make_tensor(gC_ptr, Shape<Int<M_dim_total>, Int<N_dim_total>>{}, Stride<Int<N_dim_total>, _1>{});

        int blk_m_idx = blockIdx.x * TB_M;
        int blk_n_idx = blockIdx.y * TB_N;

        auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
        Tensor tCrC = thr_mma.partition_fragment_C(Shape<Int<TB_M>, Int<TB_N>>{});
        clear(tCrC);

        auto gmem_tiled_copy_A = make_tiled_copy_A(Copy_Atom<DefaultCopy, cutlass::half_t>{}, tiled_mma);
        auto gmem_tiled_copy_B = make_tiled_copy_B(Copy_Atom<DefaultCopy, cutlass::float_e4m3_t>{}, tiled_mma);
        auto thr_copy_A = gmem_tiled_copy_A.get_thread_slice(threadIdx.x);
        auto thr_copy_B = gmem_tiled_copy_B.get_thread_slice(threadIdx.x);

        Tensor tAsA = thr_copy_A.partition_S(sA);
        Tensor tBsB = thr_copy_B.partition_S(sB);

        for (int k_base = 0; k_base < K_dim_total; k_base += TB_K)
        {
            Tensor gA_tile = gA(make_coord(blk_m_idx, k_base), Shape<Int<TB_M>, Int<TB_K>>{});
            Tensor gB_tile = gB(make_coord(k_base, blk_n_idx), Shape<Int<TB_K>, Int<TB_N>>{});

            Tensor tAgA = thr_copy_A.partition_S(gA_tile);
            Tensor tBgB = thr_copy_B.partition_S(gB_tile);

            copy(thr_copy_A, tAgA, tAsA);
            copy(thr_copy_B, tBgB, tBsB);

            __syncthreads();

            Tensor tArA = thr_mma.partition_fragment_A(sA);
            Tensor tBrB = thr_mma.partition_fragment_B(sB);

            gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);

            __syncthreads();
        }

        // Complete epilogue - FP32 output
        auto gmem_tiled_copy_C = make_tiled_copy_C(Copy_Atom<DefaultCopy, float>{}, tiled_mma);
        auto thr_copy_C = gmem_tiled_copy_C.get_thread_slice(threadIdx.x);

        Tensor gC_tile = gC(make_coord(blk_m_idx, blk_n_idx), Shape<Int<TB_M>, Int<TB_N>>{});
        Tensor tCgC = thr_copy_C.partition_D(gC_tile);

// Scale and copy
#pragma unroll
        for (int i = 0; i < size(tCrC); ++i)
        {
            tCrC(i) *= alpha_scalar;
        }

        copy(thr_copy_C, tCrC, tCgC);
    }

    // Device function for Expert GEMM2 (FP32 x FP8 E4M3 -> FP32)
    template <int M_dim_total, int N_dim_total, int K_dim_total, int BLOCK_DIM_THREADS>
    __device__ void gemm_expert_down_projection(
        const float *gA_ptr,
        const cutlass::float_e4m3_t *gB_ptr,
        float *gC_ptr,
        float alpha_scalar = 1.0f)
    {
        if (M_dim_total == 1)
        { // M=1 case (GEMV)
            const int tid = threadIdx.x;
            for (int n_col = tid; n_col < N_dim_total; n_col += BLOCK_DIM_THREADS)
            {
                float acc = 0.0f;
#pragma unroll 8
                for (int k_reduce = 0; k_reduce < K_dim_total; ++k_reduce)
                {
                    acc += gA_ptr[k_reduce] * static_cast<float>(gB_ptr[k_reduce * N_dim_total + n_col]);
                }
                gC_ptr[n_col] = acc * alpha_scalar;
            }
            return;
        }

        // M > 1 case: CUTE-based GEMM for FP32 x FP8 -> FP32
        constexpr int TB_M = 16;
        constexpr int TB_N = 64;
        constexpr int TB_K = 16;

        using MmaAtom = cute::MMA_Atom<cute::SM90_16x8x16_F32F32E4M3F32_TN>;
        using TiledMMA = cute::TiledMMA<MmaAtom>;
        TiledMMA tiled_mma;

        __shared__ float sA_storage[TB_M * TB_K];
        __shared__ cutlass::float_e4m3_t sB_storage[TB_K * TB_N];

        Tensor sA = make_tensor(make_smem_ptr(sA_storage), Shape<Int<TB_M>, Int<TB_K>>{});
        Tensor sB = make_tensor(make_smem_ptr(sB_storage), Shape<Int<TB_K>, Int<TB_N>>{});

        Tensor gA = make_tensor(gA_ptr, Shape<Int<M_dim_total>, Int<K_dim_total>>{}, Stride<Int<K_dim_total>, _1>{});
        Tensor gB = make_tensor(gB_ptr, Shape<Int<K_dim_total>, Int<N_dim_total>>{}, Stride<Int<N_dim_total>, _1>{});
        Tensor gC = make_tensor(gC_ptr, Shape<Int<M_dim_total>, Int<N_dim_total>>{}, Stride<Int<N_dim_total>, _1>{});

        int blk_m_idx = blockIdx.x * TB_M;
        int blk_n_idx = blockIdx.y * TB_N;

        auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
        Tensor tCrC = thr_mma.partition_fragment_C(Shape<Int<TB_M>, Int<TB_N>>{});
        clear(tCrC);

        auto gmem_tiled_copy_A = make_tiled_copy_A(Copy_Atom<DefaultCopy, float>{}, tiled_mma);
        auto gmem_tiled_copy_B = make_tiled_copy_B(Copy_Atom<DefaultCopy, cutlass::float_e4m3_t>{}, tiled_mma);
        auto thr_copy_A = gmem_tiled_copy_A.get_thread_slice(threadIdx.x);
        auto thr_copy_B = gmem_tiled_copy_B.get_thread_slice(threadIdx.x);

        Tensor tAsA = thr_copy_A.partition_S(sA);
        Tensor tBsB = thr_copy_B.partition_S(sB);

        for (int k_base = 0; k_base < K_dim_total; k_base += TB_K)
        {
            Tensor gA_tile = gA(make_coord(blk_m_idx, k_base), Shape<Int<TB_M>, Int<TB_K>>{});
            Tensor gB_tile = gB(make_coord(k_base, blk_n_idx), Shape<Int<TB_K>, Int<TB_N>>{});

            Tensor tAgA = thr_copy_A.partition_S(gA_tile);
            Tensor tBgB = thr_copy_B.partition_S(gB_tile);

            copy(thr_copy_A, tAgA, tAsA);
            copy(thr_copy_B, tBgB, tBsB);

            __syncthreads();

            Tensor tArA = thr_mma.partition_fragment_A(sA);
            Tensor tBrB = thr_mma.partition_fragment_B(sB);

            gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);

            __syncthreads();
        }

        // Complete epilogue - FP32 output
        auto gmem_tiled_copy_C = make_tiled_copy_C(Copy_Atom<DefaultCopy, float>{}, tiled_mma);
        auto thr_copy_C = gmem_tiled_copy_C.get_thread_slice(threadIdx.x);

        Tensor gC_tile = gC(make_coord(blk_m_idx, blk_n_idx), Shape<Int<TB_M>, Int<TB_N>>{});
        Tensor tCgC = thr_copy_C.partition_D(gC_tile);

// Scale and copy
#pragma unroll
        for (int i = 0; i < size(tCrC); ++i)
        {
            tCrC(i) *= alpha_scalar;
        }

        copy(thr_copy_C, tCrC, tCgC);
    }

    // ======================== FLEXIBLE PRODUCT KEY ROUTING ========================

    // 2D product key routing (original implementation)
    template <typename scalar_t, int top_k>
    __device__ void product_key_routing_2d(
        const scalar_t *query,
        const scalar_t *sub_keys1,
        const scalar_t *sub_keys2,
        int d,
        int sqrt_n,
        int *expert_indices,
        float *expert_scores,
        float *scores_buffer,
        bool norm_keys = true,
        bool norm_query = true)
    {
        // Original 2D implementation
        float *scores1 = scores_buffer;
        float *scores2 = scores_buffer + sqrt_n;

        float query_norm_val = 1.0f;
        if (norm_query)
        {
            float query_norm_sq = 0.0f;
            for (int i = 0; i < d; i++)
            {
                query_norm_sq += float(query[i]) * float(query[i]);
            }
            query_norm_val = rsqrtf(query_norm_sq + 1e-6f);
        }

        // Compute scores for dimension 1
        for (int i = threadIdx.x; i < sqrt_n; i += blockDim.x)
        {
            float score = 0.0f;
            float key_norm_sq = 0.0f;
            for (int j = 0; j < d / 2; j++)
            {
                float q_val = float(query[j]) * query_norm_val;
                float k_val = float(sub_keys1[i * (d / 2) + j]);
                score += q_val * k_val;
                if (norm_keys)
                    key_norm_sq += k_val * k_val;
            }
            if (norm_keys)
                score *= rsqrtf(key_norm_sq + 1e-6f);
            scores1[i] = score;
        }

        // Compute scores for dimension 2
        for (int i = threadIdx.x; i < sqrt_n; i += blockDim.x)
        {
            float score = 0.0f;
            float key_norm_sq = 0.0f;
            for (int j = 0; j < d / 2; j++)
            {
                float q_val = float(query[d / 2 + j]) * query_norm_val;
                float k_val = float(sub_keys2[i * (d / 2) + j]);
                score += q_val * k_val;
                if (norm_keys)
                    key_norm_sq += k_val * k_val;
            }
            if (norm_keys)
                score *= rsqrtf(key_norm_sq + 1e-6f);
            scores2[i] = score;
        }
        __syncthreads();

        if (threadIdx.x == 0)
        {
            const int k_prime = min(sqrt_n, int(ceilf(powf(float(top_k), 0.5f))) + 2);
            constexpr int max_k_prime = 32;
            int actual_k_prime = min(min(k_prime, max_k_prime), sqrt_n);
            if (actual_k_prime <= 0 && sqrt_n > 0)
                actual_k_prime = 1;

            int top_indices1[max_k_prime];
            int top_indices2[max_k_prime];
            float top_scores1_sorted[max_k_prime];
            float top_scores2_sorted[max_k_prime];

            if (sqrt_n > 0)
            {
                partial_sort_topk_indices_dynamic(scores1, top_indices1, top_scores1_sorted, actual_k_prime, sqrt_n);
                partial_sort_topk_indices_dynamic(scores2, top_indices2, top_scores2_sorted, actual_k_prime, sqrt_n);
            }
            else
            {
                for (int i = 0; i < actual_k_prime; ++i)
                {
                    top_indices1[i] = 0;
                    top_indices2[i] = 0;
                    top_scores1_sorted[i] = 0.f;
                    top_scores2_sorted[i] = 0.f;
                }
            }

            struct ScorePair
            {
                float value;
                int index;
            };
            ScorePair current_top_k[top_k];
            for (int i = 0; i < top_k; i++)
            {
                current_top_k[i] = {-1e20f, -1};
            }

            for (int i = 0; i < actual_k_prime; i++)
            {
                for (int j = 0; j < actual_k_prime; j++)
                {
                    float prod_score = top_scores1_sorted[i] * top_scores2_sorted[j];
                    int expert_id = top_indices1[i] * sqrt_n + top_indices2[j];
                    if (sqrt_n == 0)
                        expert_id = 0;

                    if (prod_score > current_top_k[top_k - 1].value)
                    {
                        current_top_k[top_k - 1] = {prod_score, expert_id};
                        for (int m = top_k - 1; m > 0; --m)
                        {
                            if (current_top_k[m].value > current_top_k[m - 1].value)
                            {
                                ScorePair temp = current_top_k[m];
                                current_top_k[m] = current_top_k[m - 1];
                                current_top_k[m - 1] = temp;
                            }
                            else
                            {
                                break;
                            }
                        }
                    }
                }
            }

            float sum_exp = 0.0f;
            for (int i = 0; i < top_k; i++)
            {
                if (current_top_k[i].index != -1)
                {
                    current_top_k[i].value = expf(current_top_k[i].value);
                    sum_exp += current_top_k[i].value;
                }
                else
                {
                    current_top_k[i].value = 0.f;
                }
            }

            float inv_sum_exp = (sum_exp == 0.0f) ? 0.0f : (1.0f / sum_exp);

            for (int i = 0; i < top_k; i++)
            {
                expert_indices[i] = current_top_k[i].index;
                expert_scores[i] = current_top_k[i].value * inv_sum_exp;
            }
        }
        __syncthreads();
    }

    // Helper function for partial sorting
    __device__ void partial_sort_topk_indices_dynamic(const float *scores, int *indices, float *top_scores, int k, int n)
    {
        for (int i = 0; i < n; i++)
            indices[i] = i;
        for (int i = 0; i < k; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                if (scores[indices[j]] > scores[indices[i]])
                {
                    int temp = indices[i];
                    indices[i] = indices[j];
                    indices[j] = temp;
                }
            }
            if (top_scores)
                top_scores[i] = scores[indices[i]];
        }
    }

    // ======================== ENHANCED KERNEL WITH TMA AND HOPPER OPTIMIZATIONS ========================

    template <typename T>
    __host__ __device__ inline int compute_l2_chunk_size(int input_dim)
    {
        constexpr int EFFECTIVE_L2_SIZE_BYTES = 50 * 1024 * 1024;
        int bytes_per_token = input_dim * sizeof(T);
        if (bytes_per_token <= 0)
            return 1;
        return max(1, EFFECTIVE_L2_SIZE_BYTES / bytes_per_token);
    }

    template <int MaxExperts_, int BlockM_, int BlockK_, int HiddenSize_, int OuterTiles_>
    struct PEERConfig
    {
        static constexpr int MaxExperts = MaxExperts_;
        static constexpr int BlockM_compute = BlockM_;
        static constexpr int BlockK_compute = BlockK_;
        static constexpr int HiddenSize = HiddenSize_;
        static constexpr int OuterTiles = OuterTiles_;
    };

    template
        typename Config,
        typename Element,
        int NumHeads,
        int TopK,
        int QueryDim,
        int OUT_DIM,
        int BLOCK_DIM >
            __global__
            __launch_bounds__(BLOCK_DIM, 1) void peer_kernel_enhanced(
                const Element *__restrict__ input_gbl,
                const Element *__restrict__ query_weight_gbl,
                const Element *__restrict__ query_bias_gbl,
                const Element *__restrict__ sub_keys1_gbl,
                const Element *__restrict__ sub_keys2_gbl,
                Element *__restrict__ output_gbl,
                ExpertPtrDev *d_experts_table,
                const Element *__restrict__ bn_scale_gbl,
                const Element *__restrict__ bn_bias_gbl,
                int B, int S, int IN_DIM_runtime,
                int sqrt_n_experts,
                int chunk_size,
                float dropout_rate,
                bool use_batch_norm,
                bool norm_keys,
                bool norm_query,
                bool use_fp8_experts)
    {
        static_assert(TopK <= 32, "TopK must be <= 32 for fixed-size arrays in product_key_routing");
        static_assert(QueryDim % 8 == 0, "QueryDim must be divisible by 8 for CUTE alignment/vectorization");
        static_assert(OUT_DIM % 16 == 0, "OUT_DIM must be divisible by 16 for efficient CUTE tiling/MMA");
        static_assert(Config::HiddenSize % 16 == 0, "Config::HiddenSize must be divisible by 16 for FP8 MMA efficiency");

        extern __shared__ char smem_char_ptr[];
        char *current_smem_offset = smem_char_ptr;

        // Allocate shared memory for token cache
        auto token_cache_layout = make_layout(make_shape(Int<chunk_size>{}, IN_DIM_runtime), GenRowMajor{});
        Tensor token_cache_smem = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(current_smem_offset)), token_cache_layout);
        current_smem_offset += sizeof(Element) * size(token_cache_layout);
        current_smem_offset = reinterpret_cast<char *>(align_to<128>(reinterpret_cast<uintptr_t>(current_smem_offset)));

        // Allocate shared memory for expert weights - support both FP8 and FP16
        void *u_buffer_ptr[2];
        void *v_buffer_ptr[2];
        size_t u_buffer_size, v_buffer_size;

        if (use_fp8_experts)
        {
            u_buffer_size = IN_DIM_runtime * Config::HiddenSize * sizeof(cutlass::float_e4m3_t);
            v_buffer_size = Config::HiddenSize * OUT_DIM * sizeof(cutlass::float_e4m3_t);
        }
        else
        {
            u_buffer_size = IN_DIM_runtime * Config::HiddenSize * sizeof(Element);
            v_buffer_size = Config::HiddenSize * OUT_DIM * sizeof(Element);
        }

        u_buffer_ptr[0] = current_smem_offset;
        current_smem_offset += u_buffer_size;
        current_smem_offset = reinterpret_cast<char *>(align_to<128>(reinterpret_cast<uintptr_t>(current_smem_offset)));

        u_buffer_ptr[1] = current_smem_offset;
        current_smem_offset += u_buffer_size;
        current_smem_offset = reinterpret_cast<char *>(align_to<128>(reinterpret_cast<uintptr_t>(current_smem_offset)));

        v_buffer_ptr[0] = current_smem_offset;
        current_smem_offset += v_buffer_size;
        current_smem_offset = reinterpret_cast<char *>(align_to<128>(reinterpret_cast<uintptr_t>(current_smem_offset)));

        v_buffer_ptr[1] = current_smem_offset;
        current_smem_offset += v_buffer_size;
        current_smem_offset = reinterpret_cast<char *>(align_to<128>(reinterpret_cast<uintptr_t>(current_smem_offset)));

        // Rest of shared memory allocations
        auto query_smem_layout = make_layout(make_shape(Int<1>{}, Int<QueryDim>{}));
        Tensor query_smem = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(current_smem_offset)), query_smem_layout);
        current_smem_offset += sizeof(Element) * size(query_smem_layout);
        current_smem_offset = reinterpret_cast<char *>(align_to<128>(reinterpret_cast<uintptr_t>(current_smem_offset)));

        auto hidden_smem_layout = make_layout(make_shape(Int<1>{}, Int<Config::HiddenSize>{}));
        Tensor hidden_smem = make_tensor(make_smem_ptr(reinterpret_cast<float *>(current_smem_offset)), hidden_smem_layout);
        current_smem_offset += sizeof(float) * size(hidden_smem_layout);
        current_smem_offset = reinterpret_cast<char *>(align_to<128>(reinterpret_cast<uintptr_t>(current_smem_offset)));

        auto temp_expert_output_layout = make_layout(make_shape(Int<1>{}, Int<OUT_DIM>{}));
        Tensor temp_expert_output_smem = make_tensor(make_smem_ptr(reinterpret_cast<float *>(current_smem_offset)), temp_expert_output_layout);
        current_smem_offset += sizeof(float) * size(temp_expert_output_layout);
        current_smem_offset = reinterpret_cast<char *>(align_to<128>(reinterpret_cast<uintptr_t>(current_smem_offset)));

        auto routing_scores_layout = make_layout(make_shape(Int<2 * sqrt_n_experts>{}));
        Tensor routing_scores_smem = make_tensor(make_smem_ptr(reinterpret_cast<float *>(current_smem_offset)), routing_scores_layout);

        cuda::pipeline<cuda::thread_scope_block> pipe = cuda::make_pipeline();

        const int tid = threadIdx.x;
        const int bid = blockIdx.x;
        const int warp_id = tid / 32;
        const int lane_id = tid % 32;

        __shared__ cuda::barrier<cuda::thread_scope_block> load_barrier;
        if (tid == 0)
        {
            init(&load_barrier, BLOCK_DIM);
        }
        __syncthreads();

        // Main loop over token chunks
        for (int chunk_base_token_idx = bid * chunk_size;
             chunk_base_token_idx < B * S;
             chunk_base_token_idx += gridDim.x * chunk_size)
        {

            int tokens_in_this_chunk = min(chunk_size, B * S - chunk_base_token_idx);

            // Load token chunk
            for (int i = tid; i < tokens_in_this_chunk * IN_DIM_runtime; i += BLOCK_DIM)
            {
                int token_idx_in_chunk = i / IN_DIM_runtime;
                int feature_idx = i % IN_DIM_runtime;
                int global_flat_token_idx = chunk_base_token_idx + token_idx_in_chunk;
                if (global_flat_token_idx < B * S)
                {
                    token_cache_smem(token_idx_in_chunk, feature_idx) =
                        input_gbl[global_flat_token_idx * IN_DIM_runtime + feature_idx];
                }
            }
            __syncthreads();

            // Process each token
            for (int local_token_offset = 0; local_token_offset < tokens_in_this_chunk; ++local_token_offset)
            {
                int current_global_token_idx = chunk_base_token_idx + local_token_offset;
                Tensor current_token_smem = token_cache_smem(local_token_offset, _);

                constexpr int OUT_PER_THREAD = (OUT_DIM + BLOCK_DIM - 1) / BLOCK_DIM;
                float thread_output[OUT_PER_THREAD];
                for (int i = 0; i < OUT_PER_THREAD; ++i)
                    thread_output[i] = 0.0f;

                // Process each head
                for (int h = 0; h < NumHeads; ++h)
                {
                    // Query projection
                    gemm_query_projection<1, QueryDim, IN_DIM_runtime, BLOCK_DIM>(
                        reinterpret_cast<const cutlass::half_t *>(current_token_smem.data()),
                        reinterpret_cast<const cutlass::half_t *>(query_weight_gbl + h * QueryDim * IN_DIM_runtime),
                        reinterpret_cast<cutlass::half_t *>(query_smem.data()));
                    __syncthreads();

                    // Add bias if present
                    if (query_bias_gbl != nullptr)
                    {
                        for (int i = tid; i < QueryDim; i += BLOCK_DIM)
                        {
                            query_smem(0, i) = Element(float(query_smem(0, i)) +
                                                       float(query_bias_gbl[h * QueryDim + i]));
                        }
                    }
                    __syncthreads();

                    // Batch normalization if requested
                    if (use_batch_norm)
                    {
                        __shared__ float reduction_buffer[BLOCK_DIM / 32];
                        __shared__ float shared_mean;
                        __shared__ float shared_inv_std;

                        // Compute mean
                        float thread_sum = 0.0f;
                        for (int i = tid; i < QueryDim; i += BLOCK_DIM)
                        {
                            thread_sum += static_cast<float>(query_smem(0, i));
                        }
                        thread_sum = warpReduceSum(thread_sum);
                        if (lane_id == 0)
                        {
                            reduction_buffer[warp_id] = thread_sum;
                        }
                        __syncthreads();

                        if (warp_id == 0)
                        {
                            float total_sum = 0.0f;
                            int num_warps_in_block_actual = (BLOCK_DIM + 31) / 32;
                            if (lane_id < num_warps_in_block_actual)
                            {
                                total_sum = reduction_buffer[lane_id];
                            }
                            total_sum = warpReduceSum(total_sum);
                            if (tid == 0)
                            {
                                shared_mean = (QueryDim > 0) ? (total_sum / QueryDim) : 0.0f;
                            }
                        }
                        __syncthreads();

                        // Compute variance
                        float thread_var_sum = 0.0f;
                        float current_mean = shared_mean;
                        for (int i = tid; i < QueryDim; i += BLOCK_DIM)
                        {
                            float diff = static_cast<float>(query_smem(0, i)) - current_mean;
                            thread_var_sum += diff * diff;
                        }
                        thread_var_sum = warpReduceSum(thread_var_sum);
                        if (lane_id == 0)
                        {
                            reduction_buffer[warp_id] = thread_var_sum;
                        }
                        __syncthreads();

                        if (warp_id == 0)
                        {
                            float total_var_sum = 0.0f;
                            int num_warps_in_block_actual = (BLOCK_DIM + 31) / 32;
                            if (lane_id < num_warps_in_block_actual)
                            {
                                total_var_sum = reduction_buffer[lane_id];
                            }
                            total_var_sum = warpReduceSum(total_var_sum);
                            if (tid == 0)
                            {
                                shared_inv_std = (QueryDim > 0) ? rsqrtf(total_var_sum / QueryDim + 1e-5f) : 0.0f;
                            }
                        }
                        __syncthreads();

                        // Apply normalization
                        float current_inv_std = shared_inv_std;
                        for (int i = tid; i < QueryDim; i += BLOCK_DIM)
                        {
                            float normalized_val = (static_cast<float>(query_smem(0, i)) - current_mean) * current_inv_std;
                            if (bn_scale_gbl != nullptr && bn_bias_gbl != nullptr)
                            {
                                normalized_val = normalized_val * static_cast<float>(bn_scale_gbl[h * QueryDim + i]) +
                                                 static_cast<float>(bn_bias_gbl[h * QueryDim + i]);
                            }
                            query_smem(0, i) = Element(normalized_val);
                        }
                    }
                    __syncthreads();

                    // Product key routing
                    __shared__ int expert_indices[TopK];
                    __shared__ float expert_scores[TopK];

                    if (tid == 0 && sqrt_n_experts > 0)
                    {
                        product_key_routing_2d<Element, TopK>(
                            query_smem.data(),
                            sub_keys1_gbl,
                            sub_keys2_gbl,
                            QueryDim,
                            sqrt_n_experts,
                            expert_indices,
                            expert_scores,
                            routing_scores_smem.data(),
                            norm_keys,
                            norm_query);
                    }
                    else if (tid == 0 && sqrt_n_experts == 0)
                    {
                        for (int k = 0; k < TopK; ++k)
                        {
                            expert_indices[k] = (k == 0) ? 0 : -1;
                            expert_scores[k] = (k == 0) ? 1.0f : 0.0f;
                        }
                    }
                    __syncthreads();

                    int buffer_id = 0;

                    // Process selected experts
                    for (int k_expert_loop = 0; k_expert_loop < TopK; k_expert_loop++)
                    {
                        int expert_id = expert_indices[k_expert_loop];
                        float weight = expert_scores[k_expert_loop];

                        if (expert_id == -1 || weight == 0.f)
                            continue;

                        // Fetch expert pointers
                        const cutlass::float_e4m3_t *u_ptr_global = nullptr;
                        const cutlass::float_e4m3_t *v_ptr_global = nullptr;
                        if (tid == 0)
                        {
                            fetch_expert(expert_id, d_experts_table, u_ptr_global, v_ptr_global);
                        }
                        __syncthreads();

                        __shared__ uint64_t u_addr_s, v_addr_s;
                        if (tid == 0)
                        {
                            u_addr_s = reinterpret_cast<uint64_t>(u_ptr_global);
                            v_addr_s = reinterpret_cast<uint64_t>(v_ptr_global);
                        }
                        __syncthreads();
                        u_ptr_global = reinterpret_cast<const cutlass::float_e4m3_t *>(u_addr_s);
                        v_ptr_global = reinterpret_cast<const cutlass::float_e4m3_t *>(v_addr_s);

                        if (u_ptr_global == nullptr || v_ptr_global == nullptr)
                            continue;

                        // Async copy expert weights
                        if (warp_id < 2 && use_fp8_experts)
                        {
                            if (warp_id == 0)
                            {
                                size_t u_bytes = IN_DIM_runtime * Config::HiddenSize * sizeof(cutlass::float_e4m3_t);
                                if (u_bytes > 0)
                                {
                                    pipe.producer_acquire();
                                    cuda::memcpy_async(u_buffer_ptr[buffer_id], u_ptr_global, u_bytes, pipe);
                                    pipe.producer_commit();
                                }
                            }
                            else
                            {
                                size_t v_bytes = Config::HiddenSize * OUT_DIM * sizeof(cutlass::float_e4m3_t);
                                if (v_bytes > 0)
                                {
                                    pipe.producer_acquire();
                                    cuda::memcpy_async(v_buffer_ptr[buffer_id], v_ptr_global, v_bytes, pipe);
                                    pipe.producer_commit();
                                }
                            }
                        }

                        pipe.consumer_wait();
                        load_barrier.arrive_and_wait();

                        // Apply expert networks
                        if (use_fp8_experts)
                        {
                            gemm_expert_up_projection<1, Config::HiddenSize, IN_DIM_runtime, BLOCK_DIM>(
                                reinterpret_cast<const cutlass::half_t *>(current_token_smem.data()),
                                reinterpret_cast<const cutlass::float_e4m3_t *>(u_buffer_ptr[buffer_id]),
                                hidden_smem.data());
                        }
                        else
                        {
                            // FP16 path (original)
                            gemm_query_projection<1, Config::HiddenSize, IN_DIM_runtime, BLOCK_DIM>(
                                reinterpret_cast<const cutlass::half_t *>(current_token_smem.data()),
                                reinterpret_cast<const cutlass::half_t *>(u_buffer_ptr[buffer_id]),
                                reinterpret_cast<cutlass::half_t *>(hidden_smem.data()));
                        }
                        __syncthreads();

                        // Apply GELU activation
                        for (int i = tid; i < Config::HiddenSize; i += BLOCK_DIM)
                        {
                            float x = hidden_smem(0, i);
                            float x3 = x * x * x;
                            float tanh_arg = 0.7978845608f * (x + 0.044715f * x3);
                            hidden_smem(0, i) = 0.5f * x * (1.0f + tanhf(tanh_arg));
                        }
                        __syncthreads();

                        // Apply dropout if requested
                        if (dropout_rate > 0.0f)
                        {
                            curandState_t state;
                            curand_init(clock64() + static_cast<unsigned long long>(expert_id) +
                                            static_cast<unsigned long long>(current_global_token_idx) +
                                            static_cast<unsigned long long>(tid),
                                        0, 0, &state);

                            for (int i = tid; i < Config::HiddenSize; i += BLOCK_DIM)
                            {
                                float rand_val = curand_uniform(&state);
                                if (rand_val < dropout_rate)
                                {
                                    hidden_smem(0, i) = 0.0f;
                                }
                                else
                                {
                                    hidden_smem(0, i) = hidden_smem(0, i) / (1.0f - dropout_rate);
                                }
                            }
                            __syncthreads();
                        }

                        // Apply down projection
                        if (use_fp8_experts)
                        {
                            gemm_expert_down_projection<1, OUT_DIM, Config::HiddenSize, BLOCK_DIM>(
                                hidden_smem.data(),
                                reinterpret_cast<const cutlass::float_e4m3_t *>(v_buffer_ptr[buffer_id]),
                                temp_expert_output_smem.data());
                        }
                        else
                        {
                            // FP16 path
                            gemm_query_projection<1, OUT_DIM, Config::HiddenSize, BLOCK_DIM>(
                                reinterpret_cast<const cutlass::half_t *>(hidden_smem.data()),
                                reinterpret_cast<const cutlass::half_t *>(v_buffer_ptr[buffer_id]),
                                reinterpret_cast<cutlass::half_t *>(temp_expert_output_smem.data()));
                        }
                        __syncthreads();

// Accumulate weighted expert output
#pragma unroll
                        for (int elem_idx = 0; elem_idx < OUT_PER_THREAD; ++elem_idx)
                        {
                            int out_idx = tid + elem_idx * BLOCK_DIM;
                            if (out_idx < OUT_DIM)
                            {
                                thread_output[elem_idx] += weight * temp_expert_output_smem(0, out_idx);
                            }
                        }

                        pipe.consumer_release();
                        buffer_id = 1 - buffer_id;
                        __syncthreads();
                    }
                }

                // Write output
                int current_batch_idx = current_global_token_idx / S;
                int current_seq_idx = current_global_token_idx % S;
                Element *output_token_ptr = output_gbl + current_batch_idx * S * OUT_DIM + current_seq_idx * OUT_DIM;

                for (int i = 0; i < OUT_PER_THREAD; i++)
                {
                    int out_idx = tid + i * BLOCK_DIM;
                    if (out_idx < OUT_DIM)
                    {
                        output_token_ptr[out_idx] = Element(thread_output[i]);
                    }
                }
                __syncthreads();
            }
        }
    }

    // ======================== C++ WRAPPER IMPLEMENTATION ========================

    void set_smem_config_dynamic(void *kernel_ptr, size_t smem_size)
    {
        cudaFuncAttributes attr;
        cudaError_t err_attr = cudaFuncGetAttributes(&attr, kernel_ptr);
        if (err_attr != cudaSuccess)
        {
            printf("Warning: cudaFuncGetAttributes failed for kernel %p: %s\n", kernel_ptr, cudaGetErrorString(err_attr));
            return;
        }

        if (smem_size > static_cast<size_t>(attr.maxDynamicSharedMemorySize))
        {
            printf("Warning: Requested dynamic shared memory size (%zu bytes) for kernel %p exceeds device maximum (%d bytes). Clamping to max.\n",
                   smem_size, kernel_ptr, attr.maxDynamicSharedMemorySize);
            smem_size = attr.maxDynamicSharedMemorySize;
        }

        cudaError_t err_carveout = cudaFuncSetAttribute(kernel_ptr,
                                                        cudaFuncAttributePreferredSharedMemoryCarveout,
                                                        cudaSharedmemCarveoutMaxShared);
        if (err_carveout != cudaSuccess && err_carveout != cudaErrorInvalidValue)
        {
            printf("Warning: Could not set shared memory carveout for kernel %p: %s\n", kernel_ptr, cudaGetErrorString(err_carveout));
        }

        cudaError_t err_set = cudaFuncSetAttribute(kernel_ptr,
                                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                   smem_size);
        if (err_set != cudaSuccess)
        {
            printf("Warning: Could not set dynamic shared memory size to %zu bytes for kernel %p: %s\n",
                   smem_size, kernel_ptr, cudaGetErrorString(err_set));
        }
    }

    // Implementation of HierarchicalExpertCache methods
    void HierarchicalExpertCache::update_cache(cudaStream_t stream)
    {
        cudaError_t err;
        if (num_experts_ == 0 || max_hot_experts_ == 0)
            return;

        extract_heat_deltas_kernel<<<(num_experts_ + 255) / 256, 256, 0, stream>>>(d_experts_device_, d_heat_deltas_, num_experts_);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Error extract_heat_deltas_kernel launch: %s\n", cudaGetErrorString(err));
            return;
        }

        err = cudaMemcpyAsync(h_heat_deltas_, d_heat_deltas_, num_experts_ * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess)
        {
            printf("Error cudaMemcpyAsync D2H heat_deltas: %s\n", cudaGetErrorString(err));
            return;
        }

        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess)
        {
            printf("Error cudaStreamSynchronize after heat_deltas copy: %s\n", cudaGetErrorString(err));
            return;
        }

        std::vector<std::pair<unsigned int, int>> sorted_experts(num_experts_);
        for (int i = 0; i < num_experts_; ++i)
        {
            experts_[i].cpu_heat.fetch_add(h_heat_deltas_[i], std::memory_order_relaxed);
            sorted_experts[i] = {experts_[i].cpu_heat.load(std::memory_order_relaxed), i};
        }
        std::sort(sorted_experts.rbegin(), sorted_experts.rend());

        int hot_count = 0;
        for (int i = 0; i < num_experts_; ++i)
            if (experts_[i].is_hot)
                hot_count++;

        for (const auto &p : sorted_experts)
        {
            int expert_id = p.second;
            if (experts_[expert_id].is_hot)
                continue;

            if (hot_count < max_hot_experts_)
            {
                int free_slot = -1;
                for (int s = 0; s < max_hot_experts_; ++s)
                    if (hbm_slots_free_[s])
                    {
                        free_slot = s;
                        break;
                    }
                if (free_slot != -1)
                {
                    add_expert(expert_id, free_slot, stream);
                    hot_count++;
                }
            }
            else
            {
                // CLOCK eviction
                evict_and_add_expert(expert_id, stream);
            }

            if (hot_count >= max_hot_experts_ && experts_[expert_id].is_hot)
                break;
        }

        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess)
        {
            printf("Error cudaStreamSynchronize after cache update: %s\n", cudaGetErrorString(err));
        }
    }

    void HierarchicalExpertCache::add_expert(int expert_id, int slot_idx, cudaStream_t stream)
    {
        cudaError_t err;
        if (expert_id < 0 || expert_id >= num_experts_ || slot_idx < 0 || slot_idx >= max_hot_experts_ || !hbm_pool_)
        {
            printf("Warning: Add expert failed. Invalid params: expert_id=%d, slot_idx=%d, num_experts_=%d, max_hot_experts_=%d, hbm_pool_=%p\n",
                   expert_id, slot_idx, num_experts_, max_hot_experts_, hbm_pool_);
            return;
        }

        char *slot_ptr = static_cast<char *>(hbm_pool_) + slot_idx * (expert_u_bytes_ + expert_v_bytes_);
        experts_[expert_id].dev_u_ptr = reinterpret_cast<cutlass::float_e4m3_t *>(slot_ptr);
        experts_[expert_id].dev_v_ptr = reinterpret_cast<cutlass::float_e4m3_t *>(slot_ptr + expert_u_bytes_);

        if (expert_u_bytes_ > 0)
        {
            err = cudaMemcpyAsync(experts_[expert_id].dev_u_ptr, experts_[expert_id].host_u_ptr, expert_u_bytes_, cudaMemcpyHostToDevice, stream);
            if (err != cudaSuccess)
            {
                printf("Error cudaMemcpyAsync U for expert %d: %s\n", expert_id, cudaGetErrorString(err));
            }
        }
        if (expert_v_bytes_ > 0)
        {
            err = cudaMemcpyAsync(experts_[expert_id].dev_v_ptr, experts_[expert_id].host_v_ptr, expert_v_bytes_, cudaMemcpyHostToDevice, stream);
            if (err != cudaSuccess)
            {
                printf("Error cudaMemcpyAsync V for expert %d: %s\n", expert_id, cudaGetErrorString(err));
            }
        }

        experts_[expert_id].hbm_slot = slot_idx;
        experts_[expert_id].is_hot = true;
        hbm_slots_free_[slot_idx] = false;
        if (max_hot_experts_ > 0)
        {
            clock_hand_[current_clock_idx_ % max_hot_experts_] = expert_id;
        }

        d_experts_host_[expert_id].dev_u = experts_[expert_id].dev_u_ptr;
        d_experts_host_[expert_id].dev_v = experts_[expert_id].dev_v_ptr;
        d_experts_host_[expert_id].is_hot = true;
        d_experts_host_[expert_id].hbm_slot = slot_idx;
        err = cudaMemcpyAsync(&d_experts_device_[expert_id], &d_experts_host_[expert_id], sizeof(ExpertPtrDev), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            printf("Error cudaMemcpyAsync device table for expert %d: %s\n", expert_id, cudaGetErrorString(err));
        }
    }

    void HierarchicalExpertCache::evict_expert(int expert_id_to_evict, cudaStream_t stream)
    {
        cudaError_t err;
        if (expert_id_to_evict < 0 || expert_id_to_evict >= num_experts_ || !experts_[expert_id_to_evict].is_hot)
            return;

        int slot_idx = experts_[expert_id_to_evict].hbm_slot;
        if (slot_idx < 0 || slot_idx >= max_hot_experts_)
            return;

        hbm_slots_free_[slot_idx] = true;
        experts_[expert_id_to_evict].is_hot = false;
        experts_[expert_id_to_evict].dev_u_ptr = nullptr;
        experts_[expert_id_to_evict].dev_v_ptr = nullptr;
        experts_[expert_id_to_evict].hbm_slot = -1;

        d_experts_host_[expert_id_to_evict].dev_u = nullptr;
        d_experts_host_[expert_id_to_evict].dev_v = nullptr;
        d_experts_host_[expert_id_to_evict].is_hot = false;
        d_experts_host_[expert_id_to_evict].hbm_slot = -1;
        err = cudaMemcpyAsync(&d_experts_device_[expert_id_to_evict], &d_experts_host_[expert_id_to_evict], sizeof(ExpertPtrDev), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            printf("Error cudaMemcpyAsync device table (evict expert %d): %s\n", expert_id_to_evict, cudaGetErrorString(err));
        }
    }

    void HierarchicalExpertCache::evict_and_add_expert(int new_expert_id, cudaStream_t stream)
    {
        int attempts = 0;
        bool evicted = false;

        while (attempts < max_hot_experts_ && !evicted)
        {
            int victim_candidate_slot = current_clock_idx_;
            int victim_expert_id = -1;

            for (int exp_idx = 0; exp_idx < num_experts_; ++exp_idx)
            {
                if (experts_[exp_idx].is_hot && experts_[exp_idx].hbm_slot == victim_candidate_slot)
                {
                    victim_expert_id = exp_idx;
                    break;
                }
            }

            if (victim_expert_id != -1)
            {
                if (experts_[victim_expert_id].cpu_heat.load(std::memory_order_relaxed) == 0 ||
                    experts_[victim_expert_id].cpu_heat.load(std::memory_order_relaxed) <
                        experts_[new_expert_id].cpu_heat.load(std::memory_order_relaxed))
                {
                    evict_expert(victim_expert_id, stream);
                    add_expert(new_expert_id, victim_candidate_slot, stream);
                    evicted = true;
                }
            }
            current_clock_idx_ = (current_clock_idx_ + 1) % max_hot_experts_;
            attempts++;
        }

        if (!evicted)
        {
            int lru_slot_idx = current_clock_idx_;
            int lru_expert_id = -1;
            for (int exp_idx = 0; exp_idx < num_experts_; ++exp_idx)
            {
                if (experts_[exp_idx].is_hot && experts_[exp_idx].hbm_slot == lru_slot_idx)
                {
                    lru_expert_id = exp_idx;
                    break;
                }
            }
            if (lru_expert_id != -1)
            {
                evict_expert(lru_expert_id, stream);
                add_expert(new_expert_id, lru_slot_idx, stream);
            }
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
        bool use_managed,
        bool use_fp8) : num_experts_(num_experts),
                        num_heads_(num_heads),
                        top_k_(top_k),
                        query_dim_(query_dim),
                        expert_hidden_size_(expert_hidden_size),
                        input_dim_(input_dim),
                        output_dim_(output_dim),
                        use_managed_memory_(use_managed),
                        use_fp8_(use_fp8)
    {

        sqrt_n_ = static_cast<int>(std::sqrt(static_cast<double>(num_experts_)));
        if (sqrt_n_ * sqrt_n_ != num_experts_)
        {
            throw std::runtime_error("Number of experts must be a perfect square");
        }

        cache_ = std::make_unique<HierarchicalExpertCache>(
            num_experts_, input_dim_, expert_hidden_size_, output_dim_, hbm_cache_mb);
    }

    PEEROperatorEnhancedImpl::~PEEROperatorEnhancedImpl()
    {
        if (u_weights_)
        {
            if (use_managed_memory_)
            {
                cudaFree(u_weights_);
            }
            else
            {
                cudaFreeHost(u_weights_);
            }
        }
        if (v_weights_)
        {
            if (use_managed_memory_)
            {
                cudaFree(v_weights_);
            }
            else
            {
                cudaFreeHost(v_weights_);
            }
        }
    }

    // PEEROperatorEnhanced implementation
    PEEROperatorEnhanced::PEEROperatorEnhanced(
        int num_experts,
        int num_heads,
        int top_k,
        int query_dim,
        int expert_hidden_size,
        int input_dim,
        int output_dim,
        size_t hbm_cache_mb,
        bool use_managed,
        bool use_fp8) : pImpl(std::make_unique<PEEROperatorEnhancedImpl>(num_experts, num_heads, top_k, query_dim, expert_hidden_size,
                                                                         input_dim, output_dim, hbm_cache_mb, use_managed, use_fp8))
    {
    }

    PEEROperatorEnhanced::~PEEROperatorEnhanced() = default;

    void PEEROperatorEnhanced::allocate_weights()
    {
        cudaError_t err;
        size_t u_elements_per_expert = static_cast<size_t>(pImpl->input_dim_) * pImpl->expert_hidden_size_;
        size_t v_elements_per_expert = static_cast<size_t>(pImpl->expert_hidden_size_) * pImpl->output_dim_;

        size_t u_total_bytes, v_total_bytes;

        if (pImpl->use_fp8_)
        {
            u_total_bytes = static_cast<size_t>(pImpl->num_experts_) * u_elements_per_expert * sizeof(cutlass::float_e4m3_t);
            v_total_bytes = static_cast<size_t>(pImpl->num_experts_) * v_elements_per_expert * sizeof(cutlass::float_e4m3_t);
        }
        else
        {
            u_total_bytes = static_cast<size_t>(pImpl->num_experts_) * u_elements_per_expert * sizeof(__half);
            v_total_bytes = static_cast<size_t>(pImpl->num_experts_) * v_elements_per_expert * sizeof(__half);
        }

        if (pImpl->use_managed_memory_)
        {
            err = cudaMallocManaged(&pImpl->u_weights_, u_total_bytes);
            if (err != cudaSuccess)
            {
                throw std::runtime_error(std::string("cudaMallocManaged for u_weights_ failed: ") + cudaGetErrorString(err));
            }
            err = cudaMallocManaged(&pImpl->v_weights_, v_total_bytes);
            if (err != cudaSuccess)
            {
                cudaFree(pImpl->u_weights_);
                pImpl->u_weights_ = nullptr;
                throw std::runtime_error(std::string("cudaMallocManaged for v_weights_ failed: ") + cudaGetErrorString(err));
            }

            int device;
            err = cudaGetDevice(&device);
            if (err != cudaSuccess)
            {
                printf("Warning cudaGetDevice: %s\n", cudaGetErrorString(err));
            }
            err = cudaMemAdvise(pImpl->u_weights_, u_total_bytes, cudaMemAdviseSetReadMostly, device);
            if (err != cudaSuccess)
            {
                printf("Warning cudaMemAdvise U: %s\n", cudaGetErrorString(err));
            }
            err = cudaMemAdvise(pImpl->v_weights_, v_total_bytes, cudaMemAdviseSetReadMostly, device);
            if (err != cudaSuccess)
            {
                printf("Warning cudaMemAdvise V: %s\n", cudaGetErrorString(err));
            }
        }
        else
        {
            err = cudaMallocHost(&pImpl->u_weights_, u_total_bytes);
            if (err != cudaSuccess)
            {
                throw std::runtime_error(std::string("cudaMallocHost for u_weights_ failed: ") + cudaGetErrorString(err));
            }
            err = cudaMallocHost(&pImpl->v_weights_, v_total_bytes);
            if (err != cudaSuccess)
            {
                cudaFreeHost(pImpl->u_weights_);
                pImpl->u_weights_ = nullptr;
                throw std::runtime_error(std::string("cudaMallocHost for v_weights_ failed: ") + cudaGetErrorString(err));
            }
        }

        pImpl->cache_->allocate_expert_weights(pImpl->u_weights_, pImpl->v_weights_, pImpl->use_managed_memory_);

        const char *type_str = pImpl->use_fp8_ ? "FP8" : "FP16";
        printf("Allocated %.2f GB for U-weights and %.2f GB for V-weights in %s memory (%s).\n",
               static_cast<double>(u_total_bytes) / (1024.0 * 1024.0 * 1024.0),
               static_cast<double>(v_total_bytes) / (1024.0 * 1024.0 * 1024.0),
               pImpl->use_managed_memory_ ? "managed" : "pinned host",
               type_str);
    }

    void PEEROperatorEnhanced::copy_weights_from_torch(const __half *torch_u_weights, const __half *torch_v_weights)
    {
        if (!pImpl->u_weights_ || !pImpl->v_weights_)
        {
            allocate_weights();
        }

        size_t u_total_elements = static_cast<size_t>(pImpl->num_experts_) *
                                  pImpl->input_dim_ * pImpl->expert_hidden_size_;
        size_t v_total_elements = static_cast<size_t>(pImpl->num_experts_) *
                                  pImpl->expert_hidden_size_ * pImpl->output_dim_;

        if (pImpl->use_fp8_)
        {
            // Convert FP16 to FP8
            cutlass::float_e4m3_t *u_fp8_target = reinterpret_cast<cutlass::float_e4m3_t *>(pImpl->u_weights_);
            cutlass::float_e4m3_t *v_fp8_target = reinterpret_cast<cutlass::float_e4m3_t *>(pImpl->v_weights_);

#pragma omp parallel for
            for (size_t i = 0; i < u_total_elements; i++)
            {
                u_fp8_target[i] = cutlass::float_e4m3_t(torch_u_weights[i]);
            }

#pragma omp parallel for
            for (size_t i = 0; i < v_total_elements; i++)
            {
                v_fp8_target[i] = cutlass::float_e4m3_t(torch_v_weights[i]);
            }
        }
        else
        {
            // Direct copy for FP16
            memcpy(pImpl->u_weights_, torch_u_weights, u_total_elements * sizeof(__half));
            memcpy(pImpl->v_weights_, torch_v_weights, v_total_elements * sizeof(__half));
        }

        if (pImpl->use_managed_memory_)
        {
            int device;
            cudaGetDevice(&device);
            cudaMemPrefetchAsync(pImpl->u_weights_, u_total_elements * (pImpl->use_fp8_ ? sizeof(cutlass::float_e4m3_t) : sizeof(__half)), device, 0);
            cudaMemPrefetchAsync(pImpl->v_weights_, v_total_elements * (pImpl->use_fp8_ ? sizeof(cutlass::float_e4m3_t) : sizeof(__half)), device, 0);
        }

        pImpl->cache_->update_expert_pointers(pImpl->u_weights_, pImpl->v_weights_);
    }

    void PEEROperatorEnhanced::set_weight_pointers(const __half *torch_u_weights, const __half *torch_v_weights)
    {
        printf("Warning: set_weight_pointers called with FP16 inputs. Performing conversion for FP8 mode.\n");
        copy_weights_from_torch(torch_u_weights, torch_v_weights);
    }

    void PEEROperatorEnhanced::forward(
        const __half *input,
        const __half *query_weight,
        const __half *query_bias,
        const __half *sub_keys1,
        const __half *sub_keys2,
        __half *output,
        const __half *ln_scale,
        const __half *ln_bias,
        int batch_size,
        int seq_len,
        float dropout_rate,
        cudaStream_t stream)
    {
        pImpl->dropout_rate_ = dropout_rate;

        int chunk_size = compute_l2_chunk_size<__half>(pImpl->input_dim_);
        chunk_size = std::min(chunk_size, batch_size * seq_len);
        chunk_size = std::max(chunk_size, 1);

        using KernelConfig = PEERConfig<1048576, 64, 128, 256, 64>;
        constexpr int ActualBlockDim = 128;

        int num_tokens = batch_size * seq_len;
        int grid_size = (num_tokens + chunk_size - 1) / chunk_size;
        grid_size = min(grid_size, 256);

        // JIT kernel configuration
#ifdef PEER_JIT_TOP_K
        constexpr int KernelTopK = PEER_JIT_TOP_K;
#else
        constexpr int KernelTopK = 16;
#endif

#ifdef PEER_JIT_NUM_HEADS
        constexpr int KernelNumHeads = PEER_JIT_NUM_HEADS;
#else
        constexpr int KernelNumHeads = 8;
#endif

#ifdef PEER_JIT_QUERY_DIM
        constexpr int KernelQueryDim = PEER_JIT_QUERY_DIM;
#else
        constexpr int KernelQueryDim = 256;
#endif

#ifdef PEER_JIT_OUTPUT_DIM
        constexpr int KernelOutDim = PEER_JIT_OUTPUT_DIM;
#else
        constexpr int KernelOutDim = 1024;
#endif

        // Calculate dynamic shared memory size
        size_t smem_size = 0;
        smem_size += align_to<128>(chunk_size * pImpl->input_dim_ * sizeof(__half));

        if (pImpl->use_fp8_)
        {
            size_t u_buffer_expert_size = static_cast<size_t>(pImpl->input_dim_) * KernelConfig::HiddenSize * sizeof(cutlass::float_e4m3_t);
            smem_size += 2 * align_to<128>(u_buffer_expert_size);
            size_t v_buffer_expert_size = static_cast<size_t>(KernelConfig::HiddenSize) * pImpl->output_dim_ * sizeof(cutlass::float_e4m3_t);
            smem_size += 2 * align_to<128>(v_buffer_expert_size);
        }
        else
        {
            size_t u_buffer_expert_size = static_cast<size_t>(pImpl->input_dim_) * KernelConfig::HiddenSize * sizeof(__half);
            smem_size += 2 * align_to<128>(u_buffer_expert_size);
            size_t v_buffer_expert_size = static_cast<size_t>(KernelConfig::HiddenSize) * pImpl->output_dim_ * sizeof(__half);
            smem_size += 2 * align_to<128>(v_buffer_expert_size);
        }

        smem_size += align_to<128>(pImpl->query_dim_ * sizeof(__half));
        smem_size += align_to<128>(KernelConfig::HiddenSize * sizeof(float));
        smem_size += align_to<128>(pImpl->output_dim_ * sizeof(float));
        if (pImpl->sqrt_n_ > 0)
        {
            smem_size += align_to<128>(2 * pImpl->sqrt_n_ * sizeof(float));
        }

        smem_size = size_t(smem_size * 1.05);

        auto kernel_func = peer_kernel_enhanced<KernelConfig, __half, KernelNumHeads, KernelTopK, KernelQueryDim, KernelOutDim, ActualBlockDim>;

        set_smem_config_dynamic(reinterpret_cast<void *>(kernel_func), smem_size);

        if (pImpl->num_experts_ > 1 && pImpl->sqrt_n_ * pImpl->sqrt_n_ != pImpl->num_experts_)
        {
            printf("Warning: num_experts (%d) is not a perfect square. sqrt_n_experts is %d. Routing might be suboptimal or incorrect.\n",
                   pImpl->num_experts_, pImpl->sqrt_n_);
        }

        pImpl->cache_->update_cache(stream);

        kernel_func<<<grid_size, ActualBlockDim, smem_size, stream>>>(
            input, query_weight, query_bias,
            sub_keys1, sub_keys2, output,
            pImpl->cache_->get_device_experts(),
            ln_scale, ln_bias,
            batch_size, seq_len, pImpl->input_dim_,
            pImpl->sqrt_n_,
            chunk_size,
            pImpl->dropout_rate_,
            (ln_scale != nullptr && ln_bias != nullptr),
            true,           // norm_keys
            true,           // norm_query
            pImpl->use_fp8_ // use_fp8_experts
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("Enhanced PEER kernel launch failed: ") +
                cudaGetErrorString(err));
        }
    }

    void PEEROperatorEnhanced::print_cache_stats()
    {
        if (pImpl && pImpl->cache_)
        {
            pImpl->cache_->print_stats();
        }
    }

    int PEEROperatorEnhanced::num_experts() const { return pImpl->num_experts_; }
    int PEEROperatorEnhanced::num_heads() const { return pImpl->num_heads_; }
    int PEEROperatorEnhanced::input_dim() const { return pImpl->input_dim_; }
    int PEEROperatorEnhanced::output_dim() const { return pImpl->output_dim_; }
    bool PEEROperatorEnhanced::use_fp8() const { return pImpl->use_fp8_; }

    void smoke_test()
    {
        constexpr int B = 2;
        constexpr int S = 4;
        constexpr int IN_DIM = 64;
        constexpr int OUT_DIM_TEST = 1024;
        constexpr int EXPERT_HIDDEN = 256;
        constexpr int NUM_EXPERTS = 64;
        constexpr int NUM_HEADS_TEST = 8;
        constexpr int TOP_K_TEST = 4;
        constexpr int QUERY_DIM_TEST = 256;

        // Test both FP16 and FP8 modes
        for (bool use_fp8 : {false, true})
        {
            printf("\nRunning H100-optimized smoke test with %s support: B=%d, S=%d, IN=%d, OUT=%d, Hidden=%d, Experts=%d, Heads=%d, TopK=%d, QueryDim=%d\n",
                   use_fp8 ? "FP8" : "FP16", B, S, IN_DIM, OUT_DIM_TEST, EXPERT_HIDDEN, NUM_EXPERTS, NUM_HEADS_TEST, TOP_K_TEST, QUERY_DIM_TEST);

            PEEROperatorEnhanced op(NUM_EXPERTS, NUM_HEADS_TEST, TOP_K_TEST, QUERY_DIM_TEST, EXPERT_HIDDEN, IN_DIM, OUT_DIM_TEST,
                                    1024, false, use_fp8);

            std::vector<__half> h_u_weights(static_cast<size_t>(NUM_EXPERTS) * IN_DIM * EXPERT_HIDDEN);
            std::vector<__half> h_v_weights(static_cast<size_t>(NUM_EXPERTS) * EXPERT_HIDDEN * OUT_DIM_TEST);
            for (size_t i = 0; i < h_u_weights.size(); ++i)
                h_u_weights[i] = __float2half((rand() % 100) / 200.0f - 0.25f);
            for (size_t i = 0; i < h_v_weights.size(); ++i)
                h_v_weights[i] = __float2half((rand() % 100) / 200.0f - 0.25f);

            op.copy_weights_from_torch(h_u_weights.data(), h_v_weights.data());

            __half *d_input, *d_query_weight, *d_query_bias, *d_sub_keys1, *d_sub_keys2, *d_output;
            __half *d_ln_scale, *d_ln_bias;
            cudaError_t err;

            auto check_cuda_error = [](cudaError_t e, const char *msg)
            {
                if (e != cudaSuccess)
                {
                    throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(e));
                }
            };

            check_cuda_error(cudaMalloc(&d_input, B * S * IN_DIM * sizeof(__half)), "cudaMalloc d_input failed");
            std::vector<__half> h_input(B * S * IN_DIM);
            for (size_t i = 0; i < h_input.size(); ++i)
                h_input[i] = __float2half((rand() % 100) / 100.0f);
            check_cuda_error(cudaMemcpy(d_input, h_input.data(), B * S * IN_DIM * sizeof(__half), cudaMemcpyHostToDevice), "cudaMemcpy d_input failed");

            check_cuda_error(cudaMalloc(&d_query_weight, NUM_HEADS_TEST * QUERY_DIM_TEST * IN_DIM * sizeof(__half)), "cudaMalloc d_query_weight failed");
            std::vector<__half> h_query_weight(NUM_HEADS_TEST * QUERY_DIM_TEST * IN_DIM);
            for (size_t i = 0; i < h_query_weight.size(); ++i)
                h_query_weight[i] = __float2half((rand() % 100) / 200.0f - 0.25f);
            check_cuda_error(cudaMemcpy(d_query_weight, h_query_weight.data(), NUM_HEADS_TEST * QUERY_DIM_TEST * IN_DIM * sizeof(__half), cudaMemcpyHostToDevice), "cudaMemcpy d_query_weight failed");

            check_cuda_error(cudaMalloc(&d_query_bias, NUM_HEADS_TEST * QUERY_DIM_TEST * sizeof(__half)), "cudaMalloc d_query_bias failed");
            std::vector<__half> h_query_bias(NUM_HEADS_TEST * QUERY_DIM_TEST);
            for (size_t i = 0; i < h_query_bias.size(); ++i)
                h_query_bias[i] = __float2half((rand() % 100) / 500.0f - 0.1f);
            check_cuda_error(cudaMemcpy(d_query_bias, h_query_bias.data(), NUM_HEADS_TEST * QUERY_DIM_TEST * sizeof(__half), cudaMemcpyHostToDevice), "cudaMemcpy d_query_bias failed");

            int sqrt_n = static_cast<int>(std::sqrt(static_cast<double>(NUM_EXPERTS)));
            if (sqrt_n * sqrt_n != NUM_EXPERTS)
            {
                printf("Error: NUM_EXPERTS (%d) must be a perfect square for this smoke test. sqrt_n = %d\n", NUM_EXPERTS, sqrt_n);
                cudaFree(d_input);
                cudaFree(d_query_weight);
                cudaFree(d_query_bias);
                throw std::runtime_error("NUM_EXPERTS must be a perfect square for smoke test.");
            }

            check_cuda_error(cudaMalloc(&d_sub_keys1, sqrt_n * QUERY_DIM_TEST * sizeof(__half)), "cudaMalloc d_sub_keys1 failed");
            std::vector<__half> h_sub_keys1(sqrt_n * QUERY_DIM_TEST);
            for (size_t i = 0; i < h_sub_keys1.size(); ++i)
                h_sub_keys1[i] = __float2half((rand() % 100) / 200.0f - 0.25f);
            check_cuda_error(cudaMemcpy(d_sub_keys1, h_sub_keys1.data(), sqrt_n * QUERY_DIM_TEST * sizeof(__half), cudaMemcpyHostToDevice), "cudaMemcpy d_sub_keys1 failed");

            check_cuda_error(cudaMalloc(&d_sub_keys2, sqrt_n * QUERY_DIM_TEST * sizeof(__half)), "cudaMalloc d_sub_keys2 failed");
            std::vector<__half> h_sub_keys2(sqrt_n * QUERY_DIM_TEST);
            for (size_t i = 0; i < h_sub_keys2.size(); ++i)
                h_sub_keys2[i] = __float2half((rand() % 100) / 200.0f - 0.25f);
            check_cuda_error(cudaMemcpy(d_sub_keys2, h_sub_keys2.data(), sqrt_n * QUERY_DIM_TEST * sizeof(__half), cudaMemcpyHostToDevice), "cudaMemcpy d_sub_keys2 failed");

            check_cuda_error(cudaMalloc(&d_output, B * S * OUT_DIM_TEST * sizeof(__half)), "cudaMalloc d_output failed");

            check_cuda_error(cudaMalloc(&d_ln_scale, NUM_HEADS_TEST * QUERY_DIM_TEST * sizeof(__half)), "cudaMalloc d_ln_scale failed");
            std::vector<__half> h_ln_scale(NUM_HEADS_TEST * QUERY_DIM_TEST, __float2half(1.0f));
            check_cuda_error(cudaMemcpy(d_ln_scale, h_ln_scale.data(), NUM_HEADS_TEST * QUERY_DIM_TEST * sizeof(__half), cudaMemcpyHostToDevice), "cudaMemcpy d_ln_scale failed");

            check_cuda_error(cudaMalloc(&d_ln_bias, NUM_HEADS_TEST * QUERY_DIM_TEST * sizeof(__half)), "cudaMalloc d_ln_bias failed");
            std::vector<__half> h_ln_bias(NUM_HEADS_TEST * QUERY_DIM_TEST, __float2half(0.0f));
            check_cuda_error(cudaMemcpy(d_ln_bias, h_ln_bias.data(), NUM_HEADS_TEST * QUERY_DIM_TEST * sizeof(__half), cudaMemcpyHostToDevice), "cudaMemcpy d_ln_bias failed");

            cudaStream_t stream;
            cudaStreamCreate(&stream);

            try
            {
                for (int i = 0; i < 3; i++)
                {
                    op.forward(d_input, d_query_weight, d_query_bias, d_sub_keys1, d_sub_keys2,
                               d_output, d_ln_scale, d_ln_bias, B, S, 0.0f, stream);
                }
                cudaStreamSynchronize(stream);

                cudaEvent_t start_event, stop_event;
                cudaEventCreate(&start_event);
                cudaEventCreate(&stop_event);

                cudaEventRecord(start_event, stream);
                int N_REPS = 10;
                for (int i = 0; i < N_REPS; i++)
                {
                    op.forward(d_input, d_query_weight, d_query_bias, d_sub_keys1, d_sub_keys2,
                               d_output, d_ln_scale, d_ln_bias, B, S, 0.0f, stream);
                }
                cudaEventRecord(stop_event, stream);
                cudaStreamSynchronize(stream);

                float ms;
                cudaEventElapsedTime(&ms, start_event, stop_event);
                printf("Average kernel time (%s): %.3f ms\n", use_fp8 ? "FP8" : "FP16", ms / N_REPS);

                std::vector<__half> h_output(B * S * OUT_DIM_TEST);
                check_cuda_error(cudaMemcpy(h_output.data(), d_output, B * S * OUT_DIM_TEST * sizeof(__half), cudaMemcpyDeviceToHost), "cudaMemcpy d_output failed");

                bool nan_found = false;
                double sum_output = 0.0;
                for (size_t i = 0; i < h_output.size(); ++i)
                {
                    if (std::isnan(__half2float(h_output[i])))
                    {
                        nan_found = true;
                        break;
                    }
                    sum_output += static_cast<double>(__half2float(h_output[i]));
                }

                if (nan_found)
                {
                    printf("H100-optimized smoke test with %s support FAILED: NaN found in output.\n", use_fp8 ? "FP8" : "FP16");
                }
                else if (sum_output == 0.0 && h_output.size() > 0)
                {
                    printf("H100-optimized smoke test with %s support FAILED: Output is all zeros.\n", use_fp8 ? "FP8" : "FP16");
                }
                else
                {
                    printf("H100-optimized smoke test with %s support PASSED! (Output sum: %f)\n", use_fp8 ? "FP8" : "FP16", sum_output);
                }

                err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    printf("CUDA error after kernel execution: %s\n", cudaGetErrorString(err));
                }

                cudaEventDestroy(start_event);
                cudaEventDestroy(stop_event);
            }
            catch (const std::exception &e)
            {
                printf("Exception during smoke test: %s\n", e.what());
            }

            op.print_cache_stats();

            cudaFree(d_input);
            cudaFree(d_query_weight);
            cudaFree(d_query_bias);
            cudaFree(d_sub_keys1);
            cudaFree(d_sub_keys2);
            cudaFree(d_output);
            cudaFree(d_ln_scale);
            cudaFree(d_ln_bias);
            cudaStreamDestroy(stream);
        }
    }

} // namespace peer

#ifdef COMPILE_SMOKE_TEST
int main()
{
    try
    {
        peer::smoke_test();
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "Unhandled exception in main: %s\n", e.what());
        return 1;
    }
    return 0;
}
#endif