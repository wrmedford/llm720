# PEER Fused Triton Kernel Implementation Plan

This document outlines the plan for implementing a fused Triton kernel for the PEER (Parameter Efficient Expert Retrieval) module, aiming to maximize performance by leveraging GPU memory hierarchy, specifically inspired by the ping-pong expert processing pattern described in `docs/MEMORY_HIERARCHY.md` and concepts from the Mixture-of-A-Million-Experts paper.

## 1. Current Implementation Status

*   **PyTorch Path:** A functional PyTorch implementation (`PEER._forward_pytorch`) exists, handling query projection, normalization, product-key expert selection (`_get_expert_indices_pytorch`), expert computation (`nn.Embedding` weights), scoring (softmax), and aggregation.
*   **Triton Path:** Dispatch logic (`PEER.forward`) exists, controlled by `USE_TRITON_KERNEL`. `PEER._forward_triton` calls a *stubbed* kernel (`peer_fwd_kernel` in `llm/models/kernels/peer_triton.py`) which currently returns zeros.
*   **Testing:** `tests/test_peer_integration.py` verifies the PyTorch path and confirms the Triton stub runs correctly.

## 2. Alignment with Mixture-of-A-Million-Experts (MoM)

*   **Core Concept:** PEER implements the MoM idea of using product keys for efficient retrieval from a massive pool of simple experts.
*   **Tiny Experts:** The configuration supports `expert_hidden_size=1`.
*   **Efficiency Goal:** The fused Triton kernel aims to realize the *performance* benefits of sparse activation on modern hardware, using hardware-aware techniques like the ping-pong pattern.

## 3. Requirements for the Triton Kernel

### 3.1. Accuracy

*   **Requirement:** Triton kernel output must be numerically equivalent to the PyTorch path (`_forward_pytorch`) within acceptable tolerances (e.g., `atol=1e-3`, `rtol=1e-3` for FP16/BF16).
*   **Verification:** Create `tests/test_peer_correctness.py`. Parameterize tests by configuration. Compare Triton and PyTorch outputs using `torch.allclose`.

### 3.2. Performance

*   **Requirement:** Triton kernel must significantly outperform the PyTorch path on compatible GPUs (Hopper+). Latency hiding via ping-pong buffering should be effective.
*   **Verification:**
    *   Micro-benchmarks comparing execution time (`time.time()` or `torch.cuda.Event`).
    *   Profiling with Nsight Compute (`ncu`) to measure occupancy, memory throughput, compute utilization, and memory dependency stalls.

### 3.3. Functionality

*   **Requirement:** Implement the *entire* PEER forward pass logic within the fused kernel:
    1.  Query Projection (`query_proj_weight`).
    2.  Optional Query Normalization (LayerNorm: `query_norm_weight`, `query_norm_bias`).
    3.  Split Queries into sub-queries.
    4.  Sub-key Scoring (matmul with `sub_keys`, optional normalization).
    5.  Optimized Top-K Expert Index/Score Calculation (matching `_get_expert_indices_pytorch`).
    6.  Softmax normalization of scores.
    7.  Asynchronous Fetching of Expert Weights (`expert_down_weight`, `expert_up_weight`) using ping-pong pattern.
    8.  Expert Computation: `output = activation(hidden_states @ expert_down_weight) @ expert_up_weight`.
    9.  Weighting expert outputs by normalized scores.
    10. Aggregation (summing weighted outputs over selected experts and heads).
*   **Verification:** Correctness tests covering different configurations (`product_key_dim`, `num_heads`, `top_k`, dtypes, normalizations).

## 4. Triton, L1 Cache Efficiency, and Ping-Pong Experts

*   **Goal:** Maximize SMEM (L1/Scratchpad) usage to hide HBM latency for expert weights.
*   **Mechanism:**
    1.  **Token Slice in SMEM:** Load the `hidden_states` slice relevant to the thread block into SMEM once.
    2.  **Expert Weight Buffers:** Allocate *two* SMEM buffers, each sized for one expert's weights (`down` + `up`).
    3.  **Double Buffering (Ping-Pong):**
        *   While computing with expert `k` (weights in Buffer A), asynchronously load weights for expert `k+1` into Buffer B using TMA (`tl.load` with cache hints).
        *   Switch buffers for expert `k+1` computation and prefetch `k+2` into Buffer A.
    4.  **TMA (Tensor Memory Accelerator):** Use TMA for direct HBM->SMEM transfers, bypassing L2 to avoid cache pollution.
*   **Benefit:** Keep compute units busy by minimizing stalls waiting for HBM memory access.

## 5. Constraints for Ping-Pong Pattern

*   **SMEM Budget:** The primary constraint.
    *   `SMEM_Needed = Size(Token_Slice_Per_Block) + 2 * Size(Single_Expert_Weights)`
    *   `Size(Single_Expert_Weights) = (input_dim * expert_hidden_size + expert_hidden_size * output_dim) * dtype_bytes`
    *   `Size(Token_Slice_Per_Block) = BLOCK_SIZE_M * input_dim * dtype_bytes`
    *   **Constraint:** `SMEM_Needed <= Available_SMEM` (e.g., <= 228KB on H100).
*   **Implication:** Larger model dimensions (`input_dim`, `output_dim`, `expert_hidden_size`) increase `SMEM_Needed`, potentially forcing smaller `BLOCK_SIZE_M` or making ping-pong infeasible. `expert_hidden_size=1` is highly advantageous.
*   **Verification (SMEM):**
    *   *Static:* Utility function `calculate_smem_usage(config, block_dims)` to check theoretical usage against limits.
    *   *Dynamic:* Profile with `ncu --metrics sm__shared_memory_used.sum` to verify actual usage.

## 6. Alignment Requirements

*   **TMA:** Expert weights in HBM should ideally be 128-byte aligned.
*   **Vectorized Access:** Triton `tl.load`/`tl.store` pointers should be aligned to access size (e.g., 16 bytes for float4) for best performance.
*   **Verification (Alignment):** Profile with `ncu --metrics smsp__inst_executed_shared_ld_misaligned.sum,smsp__inst_executed_shared_st_misaligned.sum` (or global memory equivalents) to detect misaligned accesses.

## 7. Token Block Sizing/Alignment (L2 Cache)

*   **Goal:** Keep the token chunk processed per grid launch within the L2 cache (e.g., <= 40MB target for H100).
*   **Sizing:** Host code calculates `CHUNK_SIZE` such that `CHUNK_SIZE * seq_len * hidden_size * dtype_bytes <= L2_Target_Size`.
*   **Verification (L2):**
    *   *Static:* Assert chunk size calculation in host code.
    *   *Dynamic:* Profile with `ncu --metrics lts__t_sectors_lookup_hit_rate.pct` to check L2 hit rates for token data.

## 8. Next Steps

*   Begin implementing the Triton kernel logic within `llm/models/kernels/peer_triton.py`, starting with the core computations (projections, scoring, expert application) before adding the asynchronous ping-pong memory transfers.
*   Develop the correctness tests (`tests/test_peer_correctness.py`) alongside the kernel implementation.
*   Iteratively benchmark and profile the kernel to ensure performance goals are met and constraints are respected.
