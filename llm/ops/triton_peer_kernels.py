#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Triton kernels for optimizing PEER expert selection.
"""

from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Function

# Try to import Triton, handle ImportError
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("Triton not found. PEER expert selection will use slower PyTorch implementation.")
    # Define dummy decorators if Triton is not available
    def autotune(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def heuristics(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def grid(*args, **kwargs):
        return lambda meta: meta # Return the meta dictionary itself

if HAS_TRITON:

    # --- Helper Function for Multi-Dimensional Indexing ---
    @triton.jit
    def get_multi_dim_indices(expert_idx, dim_sizes_ptr, num_dims: tl.constexpr, stride_ds):
        """Calculates the multi-dimensional indices for a given linear expert index."""
        indices = tl.zeros((num_dims,), dtype=tl.int32)
        current_expert_idx = expert_idx
        # Calculate indices from last dimension to first
        for d in range(num_dims - 1, -1, -1):
            dim_size = tl.load(dim_sizes_ptr + d * stride_ds)
            indices = tl.where(d == tl.arange(0, num_dims), current_expert_idx % dim_size, indices)
            current_expert_idx = current_expert_idx // dim_size
        return indices

    # --- Fused Forward Kernel ---
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE_T': 64, 'BLOCK_SIZE_E': 128, 'num_warps': 4}, num_stages=2),
            triton.Config({'BLOCK_SIZE_T': 128, 'BLOCK_SIZE_E': 64, 'num_warps': 4}, num_stages=2),
            triton.Config({'BLOCK_SIZE_T': 32, 'BLOCK_SIZE_E': 256, 'num_warps': 8}, num_stages=2),
            triton.Config({'BLOCK_SIZE_T': 256, 'BLOCK_SIZE_E': 32, 'num_warps': 8}, num_stages=2),
            # Add more configs as needed
        ],
        key=['num_tokens', 'num_experts', 'top_k', 'query_dim'],
    )
    @triton.jit
    def peer_selection_fwd_kernel(
        # Pointers
        queries_ptr, sub_keys_cat_ptr, output_indices_ptr, output_scores_ptr,
        dim_sizes_ptr, dim_offsets_ptr,
        # Dimensions & Params
        num_tokens, query_dim, num_experts, top_k,
        num_dims: tl.constexpr, sub_query_dim: tl.constexpr,
        # Strides
        stride_qt, stride_qd, stride_skt, stride_skd,
        stride_oi_t, stride_oi_k, stride_os_t, stride_os_k,
        stride_ds, stride_do,
        # Normalization
        norm_query: tl.constexpr, norm_keys: tl.constexpr,
        # Meta
        BLOCK_SIZE_T: tl.constexpr, BLOCK_SIZE_E: tl.constexpr,
    ):
        pid_t = tl.program_id(axis=0)
        token_start_idx = pid_t * BLOCK_SIZE_T
        token_offs = token_start_idx + tl.arange(0, BLOCK_SIZE_T)
        token_mask = token_offs < num_tokens

        # --- Load query block (as original dtype, e.g., float16) ---
        q_ptrs = queries_ptr + token_offs[:, None] * stride_qt + tl.arange(0, query_dim)[None, :] * stride_qd
        query_block_orig_dtype = tl.load(q_ptrs, mask=token_mask[:, None], other=0.0)
        query_block_fp32 = query_block_orig_dtype.to(tl.float32) # Keep fp32 for normalization

        # --- Normalize query block (in FP32) ---
        if norm_query:
            q_norm = tl.sqrt(tl.sum(query_block_fp32 * query_block_fp32, axis=1, keep_dims=True))
            query_block_normed_fp32 = query_block_fp32 / (q_norm + 1e-6)
        else:
            query_block_normed_fp32 = query_block_fp32 # Use unnormalized if norm_query is false

        # --- Prepare Top-K ---
        top_k_scores = tl.full((BLOCK_SIZE_T, top_k), -float('inf'), dtype=tl.float32)
        top_k_indices = tl.full((BLOCK_SIZE_T, top_k), -1, dtype=tl.int32)

        # --- Load dimension info ---
        dim_sizes = tl.load(dim_sizes_ptr + tl.arange(0, num_dims) * stride_ds)
        dim_offsets = tl.load(dim_offsets_ptr + tl.arange(0, num_dims) * stride_do)

        # --- Iterate through experts ---
        for expert_start_idx in range(0, num_experts, BLOCK_SIZE_E):
            expert_offs = expert_start_idx + tl.arange(0, BLOCK_SIZE_E)
            expert_mask = expert_offs < num_experts

            # --- Calculate scores for the expert block ---
            # [BLOCK_SIZE_T, BLOCK_SIZE_E]
            current_expert_scores = tl.zeros((BLOCK_SIZE_T, BLOCK_SIZE_E), dtype=tl.float32)

            for d_idx in range(num_dims):
                # Calculate multi-dim index for this dimension
                remaining_idx = expert_offs
                current_dim_size = tl.load(dim_sizes_ptr + d_idx * stride_ds)
                divisor = 1
                for d_inner in range(d_idx + 1, num_dims):
                   divisor *= tl.load(dim_sizes_ptr + d_inner * stride_ds)
                dim_expert_indices = (remaining_idx // divisor) % current_dim_size

                # Load sub-key for this dimension and expert index
                key_offset = tl.load(dim_offsets_ptr + d_idx * stride_do)
                # Pointers to sub-keys: [BLOCK_SIZE_E, sub_query_dim]
                sk_ptrs = sub_keys_cat_ptr + \
                          (key_offset + dim_expert_indices)[:, None] * stride_skt + \
                          tl.arange(0, sub_query_dim)[None, :] * stride_skd
                # Load sub-key block (as original dtype, e.g., float16)
                sub_key_block_orig_dtype = tl.load(sk_ptrs, mask=expert_mask[:, None], other=0.0)
                sub_key_block_fp32 = sub_key_block_orig_dtype.to(tl.float32) # Keep fp32 for normalization

                # Normalize sub-key block (in FP32)
                if norm_keys:
                    sk_norm = tl.sqrt(tl.sum(sub_key_block_fp32 * sub_key_block_fp32, axis=1, keep_dims=True))
                    sub_key_block_normed_fp32 = sub_key_block_fp32 / (sk_norm + 1e-6)
                else:
                    sub_key_block_normed_fp32 = sub_key_block_fp32

                # Load corresponding query chunk (already loaded and potentially normalized in FP32)
                q_chunk_offs = tl.arange(0, sub_query_dim) + d_idx * sub_query_dim
                query_chunk_normed_fp32 = query_block_normed_fp32[:, q_chunk_offs]

                # --- Matmul (No explicit FP8 casting here - rely on ambient precision from fp8_autocast) ---
                # Inputs (query_chunk_normed_fp32, sub_key_block_normed_fp32) are FP32
                # Perform dot product, accumulate in FP32
                # TE's fp8_autocast context should handle casting inputs to tl.dot if TE intercepts it,
                # otherwise, it runs in the input precision (FP32 here).
                # For safety, let's assume it runs in FP32 if not intercepted by TE.
                dim_scores = tl.dot(query_chunk_normed_fp32, tl.trans(sub_key_block_normed_fp32), out_dtype=tl.float32, allow_tf32=False)
                current_expert_scores += dim_scores # Accumulate in FP32

            # --- Update Top-K ---
            # Combine current top-k with new scores and indices
            combined_scores = tl.cat((top_k_scores, current_expert_scores), axis=1)
            expert_indices_block = expert_offs[None, :].to(tl.int32) # Shape [1, BLOCK_SIZE_E]
            expert_indices_block_expanded = tl.broadcast_to(expert_indices_block, (BLOCK_SIZE_T, BLOCK_SIZE_E))
            combined_indices = tl.cat((top_k_indices, expert_indices_block_expanded), axis=1) # Shape [BLOCK_SIZE_T, top_k + BLOCK_SIZE_E]

            # Sort combined scores in descending order along the last dimension (axis=1)
            # tl.sort returns (sorted_values, sorted_indices_relative_to_input)
            sorted_scores, sorted_relative_indices = tl.sort(combined_scores, axis=1, descending=True)

            # Select the top-k scores
            top_k_scores = sorted_scores[:, :top_k]

            # Use the sorted_relative_indices to gather the corresponding expert indices
            # Need to expand sorted_relative_indices to match combined_indices shape for gather
            # Gather indices: indices[sorted_relative_indices]
            # We only need the indices corresponding to the top k scores
            top_k_relative_indices = sorted_relative_indices[:, :top_k] # Shape [BLOCK_SIZE_T, top_k]

            # Gather the actual expert indices using the relative indices
            # combined_indices shape: [BLOCK_SIZE_T, top_k + BLOCK_SIZE_E]
            # top_k_relative_indices shape: [BLOCK_SIZE_T, top_k]
            # We need to gather along axis=1
            # Add offset for the token dimension (axis=0) for gather
            token_idx_offsets = tl.arange(0, BLOCK_SIZE_T)[:, None] * combined_indices.shape[1]
            gather_indices = token_idx_offsets + top_k_relative_indices
            # Flatten combined_indices for gather
            combined_indices_flat = tl.reshape(combined_indices, (BLOCK_SIZE_T * combined_indices.shape[1],))
            # Gather the top k indices
            top_k_indices = tl.load(combined_indices_flat + gather_indices) # Shape [BLOCK_SIZE_T, top_k]


        # --- Store final top-k results ---
        oi_ptrs = output_indices_ptr + token_offs[:, None] * stride_oi_t + tl.arange(0, top_k)[None, :] * stride_oi_k
        os_ptrs = output_scores_ptr + token_offs[:, None] * stride_os_t + tl.arange(0, top_k)[None, :] * stride_os_k
        tl.store(oi_ptrs, top_k_indices, mask=token_mask[:, None])
        tl.store(os_ptrs, top_k_scores.to(output_scores_ptr.dtype.element_ty), mask=token_mask[:, None])


    # --- Fused Backward Kernel ---
    @triton.jit
    def peer_selection_bwd_kernel(
        # Inputs from forward & grad_outputs
        queries_ptr, sub_keys_cat_ptr, output_indices_ptr, grad_scores_ptr,
        dim_sizes_ptr, dim_offsets_ptr,
        # Outputs: Gradients for inputs
        grad_queries_ptr, grad_sub_keys_cat_ptr,
        # Dimensions & Params
        num_tokens, query_dim, num_experts, top_k,
        num_dims: tl.constexpr, sub_query_dim: tl.constexpr,
        # Strides
        stride_qt, stride_qd, stride_skt, stride_skd,
        stride_oi_t, stride_oi_k, stride_gs_t, stride_gs_k, # grad_scores strides
        stride_gq_t, stride_gq_d, stride_gsk_t, stride_gsk_d, # grad inputs strides
        stride_ds, stride_do,
        # Normalization
        norm_query: tl.constexpr, norm_keys: tl.constexpr,
        # Meta
        BLOCK_SIZE_T: tl.constexpr, # Must match forward for consistency if using same grid
    ):
        pid_t = tl.program_id(axis=0)
        token_start_idx = pid_t * BLOCK_SIZE_T
        token_offs = token_start_idx + tl.arange(0, BLOCK_SIZE_T)
        token_mask = token_offs < num_tokens

        # --- Load query block (as original dtype, cast to FP32 for calculations) ---
        q_ptrs = queries_ptr + token_offs[:, None] * stride_qt + tl.arange(0, query_dim)[None, :] * stride_qd
        query_block_orig_dtype = tl.load(q_ptrs, mask=token_mask[:, None], other=0.0)
        query_block_fp32 = query_block_orig_dtype.to(tl.float32)

        # Normalize query block if needed (in FP32, mirror forward pass)
        if norm_query:
            q_norm = tl.sqrt(tl.sum(query_block_fp32 * query_block_fp32, axis=1, keep_dims=True))
            query_block_normed_fp32 = query_block_fp32 / (q_norm + 1e-6)
        else:
            query_block_normed_fp32 = query_block_fp32

        # --- Load top-k indices and grad_scores for this token block ---
        oi_ptrs = output_indices_ptr + token_offs[:, None] * stride_oi_t + tl.arange(0, top_k)[None, :] * stride_oi_k
        top_indices = tl.load(oi_ptrs, mask=token_mask[:, None], other=-1) # [BLOCK_SIZE_T, top_k]
        gs_ptrs = grad_scores_ptr + token_offs[:, None] * stride_gs_t + tl.arange(0, top_k)[None, :] * stride_gs_k
        # Load grad_scores (as original dtype, cast to FP32 for accumulation)
        grad_scores_orig_dtype = tl.load(gs_ptrs, mask=token_mask[:, None], other=0.0)
        grad_scores_fp32 = grad_scores_orig_dtype.to(tl.float32) # [BLOCK_SIZE_T, top_k]

        # --- Initialize gradient accumulators for queries (in FP32) ---
        grad_queries_acc = tl.zeros((BLOCK_SIZE_T, query_dim), dtype=tl.float32)

        # --- Iterate through the selected top_k experts for each token ---
        for k_idx in range(top_k):
            expert_indices = tl.load(top_indices + k_idx, axis=1) # [BLOCK_SIZE_T]
            # Load the gradient for the k-th expert score for each token in the block
            current_grad_scores = grad_scores_fp32[:, k_idx] # [BLOCK_SIZE_T]
            expert_mask = expert_indices != -1 # Mask for valid indices

            # --- Calculate gradients for each dimension ---
            for d_idx in range(num_dims):
                # Calculate multi-dim index for this dimension
                remaining_idx = expert_indices
                current_dim_size = tl.load(dim_sizes_ptr + d_idx * stride_ds)
                divisor = 1
                for d_inner in range(d_idx + 1, num_dims):
                   divisor *= tl.load(dim_sizes_ptr + d_inner * stride_ds)
                dim_expert_indices = (remaining_idx // divisor) % current_dim_size

                # Load sub-key for this dimension and expert index
                key_offset = tl.load(dim_offsets_ptr + d_idx * stride_do)
                sk_ptrs = sub_keys_cat_ptr + \
                          (key_offset + dim_expert_indices)[:, None] * stride_skt + \
                          tl.arange(0, sub_query_dim)[None, :] * stride_skd
                # Load sub-key block (as original dtype, cast to FP32 for calculations)
                sub_key_block_orig_dtype = tl.load(sk_ptrs, mask=expert_mask[:, None], other=0.0)
                sub_key_block_fp32 = sub_key_block_orig_dtype.to(tl.float32) # [BLOCK_SIZE_T, sub_query_dim]

                # Normalize sub-key block if needed (in FP32, mirror forward)
                if norm_keys:
                    sk_norm = tl.sqrt(tl.sum(sub_key_block_fp32 * sub_key_block_fp32, axis=1, keep_dims=True))
                    sub_key_block_normed_fp32 = sub_key_block_fp32 / (sk_norm + 1e-6)
                else:
                    sub_key_block_normed_fp32 = sub_key_block_fp32

                # --- Calculate Gradient w.r.t. Query Chunk (FP32 Accumulation) ---
                # grad_score * sub_key_normed
                # Perform multiplication in FP32 for stability before accumulation
                grad_q_chunk_fp32 = current_grad_scores[:, None] * sub_key_block_normed_fp32 # [BLOCK_SIZE_T, sub_query_dim]

                # Accumulate gradient into the correct query chunk position (FP32)
                q_chunk_offs = tl.arange(0, sub_query_dim) + d_idx * sub_query_dim
                # Handle query normalization gradient (in FP32)
                if norm_query:
                     query_chunk_fp32 = query_block_fp32[:, q_chunk_offs] # Get the original unnormalized chunk
                     q_chunk_normed_fp32 = query_block_normed_fp32[:, q_chunk_offs]
                     grad_q_normed_fp32 = grad_q_chunk_fp32
                     sum_grad_q_normed_times_q_normed = tl.sum(grad_q_normed_fp32 * q_chunk_normed_fp32, axis=1, keep_dims=True)
                     # Approximate norm for chunk using the full query norm for simplicity
                     q_chunk_norm_approx = q_norm + 1e-6 # Use the precomputed full query norm
                     grad_q_unnormed_fp32 = (grad_q_normed_fp32 - q_chunk_normed_fp32 * sum_grad_q_normed_times_q_normed) / q_chunk_norm_approx
                     # Accumulate gradient based on valid experts
                     grad_queries_acc = tl.where(expert_mask[:, None] & token_mask[:, None], grad_queries_acc + grad_q_unnormed_fp32[:, q_chunk_offs], grad_queries_acc)
                else:
                     # Accumulate gradient based on valid experts
                     grad_queries_acc = tl.where(expert_mask[:, None] & token_mask[:, None], grad_queries_acc + grad_q_chunk_fp32[:, q_chunk_offs], grad_queries_acc)


                # --- Calculate Gradient w.r.t. Sub-Key Chunk (FP32 Accumulation with Atomics) ---
                # grad_sk = sum_tokens(grad_score * query_chunk_normed)
                query_chunk_normed_fp32 = query_block_normed_fp32[:, q_chunk_offs] # [BLOCK_SIZE_T, sub_query_dim]
                # Calculate per-token gradient contribution (in FP32)
                grad_sk_normed_per_token_fp32 = current_grad_scores[:, None] * query_chunk_normed_fp32 # [BLOCK_SIZE_T, sub_query_dim]

                # Handle key normalization gradient (in FP32)
                if norm_keys:
                     # Backprop through normalization
                     sum_grad_sk_normed_times_sk_normed = tl.sum(grad_sk_normed_per_token_fp32 * sub_key_block_normed_fp32, axis=1, keep_dims=True) # Sum over head dim
                     grad_sk_unnormed_per_token_fp32 = (grad_sk_normed_per_token_fp32 - sub_key_block_normed_fp32 * sum_grad_sk_normed_times_sk_normed) / (sk_norm + 1e-6)
                     grad_sk_to_add_fp32 = grad_sk_unnormed_per_token_fp32
                else:
                     grad_sk_to_add_fp32 = grad_sk_normed_per_token_fp32

                # Pointers to gradient output for sub-keys
                grad_sk_ptrs = grad_sub_keys_cat_ptr + \
                               (key_offset + dim_expert_indices)[:, None] * stride_gsk_t + \
                               tl.arange(0, sub_query_dim)[None, :] * stride_gsk_d

                # Atomic Add for Sub-Key Gradients
                # Cast the FP32 gradient contribution back to the output tensor's dtype before adding
                grad_sk_to_add_output_dtype = grad_sk_to_add_fp32.to(grad_sub_keys_cat_ptr.dtype.element_ty)
                tl.atomic_add(grad_sk_ptrs, grad_sk_to_add_output_dtype, mask=expert_mask[:, None] & token_mask[:, None])


        # --- Store accumulated query gradients (cast back to original dtype) ---
        grad_q_ptrs = grad_queries_ptr + token_offs[:, None] * stride_gq_t + tl.arange(0, query_dim)[None, :] * stride_gq_d
        tl.store(grad_q_ptrs, grad_queries_acc.to(grad_queries_ptr.dtype.element_ty), mask=token_mask[:, None])


    # --- Autograd Function ---
    class PEERSelectionFunction(Function):
        @staticmethod
        def forward(ctx, queries, sub_keys_list, product_key_dim, top_k, norm_query, norm_keys):
            if not HAS_TRITON:
                raise RuntimeError("Triton is not available.")

            batch_size, seq_len, num_heads, query_dim = queries.shape
            device = queries.device
            dtype = queries.dtype

            num_tokens = batch_size * seq_len * num_heads
            num_dims = len(product_key_dim)
            num_experts = np.prod(product_key_dim).item() # Use .item()
            sub_query_dim = query_dim // num_dims
            assert query_dim % num_dims == 0, "Query dim must be divisible by num_dims"

            queries_flat = queries.reshape(num_tokens, query_dim).contiguous()
            # Ensure sub_keys are contiguous and correct dtype
            sub_keys_cat = torch.cat([sk.contiguous().to(queries.device, dtype=queries.dtype) for sk in sub_keys_list], dim=0)
            dim_sizes = torch.tensor(product_key_dim, dtype=torch.int32, device=device)
            dim_offsets = torch.zeros_like(dim_sizes)
            dim_offsets[1:] = torch.cumsum(dim_sizes[:-1], dim=0)

            output_indices = torch.empty((num_tokens, top_k), dtype=torch.int32, device=device)
            # Output scores should match query dtype for consistency with PyTorch version
            output_scores = torch.empty((num_tokens, top_k), dtype=dtype, device=device)

            grid = lambda META: (triton.cdiv(num_tokens, META['BLOCK_SIZE_T']),)
            # Ensure inputs passed to kernel match expected dtypes (e.g., FP16/BF16)
            # Kernel handles internal casting
            queries_kernel = queries_flat.to(dtype) # Ensure correct input dtype
            sub_keys_kernel = sub_keys_cat.to(dtype) # Ensure correct input dtype

            # Launch Forward Kernel
            peer_selection_fwd_kernel[grid](
                queries_kernel, sub_keys_kernel, output_indices, output_scores,
                dim_sizes, dim_offsets,
                num_tokens, query_dim, num_experts, top_k,
                num_dims, sub_query_dim,
                queries_flat.stride(0), queries_flat.stride(1),
                sub_keys_cat.stride(0), sub_keys_cat.stride(1),
                output_indices.stride(0), output_indices.stride(1),
                output_scores.stride(0), output_scores.stride(1),
                dim_sizes.stride(0), dim_offsets.stride(0),
                norm_query, norm_keys,
                # BLOCK_SIZE_T=64, BLOCK_SIZE_E=128 # Let autotuner pick
            )

            # Reshape outputs
            output_indices_reshaped = output_indices.view(batch_size, seq_len, num_heads, top_k)
            output_scores_reshaped = output_scores.view(batch_size, seq_len, num_heads, top_k)

            # Save for backward
            # Ensure all saved tensors are contiguous
            ctx.save_for_backward(
                queries_flat.contiguous(),
                sub_keys_cat.contiguous(),
                dim_sizes.contiguous(),
                dim_offsets.contiguous(),
                output_indices.contiguous() # Save flat indices
            )
            ctx.norm_query = norm_query
            ctx.norm_keys = norm_keys
            ctx.num_dims = num_dims
            ctx.sub_query_dim = sub_query_dim
            ctx.original_query_shape = queries.shape
            ctx.top_k = top_k
            ctx.num_experts = num_experts

            return output_indices_reshaped, output_scores_reshaped

        @staticmethod
        def backward(ctx, grad_indices, grad_scores):
            # grad_indices is usually None for top_k
            if grad_scores is None:
                return None, None, None, None, None, None # Match forward inputs

            queries_flat, sub_keys_cat, dim_sizes, dim_offsets, output_indices = ctx.saved_tensors
            norm_query = ctx.norm_query
            norm_keys = ctx.norm_keys
            num_dims = ctx.num_dims
            sub_query_dim = ctx.sub_query_dim
            original_query_shape = ctx.original_query_shape
            top_k = ctx.top_k
            num_experts = ctx.num_experts
            num_tokens, query_dim = queries_flat.shape
            device = queries_flat.device

            # Flatten grad_scores
            grad_scores_flat = grad_scores.reshape(num_tokens, top_k).contiguous()

            # Allocate gradient tensors for inputs (matching input dtypes)
            grad_queries = torch.zeros_like(queries_flat, dtype=queries_flat.dtype)
            grad_sub_keys_cat = torch.zeros_like(sub_keys_cat, dtype=sub_keys_cat.dtype)

            grid = lambda META: (triton.cdiv(num_tokens, META['BLOCK_SIZE_T']),)
            # Ensure inputs passed to kernel match expected dtypes
            queries_kernel = queries_flat.to(queries_flat.dtype)
            sub_keys_kernel = sub_keys_cat.to(sub_keys_cat.dtype)
            grad_scores_kernel = grad_scores_flat.to(grad_scores_flat.dtype)

            # Launch Backward Kernel
            peer_selection_bwd_kernel[grid](
                queries_kernel, sub_keys_kernel, output_indices, grad_scores_kernel,
                dim_sizes, dim_offsets,
                grad_queries, grad_sub_keys_cat, # Output grads match input dtypes
                num_tokens, query_dim, num_experts, top_k,
                num_dims, sub_query_dim,
                queries_flat.stride(0), queries_flat.stride(1),
                sub_keys_cat.stride(0), sub_keys_cat.stride(1),
                output_indices.stride(0), output_indices.stride(1),
                grad_scores_flat.stride(0), grad_scores_flat.stride(1),
                grad_queries.stride(0), grad_queries.stride(1),
                grad_sub_keys_cat.stride(0), grad_sub_keys_cat.stride(1),
                dim_sizes.stride(0), dim_offsets.stride(0),
                norm_query, norm_keys,
                # BLOCK_SIZE_T=64 # Use same block size as forward? Autotuner might handle this.
            )

            # Reshape grad_queries
            grad_queries_reshaped = grad_queries.view(original_query_shape)

            # Split grad_sub_keys_cat back into a list
            grad_sub_keys_list = []
            current_offset = 0
            dim_sizes_list = dim_sizes.tolist()
            for dim_size in dim_sizes_list:
                # Select the correct slice from the concatenated gradient tensor
                grad_slice = grad_sub_keys_cat[current_offset : current_offset + dim_size]
                grad_sub_keys_list.append(grad_slice)
                current_offset += dim_size

            # Return gradients matching forward inputs
            return (
                grad_queries_reshaped, # grad w.r.t queries
                grad_sub_keys_list,    # grad w.r.t sub_keys_list
                None,                  # grad w.r.t product_key_dim
                None,                  # grad w.r.t top_k
                None,                  # grad w.r.t norm_query
                None                   # grad w.r.t norm_keys
            )

    # Define a convenience function
    def peer_selection_triton(queries, sub_keys_list, product_key_dim, top_k, norm_query, norm_keys):
        # Ensure sub_keys_list contains tensors, not parameters, for autograd function
        sub_key_tensors = [p.data if isinstance(p, torch.nn.Parameter) else p for p in sub_keys_list]
        return PEERSelectionFunction.apply(queries, sub_key_tensors, product_key_dim, top_k, norm_query, norm_keys)

else: # If Triton is not available
    peer_selection_triton = None # Define the function as None

# Remove old functions if they exist
# compute_dimension_scores = None
# triton_matmul = None
# matmul_kernel = None
# gelu = None
