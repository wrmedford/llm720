#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parameter Efficient Expert Retrieval (PEER) module implementation.

PEER uses product keys to efficiently select experts from a large pool.
This implementation allows for variable dimensionality in the cartesian
product, and supports experts with configurable hidden sizes.
"""

from typing import List

import os
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

# Check environment variable to decide which kernel to use
_USE_TRITON_KERNEL = os.environ.get("USE_TRITON_KERNEL", "0") == "1"
_USE_CUTLASS_KERNEL = os.environ.get("USE_CUTLASS_KERNEL", "0") == "1"

if _USE_CUTLASS_KERNEL:
    try:
        # Import the CUTLASS kernel
        from .kernels.peer_cutlass import peer_forward_cutlass
        print("Using PEER CUTLASS kernel for optimized performance.")
    except ImportError as e:
        print(f"Warning: USE_CUTLASS_KERNEL is set, but CUTLASS kernel import failed: {e}. Falling back to PyTorch.")
        _USE_CUTLASS_KERNEL = False
elif _USE_TRITON_KERNEL:
    try:
        # Import the (stubbed) Triton kernel
        from .kernels.peer_triton import peer_fwd_kernel
        print("Attempting to use PEER Triton kernel (currently stubbed).")
    except ImportError:
        print("Warning: USE_TRITON_KERNEL is set, but Triton kernel import failed. Falling back to PyTorch.")
        _USE_TRITON_KERNEL = False


class PEER(nn.Module):
    """
    Parameter Efficient Expert Retrieval (PEER) module.

    PEER uses product keys to efficiently select experts from a large pool.
    This implementation allows for variable dimensionality in the cartesian
    product, and supports experts with configurable hidden sizes.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_experts: int,
        num_experts_per_tok: int = 16,
        num_heads: int = 8,
        expert_hidden_size: int = 1,  # Single neuron experts by default
        query_dim: int = 256,
        product_key_dim: List[int] = [32, 32],  # Cartesian product dimensions
        activation: str = "gelu",
        dropout: float = 0.0,
        norm_keys: bool = True,
        norm_query: bool = True,
        batch_norm_query: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_heads = num_heads
        self.expert_hidden_size = expert_hidden_size
        self.query_dim = query_dim
        self.product_key_dim = product_key_dim
        self.norm_keys = norm_keys
        self.norm_query = norm_query
        self.batch_norm_query = batch_norm_query

        # Verify that the product of the dimensions equals the number of
        # experts
        product_size = 1
        for dim in product_key_dim:
            product_size *= dim
        assert product_size == num_experts, (
            f"Product of dimensions {product_key_dim} = {product_size} must "
            f"equal the number of experts {num_experts}"
        )

        # Create the query network
        self.query_proj = nn.Linear(input_dim, num_heads * query_dim)

        if batch_norm_query:
            # Use LayerNorm instead of BatchNorm for better torch.compile compatibility
            self.query_layer_norm = nn.LayerNorm(query_dim)
        else:
            self.query_layer_norm = None # Explicitly set to None if not used

        # Create sub-key embeddings for each dimension
        self.sub_keys = nn.ParameterList()
        for i, dim_size in enumerate(product_key_dim):
            # Each dimension gets its own set of keys
            sub_key_dim = query_dim // len(product_key_dim)
            keys = nn.Parameter(torch.FloatTensor(dim_size, sub_key_dim))
            nn.init.normal_(keys, mean=0, std=0.02)
            self.sub_keys.append(keys)

        # Expert networks - each expert has a down and up projection
        # Down projection: input_dim -> expert_hidden_size
        # Up projection: expert_hidden_size -> output_dim
        self.expert_down = nn.Embedding(num_experts, input_dim * expert_hidden_size)
        self.expert_up = nn.Embedding(num_experts, output_dim * expert_hidden_size)

        # Set activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.dropout = nn.Dropout(dropout)

        # Placeholder for storing last indices/scores if needed by hooks/tracking
        self._last_expert_indices: Optional[torch.Tensor] = None
        self._last_expert_scores: Optional[torch.Tensor] = None


    def split_queries(self, queries: torch.Tensor) -> List[torch.Tensor]:
        """Split queries for multi-dim product keys."""
        # Queries shape: [batch, seq, heads, query_dim]
        chunks = []
        chunk_size = self.query_dim // len(self.product_key_dim)

        for i in range(len(self.product_key_dim)):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size
            chunks.append(queries[..., start_idx:end_idx])

        return chunks

    def _get_expert_indices_pytorch(self, queries, top_k):
        """
        PyTorch implementation for retrieving top-k experts using product keys.
        Uses an optimized approach to avoid computing all N expert scores.
        """
        batch_size, seq_len, num_heads, query_dim = queries.shape
        device = queries.device
        num_dims = len(self.product_key_dim)
        sub_query_dim = query_dim // num_dims

        # Split queries along the feature dimension
        query_chunks = self.split_queries(queries) # List of [B, S, H, sub_query_dim]

        # Compute scores for each dimension
        all_dim_scores = []
        for i, (q_chunk, sub_keys) in enumerate(zip(query_chunks, self.sub_keys)):
            current_sub_keys = sub_keys.data # Use .data to get the tensor
            # Normalize if requested
            if self.norm_keys:
                current_sub_keys = F.normalize(current_sub_keys, dim=-1)
            if self.norm_query:
                q_chunk = F.normalize(q_chunk, dim=-1)

            # Compute scores: [B, S, H, dim_size_i]
            dim_scores = torch.matmul(q_chunk, current_sub_keys.t())
            all_dim_scores.append(dim_scores)

        # --- Optimized Top-k Combination ---
        # Determine k_prime (number of candidates per dimension)
        # Use a value slightly larger than k^(1/N) for robustness
        k_prime_base = int(np.ceil(top_k**(1.0 / num_dims))) + 2 if num_dims > 0 else top_k

        top_scores_per_dim = []
        top_indices_per_dim = []
        actual_k_primes = [] # Store the actual k' used for each dim

        # Find Top-k' per Dimension
        for i, dim_scores in enumerate(all_dim_scores):
            dim_size = self.product_key_dim[i]
            if dim_size <= 0:
                 raise ValueError(f"Dimension {i} has size {dim_size}, which is invalid.")

            # Ensure k_prime does not exceed dimension size
            k_prime_i = min(k_prime_base, dim_size)
            # Ensure k_prime_i is at least top_k if dim_size allows, otherwise dim_size
            # This heuristic helps ensure final top_k are likely within candidates
            k_prime_i = min(max(k_prime_i, top_k if dim_size >= top_k else dim_size), dim_size)

            # Ensure k_prime_i is at least 1
            k_prime_i = max(1, k_prime_i)

            scores_i, indices_i = torch.topk(dim_scores, k=k_prime_i, dim=-1)
            top_scores_per_dim.append(scores_i)
            top_indices_per_dim.append(indices_i)
            actual_k_primes.append(k_prime_i) # Store actual k' used

        # Combine Top Candidates Iteratively
        if num_dims == 0:
             raise ValueError("PEER module has no product key dimensions.")
        elif num_dims == 1:
            # If only one dimension, the result is simply the top-k from that dimension
            actual_top_k = min(top_k, actual_k_primes[0])
            final_scores, final_indices = torch.topk(top_scores_per_dim[0], k=actual_top_k, dim=-1)
        else:
            # Initialize with the first dimension's candidates
            combined_scores = top_scores_per_dim[0]  # [B, S, H, k'_0]
            combined_indices = top_indices_per_dim[0] # [B, S, H, k'_0]

            # Calculate dimension multipliers for index calculation
            dim_multipliers = [1] * num_dims
            for d in range(num_dims - 2, -1, -1):
                dim_multipliers[d] = dim_multipliers[d+1] * self.product_key_dim[d+1]
            dim_multipliers_tensor = torch.tensor(dim_multipliers, device=device, dtype=torch.long)


            # Iteratively combine with subsequent dimensions
            for d in range(1, num_dims):
                scores_d = top_scores_per_dim[d]    # [B, S, H, k'_d]
                indices_d = top_indices_per_dim[d]  # [B, S, H, k'_d]
                k_prime_prev = combined_indices.shape[-1]
                k_prime_d = actual_k_primes[d] # Use actual k'

                # Expand scores and indices for broadcasting
                # combined_scores: [B, S, H, k_prime_prev] -> [B, S, H, k_prime_prev, 1]
                # scores_d:        [B, S, H, k_prime_d]    -> [B, S, H, 1, k_prime_d]
                current_combined_scores = combined_scores.unsqueeze(-1) + scores_d.unsqueeze(-2)
                # -> [B, S, H, k_prime_prev, k_prime_d]

                # combined_indices: [B, S, H, k_prime_prev] -> [B, S, H, k_prime_prev, 1]
                # indices_d:        [B, S, H, k_prime_d]    -> [B, S, H, 1, k_prime_d]
                # Use precalculated multipliers for correct global index
                # The combined_indices already hold the partial global index up to dim d-1
                current_combined_indices = combined_indices.unsqueeze(-1) + indices_d.unsqueeze(-2) * dim_multipliers_tensor[d]
                # -> [B, S, H, k_prime_prev, k_prime_d]

                # Flatten the candidate dimensions
                flat_scores = current_combined_scores.view(batch_size, seq_len, num_heads, -1)
                flat_indices = current_combined_indices.view(batch_size, seq_len, num_heads, -1)
                num_candidates = flat_scores.shape[-1]

                # Determine how many candidates to keep for the next round or final selection
                if d < num_dims - 1:
                    # Pruning heuristic: Keep more candidates than final top_k
                    # e.g., k_prime_base for the *next* dimension's combination
                    # Or simply a multiple of top_k
                    k_intermediate = min(top_k * 4, num_candidates) # Keep up to top_k * 4
                    k_intermediate = max(1, k_intermediate) # Ensure at least 1
                else:
                    # Last step: Select the final top_k
                    k_intermediate = min(top_k, num_candidates)
                    k_intermediate = max(1, k_intermediate) # Ensure at least 1


                # Prune or select final candidates
                if k_intermediate == num_candidates:
                    # If keeping all candidates, just use the flattened tensors
                    combined_scores = flat_scores
                    combined_indices = flat_indices
                else:
                    # Use topk to select the best k_intermediate candidates
                    combined_scores, topk_indices_flat = torch.topk(flat_scores, k=k_intermediate, dim=-1)
                    combined_indices = torch.gather(flat_indices, dim=-1, index=topk_indices_flat)


            # After the loop, combined_scores/indices hold the final result
            final_scores = combined_scores
            final_indices = combined_indices

            # Ensure the final number of experts matches actual_top_k if pruning occurred
            actual_top_k = min(top_k, final_indices.shape[-1])
            if final_indices.shape[-1] > actual_top_k:
                 final_scores = final_scores[..., :actual_top_k]
                 final_indices = final_indices[..., :actual_top_k]


        # Ensure final_indices and final_scores are defined and have correct shape
        if 'final_indices' not in locals() or 'final_scores' not in locals():
             raise RuntimeError("Failed to compute final expert indices and scores.")
        # Check if the last dimension is 0 when top_k > 0
        if final_indices.shape[-1] == 0 and top_k > 0:
             # This can happen if k_prime becomes 0, e.g., dim_size is 0
             # Or if top_k requested is 0
             # Let's return empty tensors in this specific case, assuming top_k=0 was intended or dim_size=0
             # If top_k > 0 and dim_size > 0, this indicates an issue elsewhere.
             # For safety, return tensors with the last dimension as 0.
             final_scores = torch.empty_like(final_indices).float() # Match shape but size 0

        return final_indices, final_scores


    def _forward_pytorch(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using the PyTorch implementation.

        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, input_dim]

        Returns:
            output: Tensor of shape [batch_size, seq_len, output_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape
        # device = hidden_states.device # Unused

        # Project input to query space
        queries = self.query_proj(hidden_states)
        queries = queries.view(batch_size, seq_len, self.num_heads, self.query_dim)

        # Apply layer norm to queries if enabled
        if self.batch_norm_query and self.query_layer_norm is not None:
            # LayerNorm operates on the last dimension(s), compatible with (B, S, H, C)
            queries = self.query_layer_norm(queries)
        elif self.batch_norm_query and self.query_layer_norm is None:
            # This case should not happen if __init__ logic is correct, but log a warning
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("batch_norm_query is True but query_layer_norm was not initialized.")


        # Use the optimized PyTorch implementation directly
        indices, scores = self._get_expert_indices_pytorch(
            queries, self.num_experts_per_tok
        )

        # Store indices and scores for tracking/debugging if needed
        self._last_expert_indices = indices
        self._last_expert_scores = scores

        # Normalize scores with softmax
        # scores shape: [B, S, H, top_k]
        scores = F.softmax(scores, dim=-1)

        # Apply expert networks
        # Get expert weights from embeddings
        # [batch, seq, heads, top_k, input_dim * expert_hidden]
        down_weights = self.expert_down(indices)
        # [batch, seq, heads, top_k, output_dim * expert_hidden]
        up_weights = self.expert_up(indices)

        # Reshape for matrix multiplication
        down_weights = down_weights.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.num_experts_per_tok,
            self.expert_hidden_size,
            self.input_dim,
        )
        up_weights = up_weights.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.num_experts_per_tok,
            self.output_dim,
            self.expert_hidden_size,
        )

        # Expand hidden states for processing with experts
        # [batch, seq, 1, 1, input_dim]
        hidden_expanded = hidden_states.unsqueeze(2).unsqueeze(3)

        # Apply down projection for each expert
        # [batch, seq, heads, top_k, expert_hidden]
        expert_inputs = torch.matmul(
            down_weights, hidden_expanded.unsqueeze(-1)
        ).squeeze(-1)

        # Apply activation
        expert_outputs = self.activation(expert_inputs)
        expert_outputs = self.dropout(expert_outputs)

        # Apply up projection for each expert
        # [batch, seq, heads, top_k, output_dim]
        expert_outputs = torch.matmul(up_weights, expert_outputs.unsqueeze(-1)).squeeze(
            -1
        )

        # Apply scores to weight expert outputs
        # [batch, seq, heads, top_k, output_dim]
        scored_outputs = expert_outputs * scores.unsqueeze(-1)

        # Sum over experts and heads
        # Sum over top_k: [batch, seq, heads, output_dim]
        outputs = scored_outputs.sum(dim=3)
        # Sum over heads: [batch, seq, output_dim]
        outputs = outputs.sum(dim=2)

        return outputs

    def _forward_triton(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using the (stubbed) Triton kernel.
        """
        # Prepare arguments for the kernel
        # Note: Weights need to be accessed directly (e.g., self.query_proj.weight)
        # Ensure sub_keys are passed as a list of tensors
        sub_key_tensors = [p for p in self.sub_keys]

        # Extract necessary config parameters
        peer_config_dict = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "num_heads": self.num_heads,
            "expert_hidden_size": self.expert_hidden_size,
            "query_dim": self.query_dim,
            "product_key_dim": self.product_key_dim,
            "norm_keys": self.norm_keys,
            "norm_query": self.norm_query,
            "batch_norm_query": self.batch_norm_query,
            # Add activation if needed by kernel logic later
        }

        # Get LayerNorm weights if applicable
        query_norm_weight = self.query_layer_norm.weight if self.query_layer_norm else None
        query_norm_bias = self.query_layer_norm.bias if self.query_layer_norm else None

        # Call the stubbed kernel
        output = peer_fwd_kernel(
            hidden_states=hidden_states,
            query_proj_weight=self.query_proj.weight,
            sub_keys=sub_key_tensors,
            expert_down_weight=self.expert_down.weight,
            expert_up_weight=self.expert_up.weight,
            peer_config=peer_config_dict,
            query_norm_weight=query_norm_weight,
            query_norm_bias=query_norm_bias,
        )
        return output

    def _forward_cutlass(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using the CUTLASS kernel.
        """
        # Extract sub-key tensors
        sub_key_1 = self.sub_keys[0]
        sub_key_2 = self.sub_keys[1] if len(self.sub_keys) > 1 else self.sub_keys[0]
        
        # Get query bias if it exists
        query_bias = self.query_proj.bias if hasattr(self.query_proj, 'bias') and self.query_proj.bias is not None else None
        
        # Get LayerNorm weights if applicable
        ln_weight = self.query_layer_norm.weight if self.query_layer_norm else None
        ln_bias = self.query_layer_norm.bias if self.query_layer_norm else None
        
        # Call the CUTLASS kernel
        output = peer_forward_cutlass(
            x=hidden_states,
            query_weight=self.query_proj.weight,
            query_bias=query_bias,
            key_weight_1=sub_key_1,
            key_weight_2=sub_key_2,
            expert_weights_u=self.expert_down.weight,
            expert_weights_v=self.expert_up.weight,
            num_heads=self.num_heads,
            num_experts=self.num_experts,
            expert_hidden_size=self.expert_hidden_size,
            top_k=self.num_experts_per_tok,
            dropout_rate=self.dropout.p,
            layer_norm=self.batch_norm_query,
            ln_weight=ln_weight,
            ln_bias=ln_bias,
        )
        return output

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Main forward pass for the PEER module.

        Dispatches to either the PyTorch implementation, CUTLASS kernel, or Triton kernel
        based on the environment flags.

        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, input_dim]

        Returns:
            output: Tensor of shape [batch_size, seq_len, output_dim]
        """
        if _USE_CUTLASS_KERNEL:
            # Call the CUTLASS kernel implementation
            return self._forward_cutlass(hidden_states)
        elif _USE_TRITON_KERNEL:
            # Call the Triton kernel implementation
            # Note: Currently raises NotImplementedError
            return self._forward_triton(hidden_states)
        else:
            # Call the PyTorch implementation
            return self._forward_pytorch(hidden_states)
