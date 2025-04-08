#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parameter Efficient Expert Retrieval (PEER) module implementation.

PEER uses product keys to efficiently select experts from a large pool.
This implementation allows for variable dimensionality in the cartesian
product, and supports experts with configurable hidden sizes.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te  # Import Transformer Engine

from llm.ops.triton_peer_kernels import HAS_TRITON, peer_selection_triton


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

        # Create the query network (replace with TE Linear)
        self.query_proj = te.Linear(input_dim, num_heads * query_dim)

        if batch_norm_query:
            # Keep standard BatchNorm for now, TE doesn't have a direct
            # replacement. Ensure it runs in FP32 (handled below in forward)
            self.query_batch_norm = nn.BatchNorm1d(query_dim)

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

    def split_queries(self, queries):
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
        Used as a fallback if Triton is not available or for comparison.
        """
        batch_size, seq_len, num_heads, _ = queries.shape
        device = queries.device
        num_dims = len(self.product_key_dim)

        # Split queries along the feature dimension
        # List of [B, S, H, sub_query_dim]
        query_chunks = self.split_queries(queries)

        # Compute scores for each dimension
        all_dim_scores = []
        for i, (q_chunk, sub_keys) in enumerate(zip(query_chunks, self.sub_keys)):
            # Normalize if requested
            if self.norm_keys:
                sub_keys = F.normalize(sub_keys, dim=-1)
            if self.norm_query:
                q_chunk = F.normalize(q_chunk, dim=-1)

            # Compute scores: [B, S, H, dim_size_i]
            dim_scores = torch.matmul(q_chunk, sub_keys.t())
            all_dim_scores.append(dim_scores)

        # Combine scores using broadcasting and summation
        # Create meshgrid of indices for each dimension
        dim_indices = [torch.arange(d, device=device) for d in self.product_key_dim]
        # Shape [d1, d2, ..., dn, n_dims]
        mesh_indices = torch.stack(torch.meshgrid(*dim_indices, indexing="ij"), dim=-1)

        # Flatten meshgrid indices to [num_experts, n_dims]
        # Shape [num_experts, n_dims]
        flat_mesh_indices = mesh_indices.view(-1, num_dims)

        # Gather scores corresponding to each expert index tuple
        # Expand all_dim_scores for gathering:
        # List of [B, S, H, 1, dim_size_i]
        # expanded_scores = [ds.unsqueeze(-2) for ds in all_dim_scores] # Unused # noqa E501

        # Gather scores for each dimension based on the flat_mesh_indices
        # flat_mesh_indices[:, i] gives the index for the i-th dimension
        # for all experts
        gathered_scores = []
        for dim_idx in range(num_dims):
            # Indices for this dimension: [num_experts]
            # Shape [1, 1, 1, num_experts]
            indices_for_dim = (
                flat_mesh_indices[:, dim_idx].unsqueeze(0).unsqueeze(0).unsqueeze(0)
            )
            # Shape [B, S, H, num_experts]
            indices_for_dim = indices_for_dim.expand(batch_size, seq_len, num_heads, -1)

            # Scores for this dimension: [B, S, H, dim_size_i]
            scores_for_dim = all_dim_scores[dim_idx]

            # Gather: [B, S, H, num_experts]
            gathered = torch.gather(scores_for_dim, dim=-1, index=indices_for_dim)
            gathered_scores.append(gathered)

        # Sum scores across dimensions: [B, S, H, num_experts]
        total_scores = torch.stack(gathered_scores, dim=0).sum(dim=0)

        # Get top-k experts based on total scores
        top_scores, top_indices = torch.topk(total_scores, k=top_k, dim=-1)

        return top_indices, top_scores

    def forward(self, hidden_states):
        """
        Forward pass for the PEER module.

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

        # Apply batch norm to queries if enabled (ensure FP32 execution)
        if self.batch_norm_query:
            # Reshape for batch norm
            orig_shape = queries.shape
            input_dtype = queries.dtype
            queries_flat = queries.view(-1, self.query_dim)
            # Cast input to BatchNorm1d to FP32
            queries_fp32 = queries_flat.to(torch.float32)
            queries_normed_fp32 = self.query_batch_norm(queries_fp32)
            # Cast output back to original dtype
            queries = queries_normed_fp32.to(input_dtype).view(*orig_shape)

        # Get expert indices and scores using the appropriate method
        if HAS_TRITON and peer_selection_triton is not None:
            try:
                # Use the fused Triton kernel via the autograd function
                indices, scores = peer_selection_triton(
                    queries,
                    self.sub_keys,  # Pass the list of Parameter tensors
                    self.product_key_dim,
                    self.num_experts_per_tok,
                    self.norm_query,
                    self.norm_keys,
                )
                # print("Using Triton PEER selection kernel.") # Debug print
            except Exception as e:
                print(
                    f"Triton PEER selection failed: {e}. " f"Falling back to PyTorch."
                )
                # Fallback to PyTorch implementation if Triton fails
                indices, scores = self._get_expert_indices_pytorch(
                    queries, self.num_experts_per_tok
                )
        else:
            # Use PyTorch implementation if Triton is not available or kernel
            # is None
            indices, scores = self._get_expert_indices_pytorch(
                queries, self.num_experts_per_tok
            )
            # print("Using PyTorch PEER selection.") # Debug print

        # Store indices for tracking hook if needed
        self._last_expert_indices = indices

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
