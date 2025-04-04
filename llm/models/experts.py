#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parameter Efficient Expert Retrieval (PEER) module implementation.

PEER uses product keys to efficiently select experts from a large pool.
This implementation allows for variable dimensionality in the cartesian product,
and supports experts with configurable hidden sizes.
"""

import itertools
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PEER(nn.Module):
    """
    Parameter Efficient Expert Retrieval (PEER) module.
    
    PEER uses product keys to efficiently select experts from a large pool.
    This implementation allows for variable dimensionality in the cartesian product,
    and supports experts with configurable hidden sizes.
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
        
        # Verify that the product of the dimensions equals the number of experts
        product_size = 1
        for dim in product_key_dim:
            product_size *= dim
        assert product_size == num_experts, (
            f"Product of dimensions {product_key_dim} = {product_size} must equal "
            f"the number of experts {num_experts}"
        )
        
        # Create the query network
        self.query_proj = nn.Linear(input_dim, num_heads * query_dim)
        
        if batch_norm_query:
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
        # Queries shape: [batch, heads, query_dim]
        chunks = []
        chunk_size = self.query_dim // len(self.product_key_dim)
        
        for i in range(len(self.product_key_dim)):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size
            chunks.append(queries[..., start_idx:end_idx])
        
        return chunks
    
    def get_indices(self, queries, top_k):
        """
        Retrieve the top-k experts using product keys.
        
        Args:
            queries: Tensor of shape [batch_size, seq_len, num_heads, query_dim]
            top_k: Number of experts to retrieve per head
            
        Returns:
            indices: Tensor of shape [batch_size, seq_len, num_heads, top_k] containing expert indices
            scores: Tensor of shape [batch_size, seq_len, num_heads, top_k] containing expert scores
        """
        batch_size, seq_len, num_heads, _ = queries.shape
        device = queries.device
        
        # Split queries along the feature dimension for each product key dimension
        query_chunks = self.split_queries(queries)
        
        # Compute scores for each dimension with its corresponding sub-keys
        dim_scores = []
        dim_indices = []
        
        for i, (q_chunk, sub_keys) in enumerate(zip(query_chunks, self.sub_keys)):
            # q_chunk: [batch_size, seq_len, num_heads, sub_query_dim]
            # sub_keys: [dim_size, sub_query_dim]
            
            # Normalize if requested
            if self.norm_keys:
                sub_keys = F.normalize(sub_keys, dim=-1)
            if self.norm_query:
                q_chunk = F.normalize(q_chunk, dim=-1)
            
            # Compute scores for this dimension
            # [batch_size, seq_len, num_heads, dim_size]
            scores = torch.matmul(q_chunk, sub_keys.t())
            
            # Get top-k for this dimension
            dim_top_k = min(self.product_key_dim[i], int(np.ceil(top_k ** (1 / len(self.product_key_dim)))))
            dim_top_scores, dim_top_indices = torch.topk(scores, k=dim_top_k, dim=-1)
            
            dim_scores.append(dim_top_scores)
            dim_indices.append(dim_top_indices)
        
        # Build the Cartesian product of top indices from each dimension
        # This gives us candidate experts to consider
        all_indices = []
        all_scores = []
        
        # Convert flattened indices to n-dimensional indices
        indices_ranges = [range(dim_size) for dim_size in self.product_key_dim]
        
        # For each item in the batch and each head
        for b in range(batch_size):
            for s in range(seq_len):
                for h in range(num_heads):
                    # Get the top indices for each dimension for this batch item and head
                    selected_indices = [indices[b, s, h].tolist() for indices in dim_indices]
                    
                    # Build Cartesian product of selected indices across dimensions
                    candidate_tuples = []
                    for idx_tuple in itertools.product(*selected_indices):
                        # Convert the n-dimensional index tuple to a flat index
                        flat_idx = 0
                        multiplier = 1
                        for i, idx in enumerate(reversed(idx_tuple)):
                            flat_idx += idx * multiplier
                            if i < len(self.product_key_dim) - 1:
                                multiplier *= self.product_key_dim[-(i+1)]
                        
                        # Compute the total score for this expert as the sum of individual dimension scores
                        total_score = sum(dim_scores[d][b, s, h, selected_indices[d].index(idx_tuple[d])]
                                        for d in range(len(self.product_key_dim)))
                        
                        candidate_tuples.append((flat_idx, total_score.item()))
                    
                    # Sort by total score and get the top_k
                    candidate_tuples.sort(key=lambda x: x[1], reverse=True)
                    top_experts = candidate_tuples[:top_k]
                    
                    # Extract indices and scores
                    expert_indices = [t[0] for t in top_experts]
                    expert_scores = [t[1] for t in top_experts]
                    
                    # Pad if necessary
                    if len(expert_indices) < top_k:
                        expert_indices.extend([0] * (top_k - len(expert_indices)))
                        expert_scores.extend([-float('inf')] * (top_k - len(expert_scores)))
                    
                    all_indices.append(expert_indices)
                    all_scores.append(expert_scores)
        
        # Reshape to [batch_size, seq_len, num_heads, top_k]
        indices = torch.tensor(all_indices, device=device).view(batch_size, seq_len, num_heads, top_k)
        scores = torch.tensor(all_scores, device=device).view(batch_size, seq_len, num_heads, top_k)
        
        return indices, scores

    def forward(self, hidden_states):
        """
        Forward pass for the PEER module.
        
        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            output: Tensor of shape [batch_size, seq_len, output_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        # Project input to query space
        queries = self.query_proj(hidden_states)
        queries = queries.view(batch_size, seq_len, self.num_heads, self.query_dim)
        
        # Apply batch norm to queries if enabled
        if self.batch_norm_query:
            # Reshape for batch norm
            orig_shape = queries.shape
            queries = queries.view(-1, self.query_dim)
            queries = self.query_batch_norm(queries)
            queries = queries.view(*orig_shape)
        
        # Get expert indices and scores
        indices, scores = self.get_indices(queries, self.num_experts_per_tok)
        
        # Normalize scores with softmax
        scores = F.softmax(scores, dim=-1)
        
        # Apply expert networks
        # Get expert weights from embeddings
        down_weights = self.expert_down(indices)  # [batch, seq, heads, top_k, input_dim * expert_hidden]
        up_weights = self.expert_up(indices)  # [batch, seq, heads, top_k, output_dim * expert_hidden]
        
        # Reshape for matrix multiplication
        down_weights = down_weights.view(
            batch_size, seq_len, self.num_heads, self.num_experts_per_tok, 
            self.expert_hidden_size, self.input_dim
        )
        up_weights = up_weights.view(
            batch_size, seq_len, self.num_heads, self.num_experts_per_tok, 
            self.output_dim, self.expert_hidden_size
        )
        
        # Expand hidden states for processing with experts
        # [batch, seq, 1, 1, input_dim]
        hidden_expanded = hidden_states.unsqueeze(2).unsqueeze(3)
        
        # Apply down projection for each expert
        # [batch, seq, heads, top_k, expert_hidden]
        expert_inputs = torch.matmul(down_weights, hidden_expanded.unsqueeze(-1)).squeeze(-1)
        
        # Apply activation
        expert_outputs = self.activation(expert_inputs)
        expert_outputs = self.dropout(expert_outputs)
        
        # Apply up projection for each expert
        # [batch, seq, heads, top_k, output_dim]
        expert_outputs = torch.matmul(up_weights, expert_outputs.unsqueeze(-1)).squeeze(-1)
        
        # Apply scores to weight expert outputs
        # [batch, seq, heads, top_k, output_dim]
        scored_outputs = expert_outputs * scores.unsqueeze(-1)
        
        # Sum over experts and heads
        # Sum over top_k: [batch, seq, heads, output_dim]
        outputs = scored_outputs.sum(dim=3)
        # Sum over heads: [batch, seq, output_dim]
        outputs = outputs.sum(dim=2)
        
        return outputs