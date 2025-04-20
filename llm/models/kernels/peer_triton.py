#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Triton kernel stubs for PEER operations.
"""

import torch
from typing import List, Dict, Any, Optional

# Placeholder for Triton import
# import triton
# import triton.language as tl


def peer_fwd_kernel(
    hidden_states: torch.Tensor,
    query_proj_weight: torch.Tensor,
    sub_keys: List[torch.Tensor], # List of sub-key tensors
    expert_down_weight: torch.Tensor, # From embedding
    expert_up_weight: torch.Tensor, # From embedding
    peer_config: Dict[str, Any], # Dictionary containing PEER config
    # Add other necessary parameters like norm weights if needed
    query_norm_weight: Optional[torch.Tensor] = None,
    query_norm_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Stub for the PEER forward pass Triton kernel.

    This function currently returns zeros with the expected output shape.
    The actual Triton implementation will replace this.

    Args:
        hidden_states: Input tensor [batch, seq_len, input_dim]
        query_proj_weight: Weight for query projection
        sub_keys: List of sub-key tensors for each dimension
        expert_down_weight: Embedding weights for expert down projection
        expert_up_weight: Embedding weights for expert up projection
        peer_config: Dictionary containing PEER configuration parameters
                     (e.g., num_experts_per_tok, num_heads, query_dim, etc.)
        query_norm_weight: Weight for query LayerNorm (if used)
        query_norm_bias: Bias for query LayerNorm (if used)

    Returns:
        Output tensor [batch, seq_len, output_dim] filled with zeros.
    """
    print("--- WARNING: Using STUBBED PEER Triton Kernel ---")
    batch_size, seq_len, _ = hidden_states.shape
    output_dim = peer_config.get("output_dim", hidden_states.shape[-1]) # Get output dim from config

    # Calculate expected output shape
    output_shape = (batch_size, seq_len, output_dim)

    # Return zeros with the correct shape, dtype, and device
    return torch.zeros(
        output_shape, dtype=hidden_states.dtype, device=hidden_states.device
    )

# Placeholder for backward kernel
# def peer_bwd_kernel(...):
#     pass
