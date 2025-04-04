#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-Headed Latent Attention (MLA) implementation.

As implemented in DeepSeek models, with:
- Separate projection for query, key, and value
- Query and key decomposed into rope and nope parts
- Support for different head dimensions for query and value
"""

import math
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_mla import flash_mla_with_kvcache, get_mla_metadata # Import get_mla_metadata
from llm.models.foundation import update_paged_kv_cache # Import placeholder update function


class MultiHeadedLatentAttention(nn.Module):
    """
    Multi-Headed Latent Attention module as implemented in DeepSeek models.
    This implementation follows the DeepSeek V3 specifications with:
    - Separate projection for query, key, and value
    - Query and key decomposed into rope and nope parts
    - Support for different head dimensions for query and value
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        q_lora_rank: int = None,
        kv_lora_rank: int = 512,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        qk_nope_head_dim: int = 128,
        dropout: float = 0.0,
        layer_idx: int = None, # Add layer_idx
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx # Store layer_idx
        self.num_heads = num_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.q_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.dropout = dropout
        
        # Initialize query projections
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(hidden_size, num_heads * self.q_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
            self.q_a_layernorm = nn.LayerNorm(q_lora_rank, eps=1e-6)
            self.q_b_proj = nn.Linear(q_lora_rank, num_heads * self.q_head_dim, bias=False)
        
        # Key-value projections
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden_size, 
            kv_lora_rank + qk_rope_head_dim, 
            bias=False
        )
        self.kv_a_layernorm = nn.LayerNorm(kv_lora_rank, eps=1e-6)
        self.kv_b_proj = nn.Linear(
            kv_lora_rank,
            num_heads * (self.q_head_dim - qk_rope_head_dim + v_head_dim),
            bias=False
        )
        
        # Output projection
        self.o_proj = nn.Linear(num_heads * v_head_dim, hidden_size, bias=False)
        
        # Scaling factor is handled internally by flash_mla_with_kvcache
    
    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def _apply_rotary_pos_emb(self, q, k, cos, sin, position_ids):
        """Applies rotary position embedding to q and k tensors."""
        # Apply RoPE logic similar to DeepSeek implementation
        cos = cos[position_ids].unsqueeze(1)  # [batch, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [batch, 1, seq_len, dim]
        
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_embed, k_embed
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        # KV cache arguments replace past_key_value, attention_mask is handled by causal flag or cache_seqlens
        kv_cache: Optional[torch.Tensor] = None, # Shape: [num_blocks, 2, num_heads, block_size, head_dim]
        block_table: Optional[torch.Tensor] = None, # Shape: [bsz, max_seq_len // block_size]
        cache_seqlens: Optional[torch.Tensor] = None, # Shape: [bsz]
        output_attentions: bool = False, # Note: FlashMLA does not return attention weights
        # use_cache is implicit when kv_cache is provided
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass using FlashMLA.
        
        Args:
            hidden_states: Input tensor.
            position_ids: Position IDs for RoPE.
            cos: Cosine tensor for RoPE.
            sin: Sine tensor for RoPE.
            kv_cache: Paged KV cache tensor.
            block_table: Block table for Paged KV cache.
            cache_seqlens: Sequence lengths for Paged KV cache.
            output_attentions: Whether to output attentions (not supported by FlashMLA).
            
        Returns:
            Tuple containing attention output, (None for attention weights), (None for past_key_value tuple).
        """
        bsz, q_len, _ = hidden_states.shape
        
        # --- Projections ---
        # Project queries
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        
        # Reshape and split query into nope and rope parts
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        
        # Process key-value
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        
        # Process the compressed KV
        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        kv = kv.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        
        # Split into key and value
        k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        
        # Apply rotary embeddings to the rope parts
        q_pe, k_pe = self._apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)
        
        # Combine the nope and rope parts
        query_states = torch.empty(bsz, self.num_heads, q_len, self.q_head_dim, device=hidden_states.device)
        query_states[..., :self.qk_nope_head_dim] = q_nope
        query_states[..., self.qk_nope_head_dim:] = q_pe
        
        key_states = torch.empty(bsz, self.num_heads, q_len, self.q_head_dim, device=hidden_states.device)
        key_states[..., :self.qk_nope_head_dim] = k_nope
        key_states[..., self.qk_nope_head_dim:] = k_pe # Shape: (bsz, num_heads, q_len, q_head_dim)
        # value_states shape: (bsz, num_heads, q_len, v_head_dim)

        # --- Paged KV Cache Update ---
        # Write current key/value to cache BEFORE attention computation
        # This requires the actual implementation of update_paged_kv_cache
        if kv_cache is not None and block_table is not None and cache_seqlens is not None:
             update_paged_kv_cache(
                 kv_cache=kv_cache,
                 block_table=block_table,
                 cache_seqlens=cache_seqlens,
                 key=key_states,
                 value=value_states,
                 layer_idx=self.layer_idx
             )

        # --- FlashMLA Call ---
        # Prepare query tensor in expected shape: (batch_size, seq_len_q, num_heads_q, head_dim)
        # Current query_states shape: (bsz, num_heads, q_len, q_head_dim) -> transpose(1, 2)
        q_for_flash = query_states.transpose(1, 2).contiguous() # Shape: (bsz, q_len, num_heads, q_head_dim)

        # Get metadata for FlashMLA
        # num_heads_per_head_k = seq_len_q * num_heads_q // num_heads_k = q_len * num_heads // num_heads = q_len
        num_heads_per_head_k = q_len
        num_heads_k = self.num_heads
        tile_scheduler_metadata, num_splits = get_mla_metadata(
            cache_seqlens=cache_seqlens,
            num_heads_per_head_k=num_heads_per_head_k,
            num_heads_k=num_heads_k,
        )

        # Call FlashMLA
        # k_cache expects the *entire* cache, not just current keys
        # Shape: (num_blocks, page_block_size, num_heads_k, head_dim) - assuming K only based on name
        # Or potentially (num_blocks, 2, num_heads, block_size, head_dim) if K/V packed
        attn_output_flash, _ = flash_mla_with_kvcache(
            q=q_for_flash,
            k_cache=kv_cache, # Pass the full cache tensor
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            head_dim_v=self.v_head_dim,
            tile_scheduler_metadata=tile_scheduler_metadata,
            num_splits=num_splits,
            causal=True,
            # softmax_scale is handled internally by default
        )
        # Output shape: (batch_size, seq_len_q, num_heads_q, head_dim_v)

        # Reshape output back to (bsz, q_len, num_heads * v_head_dim)
        attn_output = attn_output_flash.view(bsz, q_len, self.num_heads * self.v_head_dim)

        # --- Output Projection ---
        attn_output = self.o_proj(attn_output)
        
        attn_output = self.o_proj(attn_output)
        
        # Prepare outputs
        # FlashMLA does not return attention weights or the past_key_value tuple directly
        outputs = (attn_output, None, None) # (attn_output, attn_weights, past_key_value)
        
        return outputs
