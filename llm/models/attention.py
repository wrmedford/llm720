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
    ):
        super().__init__()
        self.hidden_size = hidden_size
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
        
        # Scaling factor for attention
        self.scale = 1.0 / math.sqrt(self.q_head_dim)
    
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
        attention_mask: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.shape
        
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
        key_states[..., self.qk_nope_head_dim:] = k_pe
        
        # Update keys and values if using cache
        if past_key_value is not None:
            # If past_key_value is provided, use it to augment the current key and value states
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=-2)
            value_states = torch.cat([past_value, value_states], dim=-2)
        
        kv_seq_len = key_states.shape[-2]
        
        # Calculate attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Ensure the mask has the right shape for broadcasting
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            attn_weights = attn_weights + attention_mask
        
        # Apply causal mask
        if past_key_value is None:
            # If not using cached key-values, create a causal mask
            causal_mask = torch.triu(
                torch.ones((q_len, kv_seq_len), dtype=torch.bool, device=hidden_states.device), 
                diagonal=1
            )
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Calculate attention output
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and project back to hidden size
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        
        # Prepare outputs
        outputs = (attn_output,)
        
        if output_attentions:
            outputs += (attn_weights,)
        
        if use_cache:
            outputs += ((key_states, value_states),)
        
        return outputs