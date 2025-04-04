#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Foundation model architecture for language modeling with PEER and MLA.

This module contains the main model components:
- TransformerConfig - configuration class
- PositionalEmbedding - rotary positional embeddings
- TransformerBlock - basic transformer block with PEER and MLA support
- FoundationModel - main model class integrating all components
"""

import math
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm.models.attention import MultiHeadedLatentAttention
from llm.models.experts import PEER


class PositionalEmbedding(nn.Module):
    """Rotary positional embeddings."""
    
    def __init__(self, dim, max_seq_len=2048, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Initialize freq_cis with shape (seq_len, d)
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)
        
    def forward(self, seq_len=None):
        if seq_len is None:
            seq_len = self.max_seq_len
            
        if self.cos_cached is not None and seq_len <= self.cos_cached.shape[0]:
            return self.cos_cached[:seq_len], self.sin_cached[:seq_len]
        
        # Generate position indices
        position = torch.arange(seq_len).float()
        
        # Create freqs
        freqs = self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        
        # Compute the complex embeddings: e^(i * position * freq)
        freqs = torch.outer(position, 1.0 / freqs)  # [seq_len, dim//2]
        
        # Convert to sin and cos
        cos = torch.cos(freqs)  # [seq_len, dim//2]
        sin = torch.sin(freqs)  # [seq_len, dim//2]
        
        # Cache the embeddings
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
        
        return cos, sin


class TransformerConfig:
    """Configuration for the Transformer model."""
    
    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        dropout=0.1,
        attention_dropout=0.1,
        max_position_embeddings=2048,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_peer=True,
        peer_config=None,
        use_mla=True,
        mla_config=None,
        vocab_size=50257,  # GPT-2 vocab size
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_peer = use_peer
        self.peer_config = peer_config or {}
        self.use_mla = use_mla
        self.mla_config = mla_config or {}
        self.vocab_size = vocab_size


class TransformerBlock(nn.Module):
    """Transformer block with support for MLA and PEER."""
    
    def __init__(self, config: TransformerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Layer norms
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Attention layer - use MLA if configured
        if config.use_mla:
            self.attention = MultiHeadedLatentAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                dropout=config.attention_dropout,
                **config.mla_config,
            )
        else:
            # Standard multi-head attention
            self.attention = nn.MultiheadAttention(
                embed_dim=config.hidden_size,
                num_heads=config.num_attention_heads,
                dropout=config.attention_dropout,
                batch_first=True,
            )
        
        # MLP/PEER layer
        if config.use_peer and layer_idx % 2 == 1:  # Apply PEER to alternate layers
            self.feed_forward = PEER(
                input_dim=config.hidden_size,
                output_dim=config.hidden_size,
                **config.peer_config
            )
        else:
            # Standard MLP
            self.feed_forward = nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size),
                nn.GELU(),
                nn.Linear(config.intermediate_size, config.hidden_size),
                nn.Dropout(config.dropout),
            )
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        hidden_states,
        position_ids,
        attention_mask=None,
        cos=None,
        sin=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
    ):
        residual = hidden_states
        
        # Self-attention
        hidden_states = self.ln_1(hidden_states)
        
        if self.config.use_mla:
            attention_outputs = self.attention(
                hidden_states=hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                cos=cos,
                sin=sin,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            attn_output = attention_outputs[0]
            
            # Add outputs depending on configuration
            outputs = (attn_output,)
            if output_attentions:
                outputs += (attention_outputs[1],)  # attention weights
            if use_cache:
                outputs += (attention_outputs[2],)  # past_key_value
        else:
            # Convert attention mask from [batch_size, seq_len] to attention format
            if attention_mask is not None:
                # Convert to float and invert (1->0, 0->-inf)
                attn_mask = (1.0 - attention_mask.unsqueeze(1)) * -10000.0
            else:
                attn_mask = None
            
            attn_output, attn_weights = self.attention(
                hidden_states, hidden_states, hidden_states,
                attn_mask=attn_mask,
                need_weights=output_attentions,
            )
            
            outputs = (attn_output,)
            if output_attentions:
                outputs += (attn_weights,)
            if use_cache:
                outputs += (None,)  # Placeholder for past_key_value
        
        # Residual connection
        hidden_states = residual + self.dropout(outputs[0])
        
        # Feed-forward (MLP or PEER)
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states
        
        return (hidden_states,) + outputs[1:]


class FoundationModel(nn.Module):
    """Foundation Language Model integrating PEER and Multi-Headed Latent Attention."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Positional embeddings for rotary attention
        self.wpe = PositionalEmbedding(
            dim=config.mla_config.get("qk_rope_head_dim", 64) if config.use_mla else config.hidden_size // 2,
            max_seq_len=config.max_position_embeddings,
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config, layer_idx) 
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Head for language modeling
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Tie weights
        self.wte.weight = self.lm_head.weight
    
    def _init_weights(self, module):
        """Initialize weights with small random values."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def get_input_embeddings(self):
        return self.wte
    
    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # Initialize return values
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        
        batch_size, seq_length = input_ids.shape if input_ids is not None else inputs_embeds.shape[:2]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        
        # Generate default position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0)
        
        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        
        # Generate positional embeddings
        cos, sin = self.wpe(seq_length)
        
        # Create causal attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)
        
        # Prepare for attention
        hidden_states = inputs_embeds
        
        # Apply transformer blocks
        for i, block in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            layer_outputs = block(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                cos=cos,
                sin=sin,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_self_attns = all_self_attns + (layer_outputs[1],)
            
            if use_cache:
                next_decoder_cache = next_decoder_cache + (layer_outputs[2 if output_attentions else 1],)
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # Language modeling head
        lm_logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate cross-entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Prepare outputs
        if return_dict:
            return {
                "loss": loss,
                "logits": lm_logits,
                "past_key_values": next_decoder_cache,
                "hidden_states": all_hidden_states,
                "attentions": all_self_attns,
            }
        else:
            outputs = (lm_logits,)
            if use_cache:
                outputs = outputs + (next_decoder_cache,)
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if output_attentions:
                outputs = outputs + (all_self_attns,)
            
            if loss is not None:
                outputs = (loss,) + outputs
            
            return outputs


def create_model_from_config(config: TransformerConfig):
    """Create a foundation model from a configuration."""
    return FoundationModel(config)