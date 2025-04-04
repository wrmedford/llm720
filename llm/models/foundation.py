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
                layer_idx=layer_idx, # Pass layer_idx
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
        attention_mask: Optional[torch.Tensor] = None, # Add attention_mask back
        cos=None,
        sin=None,
        # Paged KV cache arguments
        kv_cache=None,
        block_table=None,
        cache_seqlens=None,
        output_attentions=False,
        # use_cache is implicit
    ):
        residual = hidden_states
        
        # Self-attention
        hidden_states = self.ln_1(hidden_states)
        
        if self.config.use_mla:
            attention_outputs = self.attention(
                hidden_states=hidden_states,
                position_ids=position_ids,
                # Removed duplicate hidden_states and position_ids arguments
                cos=cos,
                sin=sin,
                kv_cache=kv_cache, # Pass Paged KV cache parts for the current layer
                block_table=block_table,
                cache_seqlens=cache_seqlens,
                output_attentions=output_attentions,
                # layer_idx is implicitly passed via self.attention instance
            )
            # Output signature: (attn_output, attn_weights, past_key_value)
            # FlashMLA returns (attn_output, None, None)
            attn_output = attention_outputs[0]
            attn_weights = attention_outputs[1] # Will be None
            
            # Add outputs depending on configuration
            outputs = (attn_output,)
            if output_attentions:
                 # FlashMLA doesn't return weights, append None or handle differently
                outputs += (attn_weights,) # Appending None
            # Cache is managed externally via kv_cache, block_table, cache_seqlens
            # No past_key_value tuple is returned by the attention layer anymore
        else:
            # Standard attention path - needs similar cache handling update if MLA is disabled
            # For simplicity, assuming use_mla=True is the primary path
            # Convert attention mask from [batch_size, seq_len] to attention format
            # Note: FlashAttention/FlashMLA typically handle causal masking internally
            if attention_mask is not None: # This mask might still be needed for padding
                # Convert to float and invert (1->0, 0->-inf)
                attn_mask = (1.0 - attention_mask.unsqueeze(1)) * -10000.0
            else:
                attn_mask = None
            
            attn_output, attn_weights = self.attention(
                hidden_states, hidden_states, hidden_states,
                attn_mask=attn_mask, # May need adjustment for standard attention with Paged KV
                need_weights=output_attentions,
                # Add cache arguments here if modifying standard attention
            )
            
            outputs = (attn_output,)
            if output_attentions:
                outputs += (attn_weights,)
            # No past_key_value tuple returned
        
        # Residual connection
        hidden_states = residual + self.dropout(outputs[0])
        
        # Feed-forward (MLP or PEER)
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states
        
        # Return signature adjusted: (hidden_states, attn_weights [None])
        return (hidden_states,) + outputs[1:]


# Placeholder for Paged KV Cache management logic
# Needs actual implementation based on chosen library/approach (e.g., vLLM's PagedAttention)

# Expected cache shapes based on FlashMLA interface (assuming block_size=64):
# k_cache: (num_blocks, 64, num_heads, k_head_dim)
# v_cache: (num_blocks, 64, num_heads, v_head_dim)
# OR potentially packed: kv_cache: (num_blocks, 2, 64, num_heads, head_dim) - Needs clarification!

def init_paged_kv_cache(config, batch_size, max_seq_len, device):
    """
    Initializes the Paged KV Cache tensors.
    This needs a concrete implementation based on the chosen PagedAttention library or logic.
    It should allocate the cache tensors (k_cache, v_cache or packed kv_cache) and
    the initial block_table and cache_seqlens.
    """
    print("Warning: Paged KV Cache initialization logic (init_paged_kv_cache) is a placeholder.")
    # Example placeholder dimensions (adjust based on actual cache structure)
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    # Assuming k_head_dim = q_head_dim from MLA config
    k_head_dim = config.mla_config.get("qk_nope_head_dim", 128) + config.mla_config.get("qk_rope_head_dim", 64)
    v_head_dim = config.mla_config.get("v_head_dim", 128)
    block_size = 64 # From FlashMLA README
    # num_blocks needs estimation based on max_seq_len, batch_size, and available memory
    num_blocks = 2048 # Example placeholder

    # Placeholder cache tensors (replace with actual allocation)
    # Separate K and V caches as expected by FlashMLA C++ backend
    k_cache = torch.zeros((num_blocks, block_size, num_heads, k_head_dim), dtype=torch.float16, device=device)
    v_cache = torch.zeros((num_blocks, block_size, num_heads, v_head_dim), dtype=torch.float16, device=device)
    # Placeholder block table and sequence lengths
    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    block_table = torch.zeros((batch_size, max_num_blocks_per_seq), dtype=torch.int32, device=device)
    cache_seqlens = torch.zeros((batch_size,), dtype=torch.int32, device=device)

    # Return separate K and V caches, block table, and sequence lengths
    return k_cache, v_cache, block_table, cache_seqlens


def update_paged_kv_cache(k_cache, v_cache, block_table, cache_seqlens, key, value, layer_idx):
    """
    Writes the current key and value tensors into the Paged KV cache.
    This needs a concrete implementation using the block_table to find the
    correct physical blocks and offsets based on cache_seqlens.
    The layer_idx might be needed if the cache tensors include the layer dimension.
    """
    # key shape: (bsz, num_heads, q_len, k_head_dim)
    # value shape: (bsz, num_heads, q_len, v_head_dim)
    # k_cache shape: (num_blocks, block_size, num_heads, k_head_dim)
    # v_cache shape: (num_blocks, block_size, num_heads, v_head_dim)
    # block_table shape: (bsz, max_num_blocks_per_seq)
    # cache_seqlens shape: (bsz,)
    print(f"Warning: Paged KV Cache update logic (update_paged_kv_cache) for layer {layer_idx} is a placeholder.")
    # --- Implementation needed here ---
    # Example steps:
    # 1. Determine target block indices and offsets using block_table and cache_seqlens.
    # 2. Reshape/scatter the key and value tensors into the kv_cache tensor at the calculated positions.
    pass


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
        # Determine batch size and sequence length
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
            device = input_ids.device
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
            device = inputs_embeds.device
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided.")

        # Initialize Paged KV Cache if use_cache is True and cache is not provided
        # This logic depends heavily on how generation is handled externally
        is_decode_phase = past_key_values is not None # Use past_key_values presence to infer phase
        if use_cache:
             if is_decode_phase:
                 # Assume past_key_values now contains the necessary cache structures
                 # Unpack separate K and V caches
                 k_cache, v_cache, block_table, cache_seqlens = past_key_values
             else: # Prefill phase
                 # Initialize Paged KV Cache structure
                 # max_length needs to be determined (e.g., from config or generation params)
                 max_length = self.config.max_position_embeddings # Example
                 k_cache, v_cache, block_table, cache_seqlens = init_paged_kv_cache(
                     self.config, batch_size, max_length, device
                 )
                 # Prefill phase uses full seq_length
                 # Decode phase uses seq_length=1, cache_seqlens tracks actual lengths
        else:
            k_cache, v_cache, block_table, cache_seqlens = None, None, None, None

        # Initialize return values
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        # next_decoder_cache tuple is replaced by the updated Paged KV cache structures

        # Generate default position IDs if not provided
        # Handle position_ids based on prefill/decode phase if using cache
        if position_ids is None:
             if is_decode_phase:
                 # Decode phase: position is the current total length
                 current_length = cache_seqlens[0].item() # Assuming uniform length for simplicity
                 position_ids = torch.tensor([[current_length]], dtype=torch.long, device=device)
                 seq_length = 1 # Decode phase processes one token at a time
             else: # Prefill phase
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

            # Get the Paged KV cache specific to the layer if needed, or pass the whole structure
            # Assuming the block expects the full cache structure to manage internally or via the attention layer
            
            layer_outputs = block(
                hidden_states,
                position_ids=position_ids,
                # attention_mask might still be needed for padding in prefill, but causal handled by FlashMLA
                cos=cos,
                sin=sin,
                k_cache=k_cache, # Pass K cache
                v_cache=v_cache, # Pass V cache
                block_table=block_table,
                cache_seqlens=cache_seqlens,
                output_attentions=output_attentions,
            )
            
            hidden_states = layer_outputs[0]
            
            # Update cache_seqlens after first layer in decode phase
            # This logic needs careful placement depending on cache update strategy
            if use_cache and is_decode_phase and i == 0:
                 cache_seqlens += 1

            if output_attentions:
                # layer_outputs[1] will be None from FlashMLA
                all_self_attns = all_self_attns + (layer_outputs[1],)
            
            # Cache is managed via passed structures, no tuple to collect
        
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
             output_dict = {
                 "loss": loss,
                 "logits": lm_logits,
                 "hidden_states": all_hidden_states,
                 "attentions": all_self_attns, # Will contain Nones if output_attentions=True
             }
             if use_cache:
                 # Return the updated cache structures
                 # Pack K and V caches along with block_table and cache_seqlens
                 output_dict["past_key_values"] = (k_cache, v_cache, block_table, cache_seqlens)
             return output_dict
        else:
            outputs = (lm_logits,)
            if use_cache:
                 # Return updated cache structures
                 # Pack K and V caches along with block_table and cache_seqlens
                 outputs = outputs + ((k_cache, v_cache, block_table, cache_seqlens),)
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
