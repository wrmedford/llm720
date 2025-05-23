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

from typing import Optional, Tuple

import torch
import torch.nn as nn

from llm.models.attention import MultiHeadedLatentAttention
from llm.models.experts import PEER


class PositionalEmbedding(nn.Module):
    """Rotary positional embeddings."""

    def __init__(self, dim, max_seq_len=2048, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        # Removed cache buffers (cos_cached, sin_cached)

    def forward(self, position_ids: torch.LongTensor, device: Optional[torch.device] = None): # Accept position_ids
        # Determine the maximum position ID needed using tensor operations
        # Add 1 because position IDs are 0-indexed
        req_len_tensor = position_ids.max() + 1

        # Ensure req_len doesn't exceed max_seq_len (use tensor comparison)
        if torch.any(req_len_tensor > self.max_seq_len):
             # Find the actual max value causing the error for a better message
             max_pos_val = position_ids.max().item() # .item() is ok here for error message
             raise ValueError(f"Requested sequence length {max_pos_val + 1} based on position_ids exceeds max_seq_len {self.max_seq_len}")

        # Determine the device
        if device is None:
            device = position_ids.device # Infer from position_ids

        # --- Always compute RoPE embeddings ---
        # Use the tensor req_len_tensor for arange
        # Note: torch.arange can accept a 0-dim tensor for the end argument.
        # Pass req_len_tensor directly to avoid the .item() graph break.
        position = torch.arange(req_len_tensor, device=device).float() # Use tensor directly

        # Create freqs on the target device
        freqs_base = torch.arange(0, self.dim, 2, device=device).float() / self.dim
        freqs = self.base ** freqs_base

        # Compute the complex embeddings: e^(i * position * freq)
        # freqs = torch.outer(position, 1.0 / freqs) # [seq_len, dim//2] # outer doesn't support device arg well before 2.0
        freqs = position[:, None] * (1.0 / freqs[None, :]) # Manual outer product

        # Convert to sin and cos on the target device
        cos = torch.cos(freqs)  # [req_len, dim//2]
        sin = torch.sin(freqs)  # [req_len, dim//2]

        # Return the computed tensors directly (no caching, no cloning needed)
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
        peer_start_layer=2, # Layer index from which PEER layers start
        peer_config=None,
        mla_config=None, # MLA is always used
        vocab_size=50272,  # Padded vocab size (divisible by 16)
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
        self.peer_start_layer = peer_start_layer
        self.peer_config = peer_config or {}
        self.mla_config = mla_config or {} # MLA is always used
        self.vocab_size = vocab_size


class TransformerBlock(nn.Module):
    """Transformer block implementing Multi-Headed Latent Attention (MLA) and PEER."""

    def __init__(self, config: TransformerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Layer norms
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Attention layer - always use MLA
        # Ensure mla_config is passed correctly
        mla_params = config.mla_config or {}
        self.attention = MultiHeadedLatentAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            # layer_idx=layer_idx, # Removed layer_idx argument
            **mla_params,
        )

        # MLP/PEER layer - always assign to self.feed_forward
        # Use PEER from peer_start_layer onwards, otherwise use standard MLP
        if config.use_peer and layer_idx >= config.peer_start_layer:
            # Filter peer_config to only include arguments accepted by PEER.__init__
            peer_init_args = {
                k: v for k, v in config.peer_config.items()
                if k in PEER.__init__.__code__.co_varnames # Inspect PEER init args
            }
            # Remove tracker-specific args if they somehow got included
            for tracker_arg in ["log_expert_usage", "log_freq", "usage_threshold"]:
                peer_init_args.pop(tracker_arg, None)

            self.feed_forward = PEER(
                input_dim=config.hidden_size,
                output_dim=config.hidden_size,
                **peer_init_args, # Pass only the filtered arguments
            )
        else:
            # Standard MLP wrapped in nn.Sequential
            self.feed_forward = nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.intermediate_size, config.hidden_size),
                nn.Dropout(config.dropout),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,  # Add attention_mask back
        position_ids: Optional[torch.LongTensor] = None,  # Add position_ids back
        cos: Optional[torch.Tensor] = None,  # Add cos
        sin: Optional[torch.Tensor] = None,  # Add sin
        past_key_value: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # Use standard past_key_value format
        output_attentions: bool = False,
        use_cache: bool = False,  # Add use_cache flag
    ):
        residual = hidden_states
        # Self-attention
        # TE LayerNorm handles precision internally when used with fp8_autocast
        hidden_states_ln = self.ln_1(hidden_states)

        # MLA path expects hidden_states, cos, sin, attention_mask, position_ids, past_key_value
        attn_output, attn_weights, present_key_value = self.attention(
            hidden_states=hidden_states_ln,
            cos=cos,
                sin=sin,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,  # Passed but may not be used effectively by MLA
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        # Note: present_key_value from MLA FlashAttention impl is currently None

        # Residual connection
        # Note: TE Linear layers often don't need explicit dropout after them if using TE's built-in dropout fusion
        # Check TE documentation if dropout is fused in te.Linear or needs separate application.
        # Assuming separate dropout for now.
        hidden_states = (
            residual + attn_output
        )  # Apply residual (dropout might be handled in o_proj or later)
        # Feed-forward (MLP or PEER)
        residual = hidden_states
        # TE LayerNorm handles precision
        hidden_states_ln = self.ln_2(hidden_states)

        # Apply MLP/PEER (now always assigned to self.feed_forward)
        ff_output = self.feed_forward(hidden_states_ln)

        # Apply residual
        hidden_states = residual + ff_output

        # Return signature: (hidden_states, present_key_value, optional_attn_weights)
        outputs = (hidden_states, present_key_value)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class FoundationModel(nn.Module):
    """Foundation Language Model implementing PEER and Multi-Headed Latent Attention."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)

        # Positional embeddings for rotary attention
        # Use qk_rope_head_dim if MLA is enabled, otherwise calculate based on hidden_size/num_heads
        # MLA is always used, determine RoPE dimension from mla_config
        rope_dim = config.mla_config.get("qk_rope_head_dim", 64) # Default to 64 if not specified

        self.wpe = PositionalEmbedding(
            dim=rope_dim,
            max_seq_len=config.max_position_embeddings,
            base=config.mla_config.get(
                "rope_theta", 10000.0
            ),  # Use theta from mla_config if available
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Head for language modeling
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

        # Tie weights
        self.wte.weight = self.lm_head.weight

    def _init_weights(self, module):
        """Initialize weights and set specific dtypes."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Initialize weights (typically in the model's main dtype, e.g., BF16/FP16)
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                # Initialize bias values
                module.bias.data.zero_()
                # --- Explicitly set bias dtype to FP32 ---
                # Ensures bias remains FP32 even if model is initialized in BF16/FP16
                module.bias.data = module.bias.data.to(torch.float32)
        elif isinstance(module, nn.LayerNorm):
            # Initialize LayerNorm parameters
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            # --- Explicitly set LayerNorm parameters dtype to FP32 ---
            # Ensures LayerNorm remains FP32 even if model is initialized in BF16/FP16
            module.weight.data = module.weight.data.to(torch.float32)
            module.bias.data = module.bias.data.to(torch.float32)

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

        # Handle KV cache
        past_key_values_length = 0
        if use_cache:
            # Check if past_key_values is provided and is a tuple
            if past_key_values is not None and isinstance(past_key_values, tuple):
                # Assuming past_key_values is a tuple of tuples for each layer
                # (layer_past_k, layer_past_v)
                if len(past_key_values) > 0 and len(past_key_values[0]) > 0:
                    past_key_values_length = past_key_values[0][0].shape[
                        2
                    ]  # Get seq length from first layer's key cache
            # Initialize past_key_values tuple if needed
            if past_key_values is None:
                past_key_values = tuple([None] * len(self.blocks))
        else:
            past_key_values = None

        # Generate default position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            # Ensure position_ids are correctly shaped if provided
            position_ids = position_ids.view(-1, seq_length).long()

        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        # Generate positional embeddings (cos, sin) based on position_ids
        # Pass position_ids directly to wpe. It returns the correctly sized, cloned tensors.
        cos, sin = self.wpe(position_ids=position_ids, device=device)

        # Prepare causal attention mask
        # Standard Hugging Face causal mask preparation
        # This assumes `attention_mask` is the padding mask (1 for real tokens, 0 for padding)
        _attn_implementation = self.config.mla_config.get(
            "_attn_implementation", "eager"
        )  # Check config if available
        if _attn_implementation == "flash_attention_2":
            # Flash Attention handles causal masking internally based on seq lengths.
            # Pass the 2D padding mask directly.
            if attention_mask is not None and 0 not in attention_mask:
                # If no padding, pass None to use internal causal masking
                attention_mask = None
        else:
            # Prepare 4D causal mask for standard attention
            from transformers.modeling_attn_mask_utils import \
                _prepare_4d_causal_attention_mask

            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )

        # Prepare for attention
        hidden_states = inputs_embeds
        next_decoder_cache = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        # Apply transformer blocks
        for i, block in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # Get past key value for the current layer from the tuple
            layer_past_key_value = (
                past_key_values[i] if past_key_values is not None else None
            )

            layer_outputs = block(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,  # Pass position_ids
                past_key_value=layer_past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cos=cos,  # Pass RoPE caches
                sin=sin,
            )

            hidden_states = layer_outputs[0]

            # Handle cache collection
            if use_cache:
                # layer_outputs[1] contains the present_key_value (kv_c, k_pe) for MLA
                # or None for standard MHA in this setup
                next_decoder_cache += (layer_outputs[1],)

            # Handle attention weights collection
            if output_attentions:
                # Attention weights are the last element if cache is used, otherwise second
                attn_weights_idx = (
                    2 if use_cache and layer_outputs[1] is not None else 1
                )
                if len(layer_outputs) > attn_weights_idx:
                    all_self_attns += (layer_outputs[attn_weights_idx],)
                else:
                    all_self_attns += (None,)  # Add None if weights weren't returned

        # Final layer norm (TE LayerNorm handles precision)
        hidden_states = self.ln_f(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Language modeling head (TE Linear)
        lm_logits = self.lm_head(hidden_states)

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Calculate cross-entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        # Prepare outputs
        if return_dict:
            # Filter out None values before creating dict
            output_dict = {}
            if loss is not None:
                output_dict["loss"] = loss
            if lm_logits is not None:
                 output_dict["logits"] = lm_logits
            if use_cache and next_decoder_cache is not None:
                output_dict["past_key_values"] = next_decoder_cache
            if output_hidden_states and all_hidden_states is not None:
                output_dict["hidden_states"] = all_hidden_states
            if output_attentions and all_self_attns is not None:
                output_dict["attentions"] = all_self_attns

            # Use HuggingFace's standard output class if available and desired,
            # otherwise return the dictionary.
            # from transformers.modeling_outputs import CausalLMOutputWithPast
            # return CausalLMOutputWithPast(**output_dict)
            return output_dict
        else:
            outputs = (
                (lm_logits,)
                + (next_decoder_cache,)
                + (all_hidden_states,)
                + (all_self_attns,)
            )
            # Filter out None values
            outputs = tuple(v for v in outputs if v is not None)

            if loss is not None:
                outputs = (loss,) + outputs

            return outputs


def create_model_from_config(config: TransformerConfig):
    """Create a foundation model from a configuration."""
    return FoundationModel(config)
