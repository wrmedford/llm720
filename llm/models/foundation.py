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
import transformer_engine.pytorch as te  # Import Transformer Engine

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

        # Layer norms (replace with TE LayerNorm)
        self.ln_1 = te.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln_2 = te.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Attention layer - use MLA if configured
        if config.use_mla:
            # Ensure mla_config is passed correctly
            mla_params = config.mla_config or {}
            self.attention = MultiHeadedLatentAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                dropout=config.attention_dropout,
                # layer_idx=layer_idx, # Removed layer_idx argument
                **mla_params,
            )
        else:
            # Standard multi-head attention (Keep nn.MultiheadAttention for now, or replace with te.DotProductAttention if desired)
            # Note: Standard MHA won't use FP8 unless replaced with TE version
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
                **config.peer_config,
            )
        else:
            # Standard MLP (replace with TE Linear)
            # Note: TE Linear doesn't include activation/dropout, apply separately
            self.ffn_linear1 = te.Linear(config.hidden_size, config.intermediate_size)
            self.ffn_activation = nn.GELU()  # Keep standard activation
            self.ffn_dropout1 = nn.Dropout(config.dropout)  # Keep standard dropout
            self.ffn_linear2 = te.Linear(config.intermediate_size, config.hidden_size)
            self.ffn_dropout2 = nn.Dropout(
                config.dropout
            )  # Dropout after second linear

        # self.dropout = nn.Dropout(config.dropout) # Now handled within MLP/Attention

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

        if self.config.use_mla:
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
        else:
            # Standard multi-head attention path
            # nn.MultiheadAttention expects query, key, value
            # It doesn't directly use cos, sin, position_ids in its signature
            # RoPE would need to be applied *before* passing to standard MHA if needed
            # Also, standard MHA doesn't return past_key_value tuple directly
            attn_output, attn_weights = self.attention(
                hidden_states_ln,  # query
                hidden_states_ln,  # key
                hidden_states_ln,  # value
                key_padding_mask=attention_mask[:, 0, 0, :]
                if attention_mask is not None and attention_mask.dim() == 4
                else None,  # Adjust mask format if needed
                attn_mask=attention_mask,  # Pass the 4D mask if appropriate for MHA variant
                need_weights=output_attentions,
                # Cache handling needs external management for standard nn.MultiheadAttention
            )
            # Standard MHA doesn't return cache, set present_key_value to None
            present_key_value = None  # Standard MHA doesn't return cache in this format

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

        # Apply MLP/PEER
        if isinstance(self.feed_forward, PEER):
            hidden_states = self.feed_forward(hidden_states_ln)
        else:
            # Apply standard MLP using TE layers
            hidden_states = self.ffn_linear1(hidden_states_ln)
            hidden_states = self.ffn_activation(hidden_states)
            hidden_states = self.ffn_dropout1(hidden_states)
            hidden_states = self.ffn_linear2(hidden_states)
            hidden_states = self.ffn_dropout2(hidden_states)

        # Apply residual
        hidden_states = residual + hidden_states

        # Return signature: (hidden_states, present_key_value, optional_attn_weights)
        outputs = (hidden_states, present_key_value)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class FoundationModel(nn.Module):
    """Foundation Language Model integrating PEER and Multi-Headed Latent Attention."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)

        # Positional embeddings for rotary attention
        # Use qk_rope_head_dim if MLA is enabled, otherwise calculate based on hidden_size/num_heads
        rope_dim = (
            config.hidden_size // config.num_attention_heads
        )  # Default for standard attention
        if config.use_mla and config.mla_config:
            rope_dim = config.mla_config.get("qk_rope_head_dim", 64)

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

        # Final layer norm (replace with TE LayerNorm)
        self.ln_f = te.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Head for language modeling (replace with TE Linear)
        self.lm_head = te.Linear(config.hidden_size, config.vocab_size, bias=False)

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

        # Generate positional embeddings (cos, sin caches)
        # Pass the required sequence length which might include past length
        cos, sin = self.wpe(seq_len=seq_length + past_key_values_length)
        # Slice cos/sin based on the maximum position ID needed for this pass
        max_pos = position_ids.max()
        cos = cos[: max_pos + 1]
        sin = sin[: max_pos + 1]

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
