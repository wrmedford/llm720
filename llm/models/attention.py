#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-Headed Latent Attention (MLA) implementation.

As implemented in DeepSeek models, with:
- Separate projection for query, key, and value
- Query and key decomposed into rope and nope parts
- Support for different head dimensions for query and value

This implementation uses FlashAttention to optimize execution.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te  # Import Transformer Engine

try:
    from flash_attn import flash_attn_varlen_func

    # Determine if RoPE needs to be applied manually or if kernel supports it
    # For simplicity, assume manual application for now.
    HAS_FLASH_ATTN = True
except ImportError:
    print("FlashAttention is not installed. Falling back to standard attention.")
    # Use torch.nn.functional.scaled_dot_product_attention as fallback
    from torch.nn.functional import scaled_dot_product_attention

    HAS_FLASH_ATTN = False


# Assuming RoPE implementation is available, e.g., from foundation
# from llm.models.foundation import apply_rotary_pos_emb # Example import


# RoPE application function based on VLLM/HF Transformers logic
def apply_rotary_pos_emb(
    t: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.LongTensor,
) -> torch.Tensor:
    """Applies Rotary Position Embedding to the input tensor `t`.

    Args:
        t (torch.Tensor): Input tensor of shape
            (batch_size, seq_len, num_heads, head_dim) or
            (total_tokens, num_heads, head_dim).
        cos (torch.Tensor): Cosine components cache of shape
            (max_seq_len, rotary_dim // 2).
        sin (torch.Tensor): Sine components cache of shape
            (max_seq_len, rotary_dim // 2).
        position_ids (torch.LongTensor): Tensor containing the position indices
            for `t`. Shape (batch_size, seq_len) or (total_tokens,).

    Returns:
        torch.Tensor: Tensor with RoPE applied.
    """
    # Ensure cos/sin are on the correct device
    cos = cos.to(t.device)
    sin = sin.to(t.device)

    # position_ids: (batch_size, seq_len) or (total_tokens,)
    # t: (batch_size, seq_len, num_heads, head_dim) or (total_tokens, num_heads, head_dim)

    # Flatten position_ids and gather corresponding cos/sin values
    position_ids_flat = position_ids.reshape(-1)
    cos = cos[position_ids_flat] # Shape: (total_tokens, rotary_dim // 2)
    sin = sin[position_ids_flat] # Shape: (total_tokens, rotary_dim // 2)

    # Reshape t if it's 4D
    original_shape = t.shape
    if t.dim() == 4:
        t = t.reshape(-1, t.shape[-2], t.shape[-1]) # Shape: (total_tokens, num_heads, head_dim)

    # Expand cos/sin to match the heads dimension of t
    # Shape: (total_tokens, 1, rotary_dim // 2)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    # Get the rotary dimension from the input tensor
    rotary_dim = cos.shape[-1] * 2
    t_rot = t[..., :rotary_dim]  # Part of the tensor to apply RoPE

    # Split the rotary part into two halves
    t_rot_half1 = t_rot[..., : rotary_dim // 2]
    t_rot_half2 = t_rot[..., rotary_dim // 2:]

    # Apply the rotation: (-x2, x1)
    rotated_t = torch.cat((-t_rot_half2, t_rot_half1), dim=-1)

    # Apply RoPE: (x * cos) + (rotate_half(x) * sin)
    t_rope = (t_rot * cos) + (rotated_t * sin)

    # Combine the rotated part with the non-rotated part
    return torch.cat((t_rope, t[..., rotary_dim:]), dim=-1)


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim)
    to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


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
        # Combined QK head dim
        self.q_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.dropout_p = dropout  # Use dropout_p for clarity
        self.softmax_scale = self.q_head_dim**-0.5  # Precompute scale

        # Store dimensions needed for MLA logic
        # MLA uses num_kv_heads = 1 conceptually for latent cache
        self.num_kv_heads = 1

        # Initialize query projections
        if self.q_lora_rank is None:
            # Standard Q projection (replace with TE Linear)
            self.q_proj = te.Linear(
                hidden_size, num_heads * self.q_head_dim, bias=False
            )
        else:
            # Q projection with LoRA (replace with TE Linear/LayerNorm)
            self.q_a_proj = te.Linear(hidden_size, q_lora_rank, bias=False)
            # Assuming eps=1e-6, adjust if needed
            self.q_a_layernorm = te.LayerNorm(q_lora_rank, eps=1e-6)
            self.q_b_proj = te.Linear(
                q_lora_rank, num_heads * self.q_head_dim, bias=False
            )

        # Key-value projections (MLA specific)
        # (replace with TE Linear/LayerNorm)
        # Projects hidden_states to latent kv_c and k_pe
        self.kv_a_proj_with_mqa = te.Linear(
            hidden_size, kv_lora_rank + qk_rope_head_dim, bias=False
        )
        # LayerNorm for the compressed kv_c part
        # Assuming eps=1e-6
        self.kv_a_layernorm = te.LayerNorm(kv_lora_rank, eps=1e-6)

        # Projects normalized kv_c to k_nope and v
        self.kv_b_proj = te.Linear(
            kv_lora_rank,
            # Output includes nope part of K and the V part
            num_heads * (self.qk_nope_head_dim + v_head_dim),
            bias=False,
        )

        # Output projection (replace with TE Linear)
        self.o_proj = te.Linear(num_heads * v_head_dim, hidden_size, bias=False)

        # Precompute weights for decode path (MQA-like projections)
        # These need to be updated if kv_b_proj weights change
        # (e.g., after loading checkpoint)
        self.W_UV = None
        self.W_UK_T = None
        self._prepare_kv_weights()  # Attempt initial preparation

    def _prepare_kv_weights(self):
        """
        Precomputes transposed/reshaped weights from kv_b_proj for MQA decode
        path. Call this after model initialization and after loading weights.
        """
        if not hasattr(self.kv_b_proj, "weight") or self.kv_b_proj.weight is None:
            # Weights might not be initialized yet
            return

        try:
            kv_b_proj_weight = self.kv_b_proj.weight.detach().clone()
            # kv_b_proj weight shape:
            # [num_heads * (qk_nope + v_dim), kv_lora_rank]
            # Need to reshape and split into W_UK and W_UV
            kv_b_proj_weight = kv_b_proj_weight.view(
                self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim,
                self.kv_lora_rank,
            )
            # Split into W_UK [num_heads, qk_nope, kv_lora_rank]
            # and W_UV [num_heads, v_dim, kv_lora_rank]
            W_UK, W_UV = kv_b_proj_weight.split(
                [self.qk_nope_head_dim, self.v_head_dim], dim=1
            )

            # W_UV needs shape [num_heads, kv_lora_rank, v_dim] for bmm
            self.W_UV = W_UV.permute(0, 2, 1).contiguous()
            # W_UK_T needs shape [num_heads, qk_nope, kv_lora_rank] for bmm
            # (VLLM uses permute(1, 2, 0) on (L, N, P) -> (N, P, L))
            # Here W_UK is (N, P, L), so we need transpose(1, 2) -> (N, L, P)?
            # No, bmm expects (N, B, P) x (N, P, L)
            # Let's stick to VLLM's naming convention for clarity
            # W_UK_T = W_UK.transpose(1, 2)
            # q_nope (N, B, P) @ W_UK_T (N, P, L) -> ql_nope (N, B, L)
            # Shape [num_heads, qk_nope_head_dim, kv_lora_rank]
            self.W_UK_T = W_UK.contiguous()

            print("Successfully prepared W_UV and W_UK_T for MLA decode.")
        except Exception as e:
            print(
                f"Warning: Failed to prepare W_UV and W_UK_T: {e}. "
                f"Decode path might fail."
            )
            self.W_UV = None
            self.W_UK_T = None

    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input. (Used in RoPE)"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    # Using the refined apply_rotary_pos_emb function above
    # Add a method to explicitly call _prepare_kv_weights,
    # e.g., after loading state_dict
    def post_weight_load(self):
        """Call this after loading weights to prepare decode matrices."""
        self._prepare_kv_weights()

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,  # Pass RoPE embeddings (cos)
        sin: torch.Tensor,  # Pass RoPE embeddings (sin)
        # Can be None or 2D/4D mask
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,  # Crucial for RoPE
        # Cache is (kv_c, k_pe)
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        # Optimized kernels often don't return weights
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,  # Allow for extra arguments if needed
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[Tuple[torch.Tensor, torch.Tensor]],
    ]:
        """Forward pass using FlashAttention for MLA.

        Args:
            hidden_states (`torch.Tensor`): input of shape
                `(batch, seq_len, embed_dim)`
            cos (`torch.Tensor`): Cosine components for RoPE of shape
                `(max_seq_len, rope_dim // 2)`
            sin (`torch.Tensor`): Sine components for RoPE of shape
                `(max_seq_len, rope_dim // 2)`
            attention_mask (`torch.Tensor`, *optional*): Mask for attention.
                FlashAttention typically infers causal mask if None.
                If provided, needs compatible shape.
            position_ids (`torch.LongTensor`, *optional*): Positions for RoPE
                application. Shape `(batch, seq_len)`.
            past_key_value (`Tuple(torch.Tensor, torch.Tensor)`, *optional*):
                Cached latent states `(past_kv_c, past_k_pe)`.
                `past_kv_c` shape:
                    `(batch, past_seq_len, kv_lora_rank)`
                `past_k_pe` shape:
                    `(batch, past_seq_len, qk_rope_head_dim)`
            output_attentions (`bool`, *optional*): Whether to return attention
                weights (not supported by FlashAttention).
            use_cache (`bool`, *optional*): Whether to use and return KV cache.

        Returns:
            Tuple containing:
                - attn_output (`torch.Tensor`): Output tensor of shape
                    `(batch, seq_len, hidden_size)`
                - attn_weights (`Optional[torch.Tensor]`): Always None for FlashAttention.
                - present_key_value (`Optional[Tuple[torch.Tensor, torch.Tensor]]`):
                    Updated cache `(kv_c_cache, k_pe_cache)`.
        """
        # Get logger instance - assumes logger is configured elsewhere
        import logging
        logger = logging.getLogger(__name__)

        if output_attentions:
            logger.warning(
               "FlashAttention does not support outputting attention weights."
            )

        bsz, q_len, _ = hidden_states.shape
        device = hidden_states.device
        # dtype = hidden_states.dtype # Unused

        # --- Projections ---
        # 1. Project queries
        if self.q_lora_rank is None:
            # Standard Q projection -> [bsz, q_len, num_heads * q_head_dim]
            # TE Linear returns tensor
            q_proj_output = self.q_proj(hidden_states)
        else:
            # Q projection with LoRA
            q_lora_intermediate = self.q_a_proj(hidden_states)
            q_lora_intermediate_normed = self.q_a_layernorm(q_lora_intermediate)
            q_proj_output = self.q_b_proj(q_lora_intermediate_normed)

        # Reshape query and split into nope/rope parts
        # q shape: [bsz, q_len, num_heads, q_head_dim]
        q = q_proj_output.view(bsz, q_len, self.num_heads, self.q_head_dim)
        # Split into q_nope [bsz, q_len, num_heads, qk_nope_head_dim]
        # and q_pe [bsz, q_len, num_heads, qk_rope_head_dim]
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # 2. Project hidden_states to get latent kv_c and k_pe (unrotated)
        # compressed_kv shape: [bsz, q_len, kv_lora_rank + qk_rope_head_dim]
        # TE Linear returns tensor
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        # kv_c: [bsz, q_len, kv_lora_rank]
        # k_pe_unrotated: [bsz, q_len, qk_rope_head_dim]
        kv_c, k_pe_unrotated = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        # 3. Normalize kv_c
        # kv_c_normed: [bsz, q_len, kv_lora_rank]
        # TE LayerNorm handles precision
        kv_c_normed = self.kv_a_layernorm(kv_c.contiguous())

        # --- KV Cache Handling ---
        past_kv_c = None
        past_k_pe = None
        # past_len = 0 # Unused variable
        if past_key_value is not None:
            if isinstance(past_key_value, tuple) and len(past_key_value) == 2:
                past_kv_c, past_k_pe = past_key_value
                # Infer past_len from cache shape: [bsz, seq_len, dim]
                # past_len = past_kv_c.shape[1] # Unused
            else:
                print(
                    "Warning: Received past_key_value in unexpected format "
                    "for MLA. Ignoring cache."
                )
                past_key_value = None  # Treat as no cache provided

        # --- RoPE Application ---
        # Apply RoPE to the PE part of the *current* query
        # q_pe shape: [bsz, q_len, num_heads, qk_rope_head_dim]
        # position_ids shape: [bsz, q_len] -> contains absolute positions
        # for current tokens
        q_pe_rotated = apply_rotary_pos_emb(q_pe, cos, sin, position_ids)

        # Combine query parts: [bsz, q_len, num_heads, q_head_dim]
        query_states = torch.cat([q_nope, q_pe_rotated], dim=-1)

        # Apply RoPE to the PE part of the *current* key
        # k_pe_unrotated shape: [bsz, q_len, qk_rope_head_dim]
        # Reshape for RoPE: [bsz, q_len, 1, qk_rope_head_dim]
        k_pe = apply_rotary_pos_emb(k_pe_unrotated.unsqueeze(2), cos, sin, position_ids)
        # Back to [bsz, q_len, qk_rope_head_dim]
        k_pe = k_pe.squeeze(2)

        # --- Update Cache (if use_cache=True) ---
        present_key_value = None
        if use_cache:
            if past_kv_c is not None:
                # Concatenate past and current states
                present_kv_c = torch.cat((past_kv_c, kv_c_normed), dim=1)
                present_k_pe = torch.cat((past_k_pe, k_pe), dim=1)
            else:
                # First step, cache is just the current state
                present_kv_c = kv_c_normed
                present_k_pe = k_pe
            present_key_value = (present_kv_c, present_k_pe)
            # Use the updated cache for attention calculation
            kv_c_for_attn = present_kv_c
            k_pe_for_attn = present_k_pe
        else:
            # If not using cache, attention uses only current states
            kv_c_for_attn = kv_c_normed
            k_pe_for_attn = k_pe

        # Determine the sequence length for K/V based on whether cache is used
        kv_seq_len = kv_c_for_attn.shape[1]

        # --- Prepare inputs for Attention Calculation ---
        # Reshape query_states for FA:
        # [bsz * q_len, num_heads, q_head_dim] or
        # [total_tokens, num_heads, q_head_dim]
        # FA expects sequence layout: (total_tokens, num_heads, head_dim)
        query_states = query_states.reshape(-1, self.num_heads, self.q_head_dim)

        # Determine attention mode (Prefill vs Decode) - simplified check
        is_decode = q_len == 1 and past_key_value is not None

        # Will hold the input to the final o_proj
        attn_output_proj_input = None

        # Need cu_seqlens_q and cu_seqlens_k for flash_attn_varlen_func
        # if bsz > 1
        # For standard training loop with batching, assume all sequences
        # in batch have same length for now.
        # If using packing/variable lengths, this needs adjustment.
        # cu_seqlens_q: Cumulative sequence lengths for queries in batch.
        # cu_seqlens_k: Cumulative sequence lengths for keys/values in batch.
        cu_seqlens_q = torch.arange(
            0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device=device
        )
        cu_seqlens_k = torch.arange(
            0, (bsz + 1) * kv_seq_len, step=kv_seq_len, dtype=torch.int32, device=device
        )
        max_seqlen_q = q_len
        max_seqlen_k = kv_seq_len

        if HAS_FLASH_ATTN:
            # --- FlashAttention Path ---
            if is_decode:
                # Decode Path (Data-Movement Friendly - MQA style)
                # Project q_nope using W_UK_T to get
                # ql_nope [bsz, q_len, num_heads, kv_lora_rank]
                if self.W_UK_T is None or self.W_UV is None:
                    # Attempt to prepare weights again if they weren't ready initially
                    self._prepare_kv_weights()
                    # If still None after trying again, raise error
                    if self.W_UK_T is None or self.W_UV is None:
                        raise RuntimeError(
                            "MLA decode weights (W_UK_T, W_UV) were not successfully prepared. "
                            "Ensure post_weight_load() is called after model initialization and weight loading."
                        )

                # q_nope: [bsz, q_len, num_heads, qk_nope_head_dim]
                # W_UK_T: [num_heads, qk_nope_head_dim, kv_lora_rank]
                # Need batch matrix multiply per head
                # [bsz, num_heads, q_len, qk_nope]
                q_nope_reshaped = q_nope.permute(0, 2, 1, 3)
                # [bsz, num_heads, q_len, kv_lora_rank]
                ql_nope = torch.matmul(q_nope_reshaped, self.W_UK_T)

                # q_pe_rotated: [bsz, q_len, num_heads, qk_rope_head_dim] ->
                # [bsz, num_heads, q_len, qk_rope_head_dim]
                q_pe_rotated_reshaped = q_pe_rotated.permute(0, 2, 1, 3)

                # Combine ql_nope and q_pe_rotated for MQA query
                # q_mqa: [bsz, num_heads, q_len,
                #         kv_lora_rank + qk_rope_head_dim]
                q_mqa = torch.cat([ql_nope, q_pe_rotated_reshaped], dim=-1)
                # [total_q_tokens, num_heads, head_dim_mqa]
                q_mqa = q_mqa.reshape(
                    -1, self.num_heads, self.kv_lora_rank + self.qk_rope_head_dim
                )

                # Prepare K and V for MQA using the potentially cached states
                # k_pe_for_attn: [bsz, kv_seq_len, qk_rope_head_dim]
                # kv_c_for_attn: [bsz, kv_seq_len, kv_lora_rank]
                # Reshape for FA MQA:
                # [total_kv_tokens, num_kv_heads=1, head_dim]
                k_pe_mqa = k_pe_for_attn.reshape(-1, 1, self.qk_rope_head_dim)
                kv_c_mqa = kv_c_for_attn.reshape(-1, 1, self.kv_lora_rank)

                # k_mqa: [total_kv_tokens, 1,
                #         kv_lora_rank + qk_rope_head_dim]
                k_mqa = torch.cat([kv_c_mqa, k_pe_mqa], dim=-1)
                # v_mqa: [total_kv_tokens, 1, kv_lora_rank]
                # (Value is the compressed latent vector)
                v_mqa = kv_c_mqa

                # Perform MQA using flash_attn_varlen_func
                # Output shape: [total_q_tokens, num_heads, kv_lora_rank]
                # (V dim = kv_lora_rank)
                attn_output_mqa = flash_attn_varlen_func(
                    q=q_mqa,
                    k=k_mqa,
                    v=v_mqa,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    dropout_p=0.0,  # No dropout during inference
                    # Scale should be based on the MQA query dimension used in dot product
                    softmax_scale=(self.kv_lora_rank + self.qk_rope_head_dim)**-0.5,
                    causal=True,  # Causal mask needed
                )
                # attn_output_mqa shape:
                # [bsz * q_len, num_heads, kv_lora_rank]

                # Apply final projection (VLLM's _v_up_proj_and_o_proj logic)
                # Reshape to [bsz*q_len, num_heads, kv_lora_rank] ->
                # [num_heads, bsz*q_len, kv_lora_rank]
                attn_output_mqa = attn_output_mqa.transpose(0, 1)
                # W_UV shape: [num_heads, kv_lora_rank, v_head_dim]
                # Output: [num_heads, bsz*q_len, v_head_dim]
                projected_output = torch.bmm(attn_output_mqa, self.W_UV)
                # Reshape back: [bsz*q_len, num_heads * v_head_dim]
                projected_output = projected_output.transpose(0, 1).reshape(
                    -1, self.num_heads * self.v_head_dim
                )
                attn_output_proj_input = projected_output

            else:
                # Prefill Path (Compute Friendly - MHA style)
                # Project kv_c_for_attn (potentially cached) to get k_nope
                # and v
                # kv_c_for_attn shape: [bsz, kv_seq_len, kv_lora_rank]
                # kv_b_proj input needs shape [*, kv_lora_rank]
                # TE Linear returns tensor
                kv_b_out = self.kv_b_proj(kv_c_for_attn.view(-1, self.kv_lora_rank))
                # Reshape output:
                # [bsz, kv_seq_len, num_heads, qk_nope + v_dim]
                kv_b_out = kv_b_out.view(
                    bsz,
                    kv_seq_len,
                    self.num_heads,
                    self.qk_nope_head_dim + self.v_head_dim,
                )
                # Split into k_nope and v
                # k_nope: [bsz, kv_seq_len, num_heads, qk_nope_head_dim]
                # v: [bsz, kv_seq_len, num_heads, v_head_dim]
                k_nope, v = torch.split(
                    kv_b_out, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
                )

                # Combine k_nope and k_pe_for_attn (potentially cached)
                # k_pe_for_attn: [bsz, kv_seq_len, qk_rope_head_dim] ->
                # expand for heads
                k_pe_expanded = k_pe_for_attn.unsqueeze(2).expand(
                    -1, -1, self.num_heads, -1
                )
                # k: [bsz, kv_seq_len, num_heads, q_head_dim]
                k = torch.cat([k_nope, k_pe_expanded], dim=-1)

                # Reshape K and V for FlashAttention:
                # [total_kv_tokens, num_heads, head_dim]
                k = k.reshape(-1, self.num_heads, self.q_head_dim)
                v = v.reshape(-1, self.num_heads, self.v_head_dim)

                # Pad V if v_head_dim < q_head_dim
                if self.v_head_dim < self.q_head_dim:
                    padding_size = self.q_head_dim - self.v_head_dim
                    v_padded = F.pad(v, (0, padding_size))
                else:
                    # Should be equal if q_head_dim == qk_nope + qk_rope
                    v_padded = v

                # Perform MHA using flash_attn_varlen_func
                # Output shape: [total_q_tokens, num_heads, q_head_dim]
                # (V dim = padded V dim)
                attn_output_padded = flash_attn_varlen_func(
                    # Shape: [total_q_tokens, num_heads, q_head_dim]
                    q=query_states,
                    # Shape: [total_kv_tokens, num_heads, q_head_dim]
                    k=k,
                    # Shape: [total_kv_tokens, num_heads, q_head_dim]
                    v=v_padded,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale,
                    causal=True,
                )

                # Slice off padding from output and reshape
                # attn_output shape: [total_q_tokens, num_heads, v_head_dim]
                attn_output_unpadded = attn_output_padded[..., : self.v_head_dim]
                # Reshape to [total_q_tokens, num_heads * v_head_dim]
                attn_output_proj_input = attn_output_unpadded.reshape(
                    -1, self.num_heads * self.v_head_dim
                )

        else:
            # --- Fallback to torch.nn.functional.scaled_dot_product_attention --- # noqa
            # This path won't implement the specific MLA projections
            # (MQA decode) easily.
            # It will perform standard MHA based on the projected Q, K, V
            # derived from combined states.
            # This serves as a functional fallback if FlashAttention is not
            # available.

            # Project kv_c_for_attn to get k_nope and v
            # (same as prefill path above)
            # TE Linear returns tensor
            kv_b_out = self.kv_b_proj(kv_c_for_attn.view(-1, self.kv_lora_rank))
            kv_b_out = kv_b_out.view(
                bsz, kv_seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            k_nope, v = torch.split(
                kv_b_out, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
            )

            # Combine k_nope and k_pe_for_attn (potentially cached)
            k_pe_expanded = k_pe_for_attn.unsqueeze(2).expand(
                -1, -1, self.num_heads, -1
            )
            # [bsz, kv_seq_len, num_heads, q_head_dim]
            k = torch.cat([k_nope, k_pe_expanded], dim=-1)

            # Reshape Q, K, V for scaled_dot_product_attention:
            # [bsz, num_heads, seq_len, head_dim]
            query_states_sdpa = query_states.reshape(
                bsz, q_len, self.num_heads, self.q_head_dim
            ).transpose(1, 2)
            key_states_sdpa = k.transpose(1, 2)
            value_states_sdpa = v.transpose(1, 2)

            # scaled_dot_product_attention expects attention_mask:
            # [bsz, num_heads, q_len, k_len]
            # Need to adapt the input attention_mask if it's 2D or None
            # Assuming causal mask is needed
            attn_output_sdpa = scaled_dot_product_attention(
                query_states_sdpa,
                key_states_sdpa,
                value_states_sdpa,
                # Let SDPA handle causal mask based on is_causal=True
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True,  # Enable causal masking
            )  # Output: [bsz, num_heads, q_len, v_head_dim]

            # Reshape output: [bsz, q_len, num_heads * v_head_dim]
            attn_output_proj_input = (
                attn_output_sdpa.transpose(1, 2).contiguous().view(bsz, q_len, -1)
            )

        # --- Final Output Projection ---
        # attn_output_proj_input shape:
        # [bsz * q_len, num_heads * v_head_dim] or
        # [bsz, q_len, num_heads * v_head_dim]
        # TE Linear returns tensor
        attn_output = self.o_proj(attn_output_proj_input)
        # Reshape back to [bsz, q_len, hidden_size] if necessary
        # (should be correct already)
        if attn_output.shape[0] != bsz or attn_output.shape[1] != q_len:
            attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        # Return None for attn_weights
        return attn_output, None, present_key_value
