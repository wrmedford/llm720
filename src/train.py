#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Foundation Language Model Training Script with PEER and Multi-Headed Latent Attention (MLA)

This script provides a production-ready implementation for training language models with:
- PEER (Parameter Efficient Expert Retrieval) for expert selection
- Multi-Headed Latent Attention (MLA) as in DeepSeek models
- HuggingFace streaming datasets with interleaving
- Checkpoint management with configurable intervals
- Continuous evaluation using OpenAI's evals
- Configuration file support for ablation testing
- Weights & Biases integration for metrics and logging
"""

import os
import time
import json
import math
import random
import logging
import argparse
import functools
import itertools
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim.lr_scheduler import LambdaLR

import transformers
from transformers import (
    PreTrainedTokenizerBase,
    AutoTokenizer, 
    AutoModelForCausalLM,
    get_scheduler,
    HfArgumentParser,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset, IterableDataset, interleave_datasets
import wandb
import yaml
import evals
from evals.api import DummyCompletionFn, CompletionFn
from evals.registry import Registry
from safetensors.torch import save_file, load_file
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger

# Import expert tracking module
from utils.experts.track import ExpertUsageTracker, hook_expert_tracking

logger = get_logger(__name__)

# Constants
MAX_SEQ_LEN = 2048
PAD_TOKEN_ID = 0

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


def interleave_datasets_with_weights(
    datasets: List[IterableDataset], 
    weights: List[float]
) -> IterableDataset:
    """
    Custom wrapper to interleave datasets with specified weights.
    
    Args:
        datasets: List of HuggingFace IterableDataset
        weights: List of weights for sampling from each dataset
    
    Returns:
        Interleaved dataset
    """
    # Normalize weights
    weights = [w / sum(weights) for w in weights]
    
    # Use the built-in interleave_datasets with sampling probabilities
    return interleave_datasets(datasets, probabilities=weights, stopping_strategy="all_exhausted")


class TrainerConfig:
    """Configuration for the trainer."""
    
    def __init__(self, **kwargs):
        # Model configuration
        self.model_config = {
            # Architecture
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            
            # PEER configuration
            "use_peer": True,
            "peer_config": {
                "num_experts": 1024,  # Default 1024 experts (32x32)
                "num_experts_per_tok": 16,
                "num_heads": 8,
                "expert_hidden_size": 1,
                "product_key_dim": [32, 32],  # Cartesian product dimensions
                "query_dim": 256,
                "batch_norm_query": True,
            },
            
            # MLA configuration
            "use_mla": True,
            "mla_config": {
                "q_lora_rank": 1536,
                "kv_lora_rank": 512,
                "qk_rope_head_dim": 64,
                "v_head_dim": 128,
                "qk_nope_head_dim": 128,
            },
        }
        
        # Training configuration
        self.train_config = {
            "output_dir": "./output",
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 16,
            "gradient_accumulation_steps": 1,
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "num_train_epochs": 3,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1,
            "log_level": "info",
            "logging_steps": 100,
            "save_steps": 1000,
            "eval_steps": 1000,
            "seed": 42,
            "fp16": True,
            "bf16": False,
            "tf32": False,
            "resume_from_checkpoint": None,
        }
        
        # Dataset configuration
        self.dataset_config = {
            "datasets": [
                {
                    "name": "pile",
                    "path": "EleutherAI/pile",
                    "split": "train",
                    "streaming": True,
                    "weight": 0.7,
                    "text_field": "text",
                },
                {
                    "name": "c4",
                    "path": "allenai/c4",
                    "split": "train",
                    "streaming": True,
                    "weight": 0.3,
                    "text_field": "text",
                }
            ],
            "tokenizer_name": "gpt2",
            "max_seq_length": MAX_SEQ_LEN,
        }
        
        # Evaluation configuration
        self.eval_config = {
            "evals_registry_path": "./evals/registry",
            "evals": [
                "hellaswag",
                "mmlu",
                "truthfulqa",
            ],
            "eval_batch_size": 16,
        }
        
        # Wandb configuration
        self.wandb_config = {
            "project": "foundation-model-training",
            "entity": "your-wandb-entity",
            "name": f"lm-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "log_model": "all",
        }
        
        # Update with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    @classmethod
    def from_yaml(cls, yaml_file: str) -> "TrainerConfig":
        """Load configuration from a YAML file."""
        with open(yaml_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)
    
    def save(self, filepath: str) -> None:
        """Save configuration to a YAML file."""
        config_dict = {
            "model_config": self.model_config,
            "train_config": self.train_config,
            "dataset_config": self.dataset_config,
            "eval_config": self.eval_config,
            "wandb_config": self.wandb_config,
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


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


def prepare_datasets(config: Dict) -> IterableDataset:
    """
    Prepare and interleave datasets based on configuration.
    
    Args:
        config: Dataset configuration dictionary
        
    Returns:
        Interleaved dataset
    """
    datasets = []
    weights = []
    
    for dataset_info in config["datasets"]:
        # Load the dataset
        try:
            dataset_path = dataset_info["path"]
            
            # Check if it's a local path or HuggingFace dataset
            if os.path.exists(dataset_path) or dataset_path.startswith("/"):
                # Local dataset
                logger.info(f"Loading local dataset from {dataset_path}")
                
                # Check file format (assume jsonl if not specified)
                file_format = dataset_info.get("format", "json")
                
                # Load dataset using datasets library
                dataset = load_dataset(
                    file_format,
                    data_files=dataset_path,
                    split=dataset_info.get("split", "train"),
                    streaming=dataset_info.get("streaming", True)
                )
            else:
                # HuggingFace dataset
                logger.info(f"Loading HuggingFace dataset {dataset_path}")
                dataset = load_dataset(
                    dataset_path,
                    split=dataset_info.get("split", "train"),
                    streaming=dataset_info.get("streaming", True)
                )
            
            # Map to standardize the text field name
            text_field = dataset_info.get("text_field", "text")
            
            def map_fn(example):
                # Check if text field exists
                if text_field in example:
                    return {"text": example[text_field]}
                
                # If text field doesn't exist, try to find a suitable text field
                for key in example:
                    if isinstance(example[key], str) and len(example[key]) > 0:
                        return {"text": example[key]}
                
                # If no suitable field found, return empty text
                return {"text": ""}
            
            dataset = dataset.map(map_fn)
            
            datasets.append(dataset)
            weights.append(dataset_info.get("weight", 1.0))
            
            logger.info(f"Added dataset {dataset_info['name']} with weight {dataset_info.get('weight', 1.0)}")
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_info['name']}: {e}")
            logger.error(f"Skipping dataset {dataset_info['name']}")
            continue
    
    if not datasets:
        raise ValueError("No valid datasets found in configuration")
    
    # Normalize weights if any are provided
    if any(w != 1.0 for w in weights):
        weights = [w / sum(weights) for w in weights]
    
    # Interleave datasets
    logger.info(f"Interleaving {len(datasets)} datasets with weights {weights}")
    return interleave_datasets_with_weights(datasets, weights)


def tokenize_function(examples, tokenizer, max_seq_length):
    """Tokenize function for dataset processing."""
    # Tokenize the texts
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_seq_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    
    # Create input_ids of exactly max_seq_length size for each example
    result = {"input_ids": [], "attention_mask": []}
    
    for input_ids, attention_mask in zip(tokenized["input_ids"], tokenized["attention_mask"]):
        # If shorter than max_length, pad
        if len(input_ids) < max_seq_length:
            padding_length = max_seq_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        
        result["input_ids"].append(input_ids)
        result["attention_mask"].append(attention_mask)
    
    return result


def prepare_model_inputs(batch, device):
    """Prepare model inputs from a batch."""
    # Move all tensors to the device
    batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
    
    # Generate position_ids if not present
    if "position_ids" not in batch:
        batch_size, seq_length = batch["input_ids"].shape
        batch["position_ids"] = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
    
    return batch


def save_checkpoint(model, optimizer, scheduler, global_step, epoch, filepath):
    """Save a model checkpoint with all required components."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save model weights separately (efficient saving)
    if isinstance(model, DDP):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    
    # Save model weights
    save_file(model_state_dict, f"{filepath}.safetensors")
    
    # Save additional training state
    training_state = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "global_step": global_step,
        "epoch": epoch,
    }
    
    torch.save(training_state, f"{filepath}.training_state")
    logger.info(f"Saved checkpoint to {filepath}")
    
    return filepath


def load_checkpoint(model, optimizer, scheduler, filepath):
    """Load a model checkpoint with all required components."""
    # Load model weights
    model_path = f"{filepath}.safetensors"
    if os.path.exists(model_path):
        state_dict = load_file(model_path)
        
        if isinstance(model, DDP):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    
    # Load training state
    training_state_path = f"{filepath}.training_state"
    if os.path.exists(training_state_path):
        training_state = torch.load(training_state_path, map_location="cpu")
        
        optimizer.load_state_dict(training_state["optimizer"])
        if scheduler and "scheduler" in training_state and training_state["scheduler"]:
            scheduler.load_state_dict(training_state["scheduler"])
        
        global_step = training_state["global_step"]
        epoch = training_state["epoch"]
    else:
        raise FileNotFoundError(f"Training state not found at {training_state_path}")
    
    logger.info(f"Loaded checkpoint from {filepath}")
    
    return global_step, epoch


class ModelEvaluator:
    """Evaluator for running OpenAI evals on model checkpoints."""
    
    def __init__(self, registry_path, tokenizer, device):
        self.registry = Registry(registry_path)
        self.tokenizer = tokenizer
        self.device = device
    
    def model_completion_fn(self, model, prompt, **kwargs):
        """Create a completion function for evals."""
        # Default generation parameters
        max_tokens = kwargs.pop("max_tokens", 256)
        temperature = kwargs.pop("temperature", 0.7)
        top_p = kwargs.pop("top_p", 0.9)
        
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode the response, removing the prompt
        prompt_length = len(inputs.input_ids[0])
        response = self.tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
        
        return [{"text": response}]
    
    def run_eval(self, model, eval_name):
        """Run a specific evaluation and return results."""
        # Create completion function for this model
        completion_fn = functools.partial(self.model_completion_fn, model)
        
        try:
            # Run the evaluation
            eval_spec = self.registry.get_eval(eval_name)
            eval_instance = eval_spec.cls(completion_fn=completion_fn, **eval_spec.args)
            result = eval_instance.run()
            
            return {
                "eval_name": eval_name,
                "metrics": result.metrics,
                "samples": result.sample_metrics,
            }
        except Exception as e:
            logger.error(f"Error running evaluation {eval_name}: {e}")
            return {
                "eval_name": eval_name,
                "error": str(e),
                "metrics": {"error": 1.0},
                "samples": [],
            }
    
    def run_all_evals(self, model, eval_names):
        """Run all specified evaluations and return results."""
        results = {}
        
        for eval_name in eval_names:
            logger.info(f"Running evaluation: {eval_name}")
            result = self.run_eval(model, eval_name)
            results[eval_name] = result
        
        return results


def get_train_setup(config, model):
    """Set up training components based on configuration."""
    # Get optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.train_config["learning_rate"],
        weight_decay=config.train_config["weight_decay"],
    )
    
    # Get scheduler
    total_steps = config.train_config["num_train_epochs"] * config.train_config.get("steps_per_epoch", 1000)
    warmup_steps = int(total_steps * config.train_config["warmup_ratio"])
    
    scheduler = get_scheduler(
        name=config.train_config["lr_scheduler_type"],
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    return optimizer, scheduler


def run_training(config: TrainerConfig):
    """Main training function."""
    # Set up accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config.train_config["gradient_accumulation_steps"],
        mixed_precision="fp16" if config.train_config["fp16"] else "no",
        log_with="wandb" if config.wandb_config else None,
    )
    
    # Set random seed for reproducibility
    set_seed(config.train_config["seed"])
    
    # Initialize wandb if configured
    if accelerator.is_main_process and config.wandb_config:
        accelerator.init_trackers(
            project_name=config.wandb_config["project"],
            config=vars(config),
            init_kwargs={"wandb": config.wandb_config}
        )
    
    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(config.train_config["output_dir"], exist_ok=True)
        
        # Save configuration to output directory
        config.save(os.path.join(config.train_config["output_dir"], "config.yaml"))
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.dataset_config["tokenizer_name"])
    
    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create model
    logger.info("Initializing model...")
    tf_config = TransformerConfig(**config.model_config)
    model = create_model_from_config(tf_config)
    
    # Set up expert usage tracking if enabled
    expert_tracker = None
    peer_config = config.model_config.get("peer_config", {})
    
    if config.model_config.get("use_peer", False) and peer_config.get("log_expert_usage", False):
        logger.info("Setting up expert usage tracking...")
        # Count PEER layers in the model - in our architecture, every other layer starting from layer 1
        num_peer_layers = config.model_config.get("num_hidden_layers", 12) // 2
        num_experts = peer_config.get("num_experts", 1048576)
        
        # Create tracker
        expert_tracker = ExpertUsageTracker(
            num_experts=num_experts,
            num_layers=num_peer_layers,
            log_freq=peer_config.get("log_freq", 1000),
            usage_threshold=peer_config.get("usage_threshold", 5.0),
            wandb_enabled=bool(config.wandb_config)
        )
        
        # Hook model to track expert usage
        hook_expert_tracking(model, expert_tracker)
        logger.info(f"Expert usage tracking enabled for {num_peer_layers} PEER layers with {num_experts} experts each")
    
    # Prepare datasets
    logger.info("Preparing datasets...")
    train_dataset = prepare_datasets(config.dataset_config)
    
    # Tokenize datasets
    def tokenize_map_fn(examples):
        return tokenize_function(examples, tokenizer, config.dataset_config["max_seq_length"])
    
    tokenized_dataset = train_dataset.map(
        tokenize_map_fn,
        batched=True,
        remove_columns=["text"],
    )
    
    # Create data loader
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    train_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=config.train_config["per_device_train_batch_size"],
        collate_fn=data_collator,
    )
    
    # Set up optimizer and scheduler
    optimizer, scheduler = get_train_setup(config, model)
    
    # Prepare for distributed training
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    
    # Set up evaluator
    evaluator = ModelEvaluator(
        config.eval_config["evals_registry_path"],
        tokenizer,
        accelerator.device,
    )
    
    # Load checkpoint if resuming
    global_step = 0
    epochs_trained = 0
    
    if config.train_config["resume_from_checkpoint"]:
        try:
            global_step, epochs_trained = load_checkpoint(
                model, optimizer, scheduler, config.train_config["resume_from_checkpoint"]
            )
            logger.info(f"Resuming from step {global_step} in epoch {epochs_trained}")
            
            # Skip steps that have already been trained
            steps_trained_in_current_epoch = global_step % len(train_dataloader)
            
        except FileNotFoundError:
            logger.warning(f"Checkpoint not found at {config.train_config['resume_from_checkpoint']}, starting from scratch")
    
    # Training loop
    logger.info("Starting training...")
    model.train()
    
    total_steps = config.train_config["num_train_epochs"] * len(train_dataloader)
    progress_bar = tqdm(
        total=total_steps,
        disable=not accelerator.is_local_main_process,
        initial=global_step,
    )
    
    # Main training loop
    for epoch in range(epochs_trained, config.train_config["num_train_epochs"]):
        for step, batch in enumerate(train_dataloader):
            # Skip steps if resuming from checkpoint
            if config.train_config["resume_from_checkpoint"] and epoch == epochs_trained and step < steps_trained_in_current_epoch:
                progress_bar.update(1)
                continue
            
            # Forward pass
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                
                # Backward pass
                accelerator.backward(loss)
                
                # Clip gradients
                if config.train_config["max_grad_norm"] > 0:
                    accelerator.clip_grad_norm_(model.parameters(), config.train_config["max_grad_norm"])
                
                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Update progress
            progress_bar.update(1)
            global_step += 1
            
            # Process expert usage tracking
            if expert_tracker is not None:
                expert_tracker.step_end()
            
            # Log metrics
            if global_step % config.train_config["logging_steps"] == 0:
                # Get learning rate
                lr = scheduler.get_last_lr()[0]
                
                metrics = {
                    "train/loss": loss.item(),
                    "train/learning_rate": lr,
                    "train/epoch": epoch + step / len(train_dataloader),
                    "train/global_step": global_step,
                }
                
                # Add expert usage metrics if available
                if expert_tracker is not None:
                    summary = expert_tracker.get_summary()
                    metrics.update({
                        "experts/hot_experts_count": summary["hot_experts_count"],
                        "experts/total_tokens_processed": summary["total_tokens_processed"],
                    })
                    
                    # Add per-layer statistics
                    for layer_idx, stats in summary.get("layer_stats", {}).items():
                        metrics.update({
                            f"experts/layer_{layer_idx}/experts_used": stats["experts_used"],
                            f"experts/layer_{layer_idx}/usage_coverage": stats["usage_coverage"],
                        })
                
                accelerator.log(metrics)
            
            # Save checkpoint
            if global_step % config.train_config["save_steps"] == 0:
                if accelerator.is_main_process:
                    checkpoint_path = os.path.join(
                        config.train_config["output_dir"],
                        f"checkpoint-{global_step}"
                    )
                    
                    save_checkpoint(
                        accelerator.unwrap_model(model),
                        optimizer,
                        scheduler,
                        global_step,
                        epoch,
                        checkpoint_path
                    )
                    
                    # Save expert usage statistics if tracking is enabled
                    if expert_tracker is not None:
                        expert_stats_path = os.path.join(
                            config.train_config["output_dir"],
                            f"checkpoint-{global_step}-expert-stats.json"
                        )
                        with open(expert_stats_path, 'w') as f:
                            json.dump(expert_tracker.get_summary(), f, indent=2)
            
            # Run evaluation
            if global_step % config.train_config["eval_steps"] == 0:
                if accelerator.is_main_process:
                    logger.info(f"Running evaluation at step {global_step}")
                    
                    # Set model to eval mode
                    model.eval()
                    
                    # Run evals
                    eval_results = evaluator.run_all_evals(
                        accelerator.unwrap_model(model),
                        config.eval_config["evals"]
                    )
                    
                    # Log eval results
                    for eval_name, result in eval_results.items():
                        if "metrics" in result:
                            for metric_name, value in result["metrics"].items():
                                accelerator.log({
                                    f"eval/{eval_name}/{metric_name}": value,
                                })
                    
                    # Set model back to train mode
                    model.train()
    
    # Save final model
    if accelerator.is_main_process:
        logger.info("Training completed. Saving final model...")
        
        final_checkpoint_path = os.path.join(
            config.train_config["output_dir"],
            "final-model"
        )
        
        save_checkpoint(
            accelerator.unwrap_model(model),
            optimizer,
            scheduler,
            global_step,
            config.train_config["num_train_epochs"] - 1,
            final_checkpoint_path
        )
        
        # Save final expert usage statistics if tracking is enabled
        if expert_tracker is not None:
            expert_stats_path = os.path.join(
                config.train_config["output_dir"],
                "final-model-expert-stats.json"
            )
            with open(expert_stats_path, 'w') as f:
                json.dump(expert_tracker.get_summary(), f, indent=2)
    
    # Finish tracking
    if accelerator.is_main_process and config.wandb_config:
        accelerator.end_training()
    
    logger.info("Training completed successfully!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train a foundation language model.")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML file")
    args = parser.parse_args()
    
    # Load configuration
    config = TrainerConfig.from_yaml(args.config)
    
    # Run training
    run_training(config)


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Handle errors gracefully
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise