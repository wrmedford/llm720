#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model Parameter Analyzer for Foundation Models with PEER and MLA

This script calculates and displays the total parameters and active parameters per token
for the foundation model architecture, helping understand the efficiency gains from PEER.
"""

import os
import argparse
import yaml
import math
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


def calculate_mla_params(hidden_size: int, mla_config: Dict) -> Dict:
    """Calculate parameters for Multi-Headed Latent Attention."""
    num_heads = mla_config.get("num_heads", 12)
    q_lora_rank = mla_config.get("q_lora_rank", None)
    kv_lora_rank = mla_config.get("kv_lora_rank", 512)
    qk_rope_head_dim = mla_config.get("qk_rope_head_dim", 64)
    v_head_dim = mla_config.get("v_head_dim", 128)
    qk_nope_head_dim = mla_config.get("qk_nope_head_dim", 128)
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim

    params = {}
    
    # Query projections
    if q_lora_rank is None:
        # Direct projection
        params["q_proj"] = hidden_size * num_heads * q_head_dim
    else:
        # Low-rank projection
        params["q_a_proj"] = hidden_size * q_lora_rank
        params["q_a_layernorm"] = 2 * q_lora_rank  # weight and bias
        params["q_b_proj"] = q_lora_rank * num_heads * q_head_dim
    
    # Key-value projections
    params["kv_a_proj_with_mqa"] = hidden_size * (kv_lora_rank + qk_rope_head_dim)
    params["kv_a_layernorm"] = 2 * kv_lora_rank  # weight and bias
    params["kv_b_proj"] = kv_lora_rank * num_heads * (q_head_dim - qk_rope_head_dim + v_head_dim)
    
    # Output projection
    params["o_proj"] = num_heads * v_head_dim * hidden_size
    
    total_params = sum(params.values())
    
    return {
        "breakdown": params,
        "total": total_params,
        "active": total_params  # All attention params are active
    }


def calculate_peer_params(hidden_size: int, peer_config: Dict) -> Dict:
    """Calculate parameters for PEER."""
    num_experts = peer_config.get("num_experts", 1024*1024)
    num_experts_per_tok = peer_config.get("num_experts_per_tok", 16)
    num_heads = peer_config.get("num_heads", 8)
    expert_hidden_size = peer_config.get("expert_hidden_size", 1)
    query_dim = peer_config.get("query_dim", 256)
    product_key_dim = peer_config.get("product_key_dim", [1024, 1024])
    batch_norm_query = peer_config.get("batch_norm_query", True)
    
    params = {}
    
    # Query network
    params["query_proj"] = hidden_size * num_heads * query_dim
    
    if batch_norm_query:
        params["query_batch_norm"] = 2 * query_dim  # weight and bias
    
    # Sub-keys
    sub_key_dim = query_dim // len(product_key_dim)
    sub_keys_params = 0
    for dim_size in product_key_dim:
        sub_keys_params += dim_size * sub_key_dim
    params["sub_keys"] = sub_keys_params
    
    # Expert networks
    params["expert_down"] = num_experts * hidden_size * expert_hidden_size
    params["expert_up"] = num_experts * hidden_size * expert_hidden_size
    
    total_params = sum(params.values())
    
    # Calculate active parameters
    active_experts = num_heads * num_experts_per_tok
    active_expert_params = active_experts * (hidden_size * expert_hidden_size * 2)  # down and up
    
    active_params = (
        params["query_proj"] + 
        (params.get("query_batch_norm", 0)) + 
        params["sub_keys"] + 
        active_expert_params
    )
    
    return {
        "breakdown": params,
        "total": total_params,
        "active": active_params
    }


def calculate_transformer_block_params(hidden_size: int, use_peer: bool, peer_config: Dict,
                                     use_mla: bool, mla_config: Dict, layer_idx: int) -> Dict:
    """Calculate parameters for a single transformer block."""
    intermediate_size = peer_config.get("intermediate_size", 4 * hidden_size)
    
    params = {}
    
    # Layer norms
    params["ln_1"] = 2 * hidden_size  # weight and bias
    params["ln_2"] = 2 * hidden_size  # weight and bias
    
    # Attention
    if use_mla:
        attn_params = calculate_mla_params(hidden_size, mla_config)
        params["attention"] = attn_params["total"]
        attention_active = attn_params["active"]
    else:
        # Standard multi-head attention
        num_heads = mla_config.get("num_heads", 12)
        head_dim = hidden_size // num_heads
        
        params["q_proj"] = hidden_size * hidden_size
        params["k_proj"] = hidden_size * hidden_size
        params["v_proj"] = hidden_size * hidden_size
        params["o_proj"] = hidden_size * hidden_size
        
        params["attention"] = sum([params["q_proj"], params["k_proj"], params["v_proj"], params["o_proj"]])
        attention_active = params["attention"]
    
    # Feed-forward/PEER
    if use_peer and layer_idx % 2 == 1:  # Apply PEER to alternate layers
        ff_params = calculate_peer_params(hidden_size, peer_config)
        params["feed_forward"] = ff_params["total"]
        ff_active = ff_params["active"]
    else:
        # Standard MLP
        params["mlp_in"] = hidden_size * intermediate_size
        params["mlp_out"] = intermediate_size * hidden_size
        params["feed_forward"] = params["mlp_in"] + params["mlp_out"]
        ff_active = params["feed_forward"]
    
    total_params = sum([params["ln_1"], params["ln_2"], params["attention"], params["feed_forward"]])
    active_params = params["ln_1"] + params["ln_2"] + attention_active + ff_active
    
    return {
        "breakdown": params,
        "total": total_params,
        "active": active_params
    }


def calculate_model_params(config: Dict) -> Dict:
    """Calculate parameters for the entire model."""
    model_config = config.get("model_config", {})
    
    hidden_size = model_config.get("hidden_size", 768)
    num_hidden_layers = model_config.get("num_hidden_layers", 12)
    vocab_size = model_config.get("vocab_size", 50257)  # GPT-2 vocab size
    
    use_peer = model_config.get("use_peer", True)
    peer_config = model_config.get("peer_config", {})
    use_mla = model_config.get("use_mla", True)
    mla_config = model_config.get("mla_config", {})
    
    # Calculate all parameters
    params = {}
    
    # Token embeddings
    params["wte"] = vocab_size * hidden_size
    
    # Transformer blocks
    blocks_total = 0
    blocks_active = 0
    
    for i in range(num_hidden_layers):
        block_params = calculate_transformer_block_params(
            hidden_size, use_peer, peer_config, use_mla, mla_config, i
        )
        blocks_total += block_params["total"]
        blocks_active += block_params["active"]
    
    params["blocks"] = blocks_total
    
    # Final layer norm
    params["ln_f"] = 2 * hidden_size  # weight and bias
    
    # LM head (tied with embeddings)
    # Since weights are tied, we don't count these parameters again
    
    total_params = sum(params.values())
    
    # Calculate active parameters per token
    active_params = params["wte"] + blocks_active + params["ln_f"]
    
    return {
        "breakdown": params,
        "total": total_params,
        "active": active_params,
        "blocks_active": blocks_active,
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "use_peer": use_peer,
        "use_mla": use_mla
    }


def analyze_peer_efficiency(config: Dict, param_counts: Dict) -> Dict:
    """Analyze the efficiency gains from using PEER."""
    model_config = config.get("model_config", {})
    peer_config = model_config.get("peer_config", {})
    
    num_experts = peer_config.get("num_experts", 1024*1024)
    num_experts_per_tok = peer_config.get("num_experts_per_tok", 16)
    num_heads = peer_config.get("num_heads", 8)
    expert_hidden_size = peer_config.get("expert_hidden_size", 1)
    
    # Calculate efficiency metrics
    active_ratio = param_counts["active"] / param_counts["total"]
    expert_utilization = num_heads * num_experts_per_tok / num_experts
    
    # Estimate theoretical performance improvement
    # Assuming compute is proportional to active parameters
    # and performance is proportional to total parameters
    theoretical_speedup = param_counts["total"] / param_counts["active"]
    
    return {
        "active_ratio": active_ratio,
        "expert_utilization": expert_utilization,
        "theoretical_speedup": theoretical_speedup,
        "experts_per_token": num_heads * num_experts_per_tok
    }


def format_number(num):
    """Format a number for display with appropriate units."""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return f"{num:.2f}"


def display_results(param_counts: Dict, efficiency: Dict):
    """Display the results in a readable format."""
    headers = ["Parameter Type", "Count", "Percentage"]
    
    # Model overview
    print("\n" + "=" * 80)
    print(f"MODEL ARCHITECTURE OVERVIEW")
    print("=" * 80)
    print(f"Hidden Size: {param_counts['hidden_size']}")
    print(f"Layers: {param_counts['num_hidden_layers']}")
    print(f"Vocabulary Size: {param_counts['vocab_size']}")
    print(f"PEER Enabled: {param_counts['use_peer']}")
    print(f"MLA Enabled: {param_counts['use_mla']}")
    
    # Total parameters
    print("\n" + "=" * 80)
    print(f"PARAMETER COUNTS")
    print("=" * 80)
    
    total = param_counts["total"]
    
    rows = [
        ["Token Embeddings", format_number(param_counts["breakdown"]["wte"]), f"{param_counts['breakdown']['wte']/total*100:.2f}%"],
        ["Transformer Blocks", format_number(param_counts["breakdown"]["blocks"]), f"{param_counts['breakdown']['blocks']/total*100:.2f}%"],
        ["Final Layer Norm", format_number(param_counts["breakdown"]["ln_f"]), f"{param_counts['breakdown']['ln_f']/total*100:.2f}%"],
        ["Total Parameters", format_number(total), "100.00%"]
    ]
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    
    # Active parameters
    print("\n" + "=" * 80)
    print(f"EFFICIENCY METRICS")
    print("=" * 80)
    
    rows = [
        ["Total Parameters", format_number(param_counts["total"]), "100.00%"],
        ["Active Parameters (per token)", format_number(param_counts["active"]), f"{param_counts['active']/total*100:.2f}%"],
        ["Parameter Efficiency Ratio", f"1:{format_number(1/efficiency['active_ratio'])}", f"{efficiency['active_ratio']*100:.2f}%"],
        ["Experts per Token", format_number(efficiency["experts_per_token"]), f"{efficiency['expert_utilization']*100:.6f}%"],
        ["Theoretical Speedup", f"{efficiency['theoretical_speedup']:.2f}x", ""]
    ]
    
    print(tabulate(rows, headers=["Metric", "Value", "Percentage"], tablefmt="grid"))
    
    # Generate comparative visualization
    print("\n" + "=" * 80)
    print(f"COMPARATIVE ANALYSIS")
    print("=" * 80)
    
    # Calculate equivalent dense model size (if all experts were used)
    peer_config = param_counts.get("model_config", {}).get("peer_config", {})
    dense_equivalent = param_counts["total"]
    
    print(f"This PEER model with {format_number(param_counts['total'])} parameters ")
    print(f"has the active parameter count of a {format_number(param_counts['active'])} parameter dense model")
    print(f"while theoretically having the capacity of a {format_number(dense_equivalent)} parameter model.")
    print(f"The reduction in active parameters allows for {efficiency['theoretical_speedup']:.2f}x theoretical speedup.")


def generate_comparison_chart(param_counts, efficiency, output_path=None):
    """Generate a visual comparison chart."""
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Parameters comparison
    labels = ['Total', 'Active']
    values = [param_counts['total']/1e9, param_counts['active']/1e9]
    
    ax1.bar(labels, values, color=['blue', 'green'])
    ax1.set_ylabel('Parameters (billions)')
    ax1.set_title('Total vs Active Parameters')
    
    # Add value labels on bars
    for i, v in enumerate(values):
        ax1.text(i, v + 0.1, f"{v:.2f}B", ha='center')
    
    # Efficiency chart
    labels = ['Parameter\nEfficiency', 'Expert\nUtilization']
    values = [efficiency['active_ratio']*100, efficiency['expert_utilization']*100]
    
    ax2.bar(labels, values, color=['orange', 'red'])
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Efficiency Metrics')
    ax2.set_ylim(0, 100)
    
    # Add value labels on bars
    for i, v in enumerate(values):
        ax2.text(i, v + 1, f"{v:.2f}%", ha='center')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Chart saved to {output_path}")
    else:
        plt.show()


def main():
    """Main function to analyze model parameters."""
    parser = argparse.ArgumentParser(description="Analyze model parameters for PEER architecture.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--output", type=str, help="Path to save visualization (optional)")
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"Error: Configuration file '{args.config}' not found.")
        return
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Calculate parameters
    param_counts = calculate_model_params(config)
    
    # Analyze efficiency
    efficiency = analyze_peer_efficiency(config, param_counts)
    
    # Display results
    display_results(param_counts, efficiency)
    
    # Generate comparison chart
    if args.output:
        generate_comparison_chart(param_counts, efficiency, args.output)
    else:
        generate_comparison_chart(param_counts, efficiency)


if __name__ == "__main__":
    # Handle errors gracefully
    try:
        main()
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()