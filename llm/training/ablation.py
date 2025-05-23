#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ablation test framework for foundation model architecture.

This module provides functionality for running comprehensive ablation studies
to determine optimal configurations for the foundation model.
"""

# Import all the required functionality from the old ablations.py file
# Remove train_lm.py references and use the new module imports

import copy
import itertools
import json
import os
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
import numpy as np


# Constants for the paper's ablation experiments
FP8_BYTES = 1  # 1 byte per parameter in FP8
EXPERT_SIZE_KB = 56  # Target expert size from paper (56KB in FP8)
EXPERT_SIZE_BYTES = EXPERT_SIZE_KB * 1024
COMPUTE_BUDGET = 3.0e24  # 3.0 × 10^24 FLOPs from paper
STANDARD_VOCAB_SIZE = 200064  # o200k_base vocab size


def calculate_expert_hidden_size(target_size_bytes: int, hidden_size: int, fp8: bool = True) -> int:
    """
    Calculate expert_hidden_size to achieve target expert size in bytes.
    
    For PEER experts, the size is approximately:
    size = (hidden_size * expert_hidden_size + expert_hidden_size) * bytes_per_param
    
    Args:
        target_size_bytes: Target size in bytes
        hidden_size: Model hidden size
        fp8: Whether using FP8 precision
        
    Returns:
        expert_hidden_size value
    """
    bytes_per_param = FP8_BYTES if fp8 else 4
    # Solve for expert_hidden_size
    expert_hidden_size = target_size_bytes / (bytes_per_param * (hidden_size + 1))
    return int(expert_hidden_size)


def calculate_total_params(config: Dict) -> int:
    """
    Calculate total model parameters based on configuration.
    
    Args:
        config: Model configuration dict
        
    Returns:
        Total parameter count
    """
    hidden_size = config["model_config"]["hidden_size"]
    num_layers = config["model_config"]["num_hidden_layers"]
    vocab_size = config["model_config"]["vocab_size"]
    intermediate_size = config["model_config"]["intermediate_size"]
    
    # Embedding parameters
    embedding_params = vocab_size * hidden_size
    
    # Attention parameters (simplified, assuming MLA compression)
    # MLA reduces KV parameters significantly
    mla_config = config["model_config"].get("mla_config", {})
    q_lora_rank = mla_config.get("q_lora_rank", hidden_size // 2)
    kv_lora_rank = mla_config.get("kv_lora_rank", hidden_size // 4)
    
    attention_params_per_layer = (
        hidden_size * q_lora_rank +  # Q projection
        hidden_size * kv_lora_rank * 2 +  # K,V projections
        hidden_size * hidden_size  # Output projection
    )
    
    # FFN or PEER parameters per layer
    if config["model_config"]["use_peer"]:
        peer_config = config["model_config"]["peer_config"]
        num_experts = peer_config["num_experts"]
        expert_hidden_size = peer_config["expert_hidden_size"]
        peer_start_layer = config["model_config"]["peer_start_layer"]
        
        # Dense FFN layers before PEER starts
        dense_ffn_params = intermediate_size * hidden_size * 2 * peer_start_layer
        
        # PEER layers
        peer_layers = num_layers - peer_start_layer
        expert_params = num_experts * expert_hidden_size * (hidden_size + 1)  # +1 for bias
        peer_params = expert_params * peer_layers
        
        ffn_params = dense_ffn_params + peer_params
    else:
        # All dense FFN
        ffn_params = num_layers * intermediate_size * hidden_size * 2
    
    # Layer norm parameters
    ln_params = num_layers * hidden_size * 2  # Pre and post LN
    
    total_params = embedding_params + num_layers * attention_params_per_layer + ffn_params + ln_params
    return int(total_params)


def calculate_active_params(config: Dict) -> int:
    """
    Calculate active parameters per token.
    
    Args:
        config: Model configuration dict
        
    Returns:
        Active parameter count per token
    """
    hidden_size = config["model_config"]["hidden_size"]
    num_layers = config["model_config"]["num_hidden_layers"]
    vocab_size = config["model_config"]["vocab_size"]
    intermediate_size = config["model_config"]["intermediate_size"]
    
    # Embedding parameters (always active)
    embedding_params = hidden_size  # Only one token embedding is active
    
    # Attention parameters (all active per layer)
    mla_config = config["model_config"].get("mla_config", {})
    q_lora_rank = mla_config.get("q_lora_rank", hidden_size // 2)
    kv_lora_rank = mla_config.get("kv_lora_rank", hidden_size // 4)
    
    attention_params_per_layer = (
        hidden_size * q_lora_rank +
        hidden_size * kv_lora_rank * 2 +
        hidden_size * hidden_size
    )
    
    # FFN or PEER active parameters per layer
    if config["model_config"]["use_peer"]:
        peer_config = config["model_config"]["peer_config"]
        num_experts_per_tok = peer_config["num_experts_per_tok"]
        expert_hidden_size = peer_config["expert_hidden_size"]
        peer_start_layer = config["model_config"]["peer_start_layer"]
        
        # Dense FFN layers before PEER
        dense_ffn_active = intermediate_size * hidden_size * 2 * peer_start_layer
        
        # PEER layers - only activated experts count
        peer_layers = num_layers - peer_start_layer
        peer_active_per_layer = num_experts_per_tok * expert_hidden_size * (hidden_size + 1)
        peer_active = peer_active_per_layer * peer_layers
        
        ffn_active = dense_ffn_active + peer_active
    else:
        # All dense FFN
        ffn_active = num_layers * intermediate_size * hidden_size * 2
    
    # Layer norm parameters
    ln_params = num_layers * hidden_size * 2
    
    active_params = embedding_params + num_layers * attention_params_per_layer + ffn_active + ln_params
    return int(active_params)


def activation_percentage_to_num_experts(percentage: float, num_experts: int) -> int:
    """
    Convert activation percentage to num_experts_per_tok.
    
    Args:
        percentage: Activation percentage (e.g., 1.5 for 1.5%)
        num_experts: Total number of experts
        
    Returns:
        num_experts_per_tok value
    """
    return max(1, int(num_experts * percentage / 100.0))


def generate_model_size_from_params(target_params: int, base_hidden_size: int = 768) -> Dict:
    """
    Generate model configuration to achieve target parameter count.
    
    Args:
        target_params: Target total parameter count
        base_hidden_size: Base hidden size to scale from
        
    Returns:
        Dict with hidden_size, num_hidden_layers, intermediate_size
    """
    # Simple scaling heuristic - scale hidden size and adjust layers
    # This is approximate and may need fine-tuning
    
    if target_params < 1e9:  # < 1B params
        scale = (target_params / 300e6) ** 0.5
        hidden_size = int(base_hidden_size * scale)
        hidden_size = (hidden_size // 64) * 64  # Round to multiple of 64
        num_layers = 12
        intermediate_size = hidden_size * 4
    elif target_params < 10e9:  # 1B - 10B params
        scale = (target_params / 1e9) ** 0.5
        hidden_size = int(1024 * scale)
        hidden_size = (hidden_size // 64) * 64
        num_layers = 24
        intermediate_size = hidden_size * 4
    else:  # > 10B params
        scale = (target_params / 10e9) ** 0.5
        hidden_size = int(2048 * scale)
        hidden_size = (hidden_size // 128) * 128  # Round to multiple of 128
        num_layers = 32
        intermediate_size = int(hidden_size * 3.5)
    
    return {
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
        "intermediate_size": intermediate_size
    }


# Define ablation configuration spaces for each axis
ABLATION_CONFIGS = {
    # Ablation 1: Model-Capacity Sweep at Constant P_act
    # C_act = 2048, s_exp = 56KB, vary N_tot from 0 (dense) to 4.2M
    "capacity_sweep": [
        {
            "name": "dense_baseline",
            "use_peer": False,  # Dense model (no experts)
            "target_params": 1e9,  # 1B params
        },
        {
            "name": "1B_65k_experts",
            "use_peer": True,
            "num_experts": 65536,  # 2^16
            "target_params": 1e9,
        },
        {
            "name": "4B_262k_experts",
            "use_peer": True,
            "num_experts": 262144,  # 2^18
            "target_params": 4e9,
        },
        {
            "name": "16B_1M_experts",
            "use_peer": True,
            "num_experts": 1048576,  # 2^20
            "target_params": 16e9,
        },
        {
            "name": "64B_2M_experts",
            "use_peer": True,
            "num_experts": 2097152,  # 2^21
            "target_params": 64e9,
        },
        {
            "name": "250B_4M_experts",
            "use_peer": True,
            "num_experts": 4194304,  # 2^22 ≈ 4.2M
            "target_params": 250e9,
        },
    ],
    
    # Ablation 2: Expert-Granularity Sweep at Constant Capacity
    # Start from 4B params (N_tot = 65,536), quarter s_exp and quadruple N_tot
    "granularity_sweep": [
        {
            "name": "4B_65k_56KB",
            "num_experts": 65536,
            "expert_size_kb": 56,  # Base size
            "num_experts_per_tok": 2048,  # C_act = 2048
        },
        {
            "name": "4B_262k_14KB",
            "num_experts": 262144,  # 4x experts
            "expert_size_kb": 14,  # 1/4 size
            "num_experts_per_tok": 512,  # 1/4 C_act
        },
        {
            "name": "4B_1M_3.5KB",
            "num_experts": 1048576,  # 16x experts
            "expert_size_kb": 3.5,  # 1/16 size
            "num_experts_per_tok": 128,  # 1/16 C_act
        },
        {
            "name": "4B_4M_0.875KB",
            "num_experts": 4194304,  # 64x experts
            "expert_size_kb": 0.875,  # 1/64 size
            "num_experts_per_tok": 32,  # 1/64 C_act
        },
    ],
    
    # Ablation 3: Compute-Budget Trade-off
    # Fixed compute C = 3.0 × 10^24 FLOPs
    "compute_tradeoff": [
        {
            "name": "small_model_many_tokens",
            "target_params": 1e9,  # 1B params
            "training_tokens": 3e12,  # 3T tokens
        },
        {
            "name": "medium_model_balanced",
            "target_params": 4e9,  # 4B params
            "training_tokens": 750e9,  # 750B tokens
        },
        {
            "name": "large_model_fewer_tokens",
            "target_params": 16e9,  # 16B params
            "training_tokens": 187.5e9,  # 187.5B tokens
        },
        {
            "name": "xlarge_model_minimal_tokens",
            "target_params": 64e9,  # 64B params
            "training_tokens": 46.875e9,  # 46.875B tokens
        },
    ],
    
    # Ablation 4: Activation Sweep in 4B-Parameter Regime
    # Fixed at 4B params (N_tot = 65,536), vary activation 1.5% to 12.5%
    "activation_4b": [
        {
            "name": "4B_1.5pct",
            "num_experts": 65536,
            "activation_percentage": 1.5,
        },
        {
            "name": "4B_3pct",
            "num_experts": 65536,
            "activation_percentage": 3.0,
        },
        {
            "name": "4B_6pct",
            "num_experts": 65536,
            "activation_percentage": 6.0,
        },
        {
            "name": "4B_12.5pct",
            "num_experts": 65536,
            "activation_percentage": 12.5,
        },
    ],
    
    # Ablation 5: Activation Sweep in 256B-Parameter Regime
    # Fixed at 256B params (N_tot = 4.19M), vary activation 1.5% to 12.5%
    "activation_256b": [
        {
            "name": "256B_1.5pct",
            "num_experts": 4194304,  # ~4.19M
            "activation_percentage": 1.5,
        },
        {
            "name": "256B_3pct",
            "num_experts": 4194304,
            "activation_percentage": 3.0,
        },
        {
            "name": "256B_6pct",
            "num_experts": 4194304,
            "activation_percentage": 6.0,
        },
        {
            "name": "256B_12.5pct",
            "num_experts": 4194304,
            "activation_percentage": 12.5,
        },
    ],
}


def create_experiment_folder(output_dir: str) -> str:
    """Create a unique experiment folder for this ablation run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(output_dir, f"ablation_run_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)

    # Create subdirectories
    os.makedirs(os.path.join(experiment_dir, "configs"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "results"), exist_ok=True)

    return experiment_dir


def generate_config(base_config: Dict, experiment: Dict, experiment_dir: str, ablation_type: str, gpu_count: int = 1) -> Tuple[str, str]:
    """
    Generate a configuration file for a specific experiment.

    Args:
        base_config: Base configuration dictionary
        experiment: Experiment-specific configuration dictionary
        experiment_dir: Path to experiment directory
        ablation_type: Type of ablation being run
        gpu_count: Number of GPUs to use for training

    Returns:
        Tuple of (config_path, experiment_name)
    """
    # Create a deep copy of the base configuration
    config = copy.deepcopy(base_config)
    
    # Ensure vocab_size is set correctly
    config["model_config"]["vocab_size"] = STANDARD_VOCAB_SIZE

    # Apply experiment-specific configurations based on ablation type
    experiment_name = experiment["name"]
    
    if ablation_type == "capacity_sweep":
        # Ablation 1: Model-Capacity Sweep
        model_size = generate_model_size_from_params(experiment["target_params"])
        config["model_config"]["hidden_size"] = model_size["hidden_size"]
        config["model_config"]["num_hidden_layers"] = model_size["num_hidden_layers"]
        config["model_config"]["intermediate_size"] = model_size["intermediate_size"]
        
        # Update MLA config to scale with model size
        config["model_config"]["mla_config"]["q_lora_rank"] = model_size["hidden_size"] // 2
        config["model_config"]["mla_config"]["kv_lora_rank"] = model_size["hidden_size"] // 2
        
        # Set use_peer based on experiment configuration
        config["model_config"]["use_peer"] = experiment.get("use_peer", True)
        
        if config["model_config"]["use_peer"]:
            # PEER configuration
            config["model_config"]["peer_start_layer"] = 2
            config["model_config"]["peer_config"]["num_experts"] = experiment["num_experts"]
            config["model_config"]["peer_config"]["num_experts_per_tok"] = 2048  # C_act = 2048
            
            # Calculate expert_hidden_size for 56KB experts
            expert_hidden_size = calculate_expert_hidden_size(EXPERT_SIZE_BYTES, model_size["hidden_size"])
            config["model_config"]["peer_config"]["expert_hidden_size"] = expert_hidden_size
            
            # Set product key dimensions based on number of experts
            if experiment["num_experts"] <= 65536:
                config["model_config"]["peer_config"]["product_key_dim"] = [256, 256]
            elif experiment["num_experts"] <= 1048576:
                config["model_config"]["peer_config"]["product_key_dim"] = [1024, 1024]
            else:
                config["model_config"]["peer_config"]["product_key_dim"] = [2048, 2048]
        else:
            # Dense baseline - explicitly set use_peer to False
            # No need for additional configuration since PEER is disabled
            
    elif ablation_type == "granularity_sweep":
        # Ablation 2: Expert-Granularity Sweep
        # Fixed 4B parameter model
        model_size = generate_model_size_from_params(4e9)
        config["model_config"]["hidden_size"] = model_size["hidden_size"]
        config["model_config"]["num_hidden_layers"] = model_size["num_hidden_layers"]
        config["model_config"]["intermediate_size"] = model_size["intermediate_size"]
        
        config["model_config"]["use_peer"] = True
        config["model_config"]["peer_start_layer"] = 2
        config["model_config"]["peer_config"]["num_experts"] = experiment["num_experts"]
        config["model_config"]["peer_config"]["num_experts_per_tok"] = experiment["num_experts_per_tok"]
        
        # Calculate expert_hidden_size based on target size
        target_bytes = int(experiment["expert_size_kb"] * 1024)
        expert_hidden_size = calculate_expert_hidden_size(target_bytes, model_size["hidden_size"])
        config["model_config"]["peer_config"]["expert_hidden_size"] = expert_hidden_size
        
        # Set product key dimensions
        if experiment["num_experts"] <= 65536:
            config["model_config"]["peer_config"]["product_key_dim"] = [256, 256]
        elif experiment["num_experts"] <= 1048576:
            config["model_config"]["peer_config"]["product_key_dim"] = [1024, 1024]
        else:
            config["model_config"]["peer_config"]["product_key_dim"] = [2048, 2048]
            
    elif ablation_type == "compute_tradeoff":
        # Ablation 3: Compute-Budget Trade-off
        model_size = generate_model_size_from_params(experiment["target_params"])
        config["model_config"]["hidden_size"] = model_size["hidden_size"]
        config["model_config"]["num_hidden_layers"] = model_size["num_hidden_layers"]
        config["model_config"]["intermediate_size"] = model_size["intermediate_size"]
        
        # Use PEER for all models in compute tradeoff
        config["model_config"]["use_peer"] = True
        config["model_config"]["peer_start_layer"] = 2
        
        # Scale experts with model size
        if experiment["target_params"] < 4e9:
            num_experts = 65536
        elif experiment["target_params"] < 16e9:
            num_experts = 262144
        elif experiment["target_params"] < 64e9:
            num_experts = 1048576
        else:
            num_experts = 2097152
            
        config["model_config"]["peer_config"]["num_experts"] = num_experts
        config["model_config"]["peer_config"]["num_experts_per_tok"] = 2048  # Fixed C_act
        
        expert_hidden_size = calculate_expert_hidden_size(EXPERT_SIZE_BYTES, model_size["hidden_size"])
        config["model_config"]["peer_config"]["expert_hidden_size"] = expert_hidden_size
        
        # Set training tokens - calculate max_steps with gpu_count and gradient accumulation
        grad_accum_steps = config["train_config"].get("gradient_accumulation_steps", 1)
        global_batch_size = config["train_config"]["per_device_train_batch_size"] * gpu_count * grad_accum_steps
        tokens_per_step = global_batch_size * config["dataset_config"]["max_seq_length"]
        
        if tokens_per_step > 0:
            config["train_config"]["max_steps"] = int(experiment["training_tokens"] / tokens_per_step)
        else:
            raise ValueError(f"Invalid tokens_per_step: {tokens_per_step}")
        
    elif ablation_type in ["activation_4b", "activation_256b"]:
        # Ablations 4 & 5: Activation Sweeps
        if ablation_type == "activation_4b":
            target_params = 4e9
        else:
            target_params = 256e9
            
        model_size = generate_model_size_from_params(target_params)
        config["model_config"]["hidden_size"] = model_size["hidden_size"]
        config["model_config"]["num_hidden_layers"] = model_size["num_hidden_layers"]
        config["model_config"]["intermediate_size"] = model_size["intermediate_size"]
        
        config["model_config"]["use_peer"] = True
        config["model_config"]["peer_start_layer"] = 2
        config["model_config"]["peer_config"]["num_experts"] = experiment["num_experts"]
        
        # Convert activation percentage to num_experts_per_tok
        num_experts_per_tok = activation_percentage_to_num_experts(
            experiment["activation_percentage"], experiment["num_experts"]
        )
        config["model_config"]["peer_config"]["num_experts_per_tok"] = num_experts_per_tok
        
        expert_hidden_size = calculate_expert_hidden_size(EXPERT_SIZE_BYTES, model_size["hidden_size"])
        config["model_config"]["peer_config"]["expert_hidden_size"] = expert_hidden_size
        
        # Set product key dimensions
        if experiment["num_experts"] <= 65536:
            config["model_config"]["peer_config"]["product_key_dim"] = [256, 256]
        else:
            config["model_config"]["peer_config"]["product_key_dim"] = [2048, 2048]

    # Update configuration with unique experiment name and output directory
    config["wandb_config"]["name"] = f"ablation_{ablation_type}_{experiment_name}"
    config["train_config"]["output_dir"] = os.path.join(
        experiment_dir, "checkpoints", f"{ablation_type}_{experiment_name}"
    )

    # Save configuration to file
    config_path = os.path.join(experiment_dir, "configs", f"{ablation_type}_{experiment_name}.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return config_path, f"{ablation_type}_{experiment_name}"


def run_training(
    config_path: str,
    experiment_name: str,
    experiment_dir: str,
    gpu_count: int,
    tokens_limit: int = None,
) -> Tuple[str, str]:
    """
    Run training with the specified configuration.

    Args:
        config_path: Path to configuration file
        experiment_name: Unique name for this experiment
        experiment_dir: Path to experiment directory
        gpu_count: Number of GPUs to use
        tokens_limit: Limit training to this many tokens (optional)

    Returns:
        Tuple of (log_path, checkpoint_path)
    """
    log_path = os.path.join(experiment_dir, "logs", f"{experiment_name}.log")
    max_steps = None

    # Calculate max_steps if token limit is specified
    if tokens_limit is not None:
        # Load the config to get batch size and seq length
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        batch_size = config["train_config"]["per_device_train_batch_size"] * gpu_count
        seq_length = config["dataset_config"]["max_seq_length"]
        tokens_per_step = batch_size * seq_length
        if tokens_per_step > 0:
             max_steps = tokens_limit // tokens_per_step
        else:
             print("Warning: Cannot calculate max_steps, tokens_per_step is zero.")


    # Create command to run training
    cmd = [
        "bash",
        "scripts/train.sh",
        "--config",
        config_path,
        "--gpus-per-node",
        str(gpu_count),
    ]
    # Add max_steps override if calculated
    if max_steps is not None:
         cmd.extend(["--max-steps", str(max_steps)])

    # Run the training process
    print(f"Starting training for experiment: {experiment_name}")
    print(f"Command: {' '.join(cmd)}")

    # Ensure the log directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    try:
        with open(log_path, "w") as log_file:
            process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
            
            # Wait for process to complete
            process.wait()
    except Exception as e:
        print(f"Error executing training command: {e}")
        return log_path, None

    # Check if training completed successfully
    if process.returncode != 0:
        print(f"Training failed for experiment: {experiment_name}")
        print(f"Check logs at: {log_path}")
        return log_path, None

    # Find the final checkpoint
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints", experiment_name)
    checkpoint_path = os.path.join(checkpoint_dir, "final-model.safetensors")

    # If final checkpoint not found, try to find the latest checkpoint
    if not os.path.exists(checkpoint_path):
        checkpoints = [
            d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")
        ]
        if checkpoints:
            # Sort by step number
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            latest_checkpoint = checkpoints[-1]
            checkpoint_path = os.path.join(
                checkpoint_dir, latest_checkpoint, "model.safetensors"
            )
        else:
            print(f"No checkpoint found for experiment: {experiment_name}")
            return log_path, None

    print(f"Training completed for experiment: {experiment_name}")
    print(f"Checkpoint saved at: {checkpoint_path}")

    return log_path, checkpoint_path


def evaluate_checkpoint(
    checkpoint_path: str, config_path: str, experiment_name: str, experiment_dir: str
) -> Dict:
    """
    Evaluate a trained checkpoint on multiple benchmarks.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to configuration file
        experiment_name: Unique name for this experiment
        experiment_dir: Path to experiment directory

    Returns:
        Dictionary with evaluation results
    """
    results_dir = os.path.join(experiment_dir, "results", experiment_name)
    os.makedirs(results_dir, exist_ok=True)

    # Basic results structure
    results = {
        "experiment_name": experiment_name,
        "checkpoint_path": checkpoint_path,
        "config_path": config_path,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 1. First run the model analyzer to get parameter stats
    analyzer_results_path = os.path.join(results_dir, "model_analysis.json")

    # Correct path for size.py
    size_script_path = os.path.join(
        os.path.dirname(
            os.path.dirname(os.path.dirname(__file__))
        ),  # Assumes ablation.py is in llm/training
        "llm",
        "utils",
        "eval",
        "size.py",
    )
    if not os.path.exists(size_script_path):
        # Fallback if structure assumption is wrong, try relative from project root
        size_script_path = "llm/utils/eval/size.py"
        
    # Check if the script exists at either location
    if not os.path.exists(size_script_path):
        print(f"WARNING: Model analyzer script not found at {size_script_path}")
        print("Skipping model analysis step")
        # Add basic parameter info to results without running analysis
        results.update({
            "total_params": 0,
            "active_params": 0,
            "active_params_ratio": 0,
        })
        return results

    cmd = [
        "python",
        size_script_path,
        "--config",
        config_path,
        "--checkpoint",
        checkpoint_path,
        "--output",
        os.path.join(results_dir, "model_analysis.png"),
        "--json",
        analyzer_results_path,
    ]

    print(f"Running model analysis for experiment: {experiment_name}")
    try:
        process = subprocess.run(
            cmd, capture_output=True, text=True, check=False # Use capture_output
        )

        if process.returncode != 0:
            print(f"Model analysis failed for {experiment_name}.")
            print(f"Stdout: {process.stdout}")
            print(f"Stderr: {process.stderr}")
        else:
            # Load analysis results if available
            if os.path.exists(analyzer_results_path):
                with open(analyzer_results_path, "r") as f:
                    analyzer_data = json.load(f)
                    # Add parameter counts to results
                    results.update(
                        {
                            "total_params": analyzer_data.get("total_params", 0),
                            "active_params": analyzer_data.get("active_params", 0),
                            "active_params_ratio": analyzer_data.get("active_params", 0)
                            / max(analyzer_data.get("total_params", 1), 1),
                        }
                    )
    except Exception as e:
        print(f"Error running model analysis: {e}")

    # 2. Run perplexity evaluation using llm-eval entry point
    perplexity_results_path = os.path.join(results_dir, "perplexity.json")
    cmd = [
        "llm-eval",
        "perplexity",
        "--config",
        config_path,
        "--checkpoint",
        checkpoint_path,
        "--output",
        perplexity_results_path
        # Add other perplexity args if needed, e.g., --device
    ]

    print(f"Evaluating perplexity for experiment: {experiment_name}")
    print(f"Command: {' '.join(cmd)}")
    try:
        process = subprocess.run(
            cmd, capture_output=True, text=True, check=False # Use capture_output
        )

        if process.returncode != 0:
            print(f"Perplexity evaluation failed for {experiment_name}.")
            print(f"Stdout: {process.stdout}")
            print(f"Stderr: {process.stderr}")
            results["perplexity"] = float("inf")
        else:
            # Load perplexity results
            if os.path.exists(perplexity_results_path):
                try:
                    with open(perplexity_results_path, "r") as f:
                        perplexity_data = json.load(f)
                            
                        # Handle different possible key structures in perplexity output
                        # Define a priority list of keys to check
                        perplexity_key_candidates = [
                            "perplexity",
                            "validation_perplexity",
                            "test_perplexity",
                            "eval_perplexity",
                            "ppl",
                            "validation_ppl",
                        ]
                            
                        # Try each key in order
                        found_key = False
                        for key in perplexity_key_candidates:
                            if key in perplexity_data:
                                results["perplexity"] = perplexity_data[key]
                                found_key = True
                                break
                            
                        # If not found, check nested dictionaries
                        if not found_key:
                            for section in ["metrics", "results", "validation", "eval"]:
                                if section in perplexity_data and isinstance(perplexity_data[section], dict):
                                    for key in perplexity_key_candidates:
                                        if key in perplexity_data[section]:
                                            results["perplexity"] = perplexity_data[section][key]
                                            found_key = True
                                            break
                                    if found_key:
                                        break
                            
                        # Last resort: try to find any key containing 'perplexity'
                        if not found_key:
                            perplexity_keys = [k for k in perplexity_data.keys() if 'perplexity' in k.lower()]
                            if perplexity_keys:
                                results["perplexity"] = perplexity_data[perplexity_keys[0]]
                                found_key = True
                            
                        # If still not found, use infinity
                        if not found_key:
                            print(f"Warning: Could not find perplexity value in results file: {perplexity_results_path}")
                            print(f"Available keys: {list(perplexity_data.keys())}")
                            results["perplexity"] = float("inf")
                except json.JSONDecodeError:
                    print(f"Error: Could not parse JSON from {perplexity_results_path}")
                    results["perplexity"] = float("inf")
                except Exception as e:
                    print(f"Error processing perplexity results: {e}")
                    results["perplexity"] = float("inf")
            else:
                results["perplexity"] = float("inf")
    except Exception as e:
        print(f"Error evaluating perplexity: {e}")
        results["perplexity"] = float("inf")

    # 3. Run comprehensive benchmark evaluations
    # Define our suite of benchmarks
    benchmarks = [
        # Define benchmarks using the structure expected by llm-eval benchmark
        # Format: (benchmark_name, subset, metric, display_name)
        ("aime_2024", None, "pass@1", "AIME 2024"),
        ("codeforces", None, "percentile", "Codeforces"),  # Also outputs 'rating'
        ("gpqa", "diamond", "pass@1", "GPQA Diamond"),
        ("math", "500", "pass@1", "MATH-500"),
        ("mmlu", None, "pass@1", "MMLU"),
        ("swe_bench", "verified", "resolved", "SWE-bench"),
    ]

    # Run each benchmark using llm-eval
    for benchmark_name, subset, metric, display_name in benchmarks:
        benchmark_results_path = os.path.join(
            results_dir, f"{benchmark_name}_results.json"
        )

        cmd = [
            "llm-eval",
            "benchmark",
            "--config",
            config_path,
            "--checkpoint",
            checkpoint_path,
            "--benchmark",
            benchmark_name,
            "--output",
            benchmark_results_path,
        ]
        if subset:
            cmd.extend(["--subset", subset])
        if metric:
            # Note: llm-eval might calculate multiple metrics, this specifies the primary one if needed by the script
            # Or it might be used just for result extraction below. Check llm-eval behavior.
            cmd.extend(["--metric", metric])
            primary_metric_key = (
                metric  # Assume the specified metric is the key in results JSON
            )
        else:
            # Define default primary metric if none specified (adjust as needed)
            primary_metric_key = "accuracy" if benchmark_name == "mmlu" else "score"

        print(f"Evaluating {display_name} for experiment: {experiment_name}")
        print(f"Command: {' '.join(cmd)}")
        try:
            process = subprocess.run(
                cmd,
                capture_output=True, # Use capture_output
                text=True,
                timeout=7200,  # 2 hour timeout per benchmark
                check=False
            )

            if process.returncode != 0:
                print(f"{display_name} evaluation failed for {experiment_name}.")
                print(f"Stdout: {process.stdout}")
                print(f"Stderr: {process.stderr}")
                results[benchmark_name] = None
            else:
                # Load benchmark results
                if os.path.exists(benchmark_results_path):
                    try:
                        with open(benchmark_results_path, "r") as f:
                            benchmark_data = json.load(f)

                            # Extract the primary metric from the benchmark results
                            # Handle different possible JSON structures
                            if benchmark_name == "codeforces":
                                # Special handling for codeforces which has multiple metrics
                                results[benchmark_name] = benchmark_data.get("percentile", None)
                                results[f"{benchmark_name}_rating"] = benchmark_data.get("rating", None)
                                
                                # Check nested structures if not found at top level
                                if results[benchmark_name] is None and "metrics" in benchmark_data:
                                    results[benchmark_name] = benchmark_data["metrics"].get("percentile", None)
                                    results[f"{benchmark_name}_rating"] = benchmark_data["metrics"].get("rating", None)
                            else:
                                # Define a list of common locations to check for the metric
                                found_metric = False
                                
                                # 1. Direct key access
                                if primary_metric_key in benchmark_data:
                                    results[benchmark_name] = benchmark_data[primary_metric_key]
                                    found_metric = True
                                # 2. Check if metrics is a dict containing our key
                                elif "metrics" in benchmark_data and isinstance(benchmark_data["metrics"], dict):
                                    if primary_metric_key in benchmark_data["metrics"]:
                                        results[benchmark_name] = benchmark_data["metrics"][primary_metric_key]
                                        found_metric = True
                                # 3. Check if results contains our key
                                elif "results" in benchmark_data and isinstance(benchmark_data["results"], dict):
                                    if primary_metric_key in benchmark_data["results"]:
                                        results[benchmark_name] = benchmark_data["results"][primary_metric_key]
                                        found_metric = True
                                # 4. Check for "score" key if primary_metric_key not found
                                elif "score" in benchmark_data:
                                    results[benchmark_name] = benchmark_data["score"]
                                    found_metric = True
                                elif "metrics" in benchmark_data and isinstance(benchmark_data["metrics"], dict) and "score" in benchmark_data["metrics"]:
                                    results[benchmark_name] = benchmark_data["metrics"]["score"]
                                    found_metric = True
                                # 5. Look for any key containing our metric name
                                if not found_metric:
                                    metric_keys = [k for k in benchmark_data.keys() 
                                                if primary_metric_key.lower() in k.lower()]
                                    if metric_keys:
                                        results[benchmark_name] = benchmark_data[metric_keys[0]]
                                        found_metric = True
                                    else:
                                        # Last resort: look for any key that might be a metric
                                        potential_metric_keys = ["accuracy", "score", "f1", "exact_match", "pass_rate", "success_rate"]
                                        for key in potential_metric_keys:
                                            if key in benchmark_data:
                                                results[benchmark_name] = benchmark_data[key]
                                                found_metric = True
                                                print(f"Note: Using '{key}' as metric for {benchmark_name} instead of '{primary_metric_key}'")
                                                break
                                
                                if not found_metric:
                                    print(f"Warning: Could not find metric '{primary_metric_key}' for benchmark {benchmark_name}")
                                    print(f"Available keys: {list(benchmark_data.keys())}")
                                    if "metrics" in benchmark_data:
                                        print(f"Available metrics: {list(benchmark_data['metrics'].keys()) if isinstance(benchmark_data['metrics'], dict) else benchmark_data['metrics']}")
                                    results[benchmark_name] = None
                                
                                # Extract additional metrics if present
                                if "metrics" in benchmark_data and isinstance(benchmark_data["metrics"], dict):
                                    for metric_name, value in benchmark_data["metrics"].items():
                                        if metric_name != primary_metric_key:
                                            results[f"{benchmark_name}_{metric_name}"] = value
                    except json.JSONDecodeError:
                        print(f"Error: Could not parse JSON from {benchmark_results_path}")
                        results[benchmark_name] = None
                    except Exception as e:
                        print(f"Error processing benchmark results for {benchmark_name}: {e}")
                        results[benchmark_name] = None
                else:
                    results[benchmark_name] = None
        except subprocess.TimeoutExpired:
            print(f"{display_name} evaluation timed out after 2 hours")
            results[benchmark_name] = None
        except Exception as e:
            print(f"Error evaluating {display_name}: {e}")
            results[benchmark_name] = None

    # 4. Combine all results and save to file
    combined_results_path = os.path.join(results_dir, "combined_results.json")
    with open(combined_results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Evaluation completed for experiment: {experiment_name}")
    print(f"Results saved to: {combined_results_path}")

    return results


def generate_experiment_combinations(selected_axes: List[str] = None) -> List[Tuple[str, Dict]]:
    """
    Generate all experiment configurations.

    Args:
        selected_axes: List of ablation types to include (default: all types)

    Returns:
        List of tuples (ablation_type, experiment_config)
    """
    if selected_axes is None:
        selected_axes = list(ABLATION_CONFIGS.keys())

    combinations = []
    
    # For the new structure, each ablation type has its own list of experiments
    for ablation_type in selected_axes:
        if ablation_type in ABLATION_CONFIGS:
            for experiment in ABLATION_CONFIGS[ablation_type]:
                combinations.append((ablation_type, experiment))

    return combinations


def ensure_file_exists(file_path, error_message=None):
    """Check if a file exists and print a helpful message if it doesn't."""
    if file_path and not os.path.exists(file_path):
        msg = error_message or f"File not found: {file_path}"
        print(f"WARNING: {msg}")
        return False
    return True

def add_custom_dataset_mixes(ablation_configs, custom_datasets_path=None):
    """
    Add custom dataset mixes from an external configuration file.

    Args:
        ablation_configs: The current ablation configurations
        custom_datasets_path: Path to custom dataset configuration file

    Returns:
        Updated ablation configurations
    """
    if custom_datasets_path is None or not os.path.exists(custom_datasets_path):
        return ablation_configs

    try:
        # Load custom dataset configuration
        with open(custom_datasets_path, "r") as f:
            custom_config = yaml.safe_load(f)

        # Check if custom_config contains data_mixes
        if not custom_config or "data_mixes" not in custom_config:
            print(f"No data_mixes found in {custom_datasets_path}")
            return ablation_configs

        # Add custom dataset mixes to ablation configs
        custom_mixes = []
        for mix_name, mix_config in custom_config["data_mixes"].items():
            if "datasets" not in mix_config:
                print(f"No datasets found in mix {mix_name}")
                continue

            # Create mix configuration
            custom_mix = {
                "name": mix_config.get("name", mix_name),
                "datasets": mix_config["datasets"],
            }

            custom_mixes.append(custom_mix)

        # Replace or append the custom mixes
        if custom_mixes:
            # Get existing mix names
            existing_mix_names = {mix["name"] for mix in ablation_configs["data_mix"]}

            # Add new mixes or replace existing ones
            for mix in custom_mixes:
                if mix["name"] in existing_mix_names:
                    # Replace existing mix
                    for i, existing_mix in enumerate(ablation_configs["data_mix"]):
                        if existing_mix["name"] == mix["name"]:
                            ablation_configs["data_mix"][i] = mix
                            break
                else:
                    # Add new mix
                    ablation_configs["data_mix"].append(mix)

            print(
                f"Added {len(custom_mixes)} custom dataset mixes from {custom_datasets_path}"
            )

        return ablation_configs

    except Exception as e:
        print(f"Error loading custom dataset mixes: {e}")
        return ablation_configs


def summarize_results(results_list: List[Dict], experiment_dir: str) -> None:
    """
    Summarize and visualize ablation results according to paper's hypotheses.

    Args:
        results_list: List of results from experiments
        experiment_dir: Path to experiment directory
    """
    # Create a DataFrame from results
    df = pd.DataFrame(results_list)

    # Save results to CSV
    csv_path = os.path.join(experiment_dir, "ablation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")

    # Generate visualizations
    viz_dir = os.path.join(experiment_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Group results by ablation type
    ablation_groups = df.groupby('ablation_type') if 'ablation_type' in df.columns else {}

    # 1. Ablation 1: Model-Capacity Sweep (H3: PPL ∝ P_cap^-λ)
    if 'capacity_sweep' in ablation_groups:
        capacity_df = ablation_groups.get_group('capacity_sweep')
        
        plt.figure(figsize=(12, 8))
        plt.scatter(capacity_df['total_params'], capacity_df['perplexity'], 
                   s=150, alpha=0.7, label='Data')
        
        # Fit power law PPL ∝ P_cap^-λ
        non_inf_mask = capacity_df['perplexity'] != float('inf')
        if non_inf_mask.sum() > 2:
            x = np.log10(capacity_df.loc[non_inf_mask, 'total_params'])
            y = np.log10(capacity_df.loc[non_inf_mask, 'perplexity'])
            z = np.polyfit(x, y, 1)
            lambda_val = -z[0]
            
            # Plot fit
            x_fit = np.logspace(np.log10(capacity_df['total_params'].min()), 
                               np.log10(capacity_df['total_params'].max()), 100)
            y_fit = 10**(z[1]) * x_fit**z[0]
            plt.plot(x_fit, y_fit, 'r--', label=f'Fit: λ={lambda_val:.3f}')
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Total Parameters (P_cap)')
        plt.ylabel('Perplexity')
        plt.title('Ablation 1: Model-Capacity Sweep (H3: PPL ∝ P_cap^-λ)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(viz_dir, 'ablation1_capacity_sweep.png'))
        plt.close()

    # 2. Ablation 2: Expert-Granularity Sweep (H1: TPS ∝ s_exp^-1)
    if 'granularity_sweep' in ablation_groups:
        gran_df = ablation_groups.get_group('granularity_sweep')
        
        # Calculate expert size in bytes
        gran_df['expert_size_bytes'] = gran_df['expert_size_kb'] * 1024
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # H1: Throughput vs expert size (would need TPS data)
        # For now, plot active params vs expert size
        ax1.scatter(gran_df['expert_size_bytes'], gran_df['active_params'], 
                   s=150, alpha=0.7)
        ax1.set_xscale('log')
        ax1.set_xlabel('Expert Size (bytes)')
        ax1.set_ylabel('Active Parameters')
        ax1.set_title('Active Parameters vs Expert Size')
        ax1.grid(True, alpha=0.3)
        
        # H2: PPL vs active params
        ax2.scatter(gran_df['active_params'], gran_df['perplexity'], 
                   s=150, alpha=0.7, label='Data')
        
        # Fit power law PPL ∝ P_act^-δ
        non_inf_mask = gran_df['perplexity'] != float('inf')
        if non_inf_mask.sum() > 2:
            x = np.log10(gran_df.loc[non_inf_mask, 'active_params'])
            y = np.log10(gran_df.loc[non_inf_mask, 'perplexity'])
            z = np.polyfit(x, y, 1)
            delta_val = -z[0]
            
            x_fit = np.logspace(np.log10(gran_df['active_params'].min()), 
                               np.log10(gran_df['active_params'].max()), 100)
            y_fit = 10**(z[1]) * x_fit**z[0]
            ax2.plot(x_fit, y_fit, 'r--', label=f'Fit: δ={delta_val:.3f}')
        
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('Active Parameters (P_act)')
        ax2.set_ylabel('Perplexity')
        ax2.set_title('H2: PPL ∝ P_act^-δ')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Ablation 2: Expert-Granularity Sweep')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'ablation2_granularity_sweep.png'))
        plt.close()

    # 3. Ablation 3: Compute-Budget Trade-off
    if 'compute_tradeoff' in ablation_groups:
        compute_df = ablation_groups.get_group('compute_tradeoff')
        
        plt.figure(figsize=(12, 8))
        
        # Plot perplexity vs model size, with bubble size = training tokens
        sizes = compute_df['training_tokens'] / 1e9  # Scale to billions
        scatter = plt.scatter(compute_df['target_params'], compute_df['perplexity'], 
                            s=sizes*10, alpha=0.6, c=compute_df['training_tokens'],
                            cmap='viridis', edgecolors='black', linewidth=1)
        
        # Add labels for each point
        for idx, row in compute_df.iterrows():
            plt.annotate(f"{row['training_tokens']/1e9:.0f}B tokens", 
                        (row['target_params'], row['perplexity']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        plt.xscale('log')
        plt.xlabel('Model Parameters')
        plt.ylabel('Perplexity')
        plt.title('Ablation 3: Compute-Budget Trade-off (Fixed Compute = 3.0e24 FLOPs)')
        cbar = plt.colorbar(scatter)
        cbar.set_label('Training Tokens')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(viz_dir, 'ablation3_compute_tradeoff.png'))
        plt.close()

    # 4. Ablations 4 & 5: Activation Sweeps
    for ablation_name, title_suffix in [('activation_4b', '4B Parameters'), 
                                        ('activation_256b', '256B Parameters')]:
        if ablation_name in ablation_groups:
            act_df = ablation_groups.get_group(ablation_name)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
            
            # Perplexity vs activation percentage
            ax1.scatter(act_df['activation_percentage'], act_df['perplexity'], 
                       s=150, alpha=0.7)
            ax1.set_xlabel('Activation Percentage (%)')
            ax1.set_ylabel('Perplexity')
            ax1.set_title(f'Perplexity vs Activation % ({title_suffix})')
            ax1.grid(True, alpha=0.3)
            
            # Calculate actual active params for plotting
            if 'num_experts' in act_df.columns:
                act_df['calc_active_params'] = (act_df['activation_percentage'] / 100.0 * 
                                               act_df['num_experts'] * EXPERT_SIZE_BYTES)
            
            # Perplexity vs active parameters
            if 'active_params' in act_df.columns:
                ax2.scatter(act_df['active_params'], act_df['perplexity'], 
                           s=150, alpha=0.7, label='Data')
                
                # Fit power law
                non_inf_mask = act_df['perplexity'] != float('inf')
                if non_inf_mask.sum() > 2:
                    x = np.log10(act_df.loc[non_inf_mask, 'active_params'])
                    y = np.log10(act_df.loc[non_inf_mask, 'perplexity'])
                    z = np.polyfit(x, y, 1)
                    delta_val = -z[0]
                    
                    x_fit = np.logspace(np.log10(act_df['active_params'].min()), 
                                       np.log10(act_df['active_params'].max()), 100)
                    y_fit = 10**(z[1]) * x_fit**z[0]
                    ax2.plot(x_fit, y_fit, 'r--', label=f'Fit: δ={delta_val:.3f}')
                
                ax2.set_xscale('log')
                ax2.set_yscale('log')
                ax2.set_xlabel('Active Parameters')
                ax2.set_ylabel('Perplexity')
                ax2.set_title(f'PPL vs Active Parameters ({title_suffix})')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            plt.suptitle(f'Ablation {4 if ablation_name == "activation_4b" else 5}: Activation Sweep')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f'{ablation_name}_sweep.png'))
            plt.close()

    # Summary plot: All ablations on one figure
    plt.figure(figsize=(16, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(df['ablation_type'].unique())))
    
    for i, (ablation_type, group_df) in enumerate(ablation_groups):
        non_inf_mask = group_df['perplexity'] != float('inf')
        plt.scatter(group_df.loc[non_inf_mask, 'active_params'], 
                   group_df.loc[non_inf_mask, 'perplexity'],
                   label=ablation_type.replace('_', ' ').title(),
                   alpha=0.7, s=100, color=colors[i])
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Active Parameters')
    plt.ylabel('Perplexity')
    plt.title('All Ablations: Perplexity vs Active Parameters')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'all_ablations_summary.png'))
    plt.close()
    
    # Print summary statistics
    print("\n=== Ablation Results Summary ===")
    for ablation_type, group_df in ablation_groups:
        print(f"\n{ablation_type.replace('_', ' ').title()}:")
        print(f"  Experiments: {len(group_df)}")
        print(f"  Best perplexity: {group_df['perplexity'].min():.4f}")
        print(f"  Avg perplexity: {group_df['perplexity'].mean():.4f}")


def calculate_normalized_scores(
    df: pd.DataFrame, benchmark_names: List[str]
) -> pd.DataFrame:
    """Calculate normalized scores across benchmarks and create a combined score."""
    if df.empty:
        return pd.DataFrame()

    # Create a copy to avoid modifying the original
    norm_df = df.copy()

    # Ensure we have at least some benchmark data
    valid_benchmarks = []
    for benchmark in benchmark_names:
        if benchmark in norm_df.columns and not norm_df[benchmark].isna().all():
            valid_benchmarks.append(benchmark)

    if not valid_benchmarks:
        return pd.DataFrame()

    # Normalize each benchmark to a 0-1 scale
    for benchmark in valid_benchmarks:
        # Skip if all values are the same
        if norm_df[benchmark].nunique() <= 1:
            continue

        if benchmark == "perplexity":
            # Lower is better, so normalize in reverse
            min_val = norm_df[benchmark].min()
            max_val = norm_df[benchmark].max()
            norm_df[f"{benchmark}_norm"] = 1 - (
                (norm_df[benchmark] - min_val) / (max_val - min_val)
            )
        else:
            # Higher is better
            min_val = norm_df[benchmark].min()
            max_val = norm_df[benchmark].max()
            norm_df[f"{benchmark}_norm"] = (norm_df[benchmark] - min_val) / (
                max_val - min_val
            )

    # Calculate combined score (average of normalized scores)
    norm_columns = [
        f"{b}_norm" for b in valid_benchmarks if f"{b}_norm" in norm_df.columns
    ]

    if norm_columns:
        norm_df["combined_score"] = norm_df[norm_columns].mean(axis=1)

    return norm_df


def run_ablation_study(
    base_config_path: str,
    output_dir: str,
    selected_axes: List[str] = None,
    gpu_count: int = 4,
    tokens_limit: int = 10_000_000_000,  # 10B tokens
    resume: bool = False,
    custom_datasets_path: str = None,
):
    """
    Run a comprehensive ablation study across multiple configuration axes.

    Args:
        base_config_path: Path to base configuration file
        output_dir: Directory to store results
        selected_axes: List of axes to ablate (default: all axes)
        gpu_count: Number of GPUs to use for each experiment
        tokens_limit: Limit each training run to this many tokens
        resume: Whether to resume a previous ablation study
        custom_datasets_path: Path to custom dataset configuration
    """
    # Create experiment directory
    if resume and os.path.exists(output_dir):
        # Find most recent experiment directory
        experiment_dirs = [
            d for d in os.listdir(output_dir) if d.startswith("ablation_run_")
        ]
        if not experiment_dirs:
            print("No previous experiment to resume. Starting a new one.")
            experiment_dir = create_experiment_folder(output_dir)
        else:
            experiment_dirs.sort(reverse=True)
            experiment_dir = os.path.join(output_dir, experiment_dirs[0])
            print(f"Resuming from experiment directory: {experiment_dir}")
    else:
        experiment_dir = create_experiment_folder(output_dir)

    # Load base configuration
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    # Add custom dataset mixes if provided
    global ABLATION_CONFIGS
    ABLATION_CONFIGS = add_custom_dataset_mixes(ABLATION_CONFIGS, custom_datasets_path)

    # Generate all experiment combinations
    experiments = generate_experiment_combinations(selected_axes)
    print(f"Generated {len(experiments)} experiment configurations")

    # Setup tracking
    completed_experiments = set()
    results_list = []

    # Check for existing results if resuming
    if resume:
        results_path = os.path.join(experiment_dir, "ablation_results.csv")
        if os.path.exists(results_path):
            existing_results = pd.read_csv(results_path)
            for _, row in existing_results.iterrows():
                completed_experiments.add(row["experiment_name"])
                results_list.append(row.to_dict())
            print(f"Loaded {len(results_list)} existing results")

    # Create a record of all experiments
    experiments_log_path = os.path.join(experiment_dir, "experiments.json")
    with open(experiments_log_path, "w") as f:
        json.dump(experiments, f, indent=2)

    # Run each experiment
    for i, (ablation_type, experiment) in enumerate(experiments):
        # Generate experiment name
        experiment_name = f"{ablation_type}_{experiment['name']}"

        # Skip if already completed
        if experiment_name in completed_experiments:
            print(f"Skipping already completed experiment: {experiment_name}")
            continue

        print(f"\n[{i+1}/{len(experiments)}] Running experiment: {experiment_name}")

        # Generate configuration for this experiment
        config_path, experiment_name = generate_config(
            base_config, experiment, experiment_dir, ablation_type, gpu_count
        )

        # Run training
        start_time = time.time()
        log_path, checkpoint_path = run_training(
            config_path, experiment_name, experiment_dir, gpu_count, tokens_limit
        )

        # Calculate training time
        training_time = (time.time() - start_time) / 3600  # Convert to hours

        if checkpoint_path is None:
            print(f"Failed to train model for experiment: {experiment_name}")
            continue

        # Evaluate the checkpoint
        eval_results = evaluate_checkpoint(
            checkpoint_path, config_path, experiment_name, experiment_dir
        )

        if eval_results is None:
            print(f"Failed to evaluate checkpoint for experiment: {experiment_name}")
            continue

        # Load the generated config to calculate params
        with open(config_path, "r") as f:
            experiment_config = yaml.safe_load(f)
        
        # Calculate total and active params for this experiment
        total_params = calculate_total_params(experiment_config)
        active_params = calculate_active_params(experiment_config)

        # Record results
        result = {
            "experiment_name": experiment_name,
            "ablation_type": ablation_type,
            "perplexity": eval_results.get("perplexity", float("inf")),
            "total_params": total_params,
            "active_params": active_params,
            "active_params_ratio": active_params / max(total_params, 1),
            "training_time": training_time,
        }

        # Add experiment-specific metadata
        if ablation_type == "capacity_sweep":
            result["target_params"] = experiment.get("target_params", 0)
            result["num_experts"] = experiment.get("num_experts", 0)
            result["is_dense"] = not experiment.get("use_peer", True)
        elif ablation_type == "granularity_sweep":
            result["expert_size_kb"] = experiment.get("expert_size_kb", 0)
            result["num_experts"] = experiment.get("num_experts", 0)
            result["num_experts_per_tok"] = experiment.get("num_experts_per_tok", 0)
        elif ablation_type == "compute_tradeoff":
            result["target_params"] = experiment.get("target_params", 0)
            result["training_tokens"] = experiment.get("training_tokens", 0)
        elif ablation_type in ["activation_4b", "activation_256b"]:
            result["activation_percentage"] = experiment.get("activation_percentage", 0)
            result["num_experts"] = experiment.get("num_experts", 0)

        # Add benchmark results directly from eval_results
        for key, value in eval_results.items():
            if key not in ["experiment_name", "checkpoint_path", "config_path", "timestamp", "total_params", "active_params"]:
                result[key] = value


        # Add result to list
        results_list.append(result)
        completed_experiments.add(experiment_name)

        # Update results CSV after each experiment
        df = pd.DataFrame(results_list)
        df.to_csv(os.path.join(experiment_dir, "ablation_results.csv"), index=False)

        print(f"Completed experiment {i+1}/{len(experiments)}: {experiment_name}")
        print(f"  Perplexity: {result['perplexity']:.4f}")
        print(f"  Training time: {training_time:.2f} hours")

    # Summarize all results
    summarize_results(results_list, experiment_dir)
    print("\nAblation study completed!")
