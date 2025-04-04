#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Meta-script for running ablation studies on the foundation model architecture.

This script systematically executes training runs with different configurations,
evaluates performance, and collects results. It tests combinations across four axes:
1. Initial data mix
2. Number and size of experts
3. Dimensionality of cartesian product for expert selection
4. Number of expert selection heads

Usage:
    python run_ablations.py --base-config config.yaml --output-dir ablation_results
    
    # Run a specific subset of experiments
    python run_ablations.py --base-config config.yaml --output-dir ablation_results \
                           --axes data_mix expert_count
                           
    # Resume from previous run
    python run_ablations.py --resume --output-dir ablation_results
"""

import os
import sys
import time
import json
import yaml
import copy
import random
import argparse
import subprocess
import itertools
import shutil
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# Define ablation configuration spaces for each axis
ABLATION_CONFIGS = {
    # 1. Initial data mix ablations
    "data_mix": [
        {
            "name": "baseline",
            "datasets": [
                {"name": "pile", "path": "EleutherAI/pile", "split": "train", "streaming": True, "weight": 0.7, "text_field": "text"},
                {"name": "c4", "path": "allenai/c4", "split": "train", "streaming": True, "weight": 0.2, "text_field": "text"},
                {"name": "code_alpaca", "path": "sahil2801/CodeAlpaca-20k", "split": "train", "streaming": True, "weight": 0.1, "text_field": "instruction"}
            ]
        },
        {
            "name": "code_heavy",
            "datasets": [
                {"name": "pile", "path": "EleutherAI/pile", "split": "train", "streaming": True, "weight": 0.4, "text_field": "text"},
                {"name": "c4", "path": "allenai/c4", "split": "train", "streaming": True, "weight": 0.2, "text_field": "text"},
                {"name": "code_alpaca", "path": "sahil2801/CodeAlpaca-20k", "split": "train", "streaming": True, "weight": 0.2, "text_field": "instruction"},
                {"name": "starcoder", "path": "bigcode/starcoderdata", "split": "train", "streaming": True, "weight": 0.2, "text_field": "text"}
            ]
        },
        {
            "name": "web_heavy",
            "datasets": [
                {"name": "pile", "path": "EleutherAI/pile", "split": "train", "streaming": True, "weight": 0.4, "text_field": "text"},
                {"name": "c4", "path": "allenai/c4", "split": "train", "streaming": True, "weight": 0.5, "text_field": "text"},
                {"name": "code_alpaca", "path": "sahil2801/CodeAlpaca-20k", "split": "train", "streaming": True, "weight": 0.1, "text_field": "instruction"}
            ]
        },
        {
            "name": "balanced",
            "datasets": [
                {"name": "pile", "path": "EleutherAI/pile", "split": "train", "streaming": True, "weight": 0.33, "text_field": "text"},
                {"name": "c4", "path": "allenai/c4", "split": "train", "streaming": True, "weight": 0.33, "text_field": "text"},
                {"name": "code_alpaca", "path": "sahil2801/CodeAlpaca-20k", "split": "train", "streaming": True, "weight": 0.34, "text_field": "instruction"}
            ]
        },
        {
            "name": "domain_specific",
            "datasets": [
                {"name": "pile", "path": "EleutherAI/pile", "split": "train", "streaming": True, "weight": 0.5, "text_field": "text"},
                {"name": "c4", "path": "allenai/c4", "split": "train", "streaming": True, "weight": 0.15, "text_field": "text"},
                {"name": "code_alpaca", "path": "sahil2801/CodeAlpaca-20k", "split": "train", "streaming": True, "weight": 0.15, "text_field": "instruction"},
                {"name": "math", "path": "meta-math/MetaMathQA", "split": "train", "streaming": True, "weight": 0.2, "text_field": "text"}
            ]
        },
        # Add your custom dataset mixes here
        {
            "name": "custom_mix_1",
            "datasets": [
                # Replace with your custom datasets
                {"name": "custom_dataset_1", "path": "/path/to/your/dataset_1", "split": "train", "streaming": True, "weight": 0.6, "text_field": "text"},
                {"name": "custom_dataset_2", "path": "/path/to/your/dataset_2", "split": "train", "streaming": True, "weight": 0.3, "text_field": "text"},
                {"name": "code_alpaca", "path": "sahil2801/CodeAlpaca-20k", "split": "train", "streaming": True, "weight": 0.1, "text_field": "instruction"}
            ]
        },
        {
            "name": "custom_mix_2",
            "datasets": [
                # Another custom dataset mix
                {"name": "custom_dataset_3", "path": "/path/to/your/dataset_3", "split": "train", "streaming": True, "weight": 0.5, "text_field": "text"},
                {"name": "custom_dataset_4", "path": "/path/to/your/dataset_4", "split": "train", "streaming": True, "weight": 0.3, "text_field": "text"},
                {"name": "custom_dataset_5", "path": "/path/to/your/dataset_5", "split": "train", "streaming": True, "weight": 0.2, "text_field": "content"}
            ]
        }
    ],
    
    # 2. Number of experts and size of experts
    "expert_count": [
        {
            "name": "many_tiny",
            "num_experts": 1048576,  # 1024 x 1024
            "product_key_dim": [1024, 1024],
            "expert_hidden_size": 1
        },
        {
            "name": "medium",
            "num_experts": 262144,  # 512 x 512
            "product_key_dim": [512, 512],
            "expert_hidden_size": 4
        },
        {
            "name": "fewer_larger",
            "num_experts": 65536,  # 256 x 256
            "product_key_dim": [256, 256],
            "expert_hidden_size": 16
        },
        {
            "name": "traditional_moe",
            "num_experts": 16384,  # 128 x 128
            "product_key_dim": [128, 128],
            "expert_hidden_size": 64
        }
    ],
    
    # 3. Dimensionality of cartesian product
    "cartesian_dims": [
        {
            "name": "2d_balanced",
            "num_experts": 1048576,
            "product_key_dim": [1024, 1024]
        },
        {
            "name": "3d_balanced",
            "num_experts": 1000000,
            "product_key_dim": [100, 100, 100]
        },
        {
            "name": "4d_balanced",
            "num_experts": 1048576,
            "product_key_dim": [32, 32, 32, 32]
        },
        {
            "name": "asymmetric_2d",
            "num_experts": 1048576,
            "product_key_dim": [4096, 256]
        },
        {
            "name": "very_asymmetric",
            "num_experts": 1048576,
            "product_key_dim": [65536, 16]
        }
    ],
    
    # 4. Number of expert selection heads
    "selection_heads": [
        {
            "name": "few_heads_many_experts",
            "num_heads": 4,
            "num_experts_per_tok": 32
        },
        {
            "name": "balanced",
            "num_heads": 8,
            "num_experts_per_tok": 16
        },
        {
            "name": "many_heads_few_experts",
            "num_heads": 16,
            "num_experts_per_tok": 8
        },
        {
            "name": "many_heads_very_few_experts",
            "num_heads": 32,
            "num_experts_per_tok": 4
        }
    ]
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


def generate_config(base_config: Dict, experiment: Dict, experiment_dir: str) -> str:
    """
    Generate a configuration file for a specific experiment.
    
    Args:
        base_config: Base configuration dictionary
        experiment: Experiment-specific configuration dictionary
        experiment_dir: Path to experiment directory
        
    Returns:
        Path to the generated configuration file
    """
    # Create a deep copy of the base configuration
    config = copy.deepcopy(base_config)
    
    # Apply experiment-specific configurations
    experiment_name_parts = []
    
    # 1. Apply data mix configuration if present
    if "data_mix" in experiment:
        data_mix = experiment["data_mix"]
        config["dataset_config"]["datasets"] = data_mix["datasets"]
        experiment_name_parts.append(f"data-{data_mix['name']}")
    
    # 2. Apply expert count configuration if present
    if "expert_count" in experiment:
        expert_count = experiment["expert_count"]
        config["model_config"]["peer_config"]["num_experts"] = expert_count["num_experts"]
        config["model_config"]["peer_config"]["product_key_dim"] = expert_count["product_key_dim"]
        config["model_config"]["peer_config"]["expert_hidden_size"] = expert_count["expert_hidden_size"]
        experiment_name_parts.append(f"experts-{expert_count['name']}")
    
    # 3. Apply cartesian dimensions configuration if present
    if "cartesian_dims" in experiment:
        cartesian_dims = experiment["cartesian_dims"]
        config["model_config"]["peer_config"]["num_experts"] = cartesian_dims["num_experts"]
        config["model_config"]["peer_config"]["product_key_dim"] = cartesian_dims["product_key_dim"]
        experiment_name_parts.append(f"dims-{cartesian_dims['name']}")
    
    # 4. Apply selection heads configuration if present
    if "selection_heads" in experiment:
        selection_heads = experiment["selection_heads"]
        config["model_config"]["peer_config"]["num_heads"] = selection_heads["num_heads"]
        config["model_config"]["peer_config"]["num_experts_per_tok"] = selection_heads["num_experts_per_tok"]
        experiment_name_parts.append(f"heads-{selection_heads['name']}")
    
    # Set unique experiment name
    experiment_name = "_".join(experiment_name_parts)
    
    # Update configuration with unique experiment name and output directory
    config["wandb_config"]["name"] = f"ablation_{experiment_name}"
    config["train_config"]["output_dir"] = os.path.join(experiment_dir, "checkpoints", experiment_name)
    
    # Save configuration to file
    config_path = os.path.join(experiment_dir, "configs", f"{experiment_name}.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config_path, experiment_name


def run_training(config_path: str, experiment_name: str, experiment_dir: str, 
                gpu_count: int, tokens_limit: int = None) -> Tuple[str, str]:
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
    
    # Modify the config if token limit is specified
    if tokens_limit is not None:
        # Load the config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Calculate steps based on batch size, sequence length, and token limit
        batch_size = config["train_config"]["per_device_train_batch_size"] * gpu_count
        seq_length = config["dataset_config"]["max_seq_length"]
        tokens_per_step = batch_size * seq_length
        steps = tokens_limit // tokens_per_step
        
        # Update config with the calculated number of steps
        # We'll use a single epoch with the calculated number of steps
        config["train_config"]["num_train_epochs"] = 1
        config["train_config"]["max_steps"] = steps
        
        # Save the modified config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    # Create command to run training
    cmd = [
        "bash", "script/train.sh",
        "--config", config_path,
        "--gpus-per-node", str(gpu_count)
    ]
    
    # Run the training process
    print(f"Starting training for experiment: {experiment_name}")
    print(f"Command: {' '.join(cmd)}")
    
    with open(log_path, 'w') as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT
        )
        
        # Wait for process to complete
        process.wait()
    
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
        checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            # Sort by step number
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            latest_checkpoint = checkpoints[-1]
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint, "model.safetensors")
        else:
            print(f"No checkpoint found for experiment: {experiment_name}")
            return log_path, None
    
    print(f"Training completed for experiment: {experiment_name}")
    print(f"Checkpoint saved at: {checkpoint_path}")
    
    return log_path, checkpoint_path


def evaluate_checkpoint(checkpoint_path: str, config_path: str, experiment_name: str, 
                       experiment_dir: str) -> Dict:
    """
    Evaluate a trained checkpoint on multiple benchmarks.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to configuration file
        experiment_name: Unique name for this experiment
        experiment_dir: Path to experiment directory
        
    Returns:
        Dictionary of evaluation results
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
    
    cmd = [
        "python", "utils/eval/size.py",
        "--config", config_path,
        "--checkpoint", checkpoint_path,
        "--output", os.path.join(results_dir, "model_analysis.png"),
        "--json", analyzer_results_path
    ]
    
    print(f"Running model analysis for experiment: {experiment_name}")
    try:
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if process.returncode != 0:
            print(f"Model analysis failed: {process.stderr}")
        else:
            # Load analysis results if available
            if os.path.exists(analyzer_results_path):
                with open(analyzer_results_path, 'r') as f:
                    analyzer_data = json.load(f)
                    # Add parameter counts to results
                    results.update({
                        "total_params": analyzer_data.get("total_params", 0),
                        "active_params": analyzer_data.get("active_params", 0),
                        "active_params_ratio": analyzer_data.get("active_params", 0) / max(analyzer_data.get("total_params", 1), 1),
                    })
    except Exception as e:
        print(f"Error running model analysis: {e}")
    
    # 2. Run perplexity evaluation
    perplexity_results_path = os.path.join(results_dir, "perplexity.json")
    cmd = [
        "python", "utils/eval/perplexity.py",
        "--config", config_path,
        "--checkpoint", checkpoint_path,
        "--output", perplexity_results_path
    ]
    
    print(f"Evaluating perplexity for experiment: {experiment_name}")
    try:
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if process.returncode != 0:
            print(f"Perplexity evaluation failed: {process.stderr}")
            results["perplexity"] = float('inf')
        else:
            # Load perplexity results
            if os.path.exists(perplexity_results_path):
                with open(perplexity_results_path, 'r') as f:
                    perplexity_data = json.load(f)
                    results["perplexity"] = perplexity_data.get("perplexity", float('inf'))
            else:
                results["perplexity"] = float('inf')
    except Exception as e:
        print(f"Error evaluating perplexity: {e}")
        results["perplexity"] = float('inf')
    
    # 3. Run comprehensive benchmark evaluations
    # Define our suite of benchmarks
    benchmarks = [
        {
            "name": "aime_2024",
            "display_name": "AIME 2024",
            "script": "utils/eval/benchmark.py",
            "args": ["--benchmark", "aime_2024", "--metric", "pass@1"]
        },
        {
            "name": "codeforces",
            "display_name": "Codeforces",
            "script": "utils/eval/benchmark.py",
            "args": ["--benchmark", "codeforces", "--metric", "percentile"]
        },
        {
            "name": "gpqa_diamond",
            "display_name": "GPQA Diamond",
            "script": "utils/eval/benchmark.py",
            "args": ["--benchmark", "gpqa", "--subset", "diamond", "--metric", "pass@1"]
        },
        {
            "name": "math_500",
            "display_name": "MATH-500",
            "script": "utils/eval/benchmark.py",
            "args": ["--benchmark", "math", "--subset", "500", "--metric", "pass@1"]
        },
        {
            "name": "mmlu",
            "display_name": "MMLU",
            "script": "utils/eval/benchmark.py",
            "args": ["--benchmark", "mmlu", "--metric", "pass@1"]
        },
        {
            "name": "swe_bench",
            "display_name": "SWE-bench",
            "script": "utils/eval/benchmark.py",
            "args": ["--benchmark", "swe_bench", "--subset", "verified", "--metric", "resolved"]
        }
    ]
    
    # Run each benchmark
    for benchmark in benchmarks:
        benchmark_results_path = os.path.join(results_dir, f"{benchmark['name']}_results.json")
        
        cmd = [
            "python", benchmark["script"],
            "--config", config_path,
            "--checkpoint", checkpoint_path,
            "--output", benchmark_results_path
        ] + benchmark["args"]
        
        print(f"Evaluating {benchmark['display_name']} for experiment: {experiment_name}")
        try:
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=7200  # 2 hour timeout per benchmark
            )
            
            if process.returncode != 0:
                print(f"{benchmark['display_name']} evaluation failed: {process.stderr}")
                results[benchmark["name"]] = None
            else:
                # Load benchmark results
                if os.path.exists(benchmark_results_path):
                    with open(benchmark_results_path, 'r') as f:
                        benchmark_data = json.load(f)
                        
                        # Extract the primary metric from the benchmark results
                        if benchmark["name"] == "codeforces":
                            results[benchmark["name"]] = benchmark_data.get("percentile", None)
                            results[f"{benchmark['name']}_rating"] = benchmark_data.get("rating", None)
                        else:
                            # For other benchmarks, extract pass@1 or specified metric
                            metric_name = benchmark["args"][3]  # Based on the args pattern
                            results[benchmark["name"]] = benchmark_data.get(metric_name, None)
                else:
                    results[benchmark["name"]] = None
        except subprocess.TimeoutExpired:
            print(f"{benchmark['display_name']} evaluation timed out after 2 hours")
            results[benchmark["name"]] = None
        except Exception as e:
            print(f"Error evaluating {benchmark['display_name']}: {e}")
            results[benchmark["name"]] = None
    
    # 4. Combine all results and save to file
    combined_results_path = os.path.join(results_dir, "combined_results.json")
    with open(combined_results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation completed for experiment: {experiment_name}")
    print(f"Results saved to: {combined_results_path}")
    
    return results


def generate_experiment_combinations(
    selected_axes: List[str] = None
) -> List[Dict]:
    """
    Generate all combinations of experiment configurations.
    
    Args:
        selected_axes: List of axes to include in combinations (default: all axes)
        
    Returns:
        List of experiment configurations
    """
    if selected_axes is None:
        selected_axes = list(ABLATION_CONFIGS.keys())
    
    # Get configurations for selected axes
    selected_configs = {axis: ABLATION_CONFIGS[axis] for axis in selected_axes}
    
    # Generate combinations
    axes_names = list(selected_configs.keys())
    combinations = []
    
    # Get all combinations
    if len(axes_names) == 1:
        # Only one axis selected
        axis = axes_names[0]
        for config in selected_configs[axis]:
            combinations.append({axis: config})
    else:
        # Multiple axes selected, generate all combinations
        axis_values = [selected_configs[axis] for axis in axes_names]
        for combo in itertools.product(*axis_values):
            experiment = {}
            for i, axis in enumerate(axes_names):
                experiment[axis] = combo[i]
            combinations.append(experiment)
    
    return combinations


def summarize_results(results_list: List[Dict], experiment_dir: str) -> None:
    """
    Summarize and visualize ablation results.
    
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
    
    # Get benchmark names from the first result that has them
    benchmark_names = []
    for result in results_list:
        potential_benchmarks = [
            "aime_2024", "codeforces", "gpqa_diamond", 
            "math_500", "mmlu", "swe_bench"
        ]
        for benchmark in potential_benchmarks:
            if benchmark in result and result[benchmark] is not None:
                benchmark_names.append(benchmark)
        if benchmark_names:
            break
    
    # If no benchmarks found, use defaults
    if not benchmark_names:
        benchmark_names = ["perplexity"]
    
    # 1. Analyze by number of parameters vs. perplexity
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x="total_params", y="perplexity", hue="experiment_type", 
                   size="active_params_ratio", sizes=(100, 400), alpha=0.7)
    plt.xscale('log')
    plt.title("Model Size vs. Perplexity for Different Configurations")
    plt.xlabel("Total Parameters")
    plt.ylabel("Perplexity (lower is better)")
    plt.savefig(os.path.join(viz_dir, "params_vs_perplexity.png"))
    plt.close()
    
    # 2. Analyze by active parameters ratio vs. perplexity
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x="active_params_ratio", y="perplexity", 
                   hue="experiment_type", style="dims_config" if "dims_config" in df else None, 
                   s=150, alpha=0.8)
    plt.title("Expert Activation Ratio vs. Perplexity")
    plt.xlabel("Active Parameters Ratio")
    plt.ylabel("Perplexity (lower is better)")
    plt.savefig(os.path.join(viz_dir, "active_ratio_vs_perplexity.png"))
    plt.close()
    
    # 3. Create benchmark comparison plots
    for benchmark in benchmark_names:
        if benchmark == "perplexity":
            # Already handled above
            continue
            
        # Check if we have data for this benchmark
        if benchmark not in df.columns or df[benchmark].isna().all():
            continue
            
        # Create visualization for this benchmark
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=df, x="total_params", y=benchmark, hue="experiment_type", 
                       size="active_params_ratio", sizes=(100, 400), alpha=0.7)
        plt.xscale('log')
        
        # Different benchmarks have different interpretation (higher/lower better)
        ylabel_suffix = ""
        if benchmark in ["aime_2024", "codeforces", "gpqa_diamond", "math_500", "mmlu", "swe_bench"]:
            ylabel_suffix = " (higher is better)"
        
        plt.title(f"Model Size vs. {benchmark.replace('_', ' ').title()}")
        plt.xlabel("Total Parameters")
        plt.ylabel(f"{benchmark.replace('_', ' ').title()}{ylabel_suffix}")
        plt.savefig(os.path.join(viz_dir, f"params_vs_{benchmark}.png"))
        plt.close()
        
        # Create comparison by expert configuration if applicable
        if "expert_config" in df.columns:
            plt.figure(figsize=(15, 10))
            
            # Filter out NaN values
            plot_data = df[df[benchmark].notna()].copy()
            
            if len(plot_data) < 2:
                continue
                
            # Sort by benchmark score - different direction depending on metric
            if benchmark in ["aime_2024", "codeforces", "gpqa_diamond", "math_500", "mmlu", "swe_bench"]:
                # Higher is better
                plot_data = plot_data.sort_values(benchmark, ascending=False)
            else:
                # Lower is better (perplexity)
                plot_data = plot_data.sort_values(benchmark)
            
            # Create bar plot
            plt.figure(figsize=(14, 8))
            ax = sns.barplot(data=plot_data, x="experiment_name", y=benchmark)
            plt.xticks(rotation=90)
            plt.title(f"{benchmark.replace('_', ' ').title()} by Configuration")
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f"{benchmark}_by_config.png"))
            plt.close()
    
    # 4. Create radar chart for top configurations across all benchmarks
    if len(benchmark_names) > 1:
        create_benchmark_radar_chart(df, benchmark_names, viz_dir)
    
    # Generate a table of the top configurations for each benchmark
    benchmark_tables = {}
    
    # Include perplexity in benchmarks if available
    if "perplexity" in df.columns and not df["perplexity"].isna().all():
        benchmark_names = ["perplexity"] + benchmark_names
    
    for benchmark in benchmark_names:
        if benchmark not in df.columns or df[benchmark].isna().all():
            continue
            
        # Sort appropriately based on benchmark
        if benchmark == "perplexity":
            # Lower is better
            top_configs = df.sort_values(benchmark).head(5)
        else:
            # Higher is better
            top_configs = df.sort_values(benchmark, ascending=False).head(5)
        
        # Select relevant columns
        cols = ["experiment_name", benchmark, "total_params", "active_params", "training_time"]
        if "experiment_type" in df.columns:
            cols.append("experiment_type")
        
        benchmark_tables[benchmark] = top_configs[cols].copy()
    
    # Create a comprehensive report
    with open(os.path.join(experiment_dir, "ablation_report.md"), 'w') as f:
        f.write("# PEER Model Architecture Ablation Study Results\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary of experiments
        f.write(f"## Summary\n\n")
        f.write(f"Total experiments: {len(df)}\n")
        f.write(f"Axes tested: {', '.join(sorted(set(df['experiment_type'].str.split('_'))))}\n\n")
        
        # Top configurations for each benchmark
        f.write("## Top Performing Configurations by Benchmark\n\n")
        
        for benchmark, table in benchmark_tables.items():
            f.write(f"### {benchmark.replace('_', ' ').title()}\n\n")
            f.write(table.to_markdown(index=False))
            f.write("\n\n")
        
        # Overall best configurations
        f.write("## Overall Best Configurations\n\n")
        
        # Calculate normalized scores across benchmarks
        normalized_df = calculate_normalized_scores(df, benchmark_names)
        
        if not normalized_df.empty and "combined_score" in normalized_df.columns:
            top_overall = normalized_df.sort_values("combined_score", ascending=False).head(5)
            
            cols = ["experiment_name", "combined_score"] + benchmark_names
            available_cols = [col for col in cols if col in top_overall.columns]
            
            f.write(top_overall[available_cols].to_markdown(index=False))
            f.write("\n\n")
        
        # Statistics by experiment type
        f.write("## Summary Statistics\n\n")
        
        # Statistics by experiment type for each benchmark
        for benchmark in benchmark_names:
            if benchmark not in df.columns or df[benchmark].isna().all():
                continue
                
            f.write(f"### {benchmark.replace('_', ' ').title()} by Experiment Type\n\n")
            try:
                type_stats = df.groupby("experiment_type")[benchmark].agg(["mean", "min", "max", "std"]).dropna()
                f.write(type_stats.to_markdown())
                f.write("\n\n")
            except Exception as e:
                f.write(f"Error calculating statistics: {e}\n\n")
        
        # Add visualizations
        f.write("## Visualizations\n\n")
        
        f.write("### Parameters vs. Perplexity\n\n")
        f.write(f"![Params vs. Perplexity](visualizations/params_vs_perplexity.png)\n\n")
        
        f.write("### Active Parameters Ratio vs. Perplexity\n\n")
        f.write(f"![Active Ratio vs. Perplexity](visualizations/active_ratio_vs_perplexity.png)\n\n")
        
        # Add benchmark-specific visualizations
        for benchmark in benchmark_names:
            if benchmark == "perplexity":
                continue
                
            viz_path = f"visualizations/params_vs_{benchmark}.png"
            if os.path.exists(os.path.join(experiment_dir, viz_path)):
                f.write(f"### Parameters vs. {benchmark.replace('_', ' ').title()}\n\n")
                f.write(f"![Params vs. {benchmark}]({viz_path})\n\n")
            
            bar_path = f"visualizations/{benchmark}_by_config.png"
            if os.path.exists(os.path.join(experiment_dir, bar_path)):
                f.write(f"### {benchmark.replace('_', ' ').title()} by Configuration\n\n")
                f.write(f"![{benchmark} by Configuration]({bar_path})\n\n")
        
        # Add radar chart if it exists
        radar_path = "visualizations/benchmark_radar_chart.png"
        if os.path.exists(os.path.join(experiment_dir, radar_path)):
            f.write("### Radar Chart of Top Configurations\n\n")
            f.write(f"![Benchmark Radar Chart]({radar_path})\n\n")
        
        # Detailed results
        f.write("## Detailed Results\n\n")
        for result in results_list:
            f.write(f"### {result['experiment_name']}\n\n")
            
            # Write perplexity if available
            if "perplexity" in result:
                f.write(f"- Perplexity: {result.get('perplexity', 'N/A'):.4f}\n")
            
            # Write benchmark results
            for benchmark in benchmark_names:
                if benchmark != "perplexity" and benchmark in result and result[benchmark] is not None:
                    f.write(f"- {benchmark.replace('_', ' ').title()}: {result[benchmark]:.4f}\n")
            
            # Write other metadata
            f.write(f"- Total Parameters: {result.get('total_params', 'N/A'):,}\n")
            f.write(f"- Active Parameters: {result.get('active_params', 'N/A'):,}\n")
            f.write(f"- Training Time: {result.get('training_time', 'N/A'):.2f} hours\n")
            if "experiment_type" in result:
                f.write(f"- Experiment Type: {result['experiment_type']}\n")
            f.write("\n")
    
    print(f"Comprehensive report generated at: {os.path.join(experiment_dir, 'ablation_report.md')}")


def calculate_normalized_scores(df: pd.DataFrame, benchmark_names: List[str]) -> pd.DataFrame:
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
            norm_df[f"{benchmark}_norm"] = 1 - ((norm_df[benchmark] - min_val) / (max_val - min_val))
        else:
            # Higher is better
            min_val = norm_df[benchmark].min()
            max_val = norm_df[benchmark].max()
            norm_df[f"{benchmark}_norm"] = (norm_df[benchmark] - min_val) / (max_val - min_val)
    
    # Calculate combined score (average of normalized scores)
    norm_columns = [f"{b}_norm" for b in valid_benchmarks if f"{b}_norm" in norm_df.columns]
    
    if norm_columns:
        norm_df["combined_score"] = norm_df[norm_columns].mean(axis=1)
    
    return norm_df


def create_benchmark_radar_chart(df: pd.DataFrame, benchmark_names: List[str], viz_dir: str) -> None:
    """Create a radar chart comparing top configurations across benchmarks."""
    try:
        # First calculate normalized scores
        norm_df = calculate_normalized_scores(df, benchmark_names)
        
        if norm_df.empty or "combined_score" not in norm_df.columns:
            return
            
        # Get top 5 configurations by combined score
        top_configs = norm_df.sort_values("combined_score", ascending=False).head(5)
        
        # Get normalized benchmark columns
        norm_columns = [f"{b}_norm" for b in benchmark_names if f"{b}_norm" in norm_df.columns]
        
        if len(norm_columns) < 3:  # Need at least 3 axes for a radar chart
            return
            
        # Create the radar chart
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, polar=True)
        
        # Get the angle for each benchmark
        angles = np.linspace(0, 2*np.pi, len(norm_columns), endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon
        
        # Plot each configuration
        for _, config in top_configs.iterrows():
            values = [config[col] for col in norm_columns]
            values += values[:1]  # Close the polygon
            
            # Plot the values
            ax.plot(angles, values, linewidth=2, label=config["experiment_name"])
            ax.fill(angles, values, alpha=0.1)
        
        # Set the labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([col.split("_norm")[0].replace("_", " ").title() for col in norm_columns])
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title("Top Configurations Across Benchmarks")
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "benchmark_radar_chart.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating radar chart: {e}")
        import traceback
        traceback.print_exc()


def run_ablation_study(
    base_config_path: str,
    output_dir: str,
    selected_axes: List[str] = None,
    gpu_count: int = 4,
    tokens_limit: int = 10_000_000_000,  # 10B tokens
    resume: bool = False
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
    """
    # Create experiment directory
    if resume and os.path.exists(output_dir):
        # Find most recent experiment directory
        experiment_dirs = [d for d in os.listdir(output_dir) if d.startswith("ablation_run_")]
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
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
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
    with open(experiments_log_path, 'w') as f:
        json.dump(experiments, f, indent=2)
    
    # Run each experiment
    for i, experiment in enumerate(experiments):
        # Generate experiment name
        experiment_name_parts = []
        for axis, config in experiment.items():
            experiment_name_parts.append(f"{axis}-{config['name']}")
        experiment_name = "_".join(experiment_name_parts)
        
        # Skip if already completed
        if experiment_name in completed_experiments:
            print(f"Skipping already completed experiment: {experiment_name}")
            continue
        
        print(f"\n[{i+1}/{len(experiments)}] Running experiment: {experiment_name}")
        
        # Generate configuration for this experiment
        config_path, experiment_name = generate_config(base_config, experiment, experiment_dir)
        
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
        eval_results = evaluate_checkpoint(checkpoint_path, config_path, experiment_name, experiment_dir)
        
        if eval_results is None:
            print(f"Failed to evaluate checkpoint for experiment: {experiment_name}")
            continue
        
        # Record results
        result = {
            "experiment_name": experiment_name,
            "perplexity": eval_results.get("validation_perplexity", float('inf')),
            "total_params": eval_results.get("total_params", 0),
            "active_params": eval_results.get("active_params", 0),
            "active_params_ratio": eval_results.get("active_params", 0) / max(eval_results.get("total_params", 1), 1),
            "training_time": training_time,
            "experiment_type": "_".join([axis for axis in experiment.keys()]),
        }
        
        # Add specific configuration details
        for axis, config in experiment.items():
            if axis == "data_mix":
                result["data_config"] = config["name"]
            elif axis == "expert_count":
                result["expert_config"] = f"{config['num_experts']}x{config['expert_hidden_size']}"
            elif axis == "cartesian_dims":
                result["dims_config"] = "x".join(map(str, config["product_key_dim"]))
            elif axis == "selection_heads":
                result["selection_config"] = f"{config['num_heads']}x{config['num_experts_per_tok']}"
        
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


def add_custom_dataset_mixes(ablation_configs, custom_dataset_config_path=None):
    """
    Add custom dataset mixes from an external configuration file.
    
    Args:
        ablation_configs: The current ablation configurations
        custom_dataset_config_path: Path to custom dataset configuration file
        
    Returns:
        Updated ablation configurations
    """
    if custom_dataset_config_path is None or not os.path.exists(custom_dataset_config_path):
        return ablation_configs
    
    try:
        # Load custom dataset configuration
        with open(custom_dataset_config_path, 'r') as f:
            custom_config = yaml.safe_load(f)
        
        # Check if custom_config contains data_mixes
        if not custom_config or "data_mixes" not in custom_config:
            logger.warning(f"No data_mixes found in {custom_dataset_config_path}")
            return ablation_configs
        
        # Add custom dataset mixes to ablation configs
        custom_mixes = []
        for mix_name, mix_config in custom_config["data_mixes"].items():
            if "datasets" not in mix_config:
                logger.warning(f"No datasets found in mix {mix_name}")
                continue
            
            # Create mix configuration
            custom_mix = {
                "name": mix_config.get("name", mix_name),
                "datasets": mix_config["datasets"]
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
            
            logger.info(f"Added {len(custom_mixes)} custom dataset mixes from {custom_dataset_config_path}")
        
        return ablation_configs
    
    except Exception as e:
        logger.error(f"Error loading custom dataset mixes: {e}")
        return ablation_configs


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run ablation studies for PEER architecture")
    parser.add_argument("--base-config", type=str, default="config.yaml", 
                       help="Path to base configuration file")
    parser.add_argument("--output-dir", type=str, default="./ablation_results",
                       help="Directory to store results")
    parser.add_argument("--axes", type=str, nargs="+", choices=ABLATION_CONFIGS.keys(),
                       help="Specific axes to ablate (default: all)")
    parser.add_argument("--gpus", type=int, default=4,
                       help="Number of GPUs to use for each experiment")
    parser.add_argument("--tokens", type=int, default=10_000_000_000,
                       help="Token limit for each training run (default: 10B)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from a previous ablation study")
    parser.add_argument("--custom-datasets", type=str, default=None,
                       help="Path to custom dataset configuration file")
    
    args = parser.parse_args()
    
    # Load custom dataset mixes if provided
    if args.custom_datasets:
        ABLATION_CONFIGS = add_custom_dataset_mixes(ABLATION_CONFIGS, args.custom_datasets)
    
    # Run ablation study
    run_ablation_study(
        base_config_path=args.base_config,
        output_dir=args.output_dir,
        selected_axes=args.axes,
        gpu_count=args.gpus,
        tokens_limit=args.tokens,
        resume=args.resume
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAblation study interrupted by user")
    except Exception as e:
        import traceback
        print(f"Error during ablation study: {e}")
        traceback.print_exc()