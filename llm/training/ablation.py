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
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

# Define ablation configuration spaces for each axis
# Calculated expert_hidden_size for 228KB FP8 experts with hidden_size=768
EXPERT_HIDDEN_SIZE_CALC = 152

ABLATION_CONFIGS = {
    # 1. Initial data mix ablations
    "data_mix": [
        {
            "name": "web_heavy",
            "datasets": [
                {
                    "name": "fineweb",
                    "path": "HuggingFaceFW/fineweb",
                    "subset": "sample/10BT", # Specify subset for FineWeb
                    "split": "train",
                    "streaming": True,
                    "weight": 0.6,
                    "text_field": "text",
                },
                {
                    "name": "wikipedia",
                    "path": "wikimedia/wikipedia",
                    "subset": "20231101.en", # Specify Wikipedia dump version
                    "split": "train",
                    "streaming": True,
                    "weight": 0.2,
                    "text_field": "text",
                },
                {
                    "name": "github_code",
                    "path": "codeparrot/github-code",
                    "split": "train",
                    "streaming": True,
                    "weight": 0.1,
                    "text_field": "code",
                },
                {
                    "name": "openr1_math",
                    "path": "open-r1/OpenR1-Math-220k",
                    "split": "train",
                    "streaming": True,
                    "weight": 0.1,
                    "text_field": "problem",
                },
            ],
        },
        {
            "name": "code_heavy",
            "datasets": [
                {
                    "name": "fineweb",
                    "path": "HuggingFaceFW/fineweb",
                    "subset": "sample/10BT",
                    "split": "train",
                    "streaming": True,
                    "weight": 0.3,
                    "text_field": "text",
                },
                {
                    "name": "github_code",
                    "path": "codeparrot/github-code",
                    "split": "train",
                    "streaming": True,
                    "weight": 0.4,
                    "text_field": "code",
                },
                {
                    "name": "opencode_reasoning",
                    "path": "nvidia/OpenCodeReasoning",
                    "split": "train",
                    "streaming": True,
                    "weight": 0.2,
                    "text_field": "input",
                },
                 {
                    "name": "wikipedia",
                    "path": "wikimedia/wikipedia",
                    "subset": "20231101.en",
                    "split": "train",
                    "streaming": True,
                    "weight": 0.1,
                    "text_field": "text",
                },
            ],
        },
        {
            "name": "math_heavy",
            "datasets": [
                {
                    "name": "fineweb",
                    "path": "HuggingFaceFW/fineweb",
                    "name": "sample/10BT",
                    "split": "train",
                    "streaming": True,
                    "weight": 0.4,
                    "text_field": "text",
                },
                {
                    "name": "openr1_math",
                    "path": "open-r1/OpenR1-Math-220k",
                    "split": "train",
                    "streaming": True,
                    "weight": 0.4,
                    "text_field": "problem",
                },
                {
                    "name": "wikipedia",
                    "path": "wikimedia/wikipedia",
                    "name": "20231101.en",
                    "split": "train",
                    "streaming": True,
                    "weight": 0.1,
                    "text_field": "text",
                },
                {
                    "name": "github_code",
                    "path": "codeparrot/github-code",
                    "split": "train",
                    "streaming": True,
                    "weight": 0.1,
                    "text_field": "code",
                },
            ],
        },
        {
            "name": "balanced",
            "datasets": [
                {
                    "name": "fineweb",
                    "path": "HuggingFaceFW/fineweb",
                    "name": "sample/10BT",
                    "split": "train",
                    "streaming": True,
                    "weight": 0.4,
                    "text_field": "text",
                },
                {
                    "name": "github_code",
                    "path": "codeparrot/github-code",
                    "split": "train",
                    "streaming": True,
                    "weight": 0.25,
                    "text_field": "code",
                },
                {
                    "name": "openr1_math",
                    "path": "open-r1/OpenR1-Math-220k",
                    "split": "train",
                    "streaming": True,
                    "weight": 0.2,
                    "text_field": "problem",
                },
                {
                    "name": "wikipedia",
                    "path": "wikimedia/wikipedia",
                    "name": "20231101.en",
                    "split": "train",
                    "streaming": True,
                    "weight": 0.15,
                    "text_field": "text",
                },
            ],
        },
    ],
    # 2. Expert setup (combining count, dimensions, and size)
    "expert_setup": [
        {
            "name": "2d_1M_152h",
            "num_experts": 1048576,
            "product_key_dim": [1024, 1024],
            "expert_hidden_size": EXPERT_HIDDEN_SIZE_CALC,
        },
        {
            "name": "3d_1M_152h",
            "num_experts": 1000000, # Note: 100^3 = 1,000,000
            "product_key_dim": [100, 100, 100],
            "expert_hidden_size": EXPERT_HIDDEN_SIZE_CALC,
        },
        {
            "name": "4d_1M_152h",
            "num_experts": 1048576, # Note: 32^4 = 1,048,576
            "product_key_dim": [32, 32, 32, 32],
            "expert_hidden_size": EXPERT_HIDDEN_SIZE_CALC,
        },
    ],
    # 3. Number of expert selection heads and experts per token
    "selection_heads": [
        {"name": "h4_k32", "num_heads": 4, "num_experts_per_tok": 32},
        {"name": "h8_k16", "num_heads": 8, "num_experts_per_tok": 16},
        {"name": "h16_k8", "num_heads": 16, "num_experts_per_tok": 8},
        {"name": "h32_k4", "num_heads": 32, "num_experts_per_tok": 4},
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

    # 2. Apply expert setup configuration if present
    if "expert_setup" in experiment:
        expert_setup = experiment["expert_setup"]
        config["model_config"]["peer_config"]["num_experts"] = expert_setup[
            "num_experts"
        ]
        config["model_config"]["peer_config"]["product_key_dim"] = expert_setup[
            "product_key_dim"
        ]
        config["model_config"]["peer_config"]["expert_hidden_size"] = expert_setup[
            "expert_hidden_size"
        ]
        experiment_name_parts.append(f"expert-{expert_setup['name']}")

    # 3. Apply selection heads configuration if present
    if "selection_heads" in experiment:
        selection_heads = experiment["selection_heads"]
        config["model_config"]["peer_config"]["num_heads"] = selection_heads[
            "num_heads"
        ]
        config["model_config"]["peer_config"]["num_experts_per_tok"] = selection_heads[
            "num_experts_per_tok"
        ]
        experiment_name_parts.append(f"heads-{selection_heads['name']}")

    # Set unique experiment name
    experiment_name = "_".join(experiment_name_parts)

    # Update configuration with unique experiment name and output directory
    config["wandb_config"]["name"] = f"ablation_{experiment_name}"
    config["train_config"]["output_dir"] = os.path.join(
        experiment_dir, "checkpoints", experiment_name
    )

    # Save configuration to file
    config_path = os.path.join(experiment_dir, "configs", f"{experiment_name}.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return config_path, experiment_name


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
                with open(perplexity_results_path, "r") as f:
                    perplexity_data = json.load(f)
                    # Handle different possible key structures in perplexity output
                    if "perplexity" in perplexity_data:
                        results["perplexity"] = perplexity_data["perplexity"]
                    elif "validation_perplexity" in perplexity_data:
                        results["perplexity"] = perplexity_data["validation_perplexity"]
                    else:
                        # Try to find any key containing 'perplexity'
                        perplexity_keys = [k for k in perplexity_data.keys() if 'perplexity' in k.lower()]
                        if perplexity_keys:
                            results["perplexity"] = perplexity_data[perplexity_keys[0]]
                        else:
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
                    with open(benchmark_results_path, "r") as f:
                        benchmark_data = json.load(f)

                        # Extract the primary metric from the benchmark results
                        # Handle different possible JSON structures
                        if benchmark_name == "codeforces":
                            # Special handling for codeforces which has multiple metrics
                            results[benchmark_name] = benchmark_data.get("percentile", None)
                            results[f"{benchmark_name}_rating"] = benchmark_data.get("rating", None)
                        else:
                            # Try different possible locations for the metric
                            # 1. Direct key access
                            if primary_metric_key in benchmark_data:
                                results[benchmark_name] = benchmark_data[primary_metric_key]
                            # 2. Check if metrics is a dict containing our key
                            elif "metrics" in benchmark_data and isinstance(benchmark_data["metrics"], dict):
                                results[benchmark_name] = benchmark_data["metrics"].get(primary_metric_key, None)
                            # 3. Check if results contains our key
                            elif "results" in benchmark_data and isinstance(benchmark_data["results"], dict):
                                results[benchmark_name] = benchmark_data["results"].get(primary_metric_key, None)
                            # 4. Look for any key containing our metric name
                            else:
                                metric_keys = [k for k in benchmark_data.keys() 
                                              if primary_metric_key.lower() in k.lower()]
                                if metric_keys:
                                    results[benchmark_name] = benchmark_data[metric_keys[0]]
                                else:
                                    results[benchmark_name] = None
                            
                            # Extract additional metrics if present
                            if "metrics" in benchmark_data and isinstance(benchmark_data["metrics"], dict):
                                for metric_name, value in benchmark_data["metrics"].items():
                                    if metric_name != primary_metric_key:
                                        results[f"{benchmark_name}_{metric_name}"] = value
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


def generate_experiment_combinations(selected_axes: List[str] = None) -> List[Dict]:
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
            "aime_2024",
            "codeforces",
            "gpqa_diamond",
            "math_500",
            "mmlu",
            "swe_bench",
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
    # Check if required columns exist
    if "total_params" in df.columns and "perplexity" in df.columns:
        sns.scatterplot(
            data=df,
            x="total_params",
            y="perplexity",
            hue="experiment_type" if "experiment_type" in df.columns else None, # Color by the combination of axes tested
            size="active_params_ratio" if "active_params_ratio" in df.columns else None,
            sizes=(100, 400),
            alpha=0.7,
        )
    plt.xscale("log")
    plt.title("Total Parameters vs. Perplexity")
    plt.xlabel("Total Parameters")
    plt.ylabel("Perplexity (lower is better)")
    plt.savefig(os.path.join(viz_dir, "params_vs_perplexity.png"))
    plt.close()

    # 2. Analyze by active parameters ratio vs. perplexity
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df,
        x="active_params_ratio",
        y="perplexity",
        hue="experiment_type",
        style="expert_setup_config" if "expert_setup_config" in df else None, # Style by expert setup
        s=150,
        alpha=0.8,
    )
    plt.title("Active Parameter Ratio vs. Perplexity")
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

        # Create visualization for this benchmark vs Total Params
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=df,
            x="total_params",
            y=benchmark,
            hue="experiment_type",
            style="expert_setup_config" if "expert_setup_config" in df else None,
            size="active_params_ratio",
            sizes=(100, 400),
            alpha=0.7,
        )
        plt.xscale("log")

        # Different benchmarks have different interpretation (higher/lower better)
        ylabel_suffix = ""
        if benchmark in [
            "aime_2024",
            "codeforces",
            "gpqa_diamond",
            "math_500",
            "mmlu",
            "swe_bench",
        ]:
            ylabel_suffix = " (higher is better)"

        plt.title(f"Model Size vs. {benchmark.replace('_', ' ').title()}")
        plt.xlabel("Total Parameters")
        plt.ylabel(f"{benchmark.replace('_', ' ').title()}{ylabel_suffix}")
        plt.savefig(os.path.join(viz_dir, f"total_params_vs_{benchmark}.png"))
        plt.close()

        # Create visualization for this benchmark vs Active Params Ratio
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=df,
            x="active_params_ratio",
            y=benchmark,
            hue="experiment_type",
            style="expert_setup_config" if "expert_setup_config" in df else None,
            s=150,
            alpha=0.8,
        )
        plt.title(f"Active Parameter Ratio vs. {benchmark.replace('_', ' ').title()}")
        plt.xlabel("Active Parameters Ratio")
        plt.ylabel(f"{benchmark.replace('_', ' ').title()}{ylabel_suffix}")
        plt.savefig(os.path.join(viz_dir, f"active_ratio_vs_{benchmark}.png"))
        plt.close()


        # Create comparison bar plot by experiment name
        plt.figure(figsize=(max(15, len(df) * 0.5), 8)) # Adjust width based on number of experiments
        plot_data = df[df[benchmark].notna()].copy()
        if len(plot_data) < 1:
            continue

        # Sort by benchmark score
        higher_is_better = benchmark != "perplexity"
        plot_data = plot_data.sort_values(benchmark, ascending=not higher_is_better)

        sns.barplot(data=plot_data, x="experiment_name", y=benchmark, hue="experiment_type", dodge=False)
        plt.xticks(rotation=90)
        plt.title(f"{benchmark.replace('_', ' ').title()} by Experiment")
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f"{benchmark}_by_experiment.png"))
        plt.close()


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
        config_path, experiment_name = generate_config(
            base_config, experiment, experiment_dir
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

        # Record results
        result = {
            "experiment_name": experiment_name,
            "perplexity": eval_results.get("perplexity", float("inf")),
            "total_params": eval_results.get("total_params", 0),
            "active_params": eval_results.get("active_params", 0),
            "active_params_ratio": eval_results.get("active_params", 0)
            / max(eval_results.get("total_params", 1), 1),
            "training_time": training_time,
            "experiment_type": "_".join(sorted([axis for axis in experiment.keys()])), # Sort axes for consistency
        }

        # Add specific configuration details and benchmark results
        for axis, config in experiment.items():
            if axis == "data_mix":
                result["data_config"] = config["name"]
            elif axis == "expert_setup":
                result["expert_setup_config"] = config["name"]
            elif axis == "selection_heads":
                result["selection_config"] = config["name"]

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
