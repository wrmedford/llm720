# Foundation Language Model Training Framework

This repository contains an implementation for training language models with Parameter Efficient Expert Retrieval (PEER) and Multi-Headed Latent Attention (MLA).

## About This Project

This project provides a framework for training large language models (LLMs) with a focus on parameter efficiency and cost-effective training, drawing inspiration from recent research by DeepSeek (e.g., DeepSeek-V2/V3, Mixture-of-A-Million-Experts). It implements two core architectural innovations:

1.  **Parameter Efficient Expert Retrieval (PEER):** An advanced Mixture-of-Experts (MoE) layer designed to scale to millions of tiny experts (often single neurons). Instead of traditional routing, PEER uses efficient "product key" retrieval to select a small subset of these experts for each token. This allows for a massive total parameter count while keeping the activated parameters per token low, aiming for high performance with reduced computational cost.
2.  **Multi-Headed Latent Attention (MLA):** An efficient attention mechanism based on the DeepSeek V3 architecture. It employs techniques like low-rank projections and RoPE/NoPE decomposition to optimize attention computation, particularly for inference, and integrates with optimized kernels like FlashMLA.

The goal is to build powerful LLMs that are efficient to train and run, leveraging sparse activation of a vast number of parameters. The framework provides the necessary components for training, dataset management, evaluation, and analysis of models using these architectures. It differs from standard Transformer or MoE implementations by incorporating the specific PEER expert mechanism and the DeepSeek-style MLA.

## Features

- **PEER (Parameter Efficient Expert Retrieval)** for efficient expert selection
  - Variable expert hidden size
  - Support for multi-dimensional Cartesian product (not limited to 2D)
  - Customizable number of experts with configurable retrieval heads

- **Multi-Headed Latent Attention (MLA)** as implemented in DeepSeek models
  - Separate projections for query, key, and value
  - Query and key decomposed into rope and nope parts
  - Support for different head dimensions
  - Optimized implementation using FlashMLA (from DeepSeek)

- **Dataset Management**
  - Streaming HuggingFace datasets
  - Dataset interleaving with customizable weights
  - Configurable tokenization and batching

- **Robust Training Pipeline**
  - Checkpoint management with save/load functionality
  - Configurable checkpointing intervals
  - Resumable training

- **Evaluation Integration**
  - Continuous evaluation using OpenAI's evals library
  - Multiple evaluation metrics
  - Detailed logging of evaluation results

- **Configuration System**
  - YAML-based configuration
  - Support for ablation testing
  - Comprehensive logging of training parameters

- **Weights & Biases Integration**
  - Detailed metrics and logging
  - Model checkpoints tracking
  - Visualization of training progress
  - Expert usage tracking to identify overused experts

## Installation

```bash
# Install the package and dependencies
pip install -e .

# For development
pip install -e ".[dev]"

# For evaluation (includes OpenAI evals library)
pip install -e ".[eval]"

# Note: Triton installation might require specific CUDA versions.
# If the automatic installation fails, please refer to the official Triton documentation:
# https://triton-lang.org/main/getting-started/installation.html
```

## Getting Started

1. Install the required dependencies:
```bash
pip install -e .
```

2. Prepare your configuration file:
- See `configs/config.yaml` for an example configuration

3. Start training:
```bash
python scripts/train.py --config configs/config.yaml
```

4. For distributed training:
```bash
./scripts/train.sh --config configs/config.yaml --gpus-per-node 8
```

## Project Structure

```
llm/                      # Main package
├── models/               # Model implementations (MLA, PEER, Foundation)
│   ├── __init__.py
│   ├── attention.py
│   ├── experts.py
│   └── foundation.py
├── training/             # Training loop, config, ablation logic
│   ├── __init__.py
│   ├── ablation.py
│   └── train.py
├── data/                 # Dataset handling
│   ├── __init__.py
│   └── datasets.py
├── config/               # Configuration loading/saving
│   ├── __init__.py
│   └── defaults.py
├── ops/                  # Custom operations (e.g., Triton kernels)
│   └── triton_peer_kernels.py
└── utils/                # Utility functions
    ├── __init__.py
    ├── eval/             # Evaluation scripts (perplexity, benchmarks, size)
    │   ├── __init__.py
    │   ├── benchmark.py
    │   ├── perplexity.py
    │   └── size.py
    └── experts/          # Expert utilities (tracking)
        ├── __init__.py
        └── tracking.py

scripts/                  # Entry point scripts callable from command line
├── analyze_model.py      # Model analysis script
├── run_ablations.py      # Ablation study runner script
├── run_evaluation.py     # Evaluation script (perplexity, benchmarks)
├── train.py              # Training script entry point
└── train.sh              # Distributed training launch script

configs/                  # Configuration files
├── config.yaml           # Main configuration
└── datasets.yaml         # Dataset configurations

tests/                    # Test suite
```

## Configuration

The training script is controlled through a configuration file in YAML format. The configuration file includes the following sections:

- **model_config**: Model architecture parameters including PEER and MLA configurations
- **train_config**: Training parameters such as batch size, learning rate, etc.
- **dataset_config**: Dataset specifications and tokenization settings
- **eval_config**: Evaluation parameters and metrics
- **wandb_config**: Weights & Biases integration settings

## PEER Configuration

PEER (Parameter Efficient Expert Retrieval) uses product keys to efficiently select experts from a large pool. Key configuration parameters include:

- **num_experts**: Total number of experts (product of dimensions)
- **product_key_dim**: Dimensions for the Cartesian product (e.g., [1024, 1024] for 1M experts)
- **num_experts_per_tok**: Number of experts selected per token
- **num_heads**: Number of expert retrieval heads
- **expert_hidden_size**: Size of the expert hidden layer (1 for single neuron experts)

## Multi-Headed Latent Attention Configuration

MLA as implemented in DeepSeek models includes:

- **q_lora_rank**: Rank for query low-rank projections
- **kv_lora_rank**: Rank for key-value low-rank projections
- **qk_rope_head_dim**: Dimension for rotary position embeddings
- **v_head_dim**: Dimension for value heads
- **qk_nope_head_dim**: Dimension for non-rotary position embeddings

## Model Analysis

You can analyze your model's parameter counts and active parameters per token using the included analyzer script:

```bash
# Basic usage
python scripts/analyze_model.py --config configs/config.yaml

# Save visualization to file
python scripts/analyze_model.py --config configs/config.yaml --output model_analysis.png
```

This tool provides:
- Total parameter counts and breakdowns by component
- Active parameters per token during inference
- Parameter efficiency ratios and expert utilization metrics
- Visualizations comparing total vs. active parameters

## Expert Usage Monitoring

The training script includes functionality to track expert usage patterns to identify "hot experts" that may be overused during training. To enable expert usage monitoring:

```yaml
# In your config.yaml file
model_config:
  # ... other config ...
  peer_config:
    # ... other peer config ...
    log_expert_usage: true  # Enable expert usage logging
    log_freq: 1000          # Log expert usage every N steps
```

This will:
- Track which experts are used in each layer during training
- Log usage frequency histograms to Weights & Biases
- Generate visualizations of expert utilization patterns
- Identify potential load balancing issues

## Ablation Testing

The repository includes a meta-script for running comprehensive ablation studies to determine optimal configurations. The script tests combinations across multiple axes:

1. Initial data mix
2. Number and size of experts
3. Dimensionality of cartesian product
4. Number of expert selection heads

### Running Ablation Studies

```bash
# Run full ablation study across all axes
python scripts/run_ablations.py --base-config configs/config.yaml --output-dir ablation_results --gpus 4

# Run ablation on specific axes only
python scripts/run_ablations.py --base-config configs/config.yaml --output-dir ablation_results --axes data_mix expert_count

# Limit token count for faster iterations (e.g., 5B tokens per experiment)
python scripts/run_ablations.py --base-config configs/config.yaml --tokens 5000000000

# Resume an interrupted ablation study
python scripts/run_ablations.py --resume --output-dir ablation_results

# Use custom dataset mixes
python scripts/run_ablations.py --custom-datasets configs/datasets.yaml --output-dir ablation_results
```
