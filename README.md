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

We recommend using `uv` for faster environment management.

1.  **Install `uv` (if you haven't already):**
    ```bash
    # Follow instructions at https://github.com/astral-sh/uv
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    uv venv
    source .venv/bin/activate
    # Or `source .venv/bin/activate.fish` for fish shell, etc.
    ```
3.  **Install PyTorch:** Packages like `flash-attn` require PyTorch to be installed *before* they are built. Install PyTorch first, matching your CUDA version (see [https://pytorch.org/](https://pytorch.org/)).
    ```bash
    # Example for CUDA 12.1 on x86_64 - Adjust if necessary!
    # uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    # Example for CUDA 12.1 on ARM (e.g., GH200) - Adjust if necessary!
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```
    *Verify the correct CUDA version for your system and drivers.* Find available wheels at [https://pytorch.org/](https://pytorch.org/).

4.  **Install the package and remaining dependencies:** Use `--no-build-isolation` to ensure packages like `flash-attn` and `float8_experimental` can find the already installed PyTorch during their build process.
    ```bash
    # For running ablation studies (requires dev and eval dependencies)
    # This includes flash-attn, float8_experimental, pytest, evals, etc.
    uv pip install --no-build-isolation -e ".[dev,evals]"

    # --- OR --- Install only what you need:

    # Base installation (training/inference - includes flash-attn, float8_experimental)
    # uv pip install --no-build-isolation -e .

    # For development (linters, formatters, testing tools)
    # uv pip install --no-build-isolation -e ".[dev]"

    # For running evaluations (OpenAI evals library)
    # uv pip install --no-build-isolation -e ".[eval]"
    ```

    **ARM/GH200 Installation Notes:**
    - **`flash-attn`:** Installation on ARM architectures might fail as pre-built wheels are often unavailable. You may need to:
        - **Build from source:** Follow the official instructions for `flash-attn`, which may require specific compilers (like `gcc`, `g++`) and the CUDA toolkit installed on your system.
        - **Comment out:** If you don't strictly need FlashAttention for MLA initially, you can temporarily comment out `flash-attn` in `setup.py` and reinstall using the command above. The code includes PyTorch fallbacks.
    - **`float8_experimental`:** Requires a recent PyTorch version (nightly or >= 2.2) with FP8 support. It's installed directly from GitHub.

    If installation fails, check the specific error messages and consult the documentation for `flash-attn` and `float8_experimental`.

## Building Dependencies from Source (Advanced)

In some cases, especially on specific architectures like ARM (e.g., GH200) or when using non-standard CUDA versions (like 12.8), pre-built wheels for dependencies like PyTorch and FlashAttention might not be available or compatible.

This project includes a script to build these core dependencies from source tailored to your environment.

**Prerequisites:**

*   A C++ compiler (like `g++`) and build tools (`build-essential` on Debian/Ubuntu).
*   CMake (`>=3.18`).
*   Ninja (`ninja-build` on Debian/Ubuntu).
*   The CUDA Toolkit matching the version you want to build against (e.g., 12.8) installed, with `nvcc` in your `PATH`.
*   CUDA development libraries (`cuda-cupti-dev`, `cuda-nvml-dev`, `libnccl-dev` on Debian/Ubuntu, matching your CUDA version).
*   `uv` (install via `curl -LsSf https://astral.sh/uv/install.sh | sh`).
*   For ARM builds: `scons` and `patchelf` (for building ARM Compute Library).

**Usage:**

1.  **Customize Build (Optional):** You can control the build process using environment variables before running the script:
    *   `PYTHON_VERSION`: Target Python version (default: 3.12).
    *   `CUDA_VERSION`: Target CUDA version (default: 12.8.90).
    *   `TORCH_CUDA_ARCH_LIST`: Target GPU architectures (default: 9.0a for Hopper).
    *   `SKIP_TORCH=1`, `SKIP_FLASH_ATTN=1`: Skip building specific components if they are already installed or not needed.
    *   `TORCH_REF`, `FLASH_ATTN_REF`: Specify git tags/branches for dependencies.

2.  **Run the Build Script:**
    ```bash
    bash scripts/build_from_source.sh
    ```
    This script will:
    *   Set up a virtual environment (`.venv`).
    *   Clone the source code for PyTorch and FlashAttention into the `src/` directory.
    *   Build each dependency using the specified CUDA version and architecture flags.
    *   (On ARM) Build the ARM Compute Library.
    *   Place the compiled wheels into the `wheels/` directory.
    *   Install the built wheels using `uv`.
    *   Install `float8_experimental` from GitHub.
    *   Install the `llm` project itself in editable mode.

3.  **Activate Environment:**
    ```bash
    source .venv/bin/activate
    ```

After the script completes successfully, your environment will have the core dependencies built specifically for your system, and the `llm` project installed using them.

## Getting Started

1.  Install the required dependencies. Choose **one** of the following methods:
    *   **Standard Installation (Recommended):** Follow the steps in the [Installation](#installation) section using `uv pip install`. This uses pre-built wheels if available.
    *   **Build from Source (Advanced):** If standard installation fails or you need specific versions/CUDA support, follow the steps in the [Building Dependencies from Source](#building-dependencies-from-source-advanced) section.

    Example for basic training after standard install (ARM/GH200, CUDA 12.1):
    ```bash
    # Activate environment: source .venv/bin/activate
    # Install PyTorch (example for CUDA 12.1):
    # Note: For FP8 support, use a recent nightly build or PyTorch >= 2.2
    uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
    # Install base package (handle flash-attn/triton build errors if they occur):
    uv pip install --no-build-isolation -e .
    ```
2.  Prepare your configuration file:
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
