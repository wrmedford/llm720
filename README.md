# Foundation Language Model Training Framework

This repository provides a framework for training large language models (LLMs) with a focus on parameter efficiency and cost-effective training, drawing inspiration from recent research by DeepSeek (e.g., DeepSeek-V2/V3, Mixture-of-A-Million-Experts).

## About This Project

The goal is to build powerful LLMs that are efficient to train and run, leveraging sparse activation of a vast number of parameters. This framework implements two core architectural innovations:

1.  **Parameter Efficient Expert Retrieval (PEER):** An advanced Mixture-of-Experts (MoE) layer designed to scale to millions of tiny experts, using efficient "product key" retrieval.
2.  **Multi-Headed Latent Attention (MLA):** An efficient attention mechanism based on the DeepSeek V3 architecture, employing techniques like low-rank projections and RoPE/NoPE decomposition.

See [Architecture Details](docs/ARCHITECTURE.md) for more information.

## Features

-   **PEER** for efficient expert selection with optional CUTLASS kernel acceleration.
-   **MLA** for optimized attention.
-   **Dataset Management:** Streaming, interleaving, and configurable tokenization.
-   **Robust Training Pipeline:** Checkpointing, resuming, and distributed training support.
-   **Evaluation Integration:** Perplexity and benchmark evaluations (e.g., MMLU, MATH).
-   **Configuration System:** YAML-based configuration.
-   **Weights & Biases Integration:** Detailed metrics, logging, and expert usage tracking.
-   **Ablation Testing Framework:** Systematically test different configurations.
-   **Optimized Kernels:** CUTLASS-based PEER kernel for H100/A100 GPUs with hierarchical memory management.

## Installation

We recommend using `uv` for faster environment management.

1.  **Install `uv`:**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2.  **Create and activate environment:**
    ```bash
    uv venv
    source .venv/bin/activate
    ```
3.  **Install PyTorch:** Match your CUDA version (see [pytorch.org](https://pytorch.org/)).
    ```bash
    # Example for CUDA 12.1 - Adjust if necessary!
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```
4.  **Install the package:**
    ```bash
    # Install with dev and eval dependencies (recommended for full functionality)
    uv pip install --no-build-isolation -e ".[dev,evals]"

    # Or install base package only
    # uv pip install --no-build-isolation -e .
    ```

### Building CUTLASS Kernel (Optional)

For optimized performance on NVIDIA H100/A100 GPUs, you can build the CUTLASS kernel:

```bash
# The kernel will be built automatically when needed, or you can build it manually:
python setup.py build_ext --inplace

# To disable CUTLASS kernel building:
export BUILD_CUTLASS_KERNEL=0
```

For detailed installation instructions, including **building dependencies from source** (e.g., for ARM/GH200 or specific CUDA versions), see [Installation Guide](docs/INSTALLATION.md).

## Getting Started

1.  **Install** dependencies (see above).
2.  **Prepare** your configuration file (see `configs/config.yaml` and [Configuration Guide](docs/CONFIGURATION.md)).
3.  **Start training:**
    ```bash
    # Single GPU / Single Node
    python scripts/train.py --config configs/config.yaml

    # Distributed Training (Example: 8 GPUs on one node)
    ./scripts/train.sh --config configs/config.yaml --gpus-per-node 8
    
    # Use CUTLASS kernel for PEER (for H100/A100 GPUs)
    export USE_CUTLASS_KERNEL=1
    python scripts/train.py --config configs/config.yaml
    ```

## Further Documentation

-   [Architecture Details (PEER & MLA)](docs/ARCHITECTURE.md)
-   [Installation Guide](docs/INSTALLATION.md)
-   [Configuration Guide](docs/CONFIGURATION.md)
-   [Model Analysis](docs/ANALYSIS.md)
-   [Expert Usage Monitoring](docs/EXPERT_MONITORING.md)
-   [Ablation Testing](docs/ABLATION_TESTING.md)
