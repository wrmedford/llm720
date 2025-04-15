# Installation Guide

This guide provides detailed instructions for installing the necessary dependencies for the Foundation Language Model Training Framework.

## Recommended Method: Using `uv`

We recommend using `uv` for faster and more reliable environment management.

1.  **Install `uv` (if you haven't already):**
    Follow the official instructions at [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv).
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Create and Activate Virtual Environment:**
    Navigate to the project's root directory and run:
    ```bash
    uv venv
    source .venv/bin/activate
    # Or use the appropriate activation script for your shell (e.g., .fish, .csh)
    ```

3.  **Install PyTorch:**
    Crucially, install PyTorch *before* other dependencies like `flash-attn` or `torchao`. Ensure the PyTorch version matches your CUDA toolkit version. Find the correct command for your system at [pytorch.org](https://pytorch.org/).
    ```bash
    # --- Example for CUDA 12.1 ---
    # Adjust 'cu121' based on your CUDA version (e.g., cu118)
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    # --- Example for CPU-only (if no GPU) ---
    # uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    ```
    *Note: For FP8 support via `torchao`, PyTorch >= 2.5 is recommended.*

4.  **Install the `llm` Package and Dependencies:**
    Use `--no-build-isolation` to allow packages to find the already installed PyTorch during their build process. Choose the installation option that suits your needs:

    ```bash
    # Option A: Install everything (Recommended for development and ablation studies)
    # Includes base requirements + dev tools (pytest, black, etc.) + eval tools (openai-evals)
    uv pip install --no-build-isolation -e ".[dev,evals]"

    # Option B: Install base package only (for training/inference)
    # Includes torchao, flash-attn (if on Linux)
    # uv pip install --no-build-isolation -e .

    # Option C: Install base + development tools
    # uv pip install --no-build-isolation -e ".[dev]"

    # Option D: Install base + evaluation tools
    # uv pip install --no-build-isolation -e ".[eval]"
    ```

**Troubleshooting Common Issues:**

-   **`flash-attn` Installation Failure:**
    -   **Non-Linux:** `flash-attn` often provides wheels only for Linux. You might need to skip it (comment out in `setup.py`) or build it from source (see below and official `flash-attn` docs). The code has PyTorch SDPA fallbacks.
    -   **Linux/ARM:** Pre-built wheels might be missing for ARM. Building from source is often required. Ensure you have `gcc`, `g++`, and the CUDA toolkit installed.
    -   **CUDA Version Mismatch:** Ensure the installed PyTorch CUDA version matches your system's CUDA toolkit version expected by `flash-attn`.
-   **`torchao` Installation Failure:**
    -   Requires a recent PyTorch version (>= 2.5 recommended). Ensure PyTorch was installed correctly first.
-   **Build Errors:** Ensure you have necessary build tools (`build-essential` on Debian/Ubuntu, CMake, Ninja).

## Building Dependencies from Source (Advanced)

Use this method if pre-built wheels are unavailable or incompatible (e.g., specific ARM architectures like GH200, non-standard CUDA versions like 12.8).

**Prerequisites:**

*   A C++ compiler (`g++`) and build tools (`build-essential` on Debian/Ubuntu).
*   CMake (`>=3.18`).
*   Ninja (`ninja-build` on Debian/Ubuntu).
*   The CUDA Toolkit matching the target version (e.g., 12.8) installed, with `nvcc` in your `PATH`.
*   CUDA development libraries (e.g., `cuda-cupti-dev-12-8`, `cuda-nvml-dev-12-8`, `libnccl-dev` on Debian/Ubuntu, matching your CUDA version).
*   `uv` (see step 1 above).
*   For ARM builds: `scons` and `patchelf`.

**Usage:**

1.  **Customize Build (Optional):** Set environment variables before running the script to control versions and targets:
    ```bash
    # Example: Target Python 3.10, CUDA 12.8, Hopper Arch
    export PYTHON_VERSION=3.10
    export CUDA_VERSION=12.8 # Adjust minor version if needed, e.g., 12.8.0
    export TORCH_CUDA_ARCH_LIST="9.0a"
    # export SKIP_TORCH=1 # Uncomment to skip building PyTorch
    # export SKIP_FLASH_ATTN=1 # Uncomment to skip building FlashAttention
    # export TORCH_REF=v2.6.0 # Specify PyTorch git tag/branch
    # export FLASH_ATTN_REF=v2.7.4 # Specify FlashAttention git tag/branch
    ```

2.  **Run the Build Script:**
    ```bash
    bash scripts/build_from_source.sh
    ```
    This script performs the following steps:
    *   Creates a virtual environment (`.venv`).
    *   Clones dependency source code (PyTorch, FlashAttention, etc.) into `src/`.
    *   Builds dependencies using specified settings (CUDA, Arch).
    *   (On ARM) Builds the ARM Compute Library if needed.
    *   Places compiled wheels into `wheels/`.
    *   Installs the built wheels using `uv`.
    *   Installs the `llm` project in editable mode using the built dependencies.

3.  **Activate Environment:**
    ```bash
    source .venv/bin/activate
    ```

After completion, your environment will contain dependencies specifically built for your system.
