#!/usr/bin/env bash

# Build script for llm project dependencies (torch, triton, flash-attn, transformer-engine)
# Adapted from a vllm build script.

set -euo pipefail

# --- Configuration ---

# Architecture
export BUILD_ARCH=$(arch)
echo "Building for architecture: ${BUILD_ARCH}"

# Python
export PYTHON_VERSION=${PYTHON_VERSION:-3.12}
echo "Targeting Python version: ${PYTHON_VERSION}"

# CUDA
# WARNING: CUDA 12.8 is not a standard build target for many packages. Compatibility issues may arise.
export CUDA_VERSION=${CUDA_VERSION:-12.8} # Default to 12.8 to match cu128 index
export CUDA_SHORT=${CUDA_VERSION%.*} # Will be 12.8
export CUDA_TAG=cu${CUDA_SHORT//./}
export CUDA_RELEASE=${CUDA_SHORT//./-}
echo "Targeting CUDA version: ${CUDA_VERSION} (${CUDA_TAG})"

# Prerequisites Check (Basic)
if ! command -v nvcc &> /dev/null || ! command -v cmake &> /dev/null || ! command -v ninja &> /dev/null || ! command -v uv &> /dev/null; then
    echo "Error: Missing prerequisites. Ensure CUDA Toolkit, CMake, Ninja, and uv are installed and in PATH."
    # Add specific package manager commands if desired, e.g.:
    # echo "On Ubuntu/Debian, try: sudo apt install -y build-essential cmake ninja-build"
    # echo "Install uv from https://github.com/astral-sh/uv"
    # echo "Install CUDA Toolkit from NVIDIA's website, ensuring nvcc is in PATH."
    exit 1
fi
# Check for CUDA dev libraries (adjust package names for your OS/distro if needed)
# Example for Ubuntu/Debian - requires user confirmation/sudo
# if ! dpkg -s "cuda-cupti-dev-${CUDA_RELEASE}" >/dev/null 2>&1 || \
#    ! dpkg -s "cuda-nvml-dev-${CUDA_RELEASE}" >/dev/null 2>&1 || \
#    ! dpkg -s "libnccl-dev" >/dev/null 2>&1; then
#     echo "Warning: Required CUDA development libraries (cupti, nvml, nccl) might be missing."
#     echo "On Ubuntu/Debian, try: sudo apt install -y cuda-cupti-dev-${CUDA_RELEASE} cuda-nvml-dev-${CUDA_RELEASE} libnccl-dev"
#     # Optionally exit here if these are strictly required upfront
# fi


# Job scaling
export MAX_JOBS=${MAX_JOBS:-$(nproc)}
export NVCC_THREADS=${NVCC_THREADS:-8}
echo "Using MAX_JOBS=${MAX_JOBS}, NVCC_THREADS=${NVCC_THREADS}"

# Cmake build type
export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}

# Arch lists (Targeting Hopper for GH200)
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0a}
echo "Using TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

# --- Setup ---

# Prep build venv
echo "Setting up virtual environment..."
uv venv -p ${PYTHON_VERSION} --seed --python-preference only-managed
export VIRTUAL_ENV=${PWD}/.venv
export PATH=${VIRTUAL_ENV}/bin:${PATH}
# Assume standard CUDA installation path
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-:}
echo "Activating venv and setting paths (CUDA_HOME=${CUDA_HOME})"
source ${VIRTUAL_ENV}/bin/activate

# Create directories
mkdir -p wheels src
export WHEELS=${PWD}/wheels
echo "Wheel output directory: ${WHEELS}"
echo "Source checkout directory: ${PWD}/src"

# --- Helper Function ---

# Reset repo helper
# $1 = repo, $2 = ref
reset_repo () {
    REPO_URL=${1}
    REPO_REF=${2}
    REPO_PATH=$(basename ${REPO_URL} .git)
    echo "Cloning/Resetting ${REPO_PATH} to ref ${REPO_REF}..."

    if [[ -d ${REPO_PATH} ]]; then
        echo "Repository ${REPO_PATH} exists. Resetting..."
        pushd ${REPO_PATH} > /dev/null
            git remote set-url origin ${REPO_URL}
            git fetch origin --tags # Fetch tags as well
            git reset --hard
            # Try checking out the ref directly (works for tags/branches/commits)
            if ! git checkout ${REPO_REF}; then
                echo "Checkout failed, trying origin/${REPO_REF}..."
                git checkout origin/${REPO_REF} || { echo "Failed to checkout ${REPO_REF}"; exit 1; }
            fi
            git reset --hard HEAD # Reset to the checked out ref cleanly
            # Clean untracked files and directories
            git clean -fdx
            # Update submodules
            echo "Updating submodules for ${REPO_PATH}..."
            git submodule sync --recursive
            git submodule update --init --recursive -j ${MAX_JOBS}
        popd > /dev/null
    else
        echo "Cloning ${REPO_URL}..."
        git clone ${REPO_URL}
        pushd ${REPO_PATH} > /dev/null
            git checkout ${REPO_REF} || { echo "Failed to checkout ${REPO_REF}"; exit 1; }
            echo "Updating submodules for ${REPO_PATH}..."
            git submodule sync --recursive
            git submodule update --init --recursive -j ${MAX_JOBS}
        popd > /dev/null
    fi
    echo "Finished preparing ${REPO_PATH}."
}


# --- Build Dependencies ---

echo "Installing base build tools..."
# Install build deps that aren't in project requirements files
# Make sure to upgrade setuptools to avoid triton build bug
uv pip install -U build cmake ninja pybind11 "setuptools<=76" wheel

pushd src > /dev/null

# Build architecture specific differences
if [[ ${BUILD_ARCH} == "aarch64" ]]; then
    echo "Performing ARM64 specific setup..."
    # Check if ACL build should be skipped
    if ! [[ -v SKIP_ACL ]]; then
        echo "Building ARM Compute Library (ACL)..."
        # Deps for ACL
        uv pip install patchelf scons

        # Optimize ARM linking
        export USE_PRIORITIZED_TEXT_FOR_LD=1

        # Build ARM ComputeLibrary
        export ACL_REPO=https://github.com/ARM-software/ComputeLibrary.git
        export ACL_REF=${ACL_REF:-v24.09} # Use a recent tag
        export ACL_ROOT_DIR=${PWD}/acl
        export ACL_INCLUDE_DIR=${ACL_ROOT_DIR}/include
        export ACL_LIBRARY=${ACL_ROOT_DIR}/build
        export LD_LIBRARY_PATH=${ACL_LIBRARY}:${LD_LIBRARY_PATH}
        mkdir -p acl
        reset_repo ${ACL_REPO} ${ACL_REF}
        pushd ComputeLibrary > /dev/null
            echo "Running scons for ACL..."
            # Ensure build dir exists before scons tries to use it
            mkdir -p ${ACL_LIBRARY}
            scons Werror=1 -j${MAX_JOBS} build_dir=${ACL_LIBRARY} debug=0 neon=1 opencl=0 os=linux openmp=1 cppthreads=0 arch=armv8a multi_isa=1 fixed_format_kernels=1 build=native
            echo "ACL build finished."
        popd > /dev/null
    else
        echo "Skipping ARM Compute Library (ACL) build as SKIP_ACL is set."
    fi
elif [[ ${BUILD_ARCH} == "x86_64" ]]; then
    echo "Performing x86_64 specific setup..."
    # Install MKL if needed (often helps PyTorch performance on Intel CPUs)
    # uv pip install mkl-static mkl-include # Uncomment if needed
else
    echo "Unsupported architecture: ${BUILD_ARCH}"
    exit 1
fi

# --- PyTorch ---
export TORCH_REPO=${TORCH_REPO:-https://github.com/pytorch/pytorch.git}
export TORCH_REF=${TORCH_REF:-v2.6.0-rc9} # Use a recent stable release
export TORCH_BUILD_VERSION=${TORCH_BUILD_VERSION:-${TORCH_REF#v}+${CUDA_TAG}}
export PYTORCH_BUILD_VERSION=${TORCH_BUILD_VERSION:-${TORCH_REF#v}+${CUDA_TAG}}
export PYTORCH_BUILD_NUMBER=0
if ! [[ -v SKIP_TORCH ]]; then
    echo "Building PyTorch ${TORCH_REF}..."
    reset_repo ${TORCH_REPO} ${TORCH_REF}
    pushd pytorch > /dev/null
        if [[ ${BUILD_ARCH} == "aarch64" ]]; then
            # Use NVPL Blis on ARM64 (alternative to OpenBLAS/MKL)
            export BLAS=NVPL
            echo "Set BLAS=NVPL for ARM64 PyTorch build."
        fi
        echo "Installing PyTorch build requirements..."
        uv pip install -r requirements.txt
        echo "Setting CMAKE_ARGS for PyTorch build..."
        export CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
        echo "Starting PyTorch build (version ${PYTORCH_BUILD_VERSION})..."
        # Build PyTorch using uv build
        uv build --wheel --no-build-isolation -o ${WHEELS}
        echo "PyTorch build finished."
        # Install immediately to satisfy subsequent builds
        echo "Installing built PyTorch wheel..."
        TORCH_WHEEL=$(find ${WHEELS} -name "torch*.whl")
        if [[ -z "${TORCH_WHEEL}" ]]; then
            echo "Error: PyTorch wheel not found in ${WHEELS}"
            exit 1
        fi
        uv pip install --no-deps ${TORCH_WHEEL}
        # Unset CMAKE_ARGS after PyTorch build to avoid affecting other builds
        unset CMAKE_ARGS
    popd > /dev/null
else
    echo "Skipping PyTorch build as SKIP_TORCH is set. Installing from ${WHEELS}..."
    TORCH_WHEEL=$(find ${WHEELS} -name "torch*.whl")
    if [[ -z "${TORCH_WHEEL}" ]]; then
        echo "Error: PyTorch wheel not found in ${WHEELS}"
        exit 1
    fi
    uv pip install --no-deps ${TORCH_WHEEL}
fi

# --- TorchAudio ---
export AUDIO_REPO=${AUDIO_REPO:-https://github.com/pytorch/audio.git}
export AUDIO_REF=${AUDIO_REF:-v2.6.0} # Match PyTorch version
export AUDIO_BUILD_VERSION=${AUDIO_BUILD_VERSION:-${AUDIO_REF#v}+${CUDA_TAG}}
export BUILD_VERSION=${AUDIO_BUILD_VERSION} # TorchAudio uses BUILD_VERSION
if ! [[ -v SKIP_AUDIO ]]; then
    echo "Building TorchAudio ${AUDIO_REF}..."
    reset_repo ${AUDIO_REPO} ${AUDIO_REF}
    pushd audio > /dev/null
        echo "Starting TorchAudio build..."
        uv build --wheel --no-build-isolation -o ${WHEELS}
        echo "TorchAudio build finished."
    popd > /dev/null
else
    echo "Skipping TorchAudio build as SKIP_AUDIO is set."
fi

# --- TorchVision ---
export VISION_REPO=${VISION_REPO:-https://github.com/pytorch/vision.git}
export VISION_REF=${VISION_REF:-v0.21.0} # Match PyTorch version
export VISION_BUILD_VERSION=${VISION_BUILD_VERSION:-${VISION_REF#v}+${CUDA_TAG}}
export BUILD_VERSION=${VISION_BUILD_VERSION} # TorchVision uses BUILD_VERSION
if ! [[ -v SKIP_VISION ]]; then
    echo "Building TorchVision ${VISION_REF}..."
    reset_repo ${VISION_REPO} ${VISION_REF}
    pushd vision > /dev/null
        echo "Starting TorchVision build..."
        uv build --wheel --no-build-isolation -o ${WHEELS}
        echo "TorchVision build finished."
    popd > /dev/null
else
    echo "Skipping TorchVision build as SKIP_VISION is set."
fi


# --- FlashAttention ---
# Needs specific build flags
export FLASH_ATTENTION_FORCE_BUILD=1
export FLASH_ATTENTION_WITH_CUDA=1
export FLASH_ATTN_REPO=${FLASH_ATTN_REPO:-https://github.com/Dao-AILab/flash-attention.git}
export FLASH_ATTN_REF=${FLASH_ATTN_REF:-v2.7.4.post1} # Use a known tag or main
# FlashAttention doesn't use BUILD_VERSION directly, relies on setup.py logic
if ! [[ -v SKIP_FLASH_ATTN ]]; then
    echo "Building FlashAttention ${FLASH_ATTN_REF}..."
    reset_repo ${FLASH_ATTN_REPO} ${FLASH_ATTN_REF}
    pushd flash-attention > /dev/null
        echo "Starting FlashAttention build..."
        # FlashAttention build needs specific env vars
        export MAX_JOBS=${MAX_JOBS}
        # It uses TORCH_CUDA_ARCH_LIST automatically
        uv build --wheel --no-build-isolation -o ${WHEELS}
        echo "FlashAttention build finished."
    popd > /dev/null
else
    echo "Skipping FlashAttention build as SKIP_FLASH_ATTN is set."
fi

# --- Install float8_experimental ---
echo "Installing float8_experimental from GitHub..."
uv pip install git+https://github.com/pytorch-labs/float8_experimental.git
echo "float8_experimental installation complete."


popd > /dev/null # Exit src directory

# --- Final Installation ---

echo "Installing all built wheels from ${WHEELS}..."
# Check if wheels directory has any wheels
if [ ! "$(ls -A ${WHEELS})" ]; then
    echo "Error: No wheels found in ${WHEELS} directory!"
    exit 1
fi

# Install all wheels built, respecting dependencies if possible
# First install torch if it exists
TORCH_WHEEL=$(find ${WHEELS} -name "torch*.whl" | grep -v "torchaudio\|torchvision" | head -n 1)
if [[ -n "${TORCH_WHEEL}" ]]; then
    echo "Installing PyTorch wheel: ${TORCH_WHEEL}"
    uv pip install --no-deps "${TORCH_WHEEL}"
fi

# Then install all other wheels
echo "Installing remaining wheels..."
uv pip install ${WHEELS}/*.whl

echo "Installing the llm project..."
# Install the current project in editable mode, using the built dependencies
# Use --no-build-isolation as torch is already installed
if [ -f "setup.py" ]; then
    uv pip install --no-build-isolation -e ".[dev,evals]"
else
    echo "Warning: setup.py not found in current directory. Skipping project installation."
fi

echo "Build and installation complete!"
echo "Activate the environment using: source .venv/bin/activate"
