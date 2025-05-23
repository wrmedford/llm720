#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Check if CUDA is available
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

# Define extensions
ext_modules = []

if CUDA_AVAILABLE and os.environ.get("BUILD_CUTLASS_KERNEL", "1") == "1":
    # CUTLASS kernel extension
    cutlass_ext = CUDAExtension(
        name="llm.models.kernels.peer_cutlass_module",
        sources=[
            "llm/models/kernels/peer_cutlass.cu",
            "llm/models/kernels/peer_cutlass_wrapper.cpp",
        ],
        include_dirs=[
            # Add CUTLASS include path if needed
            # os.environ.get("CUTLASS_PATH", "/usr/local/cutlass/include"),
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": [
                "-O3",
                "-std=c++17",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math",
                "-gencode=arch=compute_80,code=sm_80",  # A100
                "-gencode=arch=compute_90,code=sm_90",  # H100
            ],
        },
    )
    ext_modules.append(cutlass_ext)

setup(
    name="llm",
    version="0.1.0",
    description="Foundation Language Model with PEER and MLA",
    author="LLM Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.20.0",
        "transformers>=4.20.0",
        "datasets>=2.0.0",
        "wandb>=0.13.0",
        "accelerate>=0.10.0",
        "safetensors>=0.2.0",
        "PyYAML>=6.0", # Provides the 'yaml' module
        "tqdm>=4.62.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "pandas>=1.3.0",
        "tabulate>=0.8.0",
        "flash-attn>=2.0.0; platform_system=='Linux'", # flash-attn is often Linux-only
        "torchao>=0.10.0", # Use torchao for FP8 and other optimizations
        "pynvml>=11.0.0",  # For GPU monitoring
        "tiktoken>=0.7.0", # Added tiktoken dependency
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "pandas>=1.3.0",    # Added for analysis
            "seaborn>=0.12.0",  # Added for analysis
            "black",
            "isort",
            "flake8",
            "mypy",
        ],
        "evals": [
            "evals",  # OpenAI's evals library
        ],
    },
    entry_points={
        "console_scripts": [
            "llm-train=scripts.train:main",
            "llm-eval=scripts.run_evaluation:main",
        ],
    },
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension} if ext_modules else {},
)