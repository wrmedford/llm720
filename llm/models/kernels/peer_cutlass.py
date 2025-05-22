"""Python interface for CUTLASS PEER kernel."""
import torch
import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Dict
import warnings
import hashlib
import tempfile
import shutil

# Cache for compiled kernels
_compiled_kernels: Dict[str, any] = {}

# Try to import the compiled CUTLASS module
try:
    import peer_cutlass_module
    CUTLASS_AVAILABLE = True
except ImportError:
    CUTLASS_AVAILABLE = False
    warnings.warn(
        "CUTLASS PEER kernel not compiled. Please run 'python setup.py build_ext --inplace' "
        "or use the PyTorch implementation by unsetting USE_CUTLASS_KERNEL environment variable."
    )


def _get_kernel_config_key(num_heads: int, query_dim: int, num_experts: int, 
                           output_dim: int, top_k: int) -> str:
    """Generate a unique key for a kernel configuration."""
    sqrt_n = int(num_experts ** 0.5)
    return f"h{num_heads}_q{query_dim}_e{sqrt_n}_o{output_dim}_k{top_k}"


def compile_cutlass_kernel_jit(num_heads: int, query_dim: int, num_experts: int,
                               output_dim: int, top_k: int, input_dim: int,
                               expert_hidden_size: int):
    """JIT compile a CUTLASS kernel with specific template parameters."""
    config_key = _get_kernel_config_key(num_heads, query_dim, num_experts, output_dim, top_k)
    
    # Check if already compiled
    if config_key in _compiled_kernels:
        return _compiled_kernels[config_key]
    
    kernel_dir = Path(__file__).parent
    wrapper_file = kernel_dir / "peer_cutlass_wrapper.cpp"
    cuda_file = kernel_dir / "peer_cutlass.cu"
    header_file = kernel_dir / "peer_cutlass.h"
    
    if not all(f.exists() for f in [wrapper_file, cuda_file, header_file]):
        raise FileNotFoundError("CUTLASS kernel source files not found")
    
    # Create a temporary directory for this specific build
    import torch.utils.cpp_extension as cpp_ext
    
    sqrt_n = int(num_experts ** 0.5)
    
    # Generate a custom wrapper that instantiates the specific template
    wrapper_content = f"""
#include <torch/extension.h>
#include "peer_cutlass_wrapper.cpp"

// Force instantiation of specific template configuration
namespace {{
    const int PEER_NUM_HEADS = {num_heads};
    const int PEER_QUERY_DIM = {query_dim};
    const int PEER_SQRT_N = {sqrt_n};
    const int PEER_OUTPUT_DIM = {output_dim};
    const int PEER_TOP_K = {top_k};
    const int PEER_INPUT_DIM = {input_dim};
    const int PEER_EXPERT_HIDDEN = {expert_hidden_size};
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("peer_forward", &peer_forward, "PEER forward pass (CUTLASS)");
    m.def("print_cache_stats", &print_cache_stats, "Print cache statistics");
}}
"""
    
    # Create temporary wrapper file
    temp_dir = tempfile.mkdtemp(prefix="peer_cutlass_")
    temp_wrapper = Path(temp_dir) / "wrapper_jit.cpp"
    temp_wrapper.write_text(wrapper_content)
    
    try:
        # Compile with specific defines
        module = cpp_ext.load(
            name=f"peer_cutlass_{config_key}",
            sources=[str(temp_wrapper), str(cuda_file)],
            extra_cflags=[
                f"-DPEER_JIT_NUM_HEADS={num_heads}",
                f"-DPEER_JIT_QUERY_DIM={query_dim}",
                f"-DPEER_JIT_SQRT_N={sqrt_n}",
                f"-DPEER_JIT_OUTPUT_DIM={output_dim}",
                f"-DPEER_JIT_TOP_K={top_k}",
                "-std=c++17",
            ],
            extra_cuda_cflags=[
                f"-DPEER_JIT_NUM_HEADS={num_heads}",
                f"-DPEER_JIT_QUERY_DIM={query_dim}",
                f"-DPEER_JIT_SQRT_N={sqrt_n}",
                f"-DPEER_JIT_OUTPUT_DIM={output_dim}",
                f"-DPEER_JIT_TOP_K={top_k}",
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
            verbose=False
        )
        
        _compiled_kernels[config_key] = module
        return module
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def compile_cutlass_kernel():
    """Compile the CUTLASS kernel if not already compiled."""
    # This is now deprecated in favor of JIT compilation
    warnings.warn(
        "compile_cutlass_kernel() is deprecated. Kernels are now JIT compiled with specific configurations.",
        DeprecationWarning
    )


def peer_forward_cutlass(
    x: torch.Tensor,
    query_weight: torch.Tensor,
    query_bias: Optional[torch.Tensor],
    key_weight_1: torch.Tensor,
    key_weight_2: torch.Tensor,
    expert_weights_u: torch.Tensor,
    expert_weights_v: torch.Tensor,
    num_heads: int,
    num_experts: int,
    expert_hidden_size: int,
    top_k: int,
    dropout_rate: float = 0.0,
    layer_norm: bool = True,
    ln_weight: Optional[torch.Tensor] = None,
    ln_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    CUTLASS implementation of PEER forward pass.
    
    Args:
        x: Input tensor [batch_size, seq_len, input_dim]
        query_weight: Query projection weights [num_heads, query_dim, input_dim]
        query_bias: Query projection bias [num_heads, query_dim] or None
        key_weight_1: First product key matrix [sqrt(num_experts), query_dim]
        key_weight_2: Second product key matrix [sqrt(num_experts), query_dim]
        expert_weights_u: Down projection weights [num_experts, expert_hidden_size, input_dim]
        expert_weights_v: Up projection weights [num_experts, output_dim, expert_hidden_size]
        num_heads: Number of attention heads
        num_experts: Total number of experts
        expert_hidden_size: Hidden dimension of expert networks
        top_k: Number of experts to select
        dropout_rate: Dropout rate (applied if > 0)
        layer_norm: Whether to apply layer normalization to queries
        ln_weight: Layer norm weight [num_heads, query_dim] if layer_norm=True
        ln_bias: Layer norm bias [num_heads, query_dim] if layer_norm=True
        
    Returns:
        Output tensor [batch_size, seq_len, output_dim]
    """
    # Get dimensions
    batch_size, seq_len, input_dim = x.shape
    output_dim = expert_weights_v.shape[1]
    query_dim = query_weight.shape[2] if query_weight.dim() == 3 else query_weight.shape[1]
    
    # JIT compile kernel for this configuration
    try:
        kernel_module = compile_cutlass_kernel_jit(
            num_heads, query_dim, num_experts, output_dim, top_k,
            input_dim, expert_hidden_size
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to JIT compile CUTLASS kernel: {e}. "
            "Please use PyTorch implementation by unsetting USE_CUTLASS_KERNEL."
        )
    
    # Ensure inputs are contiguous and in correct format
    x = x.contiguous().half()
    query_weight = query_weight.contiguous().half()
    if query_bias is not None:
        query_bias = query_bias.contiguous().half()
    key_weight_1 = key_weight_1.contiguous().half()
    key_weight_2 = key_weight_2.contiguous().half()
    expert_weights_u = expert_weights_u.contiguous().half()
    expert_weights_v = expert_weights_v.contiguous().half()
    
    if layer_norm and ln_weight is not None:
        ln_weight = ln_weight.contiguous().half()
        ln_bias = ln_bias.contiguous().half() if ln_bias is not None else None
    
    # Call CUTLASS kernel with JIT compiled module
    output = kernel_module.peer_forward(
        x,
        query_weight,
        query_bias if query_bias is not None else torch.empty(0, dtype=torch.half, device=x.device),
        key_weight_1,
        key_weight_2,
        expert_weights_u,
        expert_weights_v,
        ln_weight if layer_norm and ln_weight is not None else torch.empty(0, dtype=torch.half, device=x.device),
        ln_bias if layer_norm and ln_bias is not None else torch.empty(0, dtype=torch.half, device=x.device),
        batch_size,
        seq_len,
        input_dim,
        output_dim,
        num_heads,
        num_experts,
        expert_hidden_size,
        top_k,
        layer_norm,
        dropout_rate,
    )
    
    return output