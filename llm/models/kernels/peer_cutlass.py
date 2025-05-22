"""Python interface for CUTLASS PEER kernel."""
import torch
import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple
import warnings

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


def compile_cutlass_kernel():
    """Compile the CUTLASS kernel if not already compiled."""
    kernel_dir = Path(__file__).parent
    cuda_file = kernel_dir / "peer_cutlass.cu"
    
    if not cuda_file.exists():
        raise FileNotFoundError(f"CUTLASS kernel source not found at {cuda_file}")
    
    # Get CUDA and Python paths
    import torch.utils.cpp_extension as cpp_ext
    
    # Build command
    build_dir = kernel_dir / "build"
    build_dir.mkdir(exist_ok=True)
    
    # Compile with appropriate flags
    cpp_ext.load(
        name="peer_cutlass_module",
        sources=[str(cuda_file)],
        extra_cuda_cflags=[
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
        build_directory=str(build_dir),
        verbose=True
    )
    
    global CUTLASS_AVAILABLE
    CUTLASS_AVAILABLE = True


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
    if not CUTLASS_AVAILABLE:
        # Try to compile on-demand
        try:
            compile_cutlass_kernel()
            import peer_cutlass_module
        except Exception as e:
            raise RuntimeError(
                f"Failed to compile CUTLASS kernel: {e}. "
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
    
    batch_size, seq_len, input_dim = x.shape
    output_dim = expert_weights_v.shape[1]
    
    # Allocate output tensor
    output = torch.zeros(batch_size, seq_len, output_dim, dtype=torch.half, device=x.device)
    
    # Call CUTLASS kernel
    peer_cutlass_module.peer_forward(
        x,
        query_weight,
        query_bias if query_bias is not None else torch.empty(0, dtype=torch.half, device=x.device),
        key_weight_1,
        key_weight_2,
        expert_weights_u,
        expert_weights_v,
        output,
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
        True,  # norm_keys
        True,  # norm_query
        dropout_rate,
    )
    
    return output