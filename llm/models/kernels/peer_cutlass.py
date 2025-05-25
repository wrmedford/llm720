"""Python interface for CUTLASS PEER kernel with FP8 support."""
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
                           output_dim: int, top_k: int, use_fp8: bool = False) -> str:
    """Generate a unique key for a kernel configuration."""
    sqrt_n = int(num_experts ** 0.5)
    fp8_suffix = "_fp8" if use_fp8 else ""
    return f"h{num_heads}_q{query_dim}_e{sqrt_n}_o{output_dim}_k{top_k}{fp8_suffix}"


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
    use_fp8: bool = False,  # Add FP8 flag
) -> torch.Tensor:
    """
    CUTLASS implementation of PEER forward pass with optional FP8 support.
    
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
        use_fp8: Whether to use FP8 expert weights (converts internally)
        
    Returns:
        Output tensor [batch_size, seq_len, output_dim]
    """
    # Get dimensions
    batch_size, seq_len, input_dim = x.shape
    output_dim = expert_weights_v.shape[1]
    query_dim = query_weight.shape[2] if query_weight.dim() == 3 else query_weight.shape[1]
    
    # Check if FP8 is requested
    use_fp8_env = os.environ.get("PEER_USE_FP8", "0") == "1"
    use_fp8 = use_fp8 or use_fp8_env
    
    # Use pre-compiled kernel module
    try:
        from . import peer_cutlass_module as kernel_module
    except ImportError as e:
        raise RuntimeError(
            f"Failed to import CUTLASS kernel: {e}. "
            "Please build with 'python setup.py build_ext --inplace' or "
            "use PyTorch implementation by unsetting USE_CUTLASS_KERNEL."
        )
    
    # Ensure inputs are contiguous and in correct format
    x = x.contiguous().half()
    query_weight = query_weight.contiguous().half()
    if query_bias is not None:
        query_bias = query_bias.contiguous().half()
    key_weight_1 = key_weight_1.contiguous().half()
    key_weight_2 = key_weight_2.contiguous().half()
    
    # Handle expert weights - convert to FP8 if requested
    if use_fp8:
        # The kernel will handle FP8 conversion internally
        expert_weights_u = expert_weights_u.contiguous().half()
        expert_weights_v = expert_weights_v.contiguous().half()
    else:
        expert_weights_u = expert_weights_u.contiguous().half()
        expert_weights_v = expert_weights_v.contiguous().half()
    
    if layer_norm and ln_weight is not None:
        ln_weight = ln_weight.contiguous().half()
        ln_bias = ln_bias.contiguous().half() if ln_bias is not None else None
    
    # Create an output tensor to be filled by the kernel
    output_tensor = torch.empty(batch_size, seq_len, output_dim, dtype=x.dtype, device=x.device)
    
    # Call CUTLASS kernel with pre-compiled module
    # The module should have a peer_forward_fp8 function if FP8 is supported
    forward_fn = getattr(kernel_module, 'peer_forward_fp8' if use_fp8 else 'peer_forward', None)
    if forward_fn is None:
        if use_fp8:
            warnings.warn("FP8 support not available in compiled kernel. Falling back to FP16.")
        forward_fn = kernel_module.peer_forward
    
    output = forward_fn(
        x,
        query_weight,
        query_bias if query_bias is not None else torch.empty(0, dtype=torch.half, device=x.device),
        key_weight_1,
        key_weight_2,
        expert_weights_u,
        expert_weights_v,
        output_tensor,
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
        False,  # norm_keys
        False,  # norm_query
        dropout_rate,
        use_fp8  # Pass FP8 flag to kernel
    )
    
    return output