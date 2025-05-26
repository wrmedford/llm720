#!/usr/bin/env python3
"""Simple test to verify FA3 works correctly."""

import sys
import torch

# Add FA3 to path
sys.path.insert(0, '/home/ubuntu/.local/lib/python3.10/site-packages')

# Import FA3
import flash_attn_interface

print("FA3 imported successfully!")
print("Available functions:", [x for x in dir(flash_attn_interface) if 'flash' in x])

# Test simple forward pass
if torch.cuda.is_available():
    device = torch.device("cuda")
    dtype = torch.float16  # FA3 requires FP16/BF16
    
    batch_size = 2
    seq_len = 16
    num_heads = 4
    head_dim = 64
    
    # Create random tensors
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    
    print(f"\nTesting FA3 forward pass...")
    print(f"Q shape: {q.shape}, dtype: {q.dtype}")
    print(f"K shape: {k.shape}, dtype: {k.dtype}")
    print(f"V shape: {v.shape}, dtype: {v.dtype}")
    
    try:
        # Run FA3
        output = flash_attn_interface.flash_attn_func(
            q, k, v,
            causal=True,
            window_size=(-1, -1),
            softcap=0.0,
        )
        print(f"\nSuccess! Output shape: {output[0].shape}")
    except Exception as e:
        print(f"\nError running FA3: {e}")
        import traceback
        traceback.print_exc()
else:
    print("CUDA not available!")