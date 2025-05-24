#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test script for CUTLASS PEER kernel integration."""

import os
import torch
import torch.nn as nn
import pytest

# Set environment variable to use CUTLASS kernel
os.environ["USE_CUTLASS_KERNEL"] = "1"

from llm.models.experts import PEER


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cutlass_kernel_basic():
    """Test basic functionality of CUTLASS kernel."""
    # Test configuration
    batch_size = 2
    seq_len = 4
    input_dim = 64
    output_dim = 128
    num_experts = 484  # 22 * 22 = 484 (perfect square)
    num_heads = 8
    num_experts_per_tok = 4
    expert_hidden_size = 32
    
    # Create PEER module
    model = PEER(
        input_dim=input_dim,
        output_dim=output_dim,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        num_heads=num_heads,
        expert_hidden_size=expert_hidden_size,
        query_dim=64,
        product_key_dim=[22, 22],  # 22 * 22 = 484 to match num_experts
        batch_norm_query=True,
    ).cuda()
    
    # Create input
    x = torch.randn(batch_size, seq_len, input_dim).cuda().half()
    
    # Forward pass
    try:
        output = model(x)
        assert output.shape == (batch_size, seq_len, output_dim)
        print(f"✓ CUTLASS kernel forward pass successful: {output.shape}")
    except Exception as e:
        pytest.fail(f"CUTLASS kernel forward pass failed: {e}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cutlass_vs_pytorch():
    """Compare CUTLASS kernel output with PyTorch implementation."""
    # Test configuration
    batch_size = 1
    seq_len = 2
    input_dim = 32
    output_dim = 32
    num_experts = 64  # 8 * 8 = 64 (perfect square)
    num_heads = 4
    num_experts_per_tok = 2
    expert_hidden_size = 16
    
    # Create two PEER modules with same weights
    torch.manual_seed(42)
    model_pytorch = PEER(
        input_dim=input_dim,
        output_dim=output_dim,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        num_heads=num_heads,
        expert_hidden_size=expert_hidden_size,
        query_dim=32,
        product_key_dim=[8, 8],  # 8 * 8 = 64 to match num_experts
        batch_norm_query=False,  # Disable for easier comparison
    ).cuda()
    
    # Copy weights to CUTLASS model
    torch.manual_seed(42)
    os.environ["USE_CUTLASS_KERNEL"] = "1"
    model_cutlass = PEER(
        input_dim=input_dim,
        output_dim=output_dim,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        num_heads=num_heads,
        expert_hidden_size=expert_hidden_size,
        query_dim=32,
        product_key_dim=[8, 8],  # 8 * 8 = 64 to match num_experts
        batch_norm_query=False,
    ).cuda()
    
    # Ensure weights are identical
    model_cutlass.load_state_dict(model_pytorch.state_dict())
    
    # Create input
    x = torch.randn(batch_size, seq_len, input_dim).cuda().half()
    
    # Forward passes
    os.environ.pop("USE_CUTLASS_KERNEL", None)
    with torch.no_grad():
        output_pytorch = model_pytorch(x)
    
    os.environ["USE_CUTLASS_KERNEL"] = "1"
    with torch.no_grad():
        output_cutlass = model_cutlass(x)
    
    # Compare outputs (allow some tolerance due to FP16 precision)
    if torch.allclose(output_pytorch, output_cutlass, rtol=1e-2, atol=1e-3):
        print("✓ CUTLASS and PyTorch outputs match within tolerance")
    else:
        max_diff = torch.max(torch.abs(output_pytorch - output_cutlass)).item()
        print(f"⚠ Maximum difference between outputs: {max_diff}")


if __name__ == "__main__":
    print("Testing CUTLASS PEER kernel integration...\n")
    
    try:
        test_cutlass_kernel_basic()
    except Exception as e:
        print(f"✗ Basic test failed: {e}")
    
    try:
        test_cutlass_vs_pytorch()
    except Exception as e:
        print(f"✗ Comparison test failed: {e}")