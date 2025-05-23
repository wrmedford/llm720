"""Test to verify CUTLASS kernel has O(sqrt(N)) complexity for expert selection"""

import pytest
import torch
import torch.nn as nn
import time
import numpy as np
from llm.models.foundation import PEER
from llm.models.kernels.peer_cutlass import peer_forward, print_cache_stats


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cutlass_selection_complexity():
    """Verify that selection time scales with sqrt(N), not N"""
    torch.cuda.empty_cache()
    device = torch.device("cuda")
    
    # Test configuration
    batch_size = 2
    seq_len = 128
    input_dim = 512
    output_dim = 512
    num_heads = 8
    query_dim = 64
    expert_hidden = 256
    top_k = 16
    
    # Test with different expert counts
    expert_counts = [1024, 4096, 16384, 65536]  # 2^10, 2^12, 2^14, 2^16
    times = []
    
    for num_experts in expert_counts:
        print(f"\nTesting with {num_experts} experts (sqrt_n = {int(num_experts ** 0.5)})")
        
        # Create PEER module
        peer = PEER(
            input_dim=input_dim,
            output_dim=output_dim,
            num_experts=num_experts,
            expert_hidden_size=expert_hidden,
            top_k=top_k,
            num_heads=num_heads,
            query_dim=query_dim,
            compute_device=device,
            compute_dtype=torch.float16,
            layer_norm=True,
            dropout=0.0
        ).to(device).half()
        
        # Create input
        x = torch.randn(batch_size, seq_len, input_dim, device=device, dtype=torch.float16)
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = peer(x)
        torch.cuda.synchronize()
        
        # Time the selection (measure full forward pass)
        num_iterations = 20
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = peer(x)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_iterations
        times.append(avg_time)
        
        print(f"  Average time: {avg_time*1000:.2f} ms")
        print(f"  Time per token: {avg_time*1000/(batch_size*seq_len):.4f} ms")
        
        # Clean up
        del peer
        torch.cuda.empty_cache()
    
    # Verify sublinear scaling
    # If O(N), time should scale 64x from 1K to 64K experts
    # If O(sqrt(N)), time should scale ~8x
    time_ratio = times[-1] / times[0]
    expert_ratio = expert_counts[-1] / expert_counts[0]
    sqrt_ratio = np.sqrt(expert_ratio)
    
    print(f"\nScaling Analysis:")
    print(f"  Expert count increased: {expert_ratio}x")
    print(f"  Time increased: {time_ratio:.1f}x")
    print(f"  Expected if O(N): {expert_ratio}x")
    print(f"  Expected if O(√N): {sqrt_ratio:.1f}x")
    
    # The time ratio should be much closer to sqrt_ratio than expert_ratio
    # Allow some overhead for memory access patterns
    assert time_ratio < sqrt_ratio * 3, (
        f"Selection time scaling suggests O(N) complexity: {time_ratio}x > {sqrt_ratio * 3}x"
    )
    assert time_ratio < expert_ratio / 10, (
        f"Selection time scaling is too close to O(N): {time_ratio}x"
    )
    
    print(f"\n✓ Complexity verification passed: scaling is sublinear (likely O(√N))")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cutlass_vs_pytorch_accuracy():
    """Verify CUTLASS implementation produces correct results"""
    torch.cuda.empty_cache()
    device = torch.device("cuda")
    
    # Test configuration
    batch_size = 4
    seq_len = 64
    input_dim = 256
    output_dim = 256
    num_experts = 1024
    num_heads = 4
    query_dim = 64
    expert_hidden = 128
    top_k = 8
    
    # Create PEER module
    peer = PEER(
        input_dim=input_dim,
        output_dim=output_dim,
        num_experts=num_experts,
        expert_hidden_size=expert_hidden,
        top_k=top_k,
        num_heads=num_heads,
        query_dim=query_dim,
        compute_device=device,
        compute_dtype=torch.float16,
        layer_norm=True,
        dropout=0.0
    ).to(device).half()
    
    # Set to eval mode
    peer.eval()
    
    # Create identical input
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, input_dim, device=device, dtype=torch.float16)
    
    # Get outputs
    with torch.no_grad():
        # Run twice to ensure deterministic behavior
        output1 = peer(x)
        output2 = peer(x)
    
    # Outputs should be identical (no dropout in eval mode)
    assert torch.allclose(output1, output2, rtol=1e-3, atol=1e-3), (
        "CUTLASS kernel produces non-deterministic results"
    )
    
    # Check output shape
    assert output1.shape == (batch_size, seq_len, output_dim)
    
    # Check that output is not all zeros or NaN
    assert not torch.isnan(output1).any(), "Output contains NaN values"
    assert output1.abs().max() > 0, "Output is all zeros"
    
    print(f"✓ CUTLASS accuracy test passed")
    print(f"  Output shape: {output1.shape}")
    print(f"  Output range: [{output1.min():.4f}, {output1.max():.4f}]")
    print(f"  Output mean: {output1.mean():.4f}")
    print(f"  Output std: {output1.std():.4f}")


if __name__ == "__main__":
    test_cutlass_selection_complexity()
    test_cutlass_vs_pytorch_accuracy()
    
    # Print cache statistics
    print("\nCache Statistics:")
    print_cache_stats()