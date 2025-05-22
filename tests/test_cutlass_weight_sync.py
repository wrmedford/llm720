"""Test that CUTLASS kernel uses correct weights from PyTorch tensors."""
import torch
import pytest
import os

# Set environment variable before importing
os.environ["USE_CUTLASS_KERNEL"] = "1"

from llm.models.kernels.peer_cutlass import peer_forward_cutlass


def test_weight_synchronization():
    """Test that CUTLASS kernel correctly uses PyTorch-managed weights."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Small test configuration
    batch_size = 2
    seq_len = 4
    input_dim = 64
    output_dim = 128
    num_heads = 8
    num_experts = 16
    expert_hidden_size = 32
    top_k = 4
    query_dim = input_dim // num_heads
    
    # Create input tensors
    x = torch.randn(batch_size, seq_len, input_dim, device=device, dtype=torch.float16)
    
    # Create weight tensors with known patterns
    query_weight = torch.randn(num_heads, input_dim, query_dim, device=device, dtype=torch.float16)
    query_bias = torch.randn(num_heads, query_dim, device=device, dtype=torch.float16)
    key_weight_1 = torch.randn(num_heads, input_dim, int(num_experts**0.5), device=device, dtype=torch.float16)
    key_weight_2 = torch.randn(num_heads, input_dim, int(num_experts**0.5), device=device, dtype=torch.float16)
    
    # Create expert weights with specific patterns to verify they're being used
    # U weights: [num_experts, input_dim, expert_hidden_size]
    expert_weights_u = torch.zeros(num_experts, input_dim, expert_hidden_size, device=device, dtype=torch.float16)
    # V weights: [num_experts, expert_hidden_size, output_dim]
    expert_weights_v = torch.zeros(num_experts, expert_hidden_size, output_dim, device=device, dtype=torch.float16)
    
    # Fill with specific patterns - each expert has a unique pattern
    for i in range(num_experts):
        expert_weights_u[i] = (i + 1) * 0.01  # Each expert has different magnitude
        expert_weights_v[i] = (i + 1) * 0.01
    
    # First forward pass
    output1 = peer_forward_cutlass(
        x, query_weight, query_bias, key_weight_1, key_weight_2,
        expert_weights_u, expert_weights_v,
        num_heads, num_experts, expert_hidden_size, top_k,
        dropout_rate=0.0, layer_norm=False
    )
    
    # Verify output is non-zero (weights are being used)
    assert not torch.allclose(output1, torch.zeros_like(output1)), \
        "Output should be non-zero, indicating weights are being used"
    
    # Store original output for comparison
    output1_copy = output1.clone()
    
    # Modify weights
    expert_weights_u *= 2.0
    expert_weights_v *= 2.0
    
    # Second forward pass - behavior depends on PEER_DIRECT_WEIGHT_ACCESS env var
    output2 = peer_forward_cutlass(
        x, query_weight, query_bias, key_weight_1, key_weight_2,
        expert_weights_u, expert_weights_v,
        num_heads, num_experts, expert_hidden_size, top_k,
        dropout_rate=0.0, layer_norm=False
    )
    
    # Check behavior based on environment variable
    use_direct_mode = os.environ.get("PEER_DIRECT_WEIGHT_ACCESS", "0") == "1"
    
    if use_direct_mode:
        # In direct mode, weights should be updated immediately
        assert not torch.allclose(output1_copy, output2, rtol=1e-3, atol=1e-3), \
            "In direct mode, outputs should differ after weight modification"
        # Verify the output roughly doubled (since we doubled the weights)
        ratio = torch.mean(torch.abs(output2)) / torch.mean(torch.abs(output1_copy))
        assert 1.8 < ratio < 2.2, f"Output should roughly double, got ratio {ratio:.3f}"
    else:
        # In copy mode, weights are also updated on each forward pass
        assert not torch.allclose(output1_copy, output2, rtol=1e-3, atol=1e-3), \
            "In copy mode, outputs should also differ after weight modification"
        # Verify the output roughly doubled
        ratio = torch.mean(torch.abs(output2)) / torch.mean(torch.abs(output1_copy))
        assert 1.8 < ratio < 2.2, f"Output should roughly double, got ratio {ratio:.3f}"
    
    print("Weight synchronization test passed!")


def test_direct_mode_vs_copy_mode():
    """Test the difference between direct and copy modes."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Small test configuration
    batch_size = 2
    seq_len = 4
    input_dim = 64
    output_dim = 128
    num_heads = 8
    num_experts = 16
    expert_hidden_size = 32
    top_k = 4
    query_dim = input_dim // num_heads
    
    # Create identical inputs for both tests
    x = torch.randn(batch_size, seq_len, input_dim, device=device, dtype=torch.float16)
    query_weight = torch.randn(num_heads, input_dim, query_dim, device=device, dtype=torch.float16)
    query_bias = torch.randn(num_heads, query_dim, device=device, dtype=torch.float16)
    key_weight_1 = torch.randn(num_heads, input_dim, int(num_experts**0.5), device=device, dtype=torch.float16)
    key_weight_2 = torch.randn(num_heads, input_dim, int(num_experts**0.5), device=device, dtype=torch.float16)
    
    # Create expert weights
    expert_weights_u = torch.randn(num_experts, input_dim, expert_hidden_size, device=device, dtype=torch.float16)
    expert_weights_v = torch.randn(num_experts, expert_hidden_size, output_dim, device=device, dtype=torch.float16)
    
    # Test direct mode
    os.environ["PEER_DIRECT_WEIGHT_ACCESS"] = "1"
    output_direct = peer_forward_cutlass(
        x, query_weight, query_bias, key_weight_1, key_weight_2,
        expert_weights_u.clone(), expert_weights_v.clone(),
        num_heads, num_experts, expert_hidden_size, top_k,
        dropout_rate=0.0, layer_norm=False
    )
    
    # Test copy mode
    os.environ["PEER_DIRECT_WEIGHT_ACCESS"] = "0"
    output_copy = peer_forward_cutlass(
        x, query_weight, query_bias, key_weight_1, key_weight_2,
        expert_weights_u.clone(), expert_weights_v.clone(),
        num_heads, num_experts, expert_hidden_size, top_k,
        dropout_rate=0.0, layer_norm=False
    )
    
    # Both modes should produce the same output for the same weights
    assert torch.allclose(output_direct, output_copy, rtol=1e-2, atol=1e-3), \
        "Direct and copy modes should produce identical outputs for the same weights"
    
    print("Direct mode vs copy mode test passed!")


if __name__ == "__main__":
    test_weight_synchronization()
    test_direct_mode_vs_copy_mode()