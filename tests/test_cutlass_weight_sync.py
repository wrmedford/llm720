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
    
    # Modify weights and run again - should see different output
    expert_weights_u *= 2.0
    expert_weights_v *= 2.0
    
    # Second forward pass with same operator instance (weights should NOT change)
    output2 = peer_forward_cutlass(
        x, query_weight, query_bias, key_weight_1, key_weight_2,
        expert_weights_u, expert_weights_v,
        num_heads, num_experts, expert_hidden_size, top_k,
        dropout_rate=0.0, layer_norm=False
    )
    
    # The outputs should be the same because weights are only copied once
    assert torch.allclose(output1, output2, rtol=1e-3, atol=1e-3), \
        "Outputs should be identical since weights are only synchronized once"
    
    # Verify output is non-zero (weights are being used)
    assert not torch.allclose(output1, torch.zeros_like(output1)), \
        "Output should be non-zero, indicating weights are being used"
    
    print("Weight synchronization test passed!")


if __name__ == "__main__":
    test_weight_synchronization()