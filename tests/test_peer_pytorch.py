#!/usr/bin/env python3
"""Test PyTorch PEER implementation without CUTLASS kernel."""

import os
import torch
import pytest
from llm.models.experts import PEER


class TestPeerPyTorch:
    """Test PyTorch PEER implementation without CUTLASS kernel."""
    
    @pytest.fixture(autouse=True)
    def setup_environment(self):
        """Ensure we use PyTorch implementation, not CUTLASS."""
        # Remove CUTLASS environment variable if set
        original_value = os.environ.pop("USE_CUTLASS_KERNEL", None)
        yield
        # Restore original value if it existed
        if original_value is not None:
            os.environ["USE_CUTLASS_KERNEL"] = original_value
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_pytorch_peer_forward(self):
        """Test PyTorch PEER forward pass."""
        # Test configuration
        batch_size = 2
        seq_len = 4
        input_dim = 64
        output_dim = 128
        num_experts = 512
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
            product_key_dim=[16, 32],  # 16 * 32 = 512 to match num_experts
            batch_norm_query=True,
        ).cuda()
        
        # Verify model parameters
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0
        
        # Create input
        x = torch.randn(batch_size, seq_len, input_dim).cuda().half()
        
        # Forward pass
        with torch.cuda.amp.autocast():
            output = model(x)
        
        # Verify output
        assert output.shape == (batch_size, seq_len, output_dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_pytorch_peer_gradient_flow(self):
        """Test PyTorch PEER gradient flow."""
        # Test configuration
        batch_size = 2
        seq_len = 4
        input_dim = 64
        output_dim = 128
        num_experts = 512
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
            product_key_dim=[16, 32],
            batch_norm_query=True,
        ).cuda()
        
        # Create input
        x = torch.randn(batch_size, seq_len, input_dim).cuda().half()
        
        # Forward pass and compute loss
        with torch.cuda.amp.autocast():
            output = model(x)
        loss = output.mean()
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        params_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.requires_grad)
        total_params = sum(1 for p in model.parameters() if p.requires_grad)
        
        # All parameters should have gradients
        assert params_with_grad == total_params
        
        # Check that gradients are not NaN or Inf
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])