import torch
import os
import importlib.util
import pytest


class TestCutlassConfigurations:
    """Tests for different CUTLASS expert configurations."""
    
    @pytest.fixture
    def cutlass_module(self):
        """Load CUTLASS module for testing."""
        spec = importlib.util.spec_from_file_location(
            "peer_cutlass_module", 
            "/root/llm720/llm/models/kernels/peer_cutlass_module.cpython-312-x86_64-linux-gnu.so"
        )
        peer_cutlass_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(peer_cutlass_module)
        return peer_cutlass_module
    
    def test_realistic_configuration(self, cutlass_module):
        """Test with realistic configuration (512 experts)."""
        # Set environment for direct weight access
        os.environ["PEER_DIRECT_WEIGHT_ACCESS"] = "1"
        
        # Realistic test configuration
        batch_size = 2
        seq_len = 4
        input_dim = 64
        output_dim = 128
        num_experts = 512
        num_heads = 8
        num_experts_per_tok = 4
        expert_hidden_size = 32
        query_dim = 64
        top_k = num_experts_per_tok
        
        sqrt_n = int(num_experts ** 0.5)
        assert sqrt_n == 22  # Floor of sqrt(512)
        
        # Create tensors
        x = torch.randn(batch_size, seq_len, input_dim).cuda().half().contiguous()
        query_weight = torch.randn(num_heads, query_dim, input_dim).cuda().half().contiguous()
        query_bias = torch.randn(num_heads, query_dim).cuda().half().contiguous()
        key_weight_1 = torch.randn(sqrt_n, query_dim).cuda().half().contiguous()
        key_weight_2 = torch.randn(sqrt_n, query_dim).cuda().half().contiguous()
        expert_weights_u = torch.randn(num_experts, expert_hidden_size, input_dim).cuda().half().contiguous()
        expert_weights_v = torch.randn(num_experts, output_dim, expert_hidden_size).cuda().half().contiguous()
        output = torch.zeros(batch_size, seq_len, output_dim).cuda().half().contiguous()
        ln_weight = torch.ones(num_heads, query_dim).cuda().half().contiguous()
        ln_bias = torch.zeros(num_heads, query_dim).cuda().half().contiguous()
        
        # Test forward pass
        result = cutlass_module.peer_forward(
            x, query_weight, query_bias, key_weight_1, key_weight_2,
            expert_weights_u, expert_weights_v, output, ln_weight, ln_bias,
            batch_size, seq_len, input_dim, output_dim, num_heads,
            num_experts, expert_hidden_size, top_k,
            True,   # layer_norm
            False,  # norm_keys
            False,  # norm_query
            0.0     # dropout_rate
        )
        
        assert result.shape == (batch_size, seq_len, output_dim)
        assert not torch.isnan(result).any()
        
        # Clean up
        if "PEER_DIRECT_WEIGHT_ACCESS" in os.environ:
            del os.environ["PEER_DIRECT_WEIGHT_ACCESS"]
    
    def test_perfect_square_configuration(self, cutlass_module):
        """Test with perfect square number of experts (484 = 22²)."""
        # Set environment for direct weight access
        os.environ["PEER_DIRECT_WEIGHT_ACCESS"] = "1"
        
        # Perfect square configuration
        batch_size = 2
        seq_len = 4
        input_dim = 64
        output_dim = 128
        num_experts = 484  # 22 * 22 = 484 (perfect square)
        num_heads = 8
        num_experts_per_tok = 4
        expert_hidden_size = 32
        query_dim = 64
        top_k = num_experts_per_tok
        
        sqrt_n = int(num_experts ** 0.5)
        assert sqrt_n == 22
        assert sqrt_n * sqrt_n == num_experts  # Perfect square
        
        # Create tensors
        x = torch.randn(batch_size, seq_len, input_dim).cuda().half().contiguous()
        query_weight = torch.randn(num_heads, query_dim, input_dim).cuda().half().contiguous()
        query_bias = torch.randn(num_heads, query_dim).cuda().half().contiguous()
        key_weight_1 = torch.randn(sqrt_n, query_dim).cuda().half().contiguous()
        key_weight_2 = torch.randn(sqrt_n, query_dim).cuda().half().contiguous()
        expert_weights_u = torch.randn(num_experts, expert_hidden_size, input_dim).cuda().half().contiguous()
        expert_weights_v = torch.randn(num_experts, output_dim, expert_hidden_size).cuda().half().contiguous()
        output = torch.zeros(batch_size, seq_len, output_dim).cuda().half().contiguous()
        ln_weight = torch.ones(num_heads, query_dim).cuda().half().contiguous()
        ln_bias = torch.zeros(num_heads, query_dim).cuda().half().contiguous()
        
        # Test forward pass
        result = cutlass_module.peer_forward(
            x, query_weight, query_bias, key_weight_1, key_weight_2,
            expert_weights_u, expert_weights_v, output, ln_weight, ln_bias,
            batch_size, seq_len, input_dim, output_dim, num_heads,
            num_experts, expert_hidden_size, top_k,
            True,   # layer_norm
            False,  # norm_keys
            False,  # norm_query
            0.0     # dropout_rate
        )
        
        assert result.shape == (batch_size, seq_len, output_dim)
        assert not torch.isnan(result).any()
        
        # Clean up
        if "PEER_DIRECT_WEIGHT_ACCESS" in os.environ:
            del os.environ["PEER_DIRECT_WEIGHT_ACCESS"]
    
    def test_square_keys_with_peer_model(self):
        """Test CUTLASS kernel through PEER model with square product keys."""
        # Set environment to use CUTLASS
        os.environ["USE_CUTLASS_KERNEL"] = "1"
        
        from llm.models.experts import PEER
        
        # Test configuration with square product keys
        batch_size = 2
        seq_len = 4
        input_dim = 64
        output_dim = 128
        num_experts = 484  # 22 * 22 = 484 (perfect square)
        num_heads = 8
        num_experts_per_tok = 4
        expert_hidden_size = 32
        
        # Create PEER module with square product keys
        model = PEER(
            input_dim=input_dim,
            output_dim=output_dim,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            num_heads=num_heads,
            expert_hidden_size=expert_hidden_size,
            query_dim=64,
            product_key_dim=[22, 22],  # Square factorization!
            batch_norm_query=True,
        ).cuda()
        
        # Verify product key dimensions
        assert model.product_key_dim == [22, 22]
        assert len(model.sub_keys) == 2
        assert model.sub_keys[0].shape == (22, 64)
        assert model.sub_keys[1].shape == (22, 64)
        
        # Create input
        x = torch.randn(batch_size, seq_len, input_dim).cuda().half()
        
        # Forward pass
        output = model(x)
        assert output.shape == (batch_size, seq_len, output_dim)
        assert not torch.isnan(output).any()
        
        # Clean up
        if "USE_CUTLASS_KERNEL" in os.environ:
            del os.environ["USE_CUTLASS_KERNEL"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])