import torch
import os
import sys
import importlib.util
import pytest


class TestCutlassSetup:
    """Tests for CUTLASS module loading and basic setup."""
    
    @pytest.fixture
    def cutlass_module(self):
        """Load CUTLASS module for testing."""
        # Try direct import first
        try:
            sys.path.insert(0, '/root/llm720')
            from llm.models.kernels import peer_cutlass_module
            return peer_cutlass_module
        except ImportError:
            # Try importing the .so file directly
            spec = importlib.util.spec_from_file_location(
                "peer_cutlass_module", 
                "/root/llm720/llm/models/kernels/peer_cutlass_module.cpython-312-x86_64-linux-gnu.so"
            )
            peer_cutlass_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(peer_cutlass_module)
            return peer_cutlass_module
    
    def test_module_loading(self, cutlass_module):
        """Test that CUTLASS module loads successfully."""
        assert cutlass_module is not None
        available_functions = [x for x in dir(cutlass_module) if not x.startswith('_')]
        assert 'peer_forward' in available_functions
        assert 'print_cache_stats' in available_functions
    
    def test_print_cache_stats(self, cutlass_module):
        """Test print_cache_stats function."""
        try:
            cutlass_module.print_cache_stats()
        except Exception as e:
            pytest.fail(f"print_cache_stats failed: {e}")
    
    def test_minimal_forward_pass(self, cutlass_module):
        """Test minimal forward pass with small configuration."""
        # Minimal test configuration
        batch_size = 1
        seq_len = 1
        input_dim = 64
        output_dim = 128
        num_experts = 4
        num_heads = 2
        expert_hidden_size = 32
        query_dim = 16
        top_k = 2
        
        # Create tensors with correct shapes
        x = torch.randn(batch_size, seq_len, input_dim).cuda().half().contiguous()
        query_weight = torch.randn(num_heads, query_dim, input_dim).cuda().half().contiguous()
        query_bias = torch.randn(num_heads, query_dim).cuda().half().contiguous()
        key_weight_1 = torch.randn(2, query_dim).cuda().half().contiguous()  # sqrt(4) = 2
        key_weight_2 = torch.randn(2, query_dim).cuda().half().contiguous()
        expert_weights_u = torch.randn(num_experts, expert_hidden_size, input_dim).cuda().half().contiguous()
        expert_weights_v = torch.randn(num_experts, output_dim, expert_hidden_size).cuda().half().contiguous()
        output = torch.empty(batch_size, seq_len, output_dim).cuda().half().contiguous()
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
        assert result.device.type == 'cuda'
        assert result.dtype == torch.float16
    
    def test_forward_with_contiguous_tensors(self, cutlass_module):
        """Test forward pass ensuring all tensors are contiguous."""
        # Configuration matching test_cutlass_forward.py
        batch_size = 1
        seq_len = 1
        input_dim = 64
        output_dim = 128
        num_experts = 4
        num_heads = 2
        expert_hidden_size = 32
        query_dim = 16
        top_k = 2
        
        # Create contiguous tensors
        x = torch.randn(batch_size, seq_len, input_dim).cuda().half().contiguous()
        query_weight = torch.randn(num_heads, query_dim, input_dim).cuda().half().contiguous()
        query_bias = torch.randn(num_heads, query_dim).cuda().half().contiguous()
        key_weight_1 = torch.randn(2, query_dim).cuda().half().contiguous()
        key_weight_2 = torch.randn(2, query_dim).cuda().half().contiguous()
        expert_weights_u = torch.randn(num_experts, expert_hidden_size, input_dim).cuda().half().contiguous()
        expert_weights_v = torch.randn(num_experts, output_dim, expert_hidden_size).cuda().half().contiguous()
        output = torch.empty(batch_size, seq_len, output_dim).cuda().half().contiguous()
        ln_weight = torch.ones(num_heads, query_dim).cuda().half().contiguous()
        ln_bias = torch.zeros(num_heads, query_dim).cuda().half().contiguous()
        
        # Verify contiguity
        assert all([
            x.is_contiguous(),
            query_weight.is_contiguous(),
            expert_weights_u.is_contiguous(),
            expert_weights_v.is_contiguous()
        ])
        
        result = cutlass_module.peer_forward(
            x, query_weight, query_bias, key_weight_1, key_weight_2,
            expert_weights_u, expert_weights_v, output, ln_weight, ln_bias,
            batch_size, seq_len, input_dim, output_dim, num_heads,
            num_experts, expert_hidden_size, top_k,
            True, False, False, 0.0
        )
        
        assert result.shape == (batch_size, seq_len, output_dim)
    
    def test_direct_weight_access_mode(self, cutlass_module):
        """Test CUTLASS with direct weight access mode."""
        # Set environment for direct weight access
        os.environ["PEER_DIRECT_WEIGHT_ACCESS"] = "1"
        
        # Simple configuration
        batch_size = 1
        seq_len = 1
        input_dim = 64
        output_dim = 64
        num_experts = 4
        num_heads = 1
        expert_hidden_size = 16
        query_dim = 8
        top_k = 1
        
        # Create zero tensors for debugging
        x = torch.zeros(batch_size, seq_len, input_dim).cuda().half().contiguous()
        query_weight = torch.zeros(num_heads, query_dim, input_dim).cuda().half().contiguous()
        query_bias = torch.zeros(num_heads, query_dim).cuda().half().contiguous()
        key_weight_1 = torch.zeros(2, query_dim).cuda().half().contiguous()
        key_weight_2 = torch.zeros(2, query_dim).cuda().half().contiguous()
        expert_weights_u = torch.zeros(num_experts, expert_hidden_size, input_dim).cuda().half().contiguous()
        expert_weights_v = torch.zeros(num_experts, output_dim, expert_hidden_size).cuda().half().contiguous()
        output = torch.zeros(batch_size, seq_len, output_dim).cuda().half().contiguous()
        ln_weight = torch.ones(num_heads, query_dim).cuda().half().contiguous()
        ln_bias = torch.zeros(num_heads, query_dim).cuda().half().contiguous()
        
        # Test with layer_norm=False
        result = cutlass_module.peer_forward(
            x, query_weight, query_bias, key_weight_1, key_weight_2,
            expert_weights_u, expert_weights_v, output, ln_weight, ln_bias,
            batch_size, seq_len, input_dim, output_dim, num_heads,
            num_experts, expert_hidden_size, top_k,
            False,  # layer_norm OFF
            False,  # norm_keys
            False,  # norm_query
            0.0     # dropout_rate
        )
        
        assert result.shape == (batch_size, seq_len, output_dim)
        
        # Clean up environment
        if "PEER_DIRECT_WEIGHT_ACCESS" in os.environ:
            del os.environ["PEER_DIRECT_WEIGHT_ACCESS"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])