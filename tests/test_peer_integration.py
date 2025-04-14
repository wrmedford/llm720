import pytest
import torch
import sys

# Check Python version compatibility
if sys.version_info.major != 3 or sys.version_info.minor < 10:
    pytest.skip("Tests require Python 3.10+", allow_module_level=True)

try:
    # Adjust imports based on your project structure
    from llm.models.experts import PEER
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)

# Mark all tests in this file to be skipped if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="PEER integration tests require CUDA"
)

# --- Test Configurations ---
# Define configurations to test (focus on dtypes relevant for comparison)
test_configs = [
    # Float32
    pytest.param(
        {
            "id": "fp32",
            "dtype": torch.float32,
            "num_experts": 64,
            "num_experts_per_tok": 4,
            "num_heads": 2,
            "query_dim": 32,
            "product_key_dim": [8, 8],
            "norm_keys": True,
            "norm_query": True,
            "batch_norm_query": False,
            "input_dim": 128,
            "output_dim": 128,
            "expert_hidden_size": 8,
            "activation": "gelu",
            "dropout": 0.0,
        },
        id="fp32",
    ),
    # Float16
    pytest.param(
        {
            "id": "fp16",
            "dtype": torch.float16,
            "num_experts": 64,
            "num_experts_per_tok": 4,
            "num_heads": 2,
            "query_dim": 32,
            "product_key_dim": [8, 8],
            "norm_keys": True,
            "norm_query": True,
            "batch_norm_query": False,
            "input_dim": 128,
            "output_dim": 128,
            "expert_hidden_size": 8,
            "activation": "gelu",
            "dropout": 0.0,
        },
        id="fp16",
    ),
    # BFloat16 (if supported)
    pytest.param(
        {
            "id": "bf16",
            "dtype": torch.bfloat16,
            "num_experts": 64,
            "num_experts_per_tok": 4,
            "num_heads": 2,
            "query_dim": 32,
            "product_key_dim": [8, 8],
            "norm_keys": True,
            "norm_query": True,
            "batch_norm_query": False,
            "input_dim": 128,
            "output_dim": 128,
            "expert_hidden_size": 8,
            "activation": "gelu",
            "dropout": 0.0,
        },
        id="bf16",
        marks=pytest.mark.skipif(
            not torch.cuda.is_bf16_supported(), reason="BF16 not supported"
        ),
    ),
    # Config with 3D keys
    pytest.param(
        {
            "id": "3d_fp32",
            "dtype": torch.float32,
            "num_experts": 64,
            "num_experts_per_tok": 4,
            "num_heads": 2,
            "query_dim": 30,
            "product_key_dim": [4, 4, 4],  # query_dim must be divisible by num_dims
            "norm_keys": True,
            "norm_query": True,
            "batch_norm_query": False,
            "input_dim": 128,
            "output_dim": 128,
            "expert_hidden_size": 8,
            "activation": "gelu",
            "dropout": 0.0,
        },
        id="3d_fp32",
    ),
]


# --- Test Fixtures ---
@pytest.fixture(scope="module", params=test_configs)
def peer_config(request):
    """Provides parameterized PEER configurations."""
    return request.param


@pytest.fixture(scope="module")
def peer_module(peer_config):
    """Creates a PEER module instance."""
    device = torch.device("cuda")
    dtype = peer_config["dtype"]
    config_dict = {k: v for k, v in peer_config.items() if k != "id" and k != "dtype"}

    # Create module
    peer = PEER(**config_dict).to(device).to(dtype)

    # Ensure it is in eval mode if dropout > 0
    peer.eval()

    return peer


@pytest.fixture(scope="module")
def sample_data(peer_config):
    """Generates sample input data."""
    device = torch.device("cuda")
    dtype = peer_config["dtype"]
    batch_size = 4
    seq_len = 16  # Keep sequence length reasonable for testing
    input_dim = peer_config["input_dim"]

    hidden_states = torch.randn(
        batch_size, seq_len, input_dim, device=device, dtype=dtype, requires_grad=True
    )

    return hidden_states


# --- Test Cases ---

def test_peer_forward(peer_module, sample_data, peer_config):
    """Tests the forward pass of PEER."""
    hidden_states = sample_data
    dtype = peer_config["dtype"]

    try:
        output = peer_module(hidden_states)
        
        # Basic shape and type checks
        assert output.shape[0] == hidden_states.shape[0], "Batch size mismatch"
        assert output.shape[1] == hidden_states.shape[1], "Sequence length mismatch"
        assert output.shape[2] == peer_config["output_dim"], "Output dimension mismatch"
        assert output.dtype == dtype, "Output dtype mismatch"
        
        # Check for NaNs or infinities
        assert not torch.isnan(output).any(), "Output contains NaNs"
        assert not torch.isinf(output).any(), "Output contains infinities"
        
    except Exception as e:
        pytest.fail(f"PEER forward pass failed: {e}")
