import pytest
import torch
import time
import sys

# Check Python version compatibility
if sys.version_info.major != 3 or sys.version_info.minor < 10:
    pytest.skip("Tests require Python 3.10+", allow_module_level=True)

try:
    # Adjust import based on your project structure
    from llm.models.experts import PEER
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)

# Mark all tests in this file to be skipped if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="PEER selection tests require CUDA"
)

# --- Test Configurations ---
# Define configurations focusing on product_key_dim
NUM_EXPERTS = 1024 * 1024 # 1,048,576
TOP_K = 16
QUERY_DIM = 256 # Must be divisible by num_dims (2 and 4)
NUM_HEADS = 8
INPUT_DIM = 512 # Example value, not directly used by selection method
OUTPUT_DIM = 512 # Example value

test_configs = [
    # 2D Configuration
    pytest.param(
        {
            "id": "2d_1M",
            "dtype": torch.float16, # Use FP16 for realistic performance
            "num_experts": NUM_EXPERTS,
            "num_experts_per_tok": TOP_K,
            "num_heads": NUM_HEADS,
            "query_dim": QUERY_DIM,
            "product_key_dim": [1024, 1024], # d=2
            "norm_keys": True,
            "norm_query": True,
            "batch_norm_query": False, # Keep it simple for selection test
            "input_dim": INPUT_DIM,
            "output_dim": OUTPUT_DIM,
            "expert_hidden_size": 1,
            "activation": "gelu",
            "dropout": 0.0,
        },
        id="2d_1M_fp16",
    ),
    # 4D Configuration
    pytest.param(
        {
            "id": "4d_1M",
            "dtype": torch.float16, # Use FP16 for realistic performance
            "num_experts": NUM_EXPERTS,
            "num_experts_per_tok": TOP_K,
            "num_heads": NUM_HEADS,
            "query_dim": QUERY_DIM,
            "product_key_dim": [32, 32, 32, 32], # d=4
            "norm_keys": True,
            "norm_query": True,
            "batch_norm_query": False,
            "input_dim": INPUT_DIM,
            "output_dim": OUTPUT_DIM,
            "expert_hidden_size": 1,
            "activation": "gelu",
            "dropout": 0.0,
        },
        id="4d_1M_fp16",
    ),
]

# --- Test Fixtures ---
@pytest.fixture(scope="module", params=test_configs)
def peer_config(request):
    """Provides parameterized PEER configurations."""
    return request.param

@pytest.fixture(scope="module")
def peer_instance(peer_config):
    """Creates a PEER module instance for testing the selection method."""
    device = torch.device("cuda")
    dtype = peer_config["dtype"]
    config_dict = {k: v for k, v in peer_config.items() if k not in ["id", "dtype"]}

    # Create module
    peer = PEER(**config_dict).to(device).to(dtype)
    peer.eval() # Ensure dropout is off etc.
    return peer

@pytest.fixture(scope="module")
def sample_queries(peer_config):
    """Generates sample query tensors."""
    device = torch.device("cuda")
    dtype = peer_config["dtype"]
    batch_size = 8
    seq_len = 128 # A reasonable sequence length
    num_heads = peer_config["num_heads"]
    query_dim = peer_config["query_dim"]

    queries = torch.randn(
        batch_size, seq_len, num_heads, query_dim, device=device, dtype=dtype
    )
    return queries

# --- Test Case ---

def test_peer_selection_shape_and_benchmark(peer_instance, sample_queries, peer_config):
    """
    Tests the shape of the expert selection output and benchmarks the selection method.
    """
    queries = sample_queries
    top_k = peer_config["num_experts_per_tok"]
    batch_size, seq_len, num_heads, _ = queries.shape
    dtype = peer_config["dtype"]
    config_id = peer_config["id"]

    # --- Shape Test ---
    try:
        with torch.no_grad():
            indices, scores = peer_instance._get_expert_indices_pytorch(queries, top_k)

        # Check indices shape
        assert indices.shape == (batch_size, seq_len, num_heads, top_k), \
            f"[{config_id}] Indices shape mismatch. Expected {(batch_size, seq_len, num_heads, top_k)}, Got {indices.shape}"
        assert indices.dtype == torch.long, f"[{config_id}] Indices dtype mismatch. Expected torch.long, Got {indices.dtype}"

        # Check scores shape
        assert scores.shape == (batch_size, seq_len, num_heads, top_k), \
            f"[{config_id}] Scores shape mismatch. Expected {(batch_size, seq_len, num_heads, top_k)}, Got {scores.shape}"
        assert scores.dtype == dtype, f"[{config_id}] Scores dtype mismatch. Expected {dtype}, Got {scores.dtype}"

        # Check for NaNs/Infs in scores
        assert not torch.isnan(scores).any(), f"[{config_id}] Scores contain NaNs"
        assert not torch.isinf(scores).any(), f"[{config_id}] Scores contain Infs"

    except Exception as e:
        pytest.fail(f"[{config_id}] PEER selection failed during shape test: {e}")

    # --- Benchmark ---
    num_warmup = 5
    num_runs = 20

    # Warmup runs
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = peer_instance._get_expert_indices_pytorch(queries, top_k)
        torch.cuda.synchronize()

    # Timed runs
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = peer_instance._get_expert_indices_pytorch(queries, top_k)
        # Synchronize CUDA operations
        torch.cuda.synchronize()
    end_time = time.time()

    avg_time_ms = ((end_time - start_time) / num_runs) * 1000

    # Print results clearly
    print(f"\n--- Benchmark Results [{config_id}] ---")
    print(f"Configuration: {peer_config['product_key_dim']} (d={len(peer_config['product_key_dim'])})")
    print(f"Input Shape (Queries): {tuple(queries.shape)}")
    print(f"Average execution time over {num_runs} runs: {avg_time_ms:.3f} ms")
    print("--------------------------------------")

    # Optional: Add a simple assertion for performance sanity check (e.g., not excessively slow)
    # This threshold is arbitrary and might need adjustment based on hardware.
    assert avg_time_ms < 5000, f"[{config_id}] Execution time ({avg_time_ms:.1f} ms) seems excessively high."
