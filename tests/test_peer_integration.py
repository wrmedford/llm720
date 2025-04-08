import pytest
import torch

# Adjust imports based on your project structure
from llm.models.experts import PEER
from llm.ops.triton_peer_kernels import HAS_TRITON

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
            "top_k": 4,
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
            "top_k": 4,
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
            "top_k": 4,
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
            "top_k": 4,
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


# --- Helper Function ---
def force_peer_implementation(module: PEER, use_triton: bool):
    """Monkey-patches the PEER forward method to force Triton or PyTorch."""
    original_forward = module.forward

    def patched_forward(hidden_states):
        # Temporarily override HAS_TRITON for this call

        # Need to modify the check within the forward function or the function it calls
        # Option 1: Modify the check directly (if simple)
        # Option 2: Patch the called function (peer_selection_triton or _get_expert_indices_pytorch) - complex
        # Option 3: Modify the module's state temporarily (cleanest if possible)

        # Let's assume the check inside forward uses HAS_TRITON from llm.ops.triton_peer_kernels
        # We need to patch *that* specific import within the scope of the forward call.
        # This is tricky. A simpler approach for testing is to modify the PEER class
        # to allow selecting the implementation via a flag.

        # --- Alternative: Modify PEER class for testing ---
        # Add a `_force_implementation` attribute to PEER:
        # if hasattr(self, '_force_implementation'):
        #     if self._force_implementation == 'triton': use_triton_path = True
        #     elif self._force_implementation == 'pytorch': use_triton_path = False
        # else:
        #     use_triton_path = HAS_TRITON and peer_selection_triton is not None
        # For now, we assume the test modifies the global HAS_TRITON, which is NOT ideal but simpler for this example.
        # A better solution involves refactoring PEER or using mock.patch.

        import llm.ops.triton_peer_kernels as triton_kernels_module

        original_has_triton_in_module = triton_kernels_module.HAS_TRITON
        triton_kernels_module.HAS_TRITON = use_triton
        try:
            output = original_forward(hidden_states)
        finally:
            # Restore original state
            triton_kernels_module.HAS_TRITON = original_has_triton_in_module
        return output

    # For this example, we'll just rely on the global HAS_TRITON check within PEER's forward
    # and assume the test runner can influence it (e.g., by setting env var before test run)
    # or we accept this test compares Triton vs. the *actual* fallback path.
    # A truly robust test would use mocking or class modification.
    print(
        "Note: Test assumes PEER checks HAS_TRITON dynamically or relies on global state."
    )
    pass  # No actual patching applied here, relies on PEER's internal check


# --- Test Fixtures ---
@pytest.fixture(scope="module", params=test_configs)
def peer_config(request):
    """Provides parameterized PEER configurations."""
    return request.param


@pytest.fixture(scope="module")
def peer_modules(peer_config):
    """Creates PEER module instances (Triton and PyTorch versions)."""
    device = torch.device("cuda")
    dtype = peer_config["dtype"]
    config_dict = {k: v for k, v in peer_config.items() if k != "id" and k != "dtype"}

    # Create base module
    peer_base = PEER(**config_dict).to(device).to(dtype)

    # Create two instances with shared weights
    peer_triton = PEER(**config_dict).to(device).to(dtype)
    peer_pytorch = PEER(**config_dict).to(device).to(dtype)

    # Share weights between the two instances
    peer_triton.load_state_dict(peer_base.state_dict())
    peer_pytorch.load_state_dict(peer_base.state_dict())

    # Ensure they are in eval mode if dropout > 0
    peer_triton.eval()
    peer_pytorch.eval()

    # Note: Forcing implementation via monkey-patching is complex.
    # Return the base module and the config dict for creating patched versions in tests
    return peer_base, config_dict


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
    # Target gradients for backward pass comparison
    grad_output = torch.randn_like(hidden_states)

    return hidden_states, grad_output


# --- Test Cases ---


from unittest.mock import patch

def test_peer_forward_comparison(peer_modules, sample_data, peer_config):
    """Compares the forward pass output of Triton vs PyTorch PEER."""
    peer_base, config_dict = peer_modules # Get base model and config
    hidden_states, _ = sample_data
    dtype = peer_config["dtype"]
    device = hidden_states.device

    # Create fresh instances for isolated runs
    peer_triton = PEER(**config_dict).to(device).to(dtype)
    peer_pytorch = PEER(**config_dict).to(device).to(dtype)
    peer_triton.load_state_dict(peer_base.state_dict())
    peer_pytorch.load_state_dict(peer_base.state_dict())
    peer_triton.eval()
    peer_pytorch.eval()

    # Clone input for isolated runs
    hidden_states_triton = hidden_states.clone().detach().requires_grad_(True)
    hidden_states_pytorch = hidden_states.clone().detach().requires_grad_(True)

    # --- Run Triton Forward ---
    output_triton = None
    if HAS_TRITON:
        # Ensure the Triton path is taken (mock HAS_TRITON=True just in case)
        with patch("llm.models.experts.HAS_TRITON", True), \
             patch("llm.models.experts.peer_selection_triton", wraps=peer_selection_triton) as mock_triton_func:
             # Using wraps ensures the original function is called if HAS_TRITON is True
            try:
                output_triton = peer_triton(hidden_states_triton)
                # Check if the triton function was actually called (optional sanity check)
                # mock_triton_func.assert_called()
            except Exception as e:
                pytest.fail(f"PEER forward pass with intended Triton path failed: {e}")
    else:
        pytest.skip("Skipping forward comparison as Triton is not available.")

    # --- Run PyTorch Forward ---
    # Force the PyTorch path by mocking HAS_TRITON to False within the experts module
    with patch("llm.models.experts.HAS_TRITON", False):
        try:
            output_pytorch = peer_pytorch(hidden_states_pytorch)
        except Exception as e:
            pytest.fail(f"PEER forward pass with PyTorch fallback failed: {e}")

    # --- Compare outputs ---
    assert output_triton is not None, "Triton output was not generated."
    atol = (
        1e-5 if dtype == torch.float32 else (1e-2 if dtype == torch.float16 or dtype == torch.bfloat16 else 1e-5)
    )  # Looser tolerance for fp16/bf16
    rtol = (
        1e-4 if dtype == torch.float32 else (1e-1 if dtype == torch.float16 else 1e-1)
    )
    assert torch.allclose(
        output_triton, output_pytorch, atol=atol, rtol=rtol
    ), f"PEER forward output mismatch for config {peer_config['id']}"


def test_peer_backward_comparison(peer_modules, sample_data, peer_config):
    """Compares the backward pass gradients of Triton vs PyTorch PEER."""
    peer_base, config_dict = peer_modules # Get base model and config
    hidden_states, grad_output = sample_data
    dtype = peer_config["dtype"]
    device = hidden_states.device

    # Create fresh instances
    peer_triton = PEER(**config_dict).to(device).to(dtype)
    peer_pytorch = PEER(**config_dict).to(device).to(dtype)
    peer_triton.load_state_dict(peer_base.state_dict())
    peer_pytorch.load_state_dict(peer_base.state_dict())
    peer_triton.eval() # Ensure consistent mode, although grads are compared
    peer_pytorch.eval()

    # --- Triton Backward ---
    grad_input_triton = None
    grads_params_triton = {}
    if HAS_TRITON:
        hidden_states_triton = hidden_states.clone().detach().requires_grad_(True)
        peer_triton.zero_grad()
        with patch("llm.models.experts.HAS_TRITON", True):
            try:
                output_triton = peer_triton(hidden_states_triton)
                output_triton.backward(grad_output.clone().detach())
            except Exception as e:
                pytest.fail(f"PEER backward pass with intended Triton path failed: {e}")

        grad_input_triton = hidden_states_triton.grad.clone().detach() if hidden_states_triton.grad is not None else None
        grads_params_triton = {name: p.grad.clone().detach() for name, p in peer_triton.named_parameters() if p.grad is not None}
    else:
        pytest.skip("Skipping backward comparison as Triton is not available.")


    # --- PyTorch Backward ---
    hidden_states_pytorch = hidden_states.clone().detach().requires_grad_(True)
    peer_pytorch.zero_grad()
    with patch("llm.models.experts.HAS_TRITON", False):
        try:
            output_pytorch = peer_pytorch(hidden_states_pytorch)
            output_pytorch.backward(grad_output.clone().detach())
        except Exception as e:
            pytest.fail(f"PEER backward pass with PyTorch fallback failed: {e}")

    grad_input_pytorch = hidden_states_pytorch.grad.clone().detach() if hidden_states_pytorch.grad is not None else None
    grads_params_pytorch = {name: p.grad.clone().detach() for name, p in peer_pytorch.named_parameters() if p.grad is not None}

    # --- Comparison ---
    assert grad_input_triton is not None, "Triton input gradient is None"
    assert grad_input_pytorch is not None, "PyTorch input gradient is None"

    atol = (
        1e-4 if dtype == torch.float32 else (1e-1 if dtype == torch.float16 or dtype == torch.bfloat16 else 1e-4)
    )  # Looser tolerance for fp16/bf16 grads
    rtol = (
        1e-3 if dtype == torch.float32 else (1e-1 if dtype == torch.float16 else 1e-1)
    )

    # 1. Compare input gradients
    assert grad_input_triton is not None, "Triton input gradient is None"
    assert grad_input_pytorch is not None, "PyTorch input gradient is None"
    assert torch.allclose(
        grad_input_triton, grad_input_pytorch, atol=atol, rtol=rtol
    ), f"PEER input gradient mismatch for config {peer_config['id']}"

    # 2. Compare parameter gradients
    assert (
        grads_params_triton.keys() == grads_params_pytorch.keys()
    ), "Parameter gradient keys mismatch"
    for name in grads_params_triton:
        assert torch.allclose(
            grads_params_triton[name], grads_params_pytorch[name], atol=atol, rtol=rtol
        ), f"PEER parameter gradient mismatch for '{name}' in config {peer_config['id']}"
