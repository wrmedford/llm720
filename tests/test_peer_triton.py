from typing import List

import pytest
import torch
import torch.nn.functional as F
from torch.autograd import gradcheck

# Import HAS_TRITON first
from llm.ops.triton_peer_kernels import HAS_TRITON, peer_selection_triton

# Mark all tests in this file to be skipped if CUDA is not available or Triton is not installed
pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="Triton tests require CUDA"
    ),
    pytest.mark.skipif(not HAS_TRITON, reason="Triton not installed"),
]

# PEER is not imported as it's not used in this file


# --- PyTorch Reference Implementation ---
# Copied from the original PEER._get_expert_indices_pytorch for comparison
def get_expert_indices_pytorch(
    queries: torch.Tensor,
    sub_keys_list: List[torch.nn.Parameter],
    product_key_dim: List[int],
    top_k: int,
    norm_keys: bool,
    norm_query: bool,
):
    """
    PyTorch implementation for retrieving top-k experts using product keys.
    """
    batch_size, seq_len, num_heads, query_dim = queries.shape
    # device used in mesh_indices creation below
    device = queries.device
    num_dims = len(product_key_dim)
    sub_query_dim = query_dim // num_dims

    # Split queries along the feature dimension
    query_chunks = []
    for i in range(num_dims):
        start_idx = i * sub_query_dim
        end_idx = (i + 1) * sub_query_dim
        query_chunks.append(queries[..., start_idx:end_idx])

    # Compute scores for each dimension
    all_dim_scores = []
    for i, (q_chunk, sub_keys) in enumerate(zip(query_chunks, sub_keys_list)):
        # Normalize if requested
        current_sub_keys = sub_keys.data  # Use .data to get the tensor
        if norm_keys:
            current_sub_keys = F.normalize(current_sub_keys, dim=-1)
        if norm_query:
            q_chunk = F.normalize(q_chunk, dim=-1)

        # Compute scores: [B, S, H, dim_size_i]
        dim_scores = torch.matmul(q_chunk, current_sub_keys.t())
        all_dim_scores.append(dim_scores)

    # Combine scores using broadcasting and summation
    dim_indices = [torch.arange(d, device=device) for d in product_key_dim]
    mesh_indices = torch.stack(torch.meshgrid(*dim_indices, indexing="ij"), dim=-1)
    flat_mesh_indices = mesh_indices.view(-1, num_dims)

    gathered_scores = []
    for dim_idx in range(num_dims):
        indices_for_dim = (
            flat_mesh_indices[:, dim_idx].unsqueeze(0).unsqueeze(0).unsqueeze(0)
        )
        indices_for_dim = indices_for_dim.expand(batch_size, seq_len, num_heads, -1)
        scores_for_dim = all_dim_scores[dim_idx]
        gathered = torch.gather(scores_for_dim, dim=-1, index=indices_for_dim)
        gathered_scores.append(gathered)

    total_scores = torch.stack(gathered_scores, dim=0).sum(dim=0)
    top_scores, top_indices = torch.topk(total_scores, k=top_k, dim=-1)

    return top_indices, top_scores


# --- Test Fixtures ---

# Define configurations to test
test_configs = [
    # Default config (float64 for gradcheck)
    pytest.param(
        {
            "id": "default_fp64",
            "dtype": torch.double,
            "num_experts": 64,
            "top_k": 4,
            "num_heads": 2,
            "query_dim": 32,
            "product_key_dim": [8, 8],
            "norm_keys": True,
            "norm_query": True,
        },
        marks=pytest.mark.skipif(not HAS_TRITON, reason="Triton not installed"),
    ),
    # Float32 config
    pytest.param(
        {
            "id": "default_fp32",
            "dtype": torch.float32,
            "num_experts": 64,
            "top_k": 4,
            "num_heads": 2,
            "query_dim": 32,
            "product_key_dim": [8, 8],
            "norm_keys": True,
            "norm_query": True,
        },
        marks=pytest.mark.skipif(not HAS_TRITON, reason="Triton not installed"),
    ),
    # Float16 config (gradcheck might be less stable)
    pytest.param(
        {
            "id": "default_fp16",
            "dtype": torch.float16,
            "num_experts": 64,
            "top_k": 4,
            "num_heads": 2,
            "query_dim": 32,
            "product_key_dim": [8, 8],
            "norm_keys": True,
            "norm_query": True,
        },
        marks=pytest.mark.skipif(not HAS_TRITON, reason="Triton not installed"),
    ),
    # Edge case: top_k = 1
    pytest.param(
        {
            "id": "topk1_fp32",
            "dtype": torch.float32,
            "num_experts": 64,
            "top_k": 1,
            "num_heads": 2,
            "query_dim": 32,
            "product_key_dim": [8, 8],
            "norm_keys": True,
            "norm_query": True,
        },
        marks=pytest.mark.skipif(not HAS_TRITON, reason="Triton not installed"),
    ),
    # Edge case: 3D product key
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
        },
        marks=pytest.mark.skipif(not HAS_TRITON, reason="Triton not installed"),
    ),
    # No normalization
    pytest.param(
        {
            "id": "nonorm_fp32",
            "dtype": torch.float32,
            "num_experts": 64,
            "top_k": 4,
            "num_heads": 2,
            "query_dim": 32,
            "product_key_dim": [8, 8],
            "norm_keys": False,
            "norm_query": False,
        },
        marks=pytest.mark.skipif(not HAS_TRITON, reason="Triton not installed"),
    ),
]


@pytest.fixture(
    scope="module", params=test_configs, ids=[c.values[0]["id"] for c in test_configs]
)
def config_params(request):
    """Provides parameterized configurations."""
    full_config = {
        **request.param,  # Merge with parameterized values
    }
    return full_config


@pytest.fixture(scope="module")
def sample_inputs(config_params):
    """Generates sample inputs based on the parameterized config."""
    batch_size = 2
    seq_len = 5  # Smaller sequence length for faster tests
    num_heads = config_params["num_heads"]
    query_dim = config_params["query_dim"]
    dtype = config_params["dtype"]
    device = torch.device("cuda")  # Assume CUDA is available due to pytestmark

    queries = torch.randn(
        batch_size,
        seq_len,
        num_heads,
        query_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    sub_keys_list = []
    num_dims = len(config_params["product_key_dim"])
    sub_query_dim = query_dim // num_dims
    for dim_size in config_params["product_key_dim"]:
        keys = torch.randn(
            dim_size, sub_query_dim, device=device, dtype=dtype, requires_grad=True
        )
        sub_keys_list.append(keys)

    return queries, sub_keys_list


# --- Test Cases ---


def test_forward_correctness(config_params, sample_inputs):
    """Compare Triton forward pass output with PyTorch reference."""
    queries, sub_keys_list = sample_inputs
    top_k = config_params["top_k"]
    product_key_dim = config_params["product_key_dim"]
    norm_keys = config_params["norm_keys"]
    norm_query = config_params["norm_query"]
    dtype = config_params["dtype"]
    # device not used in this function
    # queries.device

    # Clone inputs to avoid in-place modifications affecting comparison
    queries_ref = queries.clone().detach().requires_grad_(True)
    sub_keys_ref = [sk.clone().detach().requires_grad_(True) for sk in sub_keys_list]

    queries_triton = queries.clone().detach().requires_grad_(True)
    sub_keys_triton = [sk.clone().detach().requires_grad_(True) for sk in sub_keys_list]

    # PyTorch reference
    ref_indices, ref_scores = get_expert_indices_pytorch(
        queries_ref, sub_keys_ref, product_key_dim, top_k, norm_keys, norm_query
    )

    # Triton implementation (will use placeholder kernel initially)
    try:
        triton_indices, triton_scores = peer_selection_triton(
            queries_triton,
            sub_keys_triton,
            product_key_dim,
            top_k,
            norm_query,
            norm_keys,
        )
    except RuntimeError as e:
        pytest.fail(f"Triton forward pass failed for config {config_params['id']}: {e}")

    # --- Comparison ---
    # Note: The placeholder kernel returns zeros, so these asserts will fail until implemented.

    # 1. Indices Check: Should match exactly.
    assert torch.equal(
        ref_indices.to(triton_indices.dtype), triton_indices
    ), f"Indices mismatch for config: {config_params['id']}"

    # 2. Scores Check: Allow tolerance based on dtype.
    atol = 1e-5 if dtype == torch.double else (1e-3 if dtype == torch.float32 else 1e-1)
    rtol = 1e-5 if dtype == torch.double else (1e-3 if dtype == torch.float32 else 1e-2)
    assert torch.allclose(
        ref_scores.to(triton_scores.dtype), triton_scores, atol=atol, rtol=rtol
    ), f"Scores mismatch for config: {config_params['id']}"


def test_backward_gradients(config_params, sample_inputs):
    """Check gradients computed by the Triton backward pass using gradcheck."""
    queries, sub_keys_list = sample_inputs
    top_k = config_params["top_k"]
    product_key_dim = config_params["product_key_dim"]
    norm_keys = config_params["norm_keys"]
    norm_query = config_params["norm_query"]
    dtype = config_params["dtype"]

    # Gradcheck requires double precision inputs
    if dtype != torch.double:
        pytest.skip(
            "Gradcheck requires double precision. Skipping for non-fp64 configs."
        )

    # Ensure inputs require grad
    queries.requires_grad_(True)
    sub_keys_tensors = [
        sk.detach().clone().requires_grad_(True) for sk in sub_keys_list
    ]

    # Use gradcheck
    def gradcheck_func(q, *sks):
        sk_list = list(sks)
        # Run the function under test
        _, scores = peer_selection_triton(
            q, sk_list, product_key_dim, top_k, norm_query, norm_keys
        )
        # Return a scalar value derived from scores for gradcheck
        # Using a simple sum might not catch all gradient issues, but is standard.
        return scores.sum()

    # Inputs for gradcheck must be a tuple of tensors
    inputs_for_gradcheck = (queries,) + tuple(sub_keys_tensors)

    # Note: The placeholder backward returns zeros, so gradcheck will likely fail initially.
    # Adjust eps and atol based on expected precision.
    try:
        # Nondet_tol checks for non-deterministic operations which can cause issues.
        is_correct = gradcheck(
            gradcheck_func,
            inputs_for_gradcheck,
            eps=1e-6,
            atol=1e-4,
            nondet_tol=1e-5,
            raise_exception=True,
        )
        assert (
            is_correct
        ), f"Gradient check failed for PEERSelectionFunction with config: {config_params['id']}"
    except RuntimeError as e:
        pytest.fail(
            f"Gradcheck failed for config {config_params['id']} with error: {e}"
        )
