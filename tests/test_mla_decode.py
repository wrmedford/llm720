import pytest
import torch
import sys

# Check Python version compatibility
if sys.version_info.major != 3 or sys.version_info.minor < 10:
    pytest.skip("Tests require Python 3.10+", allow_module_level=True)

try:
    # Adjust imports based on your project structure
    from llm.models.attention import MultiHeadedLatentAttention
    from llm.models.foundation import \
        PositionalEmbedding  # For RoPE cache generation
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)

# Mark all tests in this file to be skipped if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="MLA tests require CUDA"
)

# --- Test Configurations ---
test_configs = [
    # Basic FP32 config
    pytest.param(
        {
            "id": "fp32",
            "dtype": torch.float32,
            "hidden_size": 128,
            "num_heads": 4,
            "q_lora_rank": None,
            "kv_lora_rank": 32,
            "qk_rope_head_dim": 16,
            "v_head_dim": 32,
            "qk_nope_head_dim": 16,
            "dropout": 0.0,
            "max_seq_len": 64,
            "rope_base": 10000.0,
        },
        id="fp32",
    ),
    # FP16 config
    pytest.param(
        {
            "id": "fp16",
            "dtype": torch.float16,
            "hidden_size": 128,
            "num_heads": 4,
            "q_lora_rank": None,
            "kv_lora_rank": 32,
            "qk_rope_head_dim": 16,
            "v_head_dim": 32,
            "qk_nope_head_dim": 16,
            "dropout": 0.0,
            "max_seq_len": 64,
            "rope_base": 10000.0,
        },
        id="fp16",
    ),
    # BF16 config
    pytest.param(
        {
            "id": "bf16",
            "dtype": torch.bfloat16,
            "hidden_size": 128,
            "num_heads": 4,
            "q_lora_rank": None,
            "kv_lora_rank": 32,
            "qk_rope_head_dim": 16,
            "v_head_dim": 32,
            "qk_nope_head_dim": 16,
            "dropout": 0.0,
            "max_seq_len": 64,
            "rope_base": 10000.0,
        },
        id="bf16",
        marks=pytest.mark.skipif(
            not torch.cuda.is_bf16_supported(), reason="BF16 not supported"
        ),
    ),
    # Config with Q LoRA
    pytest.param(
        {
            "id": "qlora_fp32",
            "dtype": torch.float32,
            "hidden_size": 128,
            "num_heads": 4,
            "q_lora_rank": 64,
            "kv_lora_rank": 32,
            "qk_rope_head_dim": 16,
            "v_head_dim": 32,
            "qk_nope_head_dim": 16,
            "dropout": 0.0,
            "max_seq_len": 64,
            "rope_base": 10000.0,
        },
        id="qlora_fp32",
    ),
]


# --- Test Fixtures ---
@pytest.fixture(scope="module", params=test_configs)
def mla_config(request):
    """Provides parameterized MLA configurations."""
    return request.param


@pytest.fixture(scope="module")
def mla_module(mla_config):
    """Creates an MLA module instance."""
    device = torch.device("cuda")
    dtype = mla_config["dtype"]
    config_dict = {
        k: v
        for k, v in mla_config.items()
        if k not in ["id", "dtype", "max_seq_len", "rope_base"]
    }

    module = MultiHeadedLatentAttention(**config_dict).to(device).to(dtype)
    # Crucially, call post_weight_load after initialization to prepare decode weights
    module.post_weight_load()
    module.eval()  # Set to eval mode
    return module


@pytest.fixture(scope="module")
def rope_caches(mla_config):
    """Generates RoPE sin/cos caches."""
    device = torch.device("cuda")
    dtype = mla_config["dtype"]  # RoPE cache often kept in FP32 for precision
    rope_dim = mla_config["qk_rope_head_dim"]
    max_seq_len = mla_config["max_seq_len"]
    base = mla_config["rope_base"]

    # Use the PositionalEmbedding class to generate caches
    pos_emb = PositionalEmbedding(dim=rope_dim, max_seq_len=max_seq_len, base=base)
    # Generate cache up to max_seq_len
    cos, sin = pos_emb(max_seq_len)
    return cos.to(device).to(dtype), sin.to(device).to(
        dtype
    )  # Move to device and potentially cast


@pytest.fixture(scope="module")
def sample_decode_data(mla_config, rope_caches):
    """Generates sample input data for a decode step (seq_len=1)."""
    device = torch.device("cuda")
    dtype = mla_config["dtype"]
    batch_size = 4
    q_len = 1  # Decode step
    past_len = mla_config["max_seq_len"] // 2  # Example past length
    hidden_size = mla_config["hidden_size"]
    kv_lora_rank = mla_config["kv_lora_rank"]
    qk_rope_head_dim = mla_config["qk_rope_head_dim"]

    # Current hidden state (for the token being generated)
    hidden_states = torch.randn(
        batch_size, q_len, hidden_size, device=device, dtype=dtype
    )

    # Position ID for the current token
    position_ids = torch.full(
        (batch_size, q_len), past_len, dtype=torch.long, device=device
    )

    # Generate dummy past_key_value cache
    # past_kv_c shape: (batch, past_seq_len, kv_lora_rank) - Note: MLA stores cache differently
    # MLA cache: (present_kv_c, present_k_pe)
    # present_kv_c shape: (batch, seq_len, kv_lora_rank)
    # present_k_pe shape: (batch, seq_len, qk_rope_head_dim)
    past_kv_c = torch.randn(
        batch_size, past_len, kv_lora_rank, device=device, dtype=dtype
    )
    past_k_pe = torch.randn(
        batch_size, past_len, qk_rope_head_dim, device=device, dtype=dtype
    )
    past_key_value = (past_kv_c, past_k_pe)

    cos, sin = rope_caches

    return hidden_states, position_ids, cos, sin, past_key_value


# --- Test Cases ---


def test_mla_decode_forward_shape_and_cache(mla_module, sample_decode_data, mla_config):
    """Tests the forward pass shape and cache handling in decode mode."""
    hidden_states, position_ids, cos, sin, past_key_value = sample_decode_data
    batch_size, q_len, hidden_size = hidden_states.shape
    past_len = past_key_value[0].shape[1]
    dtype = mla_config["dtype"]

    # Ensure decode weights are prepared (should be done in fixture)
    assert mla_module.W_UV is not None, "W_UV not prepared for decode"
    assert mla_module.W_UK_T is not None, "W_UK_T not prepared for decode"

    try:
        with torch.no_grad():  # No need for gradients in this test
            attn_output, attn_weights, present_key_value = mla_module(
                hidden_states=hidden_states,
                cos=cos,
                sin=sin,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=True,
                output_attentions=False,
            )
    except Exception as e:
        pytest.fail(
            f"MLA decode forward pass failed for config {mla_config['id']}: {e}"
        )

    # 1. Check output shape and type
    assert attn_output.shape == (
        batch_size,
        q_len,
        hidden_size,
    ), f"Unexpected output shape for config {mla_config['id']}"
    assert (
        attn_output.dtype == dtype
    ), f"Unexpected output dtype for config {mla_config['id']}"

    # 2. Check attention weights (should be None)
    assert attn_weights is None, "Attention weights should be None for FlashAttention"

    # 3. Check cache (`present_key_value`)
    assert present_key_value is not None, "Cache should be returned when use_cache=True"
    assert isinstance(present_key_value, tuple), "Cache should be a tuple"
    assert (
        len(present_key_value) == 2
    ), "Cache tuple should have 2 elements (kv_c, k_pe)"

    present_kv_c, present_k_pe = present_key_value
    expected_cache_len = past_len + q_len

    # Check kv_c cache
    assert isinstance(present_kv_c, torch.Tensor), "present_kv_c should be a Tensor"
    assert present_kv_c.shape == (
        batch_size,
        expected_cache_len,
        mla_config["kv_lora_rank"],
    ), f"Unexpected present_kv_c shape for config {mla_config['id']}"
    assert (
        present_kv_c.dtype == dtype
    ), f"Unexpected present_kv_c dtype for config {mla_config['id']}"

    # Check k_pe cache
    assert isinstance(present_k_pe, torch.Tensor), "present_k_pe should be a Tensor"
    assert present_k_pe.shape == (
        batch_size,
        expected_cache_len,
        mla_config["qk_rope_head_dim"],
    ), f"Unexpected present_k_pe shape for config {mla_config['id']}"
    assert (
        present_k_pe.dtype == dtype
    ), f"Unexpected present_k_pe dtype for config {mla_config['id']}"

    # Optional: Check if cache values seem reasonable (e.g., not all zeros/NaNs)
    assert not torch.isnan(attn_output).any(), "Output contains NaNs"
    assert not torch.isinf(attn_output).any(), "Output contains Infs"
    assert not torch.isnan(present_kv_c).any(), "Cache kv_c contains NaNs"
    assert not torch.isnan(present_k_pe).any(), "Cache k_pe contains NaNs"
