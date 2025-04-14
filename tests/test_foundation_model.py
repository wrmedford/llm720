import pytest
import torch
import sys

# Check Python version compatibility
if sys.version_info.major != 3 or sys.version_info.minor < 10:
    pytest.skip("Tests require Python 3.10+", allow_module_level=True)

try:
    # Adjust imports based on your project structure
    from llm.models.foundation import TransformerConfig, FoundationModel, create_model_from_config
    # Removed unused import: from llm.training.train import TrainerConfig
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)

# Mark all tests in this file to be skipped if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FoundationModel tests require CUDA"
)

# --- Test Configurations ---
# Use a minimal config for faster testing
@pytest.fixture(scope="module")
def minimal_config():
    """Provides a minimal model configuration dictionary."""
    return {
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 128,
        "vocab_size": 1000, # Smaller vocab for testing
        "max_position_embeddings": 128,
        "use_peer": True,
        "peer_config": {
            "num_experts": 64, # Smaller number of experts
            "num_experts_per_tok": 4,
            "num_heads": 2,
            "expert_hidden_size": 8,
            "product_key_dim": [8, 8],
            "query_dim": 16,
            "batch_norm_query": False, # Simpler for testing
        },
        "mla_config": {
            "q_lora_rank": None,
            "kv_lora_rank": 16,
            "qk_rope_head_dim": 8,
            "v_head_dim": 16,
            "qk_nope_head_dim": 8,
        },
    }

@pytest.fixture(scope="module")
def foundation_model(minimal_config):
    """Creates a FoundationModel instance."""
    device = torch.device("cuda")
    dtype = torch.float32 # Use FP32 for stability in basic tests
    
    config = TransformerConfig(**minimal_config)
    model = create_model_from_config(config).to(device).to(dtype)
    model.eval() # Set to eval mode
    return model

@pytest.fixture(scope="module")
def sample_inputs(minimal_config):
    """Generates sample input data."""
    device = torch.device("cuda")
    batch_size = 2
    seq_len = 16
    vocab_size = minimal_config["vocab_size"]

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)
    attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.long)
    # Create a simple causal mask for testing (1s on and below diagonal)
    # attention_mask = torch.tril(torch.ones(batch_size, seq_len, seq_len, device=device)).bool()
    
    # Position IDs are usually handled internally if None, but we can provide them
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    return input_ids, attention_mask, position_ids

# --- Test Cases ---

def test_foundation_model_forward_shape(foundation_model, sample_inputs, minimal_config):
    """Tests the forward pass shape and output type."""
    input_ids, attention_mask, position_ids = sample_inputs
    batch_size, seq_len = input_ids.shape
    hidden_size = minimal_config["hidden_size"]
    vocab_size = minimal_config["vocab_size"]
    dtype = torch.float32

    try:
        with torch.no_grad():
            outputs = foundation_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
    except Exception as e:
        pytest.fail(f"FoundationModel forward pass failed: {e}")

    # Check output type
    assert isinstance(outputs, dict), "Output should be a dictionary when return_dict=True"
    assert "logits" in outputs, "Output dictionary should contain 'logits'"

    # Check logits shape and type
    logits = outputs["logits"]
    assert logits.shape == (batch_size, seq_len, vocab_size), "Unexpected logits shape"
    assert logits.dtype == dtype, "Unexpected logits dtype"

    # Check for NaNs or infinities
    assert not torch.isnan(logits).any(), "Logits contain NaNs"
    assert not torch.isinf(logits).any(), "Logits contain infinities"

def test_foundation_model_forward_with_labels(foundation_model, sample_inputs, minimal_config):
    """Tests the forward pass with labels to calculate loss."""
    input_ids, attention_mask, position_ids = sample_inputs
    labels = input_ids.clone() # Use input_ids as labels for simplicity

    try:
        with torch.no_grad(): # Usually loss calculation doesn't need gradients for testing
            outputs = foundation_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels,
                return_dict=True,
            )
    except Exception as e:
        pytest.fail(f"FoundationModel forward pass with labels failed: {e}")

    # Check output type and presence of loss
    assert isinstance(outputs, dict), "Output should be a dictionary"
    assert "loss" in outputs, "Output dictionary should contain 'loss'"
    assert "logits" in outputs, "Output dictionary should contain 'logits'"

    # Check loss type and shape
    loss = outputs["loss"]
    assert isinstance(loss, torch.Tensor), "Loss should be a Tensor"
    assert loss.shape == (), "Loss should be a scalar tensor"
    assert loss.dtype == torch.float32, "Loss should be float32" # Loss is typically FP32

    # Check loss value
    assert not torch.isnan(loss).any(), "Loss is NaN"
    assert not torch.isinf(loss).any(), "Loss is Inf"
    assert loss.item() >= 0, "Loss should be non-negative" # Basic sanity check

def test_foundation_model_kv_cache(foundation_model, sample_inputs, minimal_config):
    """Tests the KV cache mechanism."""
    input_ids, attention_mask, position_ids = sample_inputs
    batch_size, seq_len = input_ids.shape
    num_layers = minimal_config["num_hidden_layers"]

    # --- Prefill Step ---
    try:
        with torch.no_grad():
            outputs_prefill = foundation_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                return_dict=True,
            )
    except Exception as e:
        pytest.fail(f"FoundationModel prefill step with cache failed: {e}")

    assert "past_key_values" in outputs_prefill, "Prefill output missing cache"
    past_key_values = outputs_prefill["past_key_values"]
    assert isinstance(past_key_values, tuple), "Cache should be a tuple"
    assert len(past_key_values) == num_layers, f"Cache tuple length ({len(past_key_values)}) should match num_layers ({num_layers})"

    # Check cache shapes for the first layer (assuming MLA cache format)
    # MLA cache per layer: (kv_c, k_pe)
    first_layer_cache = past_key_values[0]
    assert isinstance(first_layer_cache, tuple), "Layer cache should be a tuple"
    assert len(first_layer_cache) == 2, "Layer cache should have 2 elements (kv_c, k_pe)"
    
    kv_c_cache, k_pe_cache = first_layer_cache
    expected_kv_c_shape = (batch_size, seq_len, minimal_config["mla_config"]["kv_lora_rank"])
    expected_k_pe_shape = (batch_size, seq_len, minimal_config["mla_config"]["qk_rope_head_dim"])
    
    assert kv_c_cache.shape == expected_kv_c_shape, f"Unexpected kv_c cache shape in layer 0: {kv_c_cache.shape}"
    assert k_pe_cache.shape == expected_k_pe_shape, f"Unexpected k_pe cache shape in layer 0: {k_pe_cache.shape}"


    # --- Decode Step ---
    # Use only the last token's logits from prefill for next token prediction simulation
    next_token_logits = outputs_prefill["logits"][:, -1, :]
    # In a real scenario, you'd sample from next_token_logits
    next_token_input_ids = torch.argmax(next_token_logits, dim=-1, keepdim=True) # Shape: [batch_size, 1]
    
    # Prepare inputs for decode step
    decode_attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_input_ids)], dim=1)
    decode_position_ids = torch.tensor([[seq_len]] * batch_size, device=input_ids.device, dtype=torch.long)

    try:
        with torch.no_grad():
            outputs_decode = foundation_model(
                input_ids=next_token_input_ids,
                attention_mask=decode_attention_mask, # Mask includes the new token
                position_ids=decode_position_ids,
                past_key_values=past_key_values, # Pass the cache from prefill
                use_cache=True,
                return_dict=True,
            )
    except Exception as e:
        pytest.fail(f"FoundationModel decode step with cache failed: {e}")

    # Check decode output shape
    assert "logits" in outputs_decode
    assert outputs_decode["logits"].shape == (batch_size, 1, minimal_config["vocab_size"]), "Unexpected logits shape in decode step"

    # Check updated cache
    assert "past_key_values" in outputs_decode
    updated_cache = outputs_decode["past_key_values"]
    assert isinstance(updated_cache, tuple) and len(updated_cache) == num_layers

    # Check updated cache shapes for the first layer
    updated_kv_c, updated_k_pe = updated_cache[0]
    expected_updated_len = seq_len + 1
    expected_updated_kv_c_shape = (batch_size, expected_updated_len, minimal_config["mla_config"]["kv_lora_rank"])
    expected_updated_k_pe_shape = (batch_size, expected_updated_len, minimal_config["mla_config"]["qk_rope_head_dim"])
    
    assert updated_kv_c.shape == expected_updated_kv_c_shape, f"Unexpected updated kv_c cache shape: {updated_kv_c.shape}"
    assert updated_k_pe.shape == expected_updated_k_pe_shape, f"Unexpected updated k_pe cache shape: {updated_k_pe.shape}"

    # Check that the updated cache contains the original cache data
    assert torch.equal(updated_kv_c[:, :seq_len, :], kv_c_cache), "Updated kv_c cache doesn't contain original data"
    assert torch.equal(updated_k_pe[:, :seq_len, :], k_pe_cache), "Updated k_pe cache doesn't contain original data"
