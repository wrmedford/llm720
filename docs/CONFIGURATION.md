# Configuration Guide

The training, evaluation, and analysis scripts are controlled through YAML configuration files. The main configuration file (e.g., `configs/config.yaml`) orchestrates the different components.

## Main Configuration File (`config.yaml`)

This file typically includes the following top-level sections:

-   `model_config`: Defines the model architecture.
-   `train_config`: Controls the training process (optimizer, scheduler, batch sizes, logging, saving, etc.).
-   `dataset_config`: Specifies datasets, tokenizer, and preprocessing parameters.
-   `eval_config`: Configures evaluation benchmarks and settings.
-   `wandb_config`: Sets up Weights & Biases integration (optional).

### `model_config`

Defines the core transformer architecture and the specialized PEER and MLA layers.

```yaml
model_config:
  # --- Core Transformer ---
  hidden_size: 768           # Model embedding dimension
  num_hidden_layers: 12      # Number of transformer blocks
  num_attention_heads: 12    # Number of attention heads (for MLA)
  intermediate_size: 3072    # Dimension of the MLP feed-forward layer
  vocab_size: 50272          # Vocabulary size (often padded for efficiency)
  max_position_embeddings: 2048 # Max sequence length for positional embeddings
  initializer_range: 0.02    # Std dev for weight initialization
  layer_norm_eps: 1e-5       # Epsilon for Layer Normalization

  # --- PEER Configuration ---
  use_peer: true             # Enable PEER layers
  peer_start_layer: 2        # Layer index (0-based) where PEER layers begin
  peer_config:
    num_experts: 1048576       # Total number of experts (e.g., 1024*1024)
    num_experts_per_tok: 16    # Number of experts activated per token
    num_heads: 8               # Number of expert retrieval heads
    expert_hidden_size: 1      # Hidden size of each expert MLP (1 for single neuron)
    product_key_dim: [1024, 1024] # Dimensions for product key retrieval
    query_dim: 256             # Dimension of the query vector for retrieval
    batch_norm_query: true     # Apply LayerNorm to the query vector
    # --- Expert Tracking (Optional) ---
    log_expert_usage: true     # Enable logging expert usage to WandB
    log_freq: 1000             # How often (in steps) to log usage
    usage_threshold: 5.0       # Threshold multiplier to flag "hot" experts

  # --- MLA Configuration (Always Used) ---
  mla_config:
    q_lora_rank: 1536          # Rank for query LoRA projection (optional)
    kv_lora_rank: 512          # Rank for key-value latent projection
    qk_rope_head_dim: 64       # Dimension for RoPE part of Q/K
    v_head_dim: 128            # Dimension for Value head
    qk_nope_head_dim: 128      # Dimension for NoPE part of Q/K
    # rope_theta: 10000.0      # Base for RoPE calculation (optional, defaults to 10000)
```

### `train_config`

Controls the training loop behavior.

```yaml
train_config:
  output_dir: "./output"              # Directory for checkpoints and logs
  per_device_train_batch_size: 16     # Batch size per GPU for training
  per_device_eval_batch_size: 16      # Batch size per GPU for evaluation
  gradient_accumulation_steps: 4      # Accumulate gradients over N steps
  learning_rate: 5e-5                 # Peak learning rate
  weight_decay: 0.01                  # Weight decay for AdamW optimizer
  max_grad_norm: 1.0                  # Gradient clipping threshold
  # --- Training Duration (Choose ONE) ---
  # max_steps: 100000                 # Train for a fixed number of steps (Recommended for streaming)
  num_train_epochs: 3                 # Train for a fixed number of epochs (Requires 'steps_per_epoch')
  # steps_per_epoch: 10000            # Required if using num_train_epochs with streaming datasets
  # --- Scheduler ---
  lr_scheduler_type: "cosine"         # Learning rate scheduler type (e.g., linear, cosine)
  warmup_ratio: 0.1                   # Proportion of total steps for LR warmup
  # --- Logging & Saving ---
  log_level: "info"                   # Logging level (debug, info, warning, error)
  logging_steps: 100                  # Log metrics every N steps
  save_steps: 1000                    # Save checkpoint every N steps
  eval_steps: 2000                    # Run evaluation every N steps
  # --- Reproducibility & Precision ---
  seed: 42                            # Random seed
  fp16: true                          # Enable Automatic Mixed Precision (FP16) - Use False if using FP8
  bf16: false                         # Enable Automatic Mixed Precision (BF16) - Use False if using FP8
  tf32: false                         # Allow TF32 on Ampere+ GPUs (PyTorch >= 1.7)
  # --- Resuming ---
  resume_from_checkpoint: null        # Path to checkpoint directory to resume from (e.g., "./output/checkpoint-5000")
```

*Note on Precision:* When using `torchao` for Float8 training (enabled by default in `llm/training/train.py`), set `fp16` and `bf16` to `false` here, as `torchao` manages the precision internally.

### `dataset_config`

Defines the datasets used for training and the tokenizer.

```yaml
dataset_config:
  datasets:
    # --- List of datasets to interleave ---
    - name: "pile-uncopyrighted"          # Unique name for this dataset entry
      path: "monology/pile-uncopyrighted" # Path (HuggingFace dataset name or local path)
      # subset: "en"                      # Optional: Specify subset/config (e.g., for C4, wikipedia)
      split: "train"                      # Dataset split to use
      streaming: true                     # Load data progressively (recommended for large datasets)
      weight: 0.7                         # Sampling weight for interleaving
      # --- Text Extraction (Choose ONE) ---
      text_field: "text"                  # Field containing the raw text
      # text_template: "Instruction:\n{instruction}\n\nInput:\n{input}\n\nOutput:\n{output}" # OR: Format string using fields from the dataset

    - name: "c4"
      path: "allenai/c4"
      subset: "en"
      split: "train"
      streaming: true
      weight: 0.2
      text_field: "text"

    # ... more datasets ...

  tokenizer_name: "gpt2"                # Name or path of the HuggingFace tokenizer
  max_seq_length: 2048                # Maximum sequence length for tokenization
```

See `configs/datasets.yaml` for examples of defining custom dataset mixes for ablation testing.

### `eval_config`

Configures the evaluation process run periodically during training.

```yaml
eval_config:
  evals_registry_path: "./evals/registry" # Path to OpenAI Evals registry (if using)
  evals:                                  # List of benchmark names to run
    - "hellaswag"
    - "mmlu"
    - "truthfulqa"
    - "gsm8k"
    - "humaneval"
  eval_batch_size: 16                     # Batch size for evaluation runs
```

### `wandb_config`

Configures integration with Weights & Biases for logging and tracking. Set to `null` or omit to disable.

```yaml
wandb_config:
  project: "llm-research"                 # WandB project name
  entity: "your-wandb-username"           # WandB entity (username or team)
  name: "lm-training-peer-mla"            # Run name (often overridden by scripts)
  log_model: "checkpoints"                # Log model artifacts ('checkpoints', 'all', or null)
```

## Custom Dataset Configuration (`datasets.yaml`)

You can define alternative dataset mixes in a separate file (e.g., `configs/datasets.yaml`) for use in ablation studies. This file follows a specific structure:

```yaml
# configs/datasets.yaml
data_mixes:
  # --- Define one or more named mixes ---
  high_quality_mix:                 # Name of the mix
    name: "high_quality_mix"        # Display name (can be same as key)
    datasets:                       # List of datasets for this mix
      - name: "custom_dataset_1"
        path: "/local/path/to/dataset_1.jsonl" # Local path or HF name
        format: "json"              # Specify format for local files (json, csv, text)
        split: "train"
        streaming: true
        weight: 0.4
        text_field: "text"          # Or use text_template
      # ... more datasets ...

  science_mix:
    name: "science_mix"
    datasets:
      # ... datasets for science mix ...
```

The ablation script (`scripts/run_ablations.py`) can load these mixes using the `--custom-datasets` argument.
