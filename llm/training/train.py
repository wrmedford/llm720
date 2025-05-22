#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Foundation Language Model Training Script with PEER and Multi-Headed Latent Attention

This module provides a production-ready implementation for training language models with:
- PEER (Parameter Efficient Expert Retrieval) for expert selection
- Multi-Headed Latent Attention (MLA) as in DeepSeek models
- Efficient dataset handling with configurable interleaving
- Comprehensive training and evaluation pipeline
"""

import argparse
import datetime
import json
import logging
import os

import torch
import torch.compiler # Import the compiler module
import torch.nn as nn
import torch.optim as optim
import yaml
import tiktoken # Import tiktoken
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from safetensors.torch import load_file, save_file
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (DataCollatorForLanguageModeling, # Removed AutoTokenizer
                          get_scheduler)
# Use torchao for float8 support
from torchao.float8 import ( # Updated import
    convert_to_float8_training,
    Float8LinearConfig,
    ScalingType,
    ScalingGranularity,
    # Optional: Import other config elements if needed later
    # CastConfig, Float8GemmConfig
)
# from torchao.utils import TORCH_VERSION_AT_LEAST_2_5 # Optional: Add version check if needed

from llm.data.datasets import prepare_datasets, tokenize_function
from llm.models.foundation import TransformerConfig, create_model_from_config
from llm.utils.experts.tracking import ExpertUsageTracker, hook_expert_tracking

logger = get_logger(__name__)

# Constants
MAX_SEQ_LEN = 2048
# PAD_TOKEN_ID = 0 # No longer used directly, will use tiktoken's EOT


class TrainerConfig:
    """Configuration for the trainer."""

    def __init__(self, **kwargs):
        # Model configuration
        self.model_config = {
            # Architecture
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            # PEER configuration
            "use_peer": True,
            "peer_start_layer": 2, # Start PEER layers from layer index 2
            "peer_config": {
                "num_experts": 1024,  # Default 1024 experts (32x32)
                "num_experts_per_tok": 16,
                "num_heads": 8,
                "expert_hidden_size": 1,
                "product_key_dim": [32, 32],  # Cartesian product dimensions
                "query_dim": 256,
                "batch_norm_query": True,
            },
            # MLA configuration (MLA is always used)
            "mla_config": {
                "q_lora_rank": 1536,
                "kv_lora_rank": 512,
                "qk_rope_head_dim": 64,
                "v_head_dim": 128,
                "qk_nope_head_dim": 128,
            },
        }

        # Training configuration
        self.train_config = {
            "output_dir": "./output",
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 16,
            "gradient_accumulation_steps": 1,
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "num_train_epochs": 3,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1,
            "log_level": "info",
            "logging_steps": 100,
            "save_steps": 1000,
            "eval_steps": 1000,
            "seed": 42,
            "fp16": True,
            "bf16": False,
            "tf32": False,
            "resume_from_checkpoint": None,
        }

        # Dataset configuration
        self.dataset_config = {
            "datasets": [
                {
                    "name": "pile",
                    "path": "EleutherAI/pile",
                    "split": "train",
                    "streaming": True,
                    "weight": 0.7,
                    "text_field": "text",
                },
                {
                    "name": "c4",
                    "path": "allenai/c4",
                    "split": "train",
                    "streaming": True,
                    "weight": 0.3,
                    "text_field": "text",
                },
            ],
            "tokenizer_name": "o200k_base", # Default to o200k_base
            "max_seq_length": MAX_SEQ_LEN,
        }

        # Evaluation configuration
        self.eval_config = {
            "evals_registry_path": "./evals/registry",
            "evals": [
                "hellaswag",
                "mmlu",
                "truthfulqa",
            ],
            "eval_batch_size": 16,
        }

        # Wandb configuration
        self.wandb_config = {
            "project": "foundation-model-training",
            "entity": "your-wandb-entity",
            "name": f"lm-training-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "log_model": "all",
        }

        # Update with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @classmethod
    def from_yaml(cls, yaml_file: str) -> "TrainerConfig":
        """Load configuration from a YAML file."""
        with open(yaml_file, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    def save(self, filepath: str) -> None:
        """Save configuration to a YAML file."""
        config_dict = {
            "model_config": self.model_config,
            "train_config": self.train_config,
            "dataset_config": self.dataset_config,
            "eval_config": self.eval_config,
            "wandb_config": self.wandb_config,
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)


def prepare_model_inputs(batch, device):
    """Prepare model inputs from a batch, ensuring tensors are on the correct device."""
    # Move all tensors to the device
    batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}

    # Generate position_ids if not present
    if "position_ids" not in batch:
        batch_size, seq_length = batch["input_ids"].shape
        batch["position_ids"] = (
            torch.arange(seq_length, dtype=torch.long, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

    return batch


def save_checkpoint(model, optimizer, scheduler, global_step, epoch, filepath):
    """Save a model checkpoint with all required components."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Save model weights separately (efficient saving)
    if isinstance(model, DDP):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    # Handle tied weights (embedding and lm_head) before saving with safetensors
    # Check if weights are tied (common practice)
    lm_head_key = "_orig_mod.lm_head.weight" # Key might change if torch.compile isn't used
    wte_key = "_orig_mod.wte.weight"
    # Adjust keys if not using torch.compile or if model structure differs
    # Example: lm_head_key = "lm_head.weight", wte_key = "wte.weight"

    if lm_head_key in model_state_dict and wte_key in model_state_dict:
        if model_state_dict[lm_head_key].data_ptr() == model_state_dict[wte_key].data_ptr():
            logger.info(f"Detected tied weights ({wte_key} and {lm_head_key}). Removing {lm_head_key} before saving.")
            del model_state_dict[lm_head_key]

    # Save model weights using safetensors
    try:
        save_file(model_state_dict, f"{filepath}.safetensors")
    except RuntimeError as e:
        logger.error(f"Failed to save model weights with safetensors: {e}")
        logger.error("Consider checking for other tied weights or saving issues.")
        raise # Re-raise the error after logging

    # Save additional training state
    training_state = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "global_step": global_step,
        "epoch": epoch,
    }

    torch.save(training_state, f"{filepath}.training_state")
    logger.info(f"Saved checkpoint to {filepath}")

    return filepath


def load_checkpoint(model, optimizer, scheduler, filepath):
    """Load a model checkpoint with all required components."""
    # Load model weights
    model_path = f"{filepath}.safetensors"
    if os.path.exists(model_path):
        state_dict = load_file(model_path)

        # Determine the base model (handle DDP)
        base_model = model.module if isinstance(model, DDP) else model

        # Load the state dict
        base_model.load_state_dict(state_dict)

        # After loading weights, call post_weight_load for MLA layers
        if hasattr(base_model, "blocks"):
            for block in base_model.blocks:
                if hasattr(block, "attention") and hasattr(
                    block.attention, "post_weight_load"
                ):
                    logger.info(
                        f"Calling post_weight_load for layer {block.layer_idx} attention."
                    )
                    block.attention.post_weight_load()

    else:
        raise FileNotFoundError(f"Model weights not found at {model_path}")

    # Load training state
    training_state_path = f"{filepath}.training_state"
    if os.path.exists(training_state_path):
        training_state = torch.load(training_state_path, map_location="cpu")

        optimizer.load_state_dict(training_state["optimizer"])
        if scheduler and "scheduler" in training_state and training_state["scheduler"]:
            scheduler.load_state_dict(training_state["scheduler"])

        global_step = training_state["global_step"]
        epoch = training_state["epoch"]
    else:
        raise FileNotFoundError(f"Training state not found at {training_state_path}")

    logger.info(f"Loaded checkpoint from {filepath}")

    return global_step, epoch


import os # Add os import

class ModelEvaluator:
    """Evaluator for running evals on model checkpoints."""

    def __init__(self, registry_path, tokenizer, device):
        self.registry = None
        # Store the tiktoken encoder instance
        self.tokenizer: tiktoken.Encoding = tokenizer
        self.device = device
        self.evals_available = False
        self.openai_key_present = bool(os.environ.get("OPENAI_API_KEY"))
        # Define EOS/PAD token ID for tiktoken
        self.eot_token_id = self.tokenizer.eot_token

        if not self.openai_key_present:
            logger.warning(
                "OPENAI_API_KEY not found in environment. OpenAI-based evaluations will be skipped."
            )
            return # Skip evals initialization

        try:
            # Conditionally import only if key is present
            from evals.registry import Registry

            self.registry = Registry(registry_path)
            self.evals_available = True
            logger.info("Evals registry initialized successfully.")

        except ImportError:
            logger.warning(
                "Evals library not found or import failed, even though OPENAI_API_KEY is set. Evaluation will be skipped."
            )
        except Exception as e:
            # Catch potential errors during Registry initialization
            logger.warning(
                f"Failed to initialize Evals registry: {e}. Evaluation will be skipped."
            )


    def model_completion_fn(self, model, prompt, **kwargs):
        """Create a completion function for evals."""
        # Default generation parameters
        max_tokens = kwargs.pop("max_tokens", 256)
        temperature = kwargs.pop("temperature", 0.7)
        top_p = kwargs.pop("top_p", 0.9)

        # Tokenize the prompt using tiktoken
        prompt_ids = self.tokenizer.encode(prompt, allowed_special="all") # Allow all special tokens
        inputs = {"input_ids": torch.tensor([prompt_ids], dtype=torch.long, device=self.device)}

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.eot_token_id, # Use EOT token for padding during generation
                eos_token_id=self.eot_token_id, # Use EOT token for stopping generation
                **kwargs,
            )

        # Decode the response, removing the prompt
        prompt_length = len(inputs["input_ids"][0])
        # outputs[0] contains the full sequence including prompt
        output_ids = outputs[0][prompt_length:].tolist()
        # Filter out potential padding tokens if necessary, though EOS should stop generation
        output_ids = [token_id for token_id in output_ids if token_id != self.eot_token_id]
        response = self.tokenizer.decode(output_ids)

        return [{"text": response}]

    def run_eval(self, model, eval_name):
        """Run a specific evaluation and return results."""
        if not self.evals_available:
            error_reason = "Evals library not available or failed to initialize."
            if not self.openai_key_present:
                error_reason = "Skipped due to missing OPENAI_API_KEY."

            return {
                "eval_name": eval_name,
                "error": error_reason,
                "metrics": {"skipped": 1.0}, # Use 'skipped' instead of 'error'
                "samples": [],
            }

        # Import necessary modules conditionally (only if evals are available)
        import functools

        # Create completion function for this model
        completion_fn = functools.partial(self.model_completion_fn, model)

        try:
            # Run the evaluation
            eval_spec = self.registry.get_eval(eval_name)
            eval_instance = eval_spec.cls(completion_fn=completion_fn, **eval_spec.args)
            result = eval_instance.run()

            return {
                "eval_name": eval_name,
                "metrics": result.metrics,
                "samples": result.sample_metrics,
            }
        except Exception as e:
            logger.error(f"Error running evaluation {eval_name}: {e}")
            return {
                "eval_name": eval_name,
                "error": str(e),
                "metrics": {"error": 1.0},
                "samples": [],
            }

    def run_all_evals(self, model, eval_names):
        """Run all specified evaluations and return results."""
        results = {}

        for eval_name in eval_names:
            logger.info(f"Running evaluation: {eval_name}")
            result = self.run_eval(model, eval_name)
            results[eval_name] = result

        return results


def get_train_setup(config: TrainerConfig, model: nn.Module):
    """Set up training components based on configuration."""
    # Get optimizer
    # Ensure learning rate is a float
    lr_value = config.train_config["learning_rate"]
    try:
        lr_float = float(lr_value)
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid learning rate value in config: {lr_value}. Must be a number. Error: {e}")
        raise TypeError(f"Learning rate must be a float, but got {type(lr_value)}") from e

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr_float, # Use the validated float value
        weight_decay=config.train_config["weight_decay"],
    )

    # Determine total training steps
    if config.train_config.get("max_steps"):
        total_steps = config.train_config["max_steps"]
        logger.info(f"Training for a maximum of {total_steps} steps.")
    elif config.train_config.get("num_train_epochs"):
        # This path is less reliable for IterableDatasets but kept for potential compatibility
        # A steps_per_epoch value should ideally be provided in the config if using epochs with iterable datasets
        steps_per_epoch = config.train_config.get("steps_per_epoch")
        if not steps_per_epoch:
            raise ValueError(
                "`steps_per_epoch` must be specified in train_config when using `num_train_epochs` with iterable datasets."
            )
        total_steps = config.train_config["num_train_epochs"] * steps_per_epoch
        logger.warning(
            f"Using num_train_epochs with IterableDataset. Calculated total_steps={total_steps}. Consider using max_steps instead."
        )
    else:
        raise ValueError(
            "Training duration not specified. Set either `max_steps` or `num_train_epochs` (with `steps_per_epoch`) in train_config."
        )

    warmup_steps = int(total_steps * config.train_config["warmup_ratio"])

    scheduler = get_scheduler(
        name=config.train_config["lr_scheduler_type"],
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    return optimizer, scheduler


def run_training(config: TrainerConfig):
    """Main training function."""
    # Set up accelerator
    # Disable accelerate's mixed precision when using torchao's FP8 conversion,
    # as torchao handles the precision internally.
    accelerator = Accelerator(
        gradient_accumulation_steps=config.train_config["gradient_accumulation_steps"],
        mixed_precision="no", # Let torchao handle precision
        log_with="wandb" if config.wandb_config else None,
    )

    # Set random seed for reproducibility
    set_seed(config.train_config["seed"])

    # Initialize wandb if configured
    if accelerator.is_main_process and config.wandb_config:
        try:
            accelerator.init_trackers(
                project_name=config.wandb_config["project"],
                config=vars(config),
                init_kwargs={"wandb": config.wandb_config},
            )
        except ImportError:
            logger.warning("Wandb not installed. Continuing without logging.")

    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(config.train_config["output_dir"], exist_ok=True)

        # Save configuration to output directory
        config.save(os.path.join(config.train_config["output_dir"], "config.yaml"))

    # Load tiktoken encoder
    tokenizer_name = config.dataset_config["tokenizer_name"]
    logger.info(f"Loading tiktoken encoder: {tokenizer_name}")
    try:
        tokenizer = tiktoken.get_encoding(tokenizer_name)
        # Define pad_token_id using the EOT token
        pad_token_id = tokenizer.eot_token
        logger.info(f"Using Tokenizer: {tokenizer_name}, Vocab Size: {tokenizer.n_vocab}, EOT/PAD ID: {pad_token_id}")
    except ValueError as e:
        logger.error(f"Failed to load tiktoken encoder '{tokenizer_name}': {e}")
        raise

    # Create model
    logger.info("Initializing model...")
    # Verify configured vocab_size is sufficient for the tokenizer
    configured_vocab_size = config.model_config.get("vocab_size")
    actual_vocab_size = tokenizer.n_vocab
    if configured_vocab_size is None:
        logger.warning(f"vocab_size not found in model_config. Setting to tokenizer's vocab size: {actual_vocab_size}")
        config.model_config["vocab_size"] = actual_vocab_size
    elif configured_vocab_size < actual_vocab_size:
        logger.warning(
            f"Configured vocab_size ({configured_vocab_size}) is smaller than tokenizer vocab size ({actual_vocab_size}). "
            f"Increasing model vocab_size to {actual_vocab_size}."
        )
        config.model_config["vocab_size"] = actual_vocab_size
    else:
        logger.info(f"Using configured vocab_size: {configured_vocab_size} (Tokenizer actual: {actual_vocab_size})")
        # Ensure the value is an int
        config.model_config["vocab_size"] = int(configured_vocab_size)

    # Determine target primary dtype (bf16 preferred for H100+)
    fp8_dtype = torch.bfloat16 if config.train_config.get("bf16", torch.cuda.is_bf16_supported()) else torch.float16

    # Create model directly on target device and primary dtype
    logger.info(f"Initializing model on {accelerator.device} with primary dtype {fp8_dtype}...")
    tf_config = TransformerConfig(**config.model_config)
    # Initialize model on the target device and dtype.
    # The modified _init_weights method will be applied during creation,
    # setting LayerNorm/Linear biases to FP32 within this BF16/FP16 structure.
    model = create_model_from_config(tf_config).to(device=accelerator.device, dtype=fp8_dtype)
    logger.info("Model initialized with mixed precision (FP32 for LN/Bias).")

    # Convert Linear/Embedding weights to Float8 format
    logger.info(f"Converting model Linear/Embedding weights to Float8 format...")
    # Configure FP8 conversion explicitly:
    # - Use DELAYED scaling for stability.
    # - Use PER_TENSOR granularity (robust default).
    # - Enable padding for dimensions not divisible by 16 (e.g., vocab size).
    # - torchao's default uses HYBRID format (E4M3 weights, E5M2 activations).
    fp8_config = Float8LinearConfig(
        scaling_type_weights=ScalingType.DELAYED,
        scaling_type_activation=ScalingType.DELAYED,
        scaling_granularity=ScalingGranularity.PER_TENSOR,
        pad_inner_dim=True,
    )
    logger.info(f"Using explicit Float8LinearConfig: {fp8_config}")
    convert_to_float8_training(model, config=fp8_config) # Pass the config
    logger.info("Model Linear/Embedding weights converted for Float8 training.")

    # Set up expert usage tracking if enabled
    expert_tracker = None
    peer_config = config.model_config.get("peer_config", {})

    if config.model_config.get("use_peer", False) and peer_config.get(
        "log_expert_usage", False
    ):
        logger.info("Setting up expert usage tracking...")
        # Count PEER layers in the model - in our architecture, every other layer starting from layer 1
        num_peer_layers = config.model_config.get("num_hidden_layers", 12) // 2
        num_experts = peer_config.get("num_experts", 1048576)

        # Create tracker
        expert_tracker = ExpertUsageTracker(
            num_experts=num_experts,
            num_layers=num_peer_layers,
            log_freq=peer_config.get("log_freq", 1000),
            usage_threshold=peer_config.get("usage_threshold", 5.0),
            wandb_enabled=bool(config.wandb_config),
        )

        # Hook model to track expert usage
        hook_expert_tracking(model, expert_tracker)
        logger.info(
            f"Expert usage tracking enabled for {num_peer_layers} PEER layers with {num_experts} experts each"
        )

    # Prepare datasets
    logger.info("Preparing datasets...")
    train_dataset = prepare_datasets(config.dataset_config)

    # Tokenize datasets using tiktoken
    # We need to pass the tokenizer and pad_token_id to the map function
    from functools import partial
    tokenize_map_fn = partial(
        tokenize_function,
        tokenizer=tokenizer,
        max_seq_length=config.dataset_config["max_seq_length"],
        pad_token_id=pad_token_id
    )

    tokenized_dataset = train_dataset.map(
        tokenize_map_fn,
        batched=True,
        remove_columns=["text"],
    )

    # Create data loader
    # DataCollatorForLanguageModeling expects a HF tokenizer or specific dict structure.
    # Since our tokenize_function now prepares 'input_ids' and 'attention_mask' correctly
    # and handles padding, we can use a simpler default collator or pass pad_token_id.
    # Let's try passing pad_token_id directly.
    # Note: Label shifting for causal LM is handled inside the model's forward pass.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=None, # Pass None as we handle tokenization separately
        mlm=False,
        pad_to_multiple_of=config.dataset_config["max_seq_length"] # Ensure batches have consistent length
    )
    # Manually set the pad_token_id if the collator needs it (though it might not use it if tokenizer=None)
    # data_collator.tokenizer = argparse.Namespace(pad_token_id=pad_token_id) # Create a dummy tokenizer object

    # Determine appropriate number of workers
    # Use half the available CPUs per process as a starting point, minimum 1
    num_cpus = os.cpu_count() or 1
    workers_per_process = max(1, num_cpus // accelerator.num_processes // 2)
    logger.info(f"Using {workers_per_process} DataLoader workers per process.")

    train_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=config.train_config["per_device_train_batch_size"],
        collate_fn=data_collator,
        num_workers=workers_per_process,
        pin_memory=True, # Pin memory for faster CPU to GPU transfers
    )

    # Set up optimizer and scheduler
    optimizer, scheduler = get_train_setup(config, model)

    # Prepare for distributed training
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )

    # --- Ensure LayerNorm is FP32 ---
    # After accelerator.prepare and before torch.compile, explicitly set LayerNorm
    # modules to FP32, as torchao's FP8 conversion focuses on Linear/Embedding weights
    # and accelerator might have cast the model to BF16/FP16.
    logger.info("Ensuring LayerNorm modules are set to FP32...")
    for name, module in accelerator.unwrap_model(model).named_modules():
        if isinstance(module, nn.LayerNorm):
            module.to(torch.float32)
            # Log parameters to confirm dtype
            # for param_name, param in module.named_parameters():
            #     logger.debug(f"LayerNorm {name}.{param_name} set to {param.dtype}")
    logger.info("LayerNorm modules confirmed as FP32.")
    # --- End LayerNorm FP32 ---

    # Compile model for better performance
    logger.info("Compiling model with torch.compile...")
    model = torch.compile(model, mode="reduce-overhead")
    logger.info("Model compiled.")

    # Set up evaluator
    evaluator = ModelEvaluator(
        config.eval_config["evals_registry_path"],
        tokenizer,
        accelerator.device,
    )

    # Determine training duration (max_steps)
    if config.train_config.get("max_steps"):
        max_steps = config.train_config["max_steps"]
    elif config.train_config.get("num_train_epochs"):
        # This path requires steps_per_epoch to be reliable with IterableDataset
        steps_per_epoch = config.train_config.get("steps_per_epoch")
        if not steps_per_epoch:
            raise ValueError(
                "`steps_per_epoch` must be specified in train_config when using `num_train_epochs` with iterable datasets."
            )
        max_steps = config.train_config["num_train_epochs"] * steps_per_epoch
    else:
        raise ValueError(
            "Training duration not specified. Set either `max_steps` or `num_train_epochs` (with `steps_per_epoch`) in train_config."
        )

    # Load checkpoint if resuming
    global_step = 0
    if config.train_config["resume_from_checkpoint"]:
        checkpoint_path = config.train_config["resume_from_checkpoint"]
        logger.info(f"Attempting to resume from checkpoint: {checkpoint_path}")
        try:
            # Assuming load_checkpoint now only returns global_step and epoch (epoch might not be accurate for iterable)
            loaded_step, _ = load_checkpoint(
                accelerator.unwrap_model(model), optimizer, scheduler, checkpoint_path
            )
            global_step = loaded_step
            logger.info(f"Successfully resumed from global step {global_step}")

        except FileNotFoundError:
            logger.warning(
                f"Checkpoint not found at {checkpoint_path}, starting from scratch."
            )
        except Exception as e:
            logger.error(
                f"Failed to load checkpoint from {checkpoint_path}: {e}. Starting from scratch."
            )
    else:
        # If not resuming, ensure MLA weights are prepared after model is on device
        logger.info("Preparing MLA decode weights for initial model...")
        unwrapped_model = accelerator.unwrap_model(model)
        if hasattr(unwrapped_model, "blocks"):
            for block in unwrapped_model.blocks:
                if hasattr(block, "attention") and hasattr(
                    block.attention, "post_weight_load"
                ):
                    block.attention.post_weight_load()

    # Training loop
    logger.info("Starting training...")
    model.train()

    progress_bar = tqdm(
        total=max_steps,
        disable=not accelerator.is_local_main_process,
        initial=global_step,
    )

    # Main training loop - iterates up to max_steps
    while global_step < max_steps:
        model.train()  # Ensure model is in training mode at the start of each effective "epoch" or segment
        for step, batch in enumerate(train_dataloader):
            # Check if max_steps reached
            if global_step >= max_steps:
                break

            # Mark the beginning of a CUDA graph step before model invocation
            # This helps manage memory for compiled functions with dynamic parts.
            torch.compiler.cudagraph_mark_step_begin()

            # Prepare inputs
            batch = prepare_model_inputs(batch, accelerator.device)

            # Forward pass - FP8 conversion handles internal types
            # No explicit autocast needed here if model and inputs are already bf16/fp16
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            # Log loss
            if accelerator.is_main_process: # Log loss once per accumulation cycle
                # Detach loss before calling .item() to avoid warning
                accelerator.log({"train/loss": loss.detach().item()}, step=global_step)

            # Backward pass
            accelerator.backward(loss)

            # Clip gradients and Optimizer step (handled by accelerator)
            if accelerator.sync_gradients: # Operations after gradient sync
                if config.train_config["max_grad_norm"] > 0:
                    accelerator.clip_grad_norm_(
                        model.parameters(), config.train_config["max_grad_norm"]
                    )
                # Note: FP8 scale synchronization is typically handled by FSDP or similar
                # distributed strategies when using torchao's float8 with distributed training.
                # No explicit sync call needed here for single-node or basic DDP.

            # Optimizer step (happens when gradients are synced or at end of accumulation)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Update progress
            progress_bar.update(1)
            global_step += 1

            # Process expert usage tracking
            if expert_tracker is not None:
                expert_tracker.step_end()

            # Log metrics
            if global_step % config.train_config["logging_steps"] == 0:
                # Get learning rate
                if scheduler:
                    lr = scheduler.get_last_lr()[0]
                else:
                    # Handle case where scheduler might not be used (though unlikely with get_train_setup)
                    lr = config.train_config["learning_rate"]

                # Calculate approximate epoch for logging purposes if possible
                approx_epoch = -1.0
                if config.train_config.get("steps_per_epoch"):
                    approx_epoch = global_step / config.train_config["steps_per_epoch"]

                metrics = {
                    # Detach loss before calling .item()
                    "train/loss": loss.detach().item(),
                    "train/learning_rate": lr,
                    "train/global_step": global_step,
                }
                if approx_epoch >= 0:
                    metrics["train/approx_epoch"] = approx_epoch
                # Add expert usage metrics if available
                if expert_tracker is not None:
                    summary = expert_tracker.get_summary()
                    metrics.update(
                        {
                            "experts/hot_experts_count": summary["hot_experts_count"],
                            "experts/total_tokens_processed": summary[
                                "total_tokens_processed"
                            ],
                        }
                    )

                    # Add per-layer statistics
                    for layer_idx, stats in summary.get("layer_stats", {}).items():
                        metrics.update(
                            {
                                f"experts/layer_{layer_idx}/experts_used": stats[
                                    "experts_used"
                                ],
                                f"experts/layer_{layer_idx}/usage_coverage": stats[
                                    "usage_coverage"
                                ],
                            }
                        )

                accelerator.log(metrics)

            # Save checkpoint
            if global_step % config.train_config["save_steps"] == 0:
                if accelerator.is_main_process:
                    checkpoint_path = os.path.join(
                        config.train_config["output_dir"], f"checkpoint-{global_step}"
                    )

                    # Save checkpoint - epoch is less meaningful here, pass global_step or 0
                    save_checkpoint(
                        accelerator.unwrap_model(model),
                        optimizer,
                        scheduler,
                        global_step,
                        0,  # Epoch is not tracked directly in this loop
                        checkpoint_path,
                    )
                    # Save expert usage statistics if tracking is enabled
                    if expert_tracker is not None:
                        expert_stats_path = os.path.join(
                            config.train_config["output_dir"],
                            f"checkpoint-{global_step}-expert-stats.json",
                        )
                        with open(expert_stats_path, "w") as f:
                            json.dump(expert_tracker.get_summary(), f, indent=2)

            # Run evaluation
            if global_step % config.train_config["eval_steps"] == 0:
                if accelerator.is_main_process:
                    logger.info(f"Running evaluation at step {global_step}")

                    # Set model to eval mode
                    model.eval()

                    # Run evals
                    eval_results = evaluator.run_all_evals(
                        accelerator.unwrap_model(model), config.eval_config["evals"]
                    )

                    # Log eval results
                    for eval_name, result in eval_results.items():
                        if "metrics" in result:
                            for metric_name, value in result["metrics"].items():
                                accelerator.log(
                                    {
                                        f"eval/{eval_name}/{metric_name}": value,
                                    },
                                    step=global_step,
                                )  # Log eval metrics at the correct step

                    # Set model back to train mode
                    model.train()

            # Check if max_steps reached within inner loop
            if global_step >= max_steps:
                break
        # End of inner loop (dataloader iteration)

        # Check again if max_steps reached after iterating through dataloader
        if global_step >= max_steps:
            break
    # End of outer while loop

    progress_bar.close()

    # Save final model
    if accelerator.is_main_process:
        logger.info("Training completed. Saving final model...")

        final_checkpoint_path = os.path.join(
            config.train_config["output_dir"], "final-model"
        )

        save_checkpoint(
            accelerator.unwrap_model(model),
            optimizer,
            scheduler,
            global_step,  # Use final global_step
            0,  # Epoch is not tracked directly
            final_checkpoint_path,
        )
        # Save final expert usage statistics if tracking is enabled
        if expert_tracker is not None:
            expert_stats_path = os.path.join(
                config.train_config["output_dir"], "final-model-expert-stats.json"
            )
            with open(expert_stats_path, "w") as f:
                json.dump(expert_tracker.get_summary(), f, indent=2)

    # Finish tracking
    if accelerator.is_main_process and config.wandb_config:
        accelerator.end_training()

    logger.info("Training completed successfully!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train a foundation language model.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration YAML file"
    )
    args = parser.parse_args()

    # Load configuration
    config = TrainerConfig.from_yaml(args.config)

    # Run training
    run_training(config)


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Handle errors gracefully
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback

        traceback.print_exc()
        raise
