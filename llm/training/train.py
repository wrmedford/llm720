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
import torch.nn as nn
import torch.optim as optim
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from safetensors.torch import load_file, save_file
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (AutoTokenizer, DataCollatorForLanguageModeling,
                          get_scheduler)
from float8_experimental import config as float8_config
from float8_experimental.float8_linear import Float8Linear
from float8_experimental.float8_linear_utils import (
    swap_linear_with_float8_linear,
    sync_float8_amax_and_scale_history
)

from llm.data.datasets import prepare_datasets, tokenize_function
from llm.models.foundation import TransformerConfig, create_model_from_config
from llm.utils.experts.tracking import ExpertUsageTracker, hook_expert_tracking

logger = get_logger(__name__)

# Constants
MAX_SEQ_LEN = 2048
PAD_TOKEN_ID = 0


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
            "tokenizer_name": "gpt2",
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
    """Prepare model inputs from a batch."""
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

    # Save model weights
    save_file(model_state_dict, f"{filepath}.safetensors")

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


class ModelEvaluator:
    """Evaluator for running evals on model checkpoints."""

    def __init__(self, registry_path, tokenizer, device):
        try:
            from evals.registry import Registry

            self.registry = Registry(registry_path)
            self.tokenizer = tokenizer
            self.device = device
        except ImportError:
            logger.warning("Evals library not found. Evaluation will be limited.")
            self.registry = None

    def model_completion_fn(self, model, prompt, **kwargs):
        """Create a completion function for evals."""
        # Default generation parameters
        max_tokens = kwargs.pop("max_tokens", 256)
        temperature = kwargs.pop("temperature", 0.7)
        top_p = kwargs.pop("top_p", 0.9)

        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

        # Decode the response, removing the prompt
        prompt_length = len(inputs.input_ids[0])
        response = self.tokenizer.decode(
            outputs[0][prompt_length:], skip_special_tokens=True
        )

        return [{"text": response}]

    def run_eval(self, model, eval_name):
        """Run a specific evaluation and return results."""
        if not self.registry:
            return {
                "eval_name": eval_name,
                "error": "Evals library not available",
                "metrics": {"error": 1.0},
                "samples": [],
            }

        # Import necessary modules conditionally
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
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.train_config["learning_rate"],
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
    accelerator = Accelerator(
        gradient_accumulation_steps=config.train_config["gradient_accumulation_steps"],
        mixed_precision="fp16" if config.train_config["fp16"] else "no",
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

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.dataset_config["tokenizer_name"])

    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create model
    logger.info("Initializing model...")
    tf_config = TransformerConfig(**config.model_config)
    model = create_model_from_config(tf_config)
    
    # Configure float8_experimental
    logger.info("Configuring float8_experimental...")
    float8_config.enable_amax_init = False
    float8_config.amax_history_len = 16
    
    # Swap nn.Linear layers with Float8Linear
    logger.info("Swapping nn.Linear layers with Float8Linear...")
    swap_linear_with_float8_linear(model, Float8Linear)
    logger.info("Model layers swapped for FP8.")

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

    # Tokenize datasets
    def tokenize_map_fn(examples):
        return tokenize_function(
            examples, tokenizer, config.dataset_config["max_seq_length"]
        )

    tokenized_dataset = train_dataset.map(
        tokenize_map_fn,
        batched=True,
        remove_columns=["text"],
    )

    # Create data loader
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    train_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=config.train_config["per_device_train_batch_size"],
        collate_fn=data_collator,
    )

    # Set up optimizer and scheduler
    optimizer, scheduler = get_train_setup(config, model)

    # Prepare for distributed training
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    
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

            # Prepare inputs
            batch = prepare_model_inputs(batch, accelerator.device)

            # Forward pass with autocast for non-FP8 operations
            autocast_dtype = torch.bfloat16 if config.train_config.get("bf16", torch.cuda.is_bf16_supported()) else torch.float16
            with torch.autocast(device_type=accelerator.device.type, dtype=autocast_dtype):
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            # Note: Backward pass MUST be outside the fp8_autocast context
            # Accelerator handles gradient accumulation context correctly with backward outside
            if accelerator.is_main_process:  # Only log loss once per accumulation cycle
                accelerator.log({"train/loss_unscaled": loss.item()}, step=global_step)

            # Backward pass (outside fp8_autocast)
            accelerator.backward(loss)

            # Clip gradients (after backward)
            if accelerator.sync_gradients:  # Only clip when gradients are synchronized
                if config.train_config["max_grad_norm"] > 0:
                    accelerator.clip_grad_norm_(
                        model.parameters(), config.train_config["max_grad_norm"]
                    )
                # Sync FP8 scaling factors across processes and update history
                sync_float8_amax_and_scale_history(model)

            # Optimizer step
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
                    "train/loss": loss.item(),
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
