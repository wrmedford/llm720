#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Perplexity evaluation script for foundation models

This script evaluates the perplexity of a trained model on validation datasets.
It supports multiple datasets including Wikitext, C4, and Pile.

Usage:
    python evaluate_perplexity.py --config config.yaml \
                                  --checkpoint path/to/checkpoint.safetensors \
                                  --dataset wikitext \
                                  --dataset_split validation \
                                  --output results.json
"""

import os
import json
import yaml
import argparse
import logging
import math
import time
import random
from typing import Dict, List, Optional, Any, Union

import torch
import numpy as np
from tqdm.auto import tqdm
from safetensors.torch import load_file

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(config_path: str, checkpoint_path: str):
    """
    Load model and tokenizer from configuration and checkpoint.
    
    Args:
        config_path: Path to configuration file
        checkpoint_path: Path to model checkpoint
        
    Returns:
        Tuple of (model, tokenizer)
    """
    import transformers
    from transformers import AutoTokenizer
    
    # Import necessary module from train_lm.py
    import sys
    sys.path.append(".")
    from src.train import (
        TransformerConfig, 
        create_model_from_config,
    )
    
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model config
    model_config = TransformerConfig(**config["model_config"])
    
    # Create model
    logger.info("Creating model from configuration")
    model = create_model_from_config(model_config)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(config["dataset_config"]["tokenizer_name"])
    
    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load checkpoint weights
    logger.info(f"Loading weights from {checkpoint_path}")
    if checkpoint_path.endswith(".safetensors"):
        state_dict = load_file(checkpoint_path)
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    
    return model, tokenizer


def load_evaluation_dataset(
    dataset_name: str, 
    split: str = "validation", 
    max_samples: int = 1000,
    text_column: str = "text"
):
    """
    Load evaluation dataset.
    
    Args:
        dataset_name: Name of the dataset
        split: Dataset split
        max_samples: Maximum number of samples to load
        text_column: Column containing text data
        
    Returns:
        List of text samples
    """
    logger.info(f"Loading dataset {dataset_name} ({split})")
    
    if dataset_name == "wikitext":
        # Load Wikitext dataset
        dataset = load_dataset("wikitext", "wikitext-103-v1", split=split)
        samples = [sample for sample in dataset[text_column] if sample.strip()]
    
    elif dataset_name == "c4":
        # Load C4 dataset
        dataset = load_dataset("allenai/c4", "en", split=f"{split}[:1%]")
        samples = [sample for sample in dataset[text_column] if sample.strip()]
    
    elif dataset_name == "pile":
        # Load Pile dataset
        dataset = load_dataset("EleutherAI/pile", split=f"{split}[:1%]")
        samples = [sample for sample in dataset[text_column] if sample.strip()]
    
    else:
        # For custom datasets, try to load from path
        try:
            dataset = load_dataset(dataset_name, split=split)
            if text_column in dataset.column_names:
                samples = [sample for sample in dataset[text_column] if sample and sample.strip()]
            else:
                # Try to find a suitable text column
                text_columns = [col for col in dataset.column_names if "text" in col.lower()]
                if text_columns:
                    text_column = text_columns[0]
                    samples = [sample for sample in dataset[text_column] if sample and sample.strip()]
                else:
                    raise ValueError(f"Could not find text column in {dataset_name}")
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            # Use a simple fallback dataset
            samples = ["This is a fallback dataset for testing perplexity evaluation."] * 10
    
    # Limit number of samples
    if max_samples and len(samples) > max_samples:
        samples = random.sample(samples, max_samples)
    
    logger.info(f"Loaded {len(samples)} samples from {dataset_name}")
    return samples


def evaluate_perplexity(
    model, 
    tokenizer, 
    samples: List[str], 
    device: torch.device,
    max_seq_length: int = 1024,
    batch_size: int = 8
):
    """
    Evaluate model perplexity on given text samples.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        samples: List of text samples
        device: The device to use
        max_seq_length: Maximum sequence length
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary with perplexity results
    """
    model.to(device)
    model.eval()
    
    # Track stats
    total_loss = 0.0
    total_tokens = 0
    
    for i in tqdm(range(0, len(samples), batch_size), desc="Evaluating perplexity"):
        batch_samples = samples[i:i+batch_size]
        
        # Tokenize batch
        encodings = tokenizer(
            batch_samples,
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt"
        )
        
        # Move batch to device
        input_ids = encodings.input_ids.to(device)
        attention_mask = encodings.attention_mask.to(device)
        
        # Create labels (shifted input_ids)
        labels = input_ids.clone()
        
        # Calculate loss
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            
            # Get number of tokens in batch (excluding padding)
            num_tokens = attention_mask.sum().item()
            
            # Accumulate loss and token count
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    # Calculate perplexity
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss)
    
    return {
        "perplexity": perplexity,
        "loss": avg_loss,
        "total_tokens": total_tokens
    }


def main():
    """Main entry point for perplexity evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate model perplexity")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default="wikitext", 
                       help="Dataset to evaluate (wikitext, c4, pile, or custom path)")
    parser.add_argument("--dataset_split", type=str, default="validation", 
                       help="Dataset split to evaluate")
    parser.add_argument("--text_column", type=str, default="text",
                       help="Column containing text data")
    parser.add_argument("--max_samples", type=int, default=1000,
                       help="Maximum number of samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for evaluation")
    parser.add_argument("--max_seq_length", type=int, default=1024,
                       help="Maximum sequence length")
    parser.add_argument("--output", type=str, default=None, 
                       help="Path to save results")
    parser.add_argument("--device", type=str, default=None, 
                       help="Device to use (cpu, cuda, cuda:0, etc.)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    device = torch.device(args.device)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.config, args.checkpoint)
    
    # Load evaluation dataset
    samples = load_evaluation_dataset(
        args.dataset, args.dataset_split, args.max_samples, args.text_column
    )
    
    # Set output path
    if args.output is None:
        output_dir = os.path.join("results", "perplexity")
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(
            output_dir, 
            f"{os.path.basename(args.checkpoint)}_{args.dataset}_{args.dataset_split}.json"
        )
    
    # Evaluate perplexity
    logger.info(f"Evaluating perplexity on {args.dataset} ({args.dataset_split})")
    start_time = time.time()
    
    results = evaluate_perplexity(
        model, tokenizer, samples, device, args.max_seq_length, args.batch_size
    )
    
    # Add metadata to results
    results["dataset"] = args.dataset
    results["dataset_split"] = args.dataset_split
    results["max_samples"] = args.max_samples
    results["batch_size"] = args.batch_size
    results["max_seq_length"] = args.max_seq_length
    results["checkpoint"] = args.checkpoint
    results["config"] = args.config
    results["evaluation_time"] = time.time() - start_time
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {args.output}")
    logger.info(f"Evaluation completed in {results['evaluation_time']:.2f} seconds")
    logger.info(f"Perplexity: {results['perplexity']:.4f}")


if __name__ == "__main__":
    main()