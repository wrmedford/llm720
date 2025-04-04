#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset handling and preprocessing for training language models.

This module handles loading, preprocessing, and interleaving datasets
for language model training.
"""

import os
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, IterableDataset, interleave_datasets
from transformers import PreTrainedTokenizerBase


def interleave_datasets_with_weights(
    datasets: List[IterableDataset], 
    weights: List[float]
) -> IterableDataset:
    """
    Custom wrapper to interleave datasets with specified weights.
    
    Args:
        datasets: List of HuggingFace IterableDataset
        weights: List of weights for sampling from each dataset
    
    Returns:
        Interleaved dataset
    """
    # Normalize weights
    weights = [w / sum(weights) for w in weights]
    
    # Use the built-in interleave_datasets with sampling probabilities
    return interleave_datasets(datasets, probabilities=weights, stopping_strategy="all_exhausted")


def prepare_datasets(config: Dict) -> IterableDataset:
    """
    Prepare and interleave datasets based on configuration.
    
    Args:
        config: Dataset configuration dictionary
        
    Returns:
        Interleaved dataset
    """
    logger = logging.getLogger(__name__)
    datasets = []
    weights = []
    
    for dataset_info in config["datasets"]:
        # Load the dataset
        try:
            dataset_path = dataset_info["path"]
            
            # Check if it's a local path or HuggingFace dataset
            if os.path.exists(dataset_path) or dataset_path.startswith("/"):
                # Local dataset
                logger.info(f"Loading local dataset from {dataset_path}")
                
                # Check file format (assume jsonl if not specified)
                file_format = dataset_info.get("format", "json")
                
                # Load dataset using datasets library
                dataset = load_dataset(
                    file_format,
                    data_files=dataset_path,
                    split=dataset_info.get("split", "train"),
                    streaming=dataset_info.get("streaming", True)
                )
            else:
                # HuggingFace dataset
                logger.info(f"Loading HuggingFace dataset {dataset_path}")
                dataset = load_dataset(
                    dataset_path,
                    split=dataset_info.get("split", "train"),
                    streaming=dataset_info.get("streaming", True)
                )
            
            # Map to standardize the text field name
            text_field = dataset_info.get("text_field", "text")
            
            def map_fn(example):
                # Check if text field exists
                if text_field in example:
                    return {"text": example[text_field]}
                
                # If text field doesn't exist, try to find a suitable text field
                for key in example:
                    if isinstance(example[key], str) and len(example[key]) > 0:
                        return {"text": example[key]}
                
                # If no suitable field found, return empty text
                return {"text": ""}
            
            dataset = dataset.map(map_fn)
            
            datasets.append(dataset)
            weights.append(dataset_info.get("weight", 1.0))
            
            logger.info(f"Added dataset {dataset_info['name']} with weight {dataset_info.get('weight', 1.0)}")
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_info['name']}: {e}")
            logger.error(f"Skipping dataset {dataset_info['name']}")
            continue
    
    if not datasets:
        raise ValueError("No valid datasets found in configuration")
    
    # Normalize weights if any are provided
    if any(w != 1.0 for w in weights):
        weights = [w / sum(weights) for w in weights]
    
    # Interleave datasets
    logger.info(f"Interleaving {len(datasets)} datasets with weights {weights}")
    return interleave_datasets_with_weights(datasets, weights)


def tokenize_function(examples, tokenizer: PreTrainedTokenizerBase, max_seq_length: int):
    """
    Tokenize function for dataset processing.
    
    Args:
        examples: The examples to tokenize
        tokenizer: The tokenizer to use
        max_seq_length: Maximum sequence length
        
    Returns:
        Tokenized examples
    """
    # Tokenize the texts
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_seq_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    
    # Create input_ids of exactly max_seq_length size for each example
    result = {"input_ids": [], "attention_mask": []}
    
    for input_ids, attention_mask in zip(tokenized["input_ids"], tokenized["attention_mask"]):
        # If shorter than max_length, pad
        if len(input_ids) < max_seq_length:
            padding_length = max_seq_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        
        result["input_ids"].append(input_ids)
        result["attention_mask"].append(attention_mask)
    
    return result