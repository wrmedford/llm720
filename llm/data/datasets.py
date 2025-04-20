#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset handling and preprocessing for training language models.

This module handles loading, preprocessing, and interleaving datasets
for language model training.
"""

import logging
import os
import logging
import os
from typing import Dict, List

import tiktoken # Import tiktoken
from datasets import IterableDataset, interleave_datasets, load_dataset
# from transformers import PreTrainedTokenizerBase # No longer needed


def interleave_datasets_with_weights(
    datasets: List[IterableDataset], weights: List[float]
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
    return interleave_datasets(
        datasets, probabilities=weights, stopping_strategy="all_exhausted"
    )


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
                    streaming=dataset_info.get("streaming", True),
                )
            else:
                # HuggingFace dataset
                logger.info(f"Loading HuggingFace dataset {dataset_path} (subset: {dataset_info.get('subset')})")
                dataset = load_dataset(
                    dataset_path,
                    name=dataset_info.get("subset"), # Pass the subset as the 'name' argument
                    split=dataset_info.get("split", "train"),
                    streaming=dataset_info.get("streaming", True),
                )

            # Map to standardize the text field name
            text_field = dataset_info.get("text_field", "text")

            # First, check the dataset schema to see what fields are available
            try:
                # Get a sample to examine the schema
                sample = next(iter(dataset.take(1)))
                available_fields = list(sample.keys())
                
                # Log available fields for debugging
                logger.info(f"Dataset {dataset_info['name']} has fields: {available_fields}")
                
                # Verify if the specified text field exists
                if text_field not in available_fields:
                    # Try to find alternative text fields
                    text_field_candidates = ["text", "content", "document", "passage", "context", "input", "instruction", "query", "question", "code"]
                    
                    # Find the first matching field
                    alternative_field = None
                    for field in text_field_candidates:
                        if field in available_fields:
                            alternative_field = field
                            break
                    
                    # If no standard field found, look for string fields with substantial content
                    if alternative_field is None:
                        for field, value in sample.items():
                            if isinstance(value, str) and len(value) > 100:  # More conservative threshold
                                alternative_field = field
                                break
                    
                    if alternative_field:
                        logger.warning(
                            f"Specified text field '{text_field}' not found in dataset {dataset_info['name']}. "
                            f"Using '{alternative_field}' instead. Please update your configuration."
                        )
                        text_field = alternative_field
                    else:
                        logger.error(
                            f"Could not find suitable text field in dataset {dataset_info['name']}. "
                            f"Available fields: {available_fields}. Please specify the correct text_field in your configuration."
                        )
                        # Skip this dataset instead of using empty strings
                        continue
            except Exception as e:
                logger.warning(f"Error examining dataset schema for {dataset_info['name']}: {e}")
                # Continue with the specified text_field and hope for the best

            def map_fn(example):
                # Prioritize the specified text_field
                if text_field in example and isinstance(example[text_field], str):
                    return {"text": example[text_field]}

                # Fallback logic (simplified) - find the first non-empty string field
                for k, v in example.items():
                    if isinstance(v, str) and v.strip():
                        if k != text_field: # Log only if using a fallback field
                             logger.warning(
                                 f"Specified text field '{text_field}' not found or invalid. Using fallback field '{k}' for dataset {dataset_info['name']}."
                             )
                        return {"text": v}
                else:
                    # If no suitable field found, log error and return empty text
                    # This will create essentially empty examples that should be filtered out
                    logger.error(
                        f"No suitable text content found in example from dataset {dataset_info['name']}. "
                        f"Fields: {list(example.keys())}"
                    )
                    # If no suitable string field found, return empty string for filtering
                    return {"text": ""}

            # Map to standardize text field
            dataset = dataset.map(map_fn)

            # Filter out examples with empty or whitespace-only text
            def filter_empty(example):
                # Check if 'text' exists and is a string before stripping
                text_content = example.get("text")
                return isinstance(text_content, str) and len(text_content.strip()) > 0

            filtered_dataset = dataset.filter(filter_empty)

            # Log dataset readiness (logging counts might be difficult/slow with streaming)
            try:
                # This might not work for all streaming datasets
                logger.info(f"Dataset {dataset_info['name']} ready for training")
            except Exception:
                pass  # Skip count logging for streaming datasets
            
            datasets.append(filtered_dataset)
            weights.append(dataset_info.get("weight", 1.0))

            logger.info(
                f"Added dataset {dataset_info['name']} with weight "
                f"{dataset_info.get('weight', 1.0)}"
            )

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
    logger.info(f"Interleaving {len(datasets)} datasets. Weights: {weights}")
    return interleave_datasets_with_weights(datasets, weights)


def tokenize_function(
    examples, tokenizer: tiktoken.Encoding, max_seq_length: int, pad_token_id: int
):
    """
    Tokenize function for dataset processing using tiktoken.

    Args:
        examples: The examples to tokenize (dictionary with 'text' key)
        tokenizer: The tiktoken encoder instance
        max_seq_length: Maximum sequence length
        pad_token_id: The token ID to use for padding

    Returns:
        Tokenized examples with 'input_ids' and 'attention_mask'
    """
    result = {"input_ids": [], "attention_mask": []}
    texts = examples["text"]

    # Ensure texts is a list
    if isinstance(texts, str):
        texts = [texts]

    for text in texts:
        if not isinstance(text, str):
            # Handle potential non-string data (e.g., None)
            text = "" # Replace with empty string

        # Encode using tiktoken, allow endoftext special token
        # Note: tiktoken doesn't handle truncation directly in encode
        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        # Truncate
        truncated_ids = token_ids[:max_seq_length]

        # Pad
        padding_length = max_seq_length - len(truncated_ids)
        padded_ids = truncated_ids + [pad_token_id] * padding_length

        # Create attention mask
        attention_mask = [1] * len(truncated_ids) + [0] * padding_length

        result["input_ids"].append(padded_ids)
        result["attention_mask"].append(attention_mask)

    return result
