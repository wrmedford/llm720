#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset handling and preprocessing for training language models.

This module handles loading, preprocessing, and interleaving datasets
for language model training.
"""

import logging
import os
from typing import Dict, List

from datasets import IterableDataset, interleave_datasets, load_dataset
from transformers import PreTrainedTokenizerBase


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
                logger.info(f"Loading HuggingFace dataset {dataset_path}")
                dataset = load_dataset(
                    dataset_path,
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
                # Check if text field exists
                if text_field in example:
                    text_content = example[text_field]
                    # Verify the content is a string and not empty
                    if isinstance(text_content, str) and text_content.strip():
                        return {"text": text_content}
                    else:
                        logger.warning(f"Text field '{text_field}' exists but contains non-string or empty content in dataset {dataset_info['name']}.")
                
                # If we get here, either the field doesn't exist or has invalid content
                # Try to find any suitable text field as a fallback
                potential_fields = [
                    (k, v) for k, v in example.items() 
                    if isinstance(v, str) and len(v.strip()) > 50
                ]
                
                if potential_fields:
                    # Sort by length to get the field with the most content
                    potential_fields.sort(key=lambda x: len(x[1]), reverse=True)
                    chosen_field, content = potential_fields[0]
                    logger.warning(
                        f"Using fallback field '{chosen_field}' for dataset {dataset_info['name']}. "
                        f"Please update your configuration."
                    )
                    return {"text": content}
                else:
                    # If no suitable field found, log error and return empty text
                    # This will create essentially empty examples that should be filtered out
                    logger.error(
                        f"No suitable text content found in example from dataset {dataset_info['name']}. "
                        f"Fields: {list(example.keys())}"
                    )
                    return {"text": ""}

            # Map to standardize text field and then filter out empty examples
            dataset = dataset.map(map_fn)
            
            # Filter out empty examples
            def filter_empty(example):
                return example["text"] != "" and len(example["text"].strip()) > 0
            
            filtered_dataset = dataset.filter(filter_empty)
            
            # Log how many examples were filtered out (if possible)
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
    examples, tokenizer: PreTrainedTokenizerBase, max_seq_length: int
):
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

    for input_ids, attention_mask in zip(
        tokenized["input_ids"], tokenized["attention_mask"]
    ):
        # If shorter than max_length, pad
        if len(input_ids) < max_seq_length:
            padding_length = max_seq_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length

        result["input_ids"].append(input_ids)
        result["attention_mask"].append(attention_mask)

    return result
