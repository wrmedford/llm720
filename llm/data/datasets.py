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
from typing import Dict, List, Any

import tiktoken # Import tiktoken
from datasets import IterableDataset, interleave_datasets, load_dataset
# from transformers import PreTrainedTokenizerBase # No longer needed


# --- Preprocessing Functions ---

def format_reasoning_steps(data: Any) -> str:
    """
    Example processor: Concatenates 'question' and 'answer' fields
    if data is a dictionary, otherwise returns data as string.
    Handles potential nested structures or simple strings.
    """
    if isinstance(data, dict):
        question = data.get("question", "")
        answer = data.get("answer", "")
        # Simple concatenation example
        return f"Question: {question}\nAnswer: {answer}"
    elif isinstance(data, str):
        # If the field already contains a string, return it directly
        return data
    else:
        # Handle other unexpected types gracefully
        return str(data) if data is not None else ""

PREPROCESSING_FUNCTIONS = {
    "format_reasoning": format_reasoning_steps,
    # Add more functions here as needed, e.g.:
    # "extract_code_blocks": extract_code_blocks_function,
}

# --- End Preprocessing Functions ---


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
        dataset_name = dataset_info.get("name", dataset_info.get("path", "unknown")) # Get a name for logging
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

            # Determine how to extract text: template, specific field, or auto-detect
            text_template = dataset_info.get("text_template")
            text_field = dataset_info.get("text_field")
            extraction_mode = "auto" # Default
            final_text_field = None # Field used if mode is 'field' or 'auto'

            # Get available fields from a sample
            available_fields = []
            try:
                sample = next(iter(dataset.take(1)))
                available_fields = list(sample.keys())
                logger.info(f"Dataset {dataset_info['name']} - Available fields: {available_fields}")
            except StopIteration:
                logger.warning(f"Dataset {dataset_info['name']} appears to be empty. Skipping.")
                continue
            except Exception as e:
                logger.warning(f"Could not get sample from dataset {dataset_name}: {e}. Proceeding with caution.")
                # Assume standard fields might exist if we can't get a sample

            # --- Determine Text Extraction/Processing Method ---
            preprocessing_function_name = dataset_info.get("preprocessing_function_name")
            text_template = dataset_info.get("text_template")
            text_field = dataset_info.get("text_field")
            processed_via_map = False # Flag to track if dataset was processed by a map function

            if preprocessing_function_name:
                if not text_field:
                    logger.error(
                        f"Dataset {dataset_name} specifies 'preprocessing_function_name' "
                        f"but not 'text_field'. Preprocessing requires a source field. Skipping dataset."
                    )
                    continue
                if text_field not in available_fields:
                    logger.error(
                        f"Specified text_field '{text_field}' for preprocessing not found in dataset {dataset_name}. "
                        f"Available fields: {available_fields}. Skipping dataset."
                    )
                    continue
                if preprocessing_function_name not in PREPROCESSING_FUNCTIONS:
                    logger.error(
                        f"Unknown preprocessing_function_name '{preprocessing_function_name}' for dataset {dataset_name}. "
                        f"Available functions: {list(PREPROCESSING_FUNCTIONS.keys())}. Skipping dataset."
                    )
                    continue

                logger.info(f"Applying preprocessing function '{preprocessing_function_name}' to field '{text_field}' for dataset {dataset_name}")
                processor = PREPROCESSING_FUNCTIONS[preprocessing_function_name]

                def preprocess_map_fn(example):
                    try:
                        source_data = example.get(text_field)
                        # Pass the entire example to the processor? Or just the field?
                        # Let's assume processor takes just the field content for now.
                        if source_data is None:
                            logger.warning(f"Field '{text_field}' is None in an example from {dataset_name}. Skipping preprocessing for this example.")
                            return {"text": ""} # Mark for filtering
                        processed_text = processor(source_data) # Apply the function
                        # Ensure output is string
                        return {"text": str(processed_text) if processed_text is not None else ""}
                    except Exception as e:
                        logger.error(f"Error applying preprocessing function '{preprocessing_function_name}' to example in {dataset_name}: {e}")
                        return {"text": ""} # Mark for filtering

                # Apply the preprocessing map function
                dataset = dataset.map(preprocess_map_fn, batched=False) # Process individually for robustness
                processed_via_map = True
                # The 'text' field is now ready, skip further template/field extraction logic

            elif text_template:
                # Validate template fields exist in the sample (if sample available)
                if sample: # Only validate if sample is available
                    try:
                        # Basic check: does the template string format with the sample?
                        _ = text_template.format(**sample)
                        logger.info(f"Using text_template for dataset {dataset_name}")
                    except KeyError as e:
                        logger.error(
                            f"Field {e} required by text_template not found in dataset {dataset_name}. "
                            f"Available fields: {available_fields}. Skipping dataset."
                        )
                        continue
                    except Exception as e:
                        logger.warning(
                            f"Error validating text_template for dataset {dataset_name}: {e}. Proceeding with caution."
                        )
                else:
                     logger.info(f"Using text_template for dataset {dataset_name} (sample unavailable for validation).")

                def template_map_fn(example):
                    try:
                        text = text_template.format(**example)
                        return {"text": str(text) if text is not None else ""}
                    except KeyError as e:
                        logger.warning(f"Missing field {e} in example from {dataset_name} when applying template. Skipping example.")
                        return {"text": ""}
                    except Exception as e:
                        logger.error(f"Error applying template to example in {dataset_name}: {e}")
                        return {"text": ""}
                dataset = dataset.map(template_map_fn, batched=False)
                processed_via_map = True

            elif text_field:
                if text_field in available_fields:
                    logger.info(f"Using specified text_field '{text_field}' for dataset {dataset_name}")
                    # Check if the field is already 'text', if so, no mapping needed unless type conversion is required
                    if text_field == "text" and sample and isinstance(sample.get("text"), str):
                         logger.info(f"Field '{text_field}' is already the target 'text' field and is a string. No mapping applied.")
                    else:
                        # Map to rename/convert field to 'text'
                        def field_map_fn(example):
                            text = example.get(text_field)
                            return {"text": str(text) if text is not None else ""}
                        dataset = dataset.map(field_map_fn, batched=False)
                        processed_via_map = True
                else:
                    logger.error(
                        f"Specified text_field '{text_field}' not found in dataset {dataset_name}. "
                        f"Available fields: {available_fields}. Skipping dataset."
                    )
                    continue
            else:
                # Auto-detect: Try common field names
                logger.info(f"No preprocessing, template, or text_field specified for {dataset_name}. Attempting auto-detection.")
                detected_field = None
                common_fields = ["text", "content", "document", "passage", "code", "instruction", "input", "output", "prompt", "completion"]
                if sample: # Only auto-detect if sample is available
                    for field in common_fields:
                        if field in available_fields:
                            # Check if the field contains string data in the sample
                            if isinstance(sample.get(field), str):
                                detected_field = field
                                logger.warning(
                                    f"Auto-detected text field '{detected_field}' for dataset {dataset_name}. "
                                    "Consider specifying 'text_field' or 'text_template' in config for clarity."
                                )
                                break
                            else:
                                logger.warning(f"Field '{field}' found but is not a string in sample. Skipping this candidate.")

                if detected_field:
                    # Map if detected field is not 'text'
                    if detected_field != "text":
                        def auto_detect_map_fn(example):
                            text = example.get(detected_field)
                            return {"text": str(text) if text is not None else ""}
                        dataset = dataset.map(auto_detect_map_fn, batched=False)
                        processed_via_map = True
                else:
                    logger.error(
                        f"Could not auto-detect a suitable text field for dataset {dataset_name}. "
                        f"Available fields: {available_fields}. Please specify 'text_field' or 'text_template'. Skipping dataset."
                    )
                    continue

            # --- Common Post-processing ---
            # Filter out examples with empty or whitespace-only text from the 'text' field
            def filter_empty(example):
                # Check if 'text' exists and is a string before stripping
                text_content = example.get("text")
                return isinstance(text_content, str) and len(text_content.strip()) > 0

            filtered_dataset = dataset.filter(filter_empty)

            # Log dataset readiness
            logger.info(f"Dataset {dataset_name} processed and ready for interleaving.")

            datasets.append(filtered_dataset)
            weights.append(dataset_info.get("weight", 1.0))

            logger.info(
                f"Added dataset {dataset_name} with weight "
                f"{dataset_info.get('weight', 1.0)}"
            )

        except Exception as e:
            logger.exception(f"Error processing dataset {dataset_name}: {e}") # Use logger.exception to include traceback
            logger.error(f"Skipping dataset {dataset_name}")
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
