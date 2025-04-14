#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Entry point script for training a foundation language model.

This script is a simple wrapper around the main training functionality
that handles command line argument parsing and configuration loading.
"""

import argparse
import logging
import os
import sys

# Add parent directory to path to ensure imports work correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# These imports are intentionally placed here after modifying the path
# flake8: noqa: E402
from llm.training.train import TrainerConfig, run_training

# Configure logger
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the training script."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train a foundation language model.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Override max_steps in config"
    )
    parser.add_argument(
        "--extra-index-url", type=str, default=None, 
        help="Extra PyPI index URL for PyTorch nightly builds"
    )
    args = parser.parse_args()

    # Handle PyTorch nightly builds if specified
    if args.extra_index_url:
        logger.info(f"Using extra index URL for PyTorch: {args.extra_index_url}")
        # This is just informational as the packages should already be installed
        # by the train.sh script, but we can verify PyTorch is available
        try:
            import torch
            logger.info(f"Using PyTorch version: {torch.__version__}")
            if hasattr(torch.cuda, 'is_available') and torch.cuda.is_available():
                logger.info(f"CUDA available: {torch.cuda.is_available()}")
                logger.info(f"CUDA version: {torch.version.cuda}")
                logger.info(f"GPU device count: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        except ImportError:
            logger.warning("PyTorch not found. Please install it using the specified index URL.")
            sys.exit(1)

    # Load configuration
    config = TrainerConfig.from_yaml(args.config)

    # Override max_steps if provided via command line
    if args.max_steps is not None:
        logger.info(f"Overriding max_steps from config with command line value: {args.max_steps}")
        config.train_config["max_steps"] = args.max_steps
        # Ensure num_train_epochs is not used if max_steps is set
        if "num_train_epochs" in config.train_config:
            del config.train_config["num_train_epochs"]

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
        logging.info("Training interrupted by user")
    except Exception as e:
        logging.error(f"Error during training: {e}")
        import traceback

        traceback.print_exc()
        raise
