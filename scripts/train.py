#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Entry point script for training a foundation language model.

This script is a simple wrapper around the main training functionality
that handles command line argument parsing and configuration loading.
"""

import argparse
import logging

from llm.training.train import TrainerConfig, run_training


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
    args = parser.parse_args()

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
