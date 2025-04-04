#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Meta-script for running ablation studies on the foundation model architecture.

This script systematically executes training runs with different configurations,
evaluates performance, and collects results. It tests combinations across four axes:
1. Initial data mix
2. Number and size of experts
3. Dimensionality of cartesian product for expert selection
4. Number of expert selection heads

Usage:
    python run_ablations.py --base-config configs/config.yaml --output-dir ablation_results
    
    # Run a specific subset of experiments
    python run_ablations.py --base-config configs/config.yaml --output-dir ablation_results \
                           --axes data_mix expert_count
                           
    # Resume from previous run
    python run_ablations.py --resume --output-dir ablation_results
"""

import argparse
import logging
import sys
import os

# Add parent directory to path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.training.ablation import run_ablation_study


def main():
    """Main entry point for the ablation script."""
    parser = argparse.ArgumentParser(description="Run ablation studies for PEER architecture")
    parser.add_argument("--base-config", type=str, default="configs/config.yaml", 
                      help="Path to base configuration file")
    parser.add_argument("--output-dir", type=str, default="./ablation_results",
                      help="Directory to store results")
    parser.add_argument("--axes", type=str, nargs="+",
                      choices=["data_mix", "expert_count", "cartesian_dims", "selection_heads"],
                      help="Specific axes to ablate (default: all)")
    parser.add_argument("--gpus", type=int, default=4,
                      help="Number of GPUs to use for each experiment")
    parser.add_argument("--tokens", type=int, default=10_000_000_000,
                      help="Token limit for each training run (default: 10B)")
    parser.add_argument("--resume", action="store_true",
                      help="Resume from a previous ablation study")
    parser.add_argument("--custom-datasets", type=str, default=None,
                      help="Path to custom dataset configuration file")
    
    args = parser.parse_args()
    
    # Run ablation study
    run_ablation_study(
        base_config_path=args.base_config,
        output_dir=args.output_dir,
        selected_axes=args.axes,
        gpu_count=args.gpus,
        tokens_limit=args.tokens,
        resume=args.resume,
        custom_datasets_path=args.custom_datasets
    )


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
        logging.info("Ablation study interrupted by user")
    except Exception as e:
        logging.error(f"Error during ablation study: {e}")
        import traceback
        traceback.print_exc()
        raise