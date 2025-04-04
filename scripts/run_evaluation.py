#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Entry point script for evaluating a trained foundation model.

This script provides an easy way to evaluate trained models on
various benchmarks and perplexity measurements.
"""

import argparse
import logging
import sys
import os

# Add parent directory to path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.utils.eval.benchmark import main as benchmark_main
from llm.utils.eval.perplexity import main as perplexity_main


def main():
    """Main entry point for the evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate a trained foundation model.")
    subparsers = parser.add_subparsers(dest="command", help="Evaluation command")
    
    # Benchmark evaluation subparser
    benchmark_parser = subparsers.add_parser("benchmark", help="Evaluate on benchmarks")
    benchmark_parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    benchmark_parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    benchmark_parser.add_argument("--benchmark", type=str, required=True, help="Benchmark to evaluate")
    benchmark_parser.add_argument("--subset", type=str, default=None, help="Benchmark subset")
    benchmark_parser.add_argument("--metric", type=str, default=None, help="Metric to use")
    benchmark_parser.add_argument("--output", type=str, default=None, help="Path to save results")
    benchmark_parser.add_argument("--device", type=str, default=None, help="Device to use (cpu, cuda, cuda:0, etc.)")
    
    # Perplexity evaluation subparser
    perplexity_parser = subparsers.add_parser("perplexity", help="Evaluate perplexity")
    perplexity_parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    perplexity_parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    perplexity_parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset to evaluate on")
    perplexity_parser.add_argument("--dataset_split", type=str, default="validation", help="Dataset split")
    perplexity_parser.add_argument("--text_column", type=str, default="text", help="Text column name")
    perplexity_parser.add_argument("--max_samples", type=int, default=1000, help="Maximum number of samples")
    perplexity_parser.add_argument("--output", type=str, default=None, help="Path to save results")
    perplexity_parser.add_argument("--device", type=str, default=None, help="Device to use (cpu, cuda, cuda:0, etc.)")
    
    args = parser.parse_args()
    
    if args.command == "benchmark":
        # Prepare arguments for benchmark evaluation
        benchmark_args = argparse.Namespace(
            config=args.config,
            checkpoint=args.checkpoint,
            benchmark=args.benchmark,
            subset=args.subset,
            metric=args.metric,
            output=args.output,
            device=args.device
        )
        benchmark_main(benchmark_args)
    
    elif args.command == "perplexity":
        # Prepare arguments for perplexity evaluation
        perplexity_args = argparse.Namespace(
            config=args.config,
            checkpoint=args.checkpoint,
            dataset=args.dataset,
            dataset_split=args.dataset_split,
            text_column=args.text_column,
            max_samples=args.max_samples,
            output=args.output,
            device=args.device
        )
        perplexity_main(perplexity_args)
    
    else:
        parser.print_help()


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
        logging.info("Evaluation interrupted by user")
    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        raise