#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model Parameter Analyzer for Foundation Models with PEER and MLA

This script calculates and displays the total parameters and active parameters per token
for the foundation model architecture, helping understand the efficiency gains from PEER.
"""

import argparse
import logging
import os
import sys

# Add parent directory to path to ensure imports work correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# This import is intentionally placed here after modifying the path
# flake8: noqa: E402
from llm.utils.eval.size import main as size_main


def main():
    """Main entry point for the model analysis script."""
    parser = argparse.ArgumentParser(
        description="Analyze model parameters for PEER architecture."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--checkpoint", type=str, help="Path to model checkpoint (optional)"
    )
    parser.add_argument(
        "--output", type=str, help="Path to save visualization (optional)"
    )
    parser.add_argument("--json", type=str, help="Path to save JSON results (optional)")
    args = parser.parse_args()

    # Call the size module's main function
    size_main(args)


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
        logging.info("Analysis interrupted by user")
    except Exception as e:
        logging.error(f"Error during analysis: {e}")
        import traceback

        traceback.print_exc()
        raise
