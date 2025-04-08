#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration handling utilities for the LLM package.

This module provides functions for loading and saving configuration
from YAML files, with support for overriding default values.
"""

import os
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a configuration from a YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Dictionary containing the configuration
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def save_config(config: Dict[str, Any], filepath: str) -> None:
    """
    Save a configuration to a YAML file.

    Args:
        config: Configuration dictionary
        filepath: Path where to save the configuration
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
