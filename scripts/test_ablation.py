#!/usr/bin/env python3
"""Test script for the updated ablation framework."""

import os
import sys
import yaml
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.training.ablation import (
    ABLATION_CONFIGS,
    generate_experiment_combinations,
    generate_config,
    create_experiment_folder,
    calculate_expert_hidden_size,
    calculate_total_params,
    calculate_active_params,
    EXPERT_SIZE_BYTES,
    STANDARD_VOCAB_SIZE
)

def test_ablation_configs():
    """Test that all ablation configurations are valid."""
    print("=== Testing Ablation Configurations ===\n")
    
    for ablation_type, experiments in ABLATION_CONFIGS.items():
        print(f"{ablation_type}: {len(experiments)} experiments")
        for exp in experiments:
            print(f"  - {exp['name']}")
    
    print(f"\nTotal experiments: {sum(len(exps) for exps in ABLATION_CONFIGS.values())}")


def test_config_generation():
    """Test configuration generation for each ablation type."""
    print("\n=== Testing Configuration Generation ===\n")
    
    # Load base config
    base_config_path = "configs/config.yaml"
    if not os.path.exists(base_config_path):
        print(f"Error: Base config not found at {base_config_path}")
        return
    
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)
    
    # Create temporary experiment directory
    experiment_dir = create_experiment_folder("./test_ablation_output")
    
    # Test one experiment from each ablation type
    test_experiments = [
        ("capacity_sweep", ABLATION_CONFIGS["capacity_sweep"][0]),
        ("granularity_sweep", ABLATION_CONFIGS["granularity_sweep"][0]),
        ("compute_tradeoff", ABLATION_CONFIGS["compute_tradeoff"][0]),
        ("activation_4b", ABLATION_CONFIGS["activation_4b"][0]),
        ("activation_256b", ABLATION_CONFIGS["activation_256b"][0]),
    ]
    
    for ablation_type, experiment in test_experiments:
        print(f"\nTesting {ablation_type} - {experiment['name']}:")
        
        try:
            config_path, exp_name = generate_config(
                base_config, experiment, experiment_dir, ablation_type
            )
            
            # Load generated config and analyze
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            
            # Calculate parameters
            total_params = calculate_total_params(config)
            active_params = calculate_active_params(config)
            
            print(f"  Config generated: {config_path}")
            print(f"  Model size: {config['model_config']['hidden_size']}h x {config['model_config']['num_hidden_layers']}L")
            print(f"  Total params: {total_params/1e9:.2f}B")
            print(f"  Active params: {active_params/1e6:.2f}M")
            print(f"  Active ratio: {active_params/total_params:.4f}")
            
            if config["model_config"]["use_peer"] and config["model_config"]["peer_start_layer"] < config["model_config"]["num_hidden_layers"]:
                peer_config = config["model_config"]["peer_config"]
                print(f"  Experts: {peer_config['num_experts']:,}")
                print(f"  Experts per token: {peer_config['num_experts_per_tok']}")
                print(f"  Expert hidden size: {peer_config['expert_hidden_size']}")
                
                # Calculate actual expert size
                expert_size = (config['model_config']['hidden_size'] * peer_config['expert_hidden_size'] + 
                              peer_config['expert_hidden_size']) * 1  # FP8 = 1 byte
                print(f"  Expert size: {expert_size/1024:.2f}KB (target: {EXPERT_SIZE_BYTES/1024}KB)")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()


def test_parameter_calculations():
    """Test parameter calculation functions."""
    print("\n=== Testing Parameter Calculations ===\n")
    
    # Test expert_hidden_size calculation
    test_cases = [
        (768, 56 * 1024),   # 768 hidden, 56KB target
        (1024, 56 * 1024),  # 1024 hidden, 56KB target
        (2048, 56 * 1024),  # 2048 hidden, 56KB target
    ]
    
    for hidden_size, target_bytes in test_cases:
        expert_hidden = calculate_expert_hidden_size(target_bytes, hidden_size, fp8=True)
        actual_size = (hidden_size * expert_hidden + expert_hidden) * 1
        print(f"Hidden={hidden_size}, Target={target_bytes/1024}KB:")
        print(f"  Expert hidden size: {expert_hidden}")
        print(f"  Actual size: {actual_size/1024:.2f}KB")
        print(f"  Error: {abs(actual_size - target_bytes)/target_bytes * 100:.1f}%")


def test_experiment_combinations():
    """Test experiment combination generation."""
    print("\n=== Testing Experiment Combinations ===\n")
    
    # Test all combinations
    all_combos = generate_experiment_combinations()
    print(f"Total combinations (all types): {len(all_combos)}")
    
    # Test specific types
    for ablation_type in ABLATION_CONFIGS.keys():
        combos = generate_experiment_combinations([ablation_type])
        print(f"{ablation_type}: {len(combos)} experiments")
    
    # Print first few combinations
    print("\nFirst 5 experiment combinations:")
    for i, (ablation_type, exp) in enumerate(all_combos[:5]):
        print(f"  {i+1}. {ablation_type} - {exp['name']}")


def main():
    """Run all tests."""
    print("Testing Updated Ablation Script\n")
    print("=" * 50)
    
    test_ablation_configs()
    test_parameter_calculations()
    test_experiment_combinations()
    test_config_generation()
    
    print("\n" + "=" * 50)
    print("Testing complete!")


if __name__ == "__main__":
    main()