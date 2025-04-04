#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Benchmark evaluation script for foundation models

This script evaluates trained models on various benchmarks including:
- AIME 2024 (Math competition)
- Codeforces (Coding challenges)
- GPQA Diamond (Scientific reasoning)
- MATH-500 (Math problems)
- MMLU (General knowledge)
- SWE-bench (Software engineering)

Usage:
    python evaluate_benchmark.py --config config.yaml \
                                --checkpoint path/to/checkpoint.safetensors \
                                --benchmark mmlu \
                                --metric pass@1 \
                                --output results.json
"""

import os
import json
import yaml
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import torch
import numpy as np
from tqdm.auto import tqdm
from safetensors.torch import load_file

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# Benchmark configurations
BENCHMARKS = {
    "aime_2024": {
        "display_name": "AIME 2024",
        "default_metric": "pass@1",
        "available_metrics": ["pass@1", "cons@64"],
        "default_subset": None,
        "available_subsets": None,
        "description": "American Invitational Mathematics Examination 2024"
    },
    "codeforces": {
        "display_name": "Codeforces",
        "default_metric": "rating",
        "available_metrics": ["percentile", "rating"],
        "default_subset": None,
        "available_subsets": None,
        "description": "Codeforces programming competition problems"
    },
    "gpqa": {
        "display_name": "GPQA",
        "default_metric": "pass@1",
        "available_metrics": ["pass@1"],
        "default_subset": "diamond",
        "available_subsets": ["diamond", "full"],
        "description": "Graduate-level physics and quantitative reasoning benchmark"
    },
    "math": {
        "display_name": "MATH",
        "default_metric": "pass@1",
        "available_metrics": ["pass@1"],
        "default_subset": "500",
        "available_subsets": ["500", "full"],
        "description": "MATH dataset with challenging math problems"
    },
    "mmlu": {
        "display_name": "MMLU",
        "default_metric": "pass@1",
        "available_metrics": ["pass@1"],
        "default_subset": None,
        "available_subsets": None,
        "description": "Massive Multitask Language Understanding benchmark"
    },
    "swe_bench": {
        "display_name": "SWE-bench",
        "default_metric": "resolved",
        "available_metrics": ["resolved", "success@1"],
        "default_subset": "verified",
        "available_subsets": ["verified", "full"],
        "description": "Software engineering tasks benchmark"
    }
}


def load_model_and_tokenizer(config_path: str, checkpoint_path: str):
    """
    Load model and tokenizer from configuration and checkpoint.
    
    Args:
        config_path: Path to configuration file
        checkpoint_path: Path to model checkpoint
        
    Returns:
        Tuple of (model, tokenizer)
    """
    import transformers
    from transformers import AutoTokenizer
    
    # Import necessary module from train_lm.py
    import sys
    sys.path.append(".")
    from src.train import (
        TransformerConfig, 
        create_model_from_config,
    )
    
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model config
    model_config = TransformerConfig(**config["model_config"])
    
    # Create model
    logger.info("Creating model from configuration")
    model = create_model_from_config(model_config)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(config["dataset_config"]["tokenizer_name"])
    
    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load checkpoint weights
    logger.info(f"Loading weights from {checkpoint_path}")
    if checkpoint_path.endswith(".safetensors"):
        state_dict = load_file(checkpoint_path)
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    
    return model, tokenizer


def load_benchmark_data(benchmark: str, subset: Optional[str] = None) -> Dict:
    """
    Load benchmark data based on benchmark name and subset.
    
    Args:
        benchmark: Name of the benchmark
        subset: Optional subset name
        
    Returns:
        Dictionary containing benchmark data
    """
    # Build path to benchmark data
    benchmark_dir = os.path.join("benchmarks", benchmark)
    if subset:
        benchmark_path = os.path.join(benchmark_dir, f"{benchmark}_{subset}.json")
    else:
        benchmark_path = os.path.join(benchmark_dir, f"{benchmark}.json")
    
    # Check if benchmark exists
    if not os.path.exists(benchmark_path):
        # Create directory and download or create stub data
        os.makedirs(os.path.dirname(benchmark_path), exist_ok=True)
        logger.info(f"Benchmark data not found at {benchmark_path}. Downloading...")
        
        try:
            # For demonstration purposes - in a real scenario, you'd implement proper downloading
            # or use benchmark-specific libraries (like lm-eval-harness, EleutherAI/lm-evaluation-harness, etc.)
            download_benchmark_data(benchmark, subset, benchmark_path)
        except Exception as e:
            logger.error(f"Failed to download benchmark data: {e}")
            # Create stub data for testing
            create_stub_benchmark_data(benchmark, subset, benchmark_path)
    
    # Load benchmark data
    logger.info(f"Loading benchmark data from {benchmark_path}")
    with open(benchmark_path, 'r') as f:
        data = json.load(f)
    
    return data


def download_benchmark_data(benchmark: str, subset: Optional[str], output_path: str):
    """
    Download benchmark data from appropriate sources.
    
    Args:
        benchmark: Name of the benchmark
        subset: Optional subset name
        output_path: Path to save the benchmark data
    """
    # In a real implementation, you would download the appropriate benchmark data
    # For example, using the lm-evaluation-harness or other benchmark libraries
    
    # Example implementation using the datasets library for some benchmarks
    import datasets
    
    try:
        if benchmark == "mmlu":
            from datasets import load_dataset
            dataset = load_dataset("cais/mmlu", "all")
            # Process and save the dataset
            questions = []
            for split in ["validation", "test"]:
                for item in dataset[split]:
                    questions.append({
                        "question": item["question"],
                        "choices": item["choices"],
                        "answer": item["answer"],
                        "subject": item["subject"]
                    })
            
            with open(output_path, 'w') as f:
                json.dump({"questions": questions}, f)
                
        elif benchmark == "math" and subset == "500":
            # Create a minimal MATH-500 dataset (in reality, you'd download the actual dataset)
            from datasets import load_dataset
            dataset = load_dataset("hendrycks/math", split="test")
            # Take 500 items from the dataset
            questions = []
            for i, item in enumerate(dataset):
                if i >= 500:
                    break
                questions.append({
                    "problem": item["problem"],
                    "solution": item["solution"],
                    "answer": item["answer"],
                    "level": item["level"],
                    "type": item["type"]
                })
            
            with open(output_path, 'w') as f:
                json.dump({"questions": questions}, f)
        
        # Add more benchmarks as needed
        
        else:
            raise NotImplementedError(f"Download not implemented for {benchmark} {subset}")
    
    except Exception as e:
        logger.error(f"Failed to download benchmark data: {e}")
        raise


def create_stub_benchmark_data(benchmark: str, subset: Optional[str], output_path: str):
    """
    Create stub benchmark data for testing.
    
    Args:
        benchmark: Name of the benchmark
        subset: Optional subset name
        output_path: Path to save the benchmark data
    """
    # Create stub data based on benchmark
    stub_data = {"questions": []}
    
    if benchmark == "aime_2024":
        # Create stub math problems
        for i in range(15):
            stub_data["questions"].append({
                "problem": f"AIME 2024 Problem {i+1}: Find the value of x such that 2x + 3 = 7.",
                "answer": "2"
            })
    
    elif benchmark == "codeforces":
        # Create stub coding problems
        for i in range(10):
            stub_data["questions"].append({
                "problem": f"Problem {i+1}: Write a function to find the sum of two numbers.",
                "test_cases": [{"input": "1 2", "output": "3"}, {"input": "3 4", "output": "7"}]
            })
    
    elif benchmark == "gpqa":
        # Create stub physics problems
        for i in range(10):
            stub_data["questions"].append({
                "question": f"Question {i+1}: What is the formula for kinetic energy?",
                "choices": ["A. E = mc²", "B. E = ½mv²", "C. F = ma", "D. p = mv"],
                "answer": "B"
            })
    
    elif benchmark == "math":
        # Create stub math problems
        for i in range(10):
            stub_data["questions"].append({
                "problem": f"Problem {i+1}: Solve for x: 3x + 2 = 11",
                "solution": "3x + 2 = 11\n3x = 9\nx = 3",
                "answer": "3"
            })
    
    elif benchmark == "mmlu":
        # Create stub MMLU problems
        subjects = ["math", "physics", "computer science", "biology", "history"]
        for i in range(10):
            stub_data["questions"].append({
                "question": f"Question {i+1}: What is the capital of France?",
                "choices": ["A. London", "B. Berlin", "C. Paris", "D. Madrid"],
                "answer": "C",
                "subject": subjects[i % len(subjects)]
            })
    
    elif benchmark == "swe_bench":
        # Create stub software engineering problems
        for i in range(5):
            stub_data["questions"].append({
                "description": f"Bug {i+1}: Fix the bug in the following function that calculates the sum of numbers.",
                "code": "def sum_numbers(numbers):\n    total = 0\n    for num in numbers:\n        total + num\n    return total",
                "correct_solution": "def sum_numbers(numbers):\n    total = 0\n    for num in numbers:\n        total += num\n    return total"
            })
    
    # Save stub data
    with open(output_path, 'w') as f:
        json.dump(stub_data, f)
    
    logger.warning(f"Created stub benchmark data for {benchmark} at {output_path}")


def evaluate_aime(model, tokenizer, benchmark_data, device, metric="pass@1"):
    """
    Evaluate model on AIME 2024 benchmark.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        benchmark_data: The benchmark data
        device: The device to use
        metric: The metric to use (pass@1 or cons@64)
        
    Returns:
        Dictionary of benchmark results
    """
    model.to(device)
    model.eval()
    
    correct = 0
    total = len(benchmark_data["questions"])
    
    # If metric is cons@64, we'll generate multiple responses and use majority voting
    num_samples = 64 if metric == "cons@64" else 1
    
    for question in tqdm(benchmark_data["questions"], desc="Evaluating AIME"):
        problem = question["problem"]
        correct_answer = question["answer"]
        
        prompt = f"Solve the following AIME problem step by step.\n\nProblem: {problem}\n\nShow your work and provide the final answer."
        
        # Track answers for this problem
        problem_answers = []
        
        for _ in range(num_samples):
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    temperature=0.7 if num_samples > 1 else 0.0,
                    top_p=0.95,
                    do_sample=num_samples > 1
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Extract answer - assuming the final answer is preceded by "The answer is" or similar
            answer = None
            for line in response.split("\n"):
                if "answer" in line.lower() and "=" in line:
                    answer = line.split("=")[-1].strip()
                    break
            
            if answer:
                problem_answers.append(answer)
        
        # If using consensus, take the most common answer
        if metric == "cons@64":
            from collections import Counter
            if problem_answers:
                most_common_answer = Counter(problem_answers).most_common(1)[0][0]
                if most_common_answer == correct_answer:
                    correct += 1
        else:
            # For pass@1, just check the first answer
            if problem_answers and problem_answers[0] == correct_answer:
                correct += 1
    
    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0
    
    return {
        metric: accuracy,
        "num_correct": correct,
        "num_total": total
    }


def evaluate_codeforces(model, tokenizer, benchmark_data, device, metric="percentile"):
    """
    Evaluate model on Codeforces benchmark.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        benchmark_data: The benchmark data
        device: The device to use
        metric: The metric to use (percentile or rating)
        
    Returns:
        Dictionary of benchmark results
    """
    model.to(device)
    model.eval()
    
    correct = 0
    total = len(benchmark_data["questions"])
    
    for question in tqdm(benchmark_data["questions"], desc="Evaluating Codeforces"):
        problem = question["problem"]
        test_cases = question["test_cases"]
        
        prompt = f"Write a Python function to solve the following problem:\n\n{problem}\n\nProvide only the code without explanation."
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.2,
                top_p=0.95
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Extract code - assuming the code is the entire response or enclosed in ```python ```
        code = response
        if "```python" in response and "```" in response.split("```python", 1)[1]:
            code = response.split("```python", 1)[1].split("```", 1)[0].strip()
        elif "```" in response:
            code = response.split("```", 1)[1].split("```", 1)[0].strip()
        
        # Evaluate code on test cases
        try:
            # Create a temporary namespace to execute the code
            namespace = {}
            exec(code, namespace)
            
            # For simplicity, assume the main function is called "solution"
            solution_func = None
            for name, func in namespace.items():
                if callable(func) and name != "exec" and not name.startswith("__"):
                    solution_func = func
                    break
            
            if solution_func:
                is_correct = True
                for test_case in test_cases:
                    # Parse input
                    inputs = test_case["input"].split()
                    # Assume the output is a single value for simplicity
                    expected = test_case["output"].strip()
                    
                    # Call function with inputs and check output
                    # This is a simplification - in a real implementation you would parse inputs based on problem specification
                    try:
                        output = str(solution_func(*inputs)).strip()
                        if output != expected:
                            is_correct = False
                            break
                    except Exception:
                        is_correct = False
                        break
                
                if is_correct:
                    correct += 1
        except Exception:
            # Code execution failed
            pass
    
    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0
    
    # Convert accuracy to Codeforces rating
    # This is a simplified mapping - in a real implementation you would use a more accurate model
    rating = 1200 + 800 * accuracy
    
    # Convert rating to percentile
    # This is a simplified mapping - in a real implementation you would use actual distribution data
    percentile = min(100, max(0, 100 * (rating - 1200) / 1600))
    
    return {
        "percentile": percentile,
        "rating": rating,
        "accuracy": accuracy,
        "num_correct": correct,
        "num_total": total
    }


def evaluate_gpqa(model, tokenizer, benchmark_data, device, metric="pass@1"):
    """
    Evaluate model on GPQA benchmark.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        benchmark_data: The benchmark data
        device: The device to use
        metric: The metric to use (pass@1)
        
    Returns:
        Dictionary of benchmark results
    """
    model.to(device)
    model.eval()
    
    correct = 0
    total = len(benchmark_data["questions"])
    
    for question in tqdm(benchmark_data["questions"], desc="Evaluating GPQA"):
        problem = question["question"]
        choices = question["choices"]
        correct_answer = question["answer"]
        
        # Format choices as A, B, C, D
        formatted_choices = "\n".join([f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)])
        
        prompt = f"Answer the following physics question by selecting the correct option.\n\nQuestion: {problem}\n\nOptions:\n{formatted_choices}\n\nProvide your answer as a single letter (A, B, C, or D)."
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.0,
                top_p=1.0
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Extract answer - looking for a single letter A, B, C, or D
        answer = None
        for line in response.split("\n"):
            if any(option in line for option in ["A", "B", "C", "D"]):
                for option in ["A", "B", "C", "D"]:
                    if option in line:
                        answer = option
                        break
                if answer:
                    break
        
        # If no clear answer found, try to extract the first letter
        if not answer:
            for char in response:
                if char in ["A", "B", "C", "D"]:
                    answer = char
                    break
        
        # Check if the answer is correct
        if answer and answer == correct_answer:
            correct += 1
    
    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0
    
    return {
        metric: accuracy,
        "num_correct": correct,
        "num_total": total
    }


def evaluate_math(model, tokenizer, benchmark_data, device, metric="pass@1"):
    """
    Evaluate model on MATH benchmark.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        benchmark_data: The benchmark data
        device: The device to use
        metric: The metric to use (pass@1)
        
    Returns:
        Dictionary of benchmark results
    """
    model.to(device)
    model.eval()
    
    correct = 0
    total = len(benchmark_data["questions"])
    
    for question in tqdm(benchmark_data["questions"], desc="Evaluating MATH"):
        problem = question["problem"]
        correct_answer = question["answer"]
        
        prompt = f"Solve the following mathematics problem step by step.\n\nProblem: {problem}\n\nShow your work and provide the final answer."
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.0,
                top_p=1.0
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Extract answer - assuming the final answer is indicated clearly
        answer = None
        for line in response.split("\n"):
            if "answer" in line.lower() or "=" in line:
                # Try to extract numerical answer
                import re
                matches = re.findall(r'[-+]?\d*\.\d+|\d+', line)
                if matches:
                    answer = matches[-1]
                    break
        
        # If no answer found, try to extract from the last line
        if not answer:
            last_line = response.strip().split("\n")[-1]
            import re
            matches = re.findall(r'[-+]?\d*\.\d+|\d+', last_line)
            if matches:
                answer = matches[-1]
        
        # Check if the answer is correct
        if answer and answer == correct_answer:
            correct += 1
    
    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0
    
    return {
        metric: accuracy,
        "num_correct": correct,
        "num_total": total
    }


def evaluate_mmlu(model, tokenizer, benchmark_data, device, metric="pass@1"):
    """
    Evaluate model on MMLU benchmark.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        benchmark_data: The benchmark data
        device: The device to use
        metric: The metric to use (pass@1)
        
    Returns:
        Dictionary of benchmark results
    """
    model.to(device)
    model.eval()
    
    correct = 0
    total = len(benchmark_data["questions"])
    
    # Track performance by subject
    subject_performance = {}
    
    for question in tqdm(benchmark_data["questions"], desc="Evaluating MMLU"):
        problem = question["question"]
        choices = question["choices"]
        correct_answer = question["answer"]
        subject = question.get("subject", "unknown")
        
        # Format choices as A, B, C, D
        formatted_choices = "\n".join([f"{choice}" for choice in choices])
        
        prompt = f"The following is a multiple-choice question. Please select the correct answer.\n\nQuestion: {problem}\n\nOptions:\n{formatted_choices}\n\nAnswer:"
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.0,
                top_p=1.0
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Extract answer - looking for a pattern like "The answer is A" or just "A"
        answer = None
        for line in response.split("\n"):
            if any(option in line for option in ["A", "B", "C", "D"]):
                for option in ["A", "B", "C", "D"]:
                    if option in line:
                        answer = option
                        break
                if answer:
                    break
        
        # If no clear answer found, try to extract the first letter
        if not answer:
            for char in response:
                if char in ["A", "B", "C", "D"]:
                    answer = char
                    break
        
        # Check if the answer is correct
        is_correct = (answer == correct_answer)
        if is_correct:
            correct += 1
        
        # Update subject performance
        if subject not in subject_performance:
            subject_performance[subject] = {"correct": 0, "total": 0}
        
        subject_performance[subject]["total"] += 1
        if is_correct:
            subject_performance[subject]["correct"] += 1
    
    # Calculate overall accuracy
    accuracy = correct / total if total > 0 else 0
    
    # Calculate per-subject accuracy
    for subject in subject_performance:
        if subject_performance[subject]["total"] > 0:
            subject_performance[subject]["accuracy"] = (
                subject_performance[subject]["correct"] / subject_performance[subject]["total"]
            )
        else:
            subject_performance[subject]["accuracy"] = 0
    
    return {
        metric: accuracy,
        "num_correct": correct,
        "num_total": total,
        "subject_performance": subject_performance
    }


def evaluate_swe_bench(model, tokenizer, benchmark_data, device, metric="resolved"):
    """
    Evaluate model on SWE-bench.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        benchmark_data: The benchmark data
        device: The device to use
        metric: The metric to use (resolved or success@1)
        
    Returns:
        Dictionary of benchmark results
    """
    model.to(device)
    model.eval()
    
    correct = 0
    total = len(benchmark_data["questions"])
    
    for question in tqdm(benchmark_data["questions"], desc="Evaluating SWE-bench"):
        description = question["description"]
        code = question["code"]
        correct_solution = question["correct_solution"]
        
        prompt = f"Fix the following code based on the description:\n\nDescription: {description}\n\nCode:\n```\n{code}\n```\n\nProvide the corrected code without explanation."
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.0 if metric == "success@1" else 0.2,
                top_p=1.0 if metric == "success@1" else 0.95
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Extract code - assuming the code is enclosed in ```python ``` or just ```
        if "```python" in response and "```" in response.split("```python", 1)[1]:
            fixed_code = response.split("```python", 1)[1].split("```", 1)[0].strip()
        elif "```" in response:
            fixed_code = response.split("```", 1)[1].split("```", 1)[0].strip()
        else:
            fixed_code = response.strip()
        
        # Check if the solution is correct
        # This is a simplified check - in a real implementation you would run tests
        # For this demo, we'll just check if the correct solution is in the response
        if correct_solution in fixed_code:
            correct += 1
    
    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0
    
    return {
        metric: accuracy,
        "num_correct": correct,
        "num_total": total
    }


def main():
    """Main entry point for benchmark evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate model on benchmarks")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--benchmark", type=str, required=True, choices=BENCHMARKS.keys(), 
                       help="Benchmark to evaluate")
    parser.add_argument("--subset", type=str, default=None, help="Benchmark subset")
    parser.add_argument("--metric", type=str, default=None, help="Metric to use")
    parser.add_argument("--output", type=str, default=None, help="Path to save results")
    parser.add_argument("--device", type=str, default=None, 
                       help="Device to use (cpu, cuda, cuda:0, etc.)")
    
    args = parser.parse_args()
    
    # Validate benchmark and subset
    benchmark_info = BENCHMARKS[args.benchmark]
    
    if args.subset is None:
        args.subset = benchmark_info["default_subset"]
    elif benchmark_info["available_subsets"] and args.subset not in benchmark_info["available_subsets"]:
        raise ValueError(f"Invalid subset '{args.subset}' for benchmark '{args.benchmark}'. "
                        f"Available subsets: {benchmark_info['available_subsets']}")
    
    # Validate metric
    if args.metric is None:
        args.metric = benchmark_info["default_metric"]
    elif args.metric not in benchmark_info["available_metrics"]:
        raise ValueError(f"Invalid metric '{args.metric}' for benchmark '{args.benchmark}'. "
                        f"Available metrics: {benchmark_info['available_metrics']}")
    
    # Set device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    device = torch.device(args.device)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.config, args.checkpoint)
    
    # Load benchmark data
    benchmark_data = load_benchmark_data(args.benchmark, args.subset)
    
    # Set output path
    if args.output is None:
        output_dir = os.path.join("results", args.benchmark)
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(output_dir, f"{Path(args.checkpoint).stem}_{args.metric}.json")
    
    # Evaluate based on benchmark
    logger.info(f"Evaluating on {args.benchmark} with metric {args.metric}")
    start_time = time.time()
    
    if args.benchmark == "aime_2024":
        results = evaluate_aime(model, tokenizer, benchmark_data, device, args.metric)
    elif args.benchmark == "codeforces":
        results = evaluate_codeforces(model, tokenizer, benchmark_data, device, args.metric)
    elif args.benchmark == "gpqa":
        results = evaluate_gpqa(model, tokenizer, benchmark_data, device, args.metric)
    elif args.benchmark == "math":
        results = evaluate_math(model, tokenizer, benchmark_data, device, args.metric)
    elif args.benchmark == "mmlu":
        results = evaluate_mmlu(model, tokenizer, benchmark_data, device, args.metric)
    elif args.benchmark == "swe_bench":
        results = evaluate_swe_bench(model, tokenizer, benchmark_data, device, args.metric)
    else:
        raise ValueError(f"Evaluation not implemented for benchmark '{args.benchmark}'")
    
    # Add metadata to results
    results["benchmark"] = args.benchmark
    results["subset"] = args.subset
    results["metric"] = args.metric
    results["checkpoint"] = args.checkpoint
    results["config"] = args.config
    results["evaluation_time"] = time.time() - start_time
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {args.output}")
    logger.info(f"Evaluation completed in {results['evaluation_time']:.2f} seconds")
    logger.info(f"Results: {results[args.metric]:.4f} {args.metric}")


if __name__ == "__main__":
    main()