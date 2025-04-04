#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Expert Usage Tracker for PEER (Parameter Efficient Expert Retrieval)

This module implements tracking and monitoring of expert usage patterns
in PEER-based models, helping identify overused "hot" experts during training.
"""

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict, Counter
import wandb
import seaborn as sns


class ExpertUsageTracker:
    """
    Tracks and logs expert usage patterns in PEER layers.
    
    This class monitors which experts are selected during forward passes,
    tracks usage frequency by layer and expert index, and logs these
    patterns to Weights & Biases for visualization.
    """
    
    def __init__(
        self,
        num_experts: int,
        num_layers: int,
        log_freq: int = 1000,
        usage_threshold: float = 5.0,
        wandb_enabled: bool = True,
    ):
        """
        Initialize the expert usage tracker.
        
        Args:
            num_experts: Total number of experts in each PEER layer
            num_layers: Number of layers with PEER in the model
            log_freq: Frequency (in steps) to log usage patterns
            usage_threshold: Threshold multiplier for flagging "hot" experts
            wandb_enabled: Whether to log to Weights & Biases
        """
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.log_freq = log_freq
        self.usage_threshold = usage_threshold
        self.wandb_enabled = wandb_enabled
        
        # Initialize tracking structures
        self._reset_counters()
        
        self.step = 0
        self.total_tokens_processed = 0
        self.hot_experts = set()  # Set of (layer_idx, expert_idx) tuples for hot experts
        
        # Statistics
        self.layer_usage_history = defaultdict(list)  # layer_idx -> list of usage counts over time
        self.expert_usage_history = defaultdict(list)  # (layer_idx, expert_idx) -> list of usage counts over time
    
    def _reset_counters(self):
        """Reset usage counters for the current logging period."""
        # expert_counters[layer_idx][expert_idx] = count
        self.expert_counters = defaultdict(Counter)
        
        # Count of tokens processed in this period
        self.period_tokens = 0
    
    def record_usage(
        self, 
        layer_idx: int, 
        expert_indices: torch.Tensor,
        batch_size: int,
        seq_len: int
    ):
        """
        Record expert usage from a forward pass.
        
        Args:
            layer_idx: Index of the layer
            expert_indices: Tensor of expert indices used [batch, seq_len, num_heads, top_k]
            batch_size: Batch size
            seq_len: Sequence length
        """
        # Move to CPU and convert to numpy for counting
        expert_indices_cpu = expert_indices.detach().cpu().numpy()
        
        # Count each expert usage
        for expert_idx in expert_indices_cpu.flatten():
            self.expert_counters[layer_idx][expert_idx] += 1
        
        # Update token count
        self.period_tokens += batch_size * seq_len
        self.total_tokens_processed += batch_size * seq_len
    
    def step_end(self) -> bool:
        """
        Call at the end of each training step.
        
        Returns:
            bool: True if logging was performed in this step
        """
        self.step += 1
        
        # Check if it's time to log
        if self.step % self.log_freq == 0:
            self._process_counters()
            self._log_usage_patterns()
            self._reset_counters()
            return True
        
        return False
    
    def _process_counters(self):
        """Process the current counters to update statistics."""
        # Skip if no tokens were processed
        if self.period_tokens == 0:
            return
        
        # Identify hot experts (significantly above average usage)
        for layer_idx, counter in self.expert_counters.items():
            # Calculate average usage per expert for this layer
            total_usages = sum(counter.values())
            num_experts_used = len(counter)
            
            if num_experts_used == 0:
                continue
                
            avg_usage = total_usages / self.num_experts  # Average over all experts, including unused
            
            # Update usage history
            self.layer_usage_history[layer_idx].append({
                'step': self.step,
                'total_usages': total_usages,
                'experts_used': num_experts_used,
                'usage_coverage': num_experts_used / self.num_experts,
                'avg_usage': avg_usage
            })
            
            # Identify hot experts (usage > threshold * average)
            for expert_idx, count in counter.items():
                if count > avg_usage * self.usage_threshold:
                    self.hot_experts.add((layer_idx, expert_idx))
                
                # Update expert usage history
                self.expert_usage_history[(layer_idx, expert_idx)].append({
                    'step': self.step,
                    'count': count,
                    'norm_count': count / self.period_tokens,
                })
    
    def _log_usage_patterns(self):
        """Log usage patterns to Weights & Biases."""
        if not self.wandb_enabled or not self.expert_counters:
            return
            
        try:
            # Create usage histogram data for each layer
            for layer_idx, counter in self.expert_counters.items():
                # Skip empty layers
                if not counter:
                    continue
                    
                # Sort by usage count
                sorted_counts = sorted(counter.items(), key=lambda x: x[1], reverse=True)
                
                # Only log the top 100 most used experts to avoid overwhelming wandb
                top_experts = sorted_counts[:100]
                
                # Prepare histogram data
                expert_ids = [f"{ex}" for ex, _ in top_experts]
                usage_counts = [count for _, count in top_experts]
                
                # Create histogram figure
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.bar(expert_ids, usage_counts)
                ax.set_title(f"Expert Usage - Layer {layer_idx} (Step {self.step})")
                ax.set_xlabel("Expert ID")
                ax.set_ylabel("Usage Count")
                plt.xticks(rotation=90)
                plt.tight_layout()
                
                # Log to wandb
                wandb.log({
                    f"expert_usage/layer_{layer_idx}_histogram": wandb.Image(fig),
                    f"expert_usage/layer_{layer_idx}_experts_used": len(counter),
                    f"expert_usage/layer_{layer_idx}_coverage": len(counter) / self.num_experts,
                }, step=self.step)
                
                plt.close(fig)
                
                # Create heatmap of expert usage across top experts and layers
                if layer_idx == 0:  # Only create heatmap once per logging period
                    self._create_expert_heatmap()
                
            # Log hot experts
            if self.hot_experts:
                hot_expert_data = [
                    f"Layer {layer}, Expert {expert}" 
                    for layer, expert in sorted(self.hot_experts)
                ]
                wandb.log({
                    "expert_usage/hot_experts_count": len(self.hot_experts),
                    "expert_usage/hot_experts": wandb.Table(
                        columns=["Hot Experts"], 
                        data=[[expert] for expert in hot_expert_data]
                    )
                }, step=self.step)
        
        except Exception as e:
            print(f"Error logging expert usage to wandb: {e}")
    
    def _create_expert_heatmap(self):
        """Create and log a heatmap of expert usage patterns."""
        # Get top 20 experts by usage across all layers
        all_expert_usage = []
        
        for layer_idx, counter in self.expert_counters.items():
            for expert_idx, count in counter.items():
                all_expert_usage.append((layer_idx, expert_idx, count))
        
        # Sort by count and take top 20
        top_experts = sorted(all_expert_usage, key=lambda x: x[2], reverse=True)[:20]
        
        if not top_experts:
            return
            
        # Create heatmap data
        heatmap_data = {}
        for layer_idx, expert_idx, count in top_experts:
            layer_name = f"Layer {layer_idx}"
            expert_name = f"Expert {expert_idx}"
            
            if layer_name not in heatmap_data:
                heatmap_data[layer_name] = {}
            
            heatmap_data[layer_name][expert_name] = count
        
        # Convert to DataFrame for seaborn
        import pandas as pd
        
        # Create structured data
        rows = []
        for layer_name, experts in heatmap_data.items():
            for expert_name, count in experts.items():
                rows.append({
                    "Layer": layer_name,
                    "Expert": expert_name,
                    "Usage Count": count
                })
        
        df = pd.DataFrame(rows)
        
        # Create pivot table
        pivot_data = df.pivot(index="Layer", columns="Expert", values="Usage Count")
        
        # Fill NaN values with 0
        pivot_data = pivot_data.fillna(0)
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_data, annot=True, fmt=".0f", cmap="YlGnBu")
        plt.title(f"Top Expert Usage Across Layers (Step {self.step})")
        plt.tight_layout()
        
        # Log to wandb
        wandb.log({
            "expert_usage/top_experts_heatmap": wandb.Image(plt),
        }, step=self.step)
        
        plt.close()
    
    def get_summary(self) -> Dict:
        """
        Get a summary of expert usage patterns.
        
        Returns:
            Dict containing usage statistics
        """
        summary = {
            "total_tokens_processed": self.total_tokens_processed,
            "hot_experts_count": len(self.hot_experts),
            "hot_experts": [(layer, expert) for layer, expert in sorted(self.hot_experts)],
            "layer_stats": {},
        }
        
        # Add layer statistics
        for layer_idx, history in self.layer_usage_history.items():
            if not history:
                continue
                
            # Most recent stats
            latest = history[-1]
            
            summary["layer_stats"][layer_idx] = {
                "experts_used": latest["experts_used"],
                "usage_coverage": latest["usage_coverage"],
                "avg_usage": latest["avg_usage"],
            }
        
        return summary


def hook_expert_tracking(module, tracker):
    """
    Add hooks to track expert usage in PEER modules.
    
    Args:
        module: The model module to hook
        tracker: ExpertUsageTracker instance
    """
    from functools import partial
    
    def _expert_forward_hook(layer_idx, mod, input, output):
        """Hook function to record expert usage."""
        # Check if expert indices are available (saved during forward pass)
        if hasattr(mod, '_last_expert_indices'):
            batch_size, seq_len = input[0].shape[:2]
            tracker.record_usage(
                layer_idx=layer_idx,
                expert_indices=mod._last_expert_indices,
                batch_size=batch_size,
                seq_len=seq_len
            )
    
    def _find_and_hook_peer_layers(module, prefix=""):
        """Recursively find and hook PEER layers."""
        layer_idx = 0
        for name, child in module.named_children():
            if "PEER" in child.__class__.__name__:
                # Modify forward method to save expert indices
                original_forward = child.forward
                
                def _save_indices_wrapper(self, original_fn, hidden_states):
                    # Call original forward
                    output = original_fn(self, hidden_states)
                    # Store the last indices used for the hook to access
                    if hasattr(self, '_last_expert_indices'):
                        delattr(self, '_last_expert_indices')
                    return output
                
                # Store the indices during get_indices function
                original_get_indices = child.get_indices
                
                def _modified_get_indices(self, original_fn, queries, top_k):
                    indices, scores = original_fn(self, queries, top_k)
                    # Save the indices for the forward hook to access
                    self._last_expert_indices = indices
                    return indices, scores
                
                # Apply monkey patches
                child.get_indices = partial(_modified_get_indices, child, original_get_indices)
                
                # Register forward hook for this layer
                child.register_forward_hook(partial(_expert_forward_hook, layer_idx))
                layer_idx += 1
            
            # Recursively process child modules
            elif len(list(child.children())) > 0:
                layer_idx = _find_and_hook_peer_layers(child, prefix=f"{prefix}.{name}")
        
        return layer_idx
    
    # Find and hook all PEER layers
    _find_and_hook_peer_layers(module)


if __name__ == "__main__":
    # Simple test/example of the ExpertUsageTracker
    tracker = ExpertUsageTracker(num_experts=1024, num_layers=12)
    
    # Simulate expert usage
    for step in range(100):
        # Simulate batch with random expert selection
        batch_size = 16
        seq_len = 128
        num_heads = 8
        top_k = 16
        
        for layer_idx in range(3):  # Simulate 3 layers
            # Create random expert indices
            expert_indices = torch.randint(
                0, 1024, (batch_size, seq_len, num_heads, top_k)
            )
            
            # Record usage
            tracker.record_usage(layer_idx, expert_indices, batch_size, seq_len)
        
        # End step
        tracker.step_end()
    
    # Print summary
    print(tracker.get_summary())