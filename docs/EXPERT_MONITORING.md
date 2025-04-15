# Expert Usage Monitoring

When training Mixture-of-Experts (MoE) models like those using PEER, it's important to monitor how experts are utilized. Uneven usage, where some experts ("hot experts") are selected far more often than others, can indicate load balancing issues and potentially limit model capacity.

This framework includes tools to track and visualize expert usage during training.

## Enabling Expert Monitoring

Expert usage tracking is configured within the `peer_config` section of your main YAML configuration file (`config.yaml`).

```yaml
model_config:
  # ... other model config ...
  use_peer: true
  peer_start_layer: 2 # Layer index where PEER starts
  peer_config:
    # ... other peer config ...
    log_expert_usage: true     # <<< Enable expert usage logging
    log_freq: 1000             # <<< How often (in training steps) to log usage patterns
    usage_threshold: 5.0       # <<< Multiplier for flagging "hot" experts (usage > threshold * average)
```

**Parameters:**

-   `log_expert_usage: true`: Enables the tracking mechanism.
-   `log_freq: 1000`: Specifies how frequently (in training steps) the collected usage data should be processed and logged (e.g., to Weights & Biases).
-   `usage_threshold: 5.0`: Defines the threshold for identifying "hot" experts. An expert is flagged if its usage count within a logging period exceeds this multiplier times the average usage across all experts in that layer.

## How it Works

When enabled, the `ExpertUsageTracker` class (`llm/utils/experts/tracking.py`) is initialized and hooks are attached to the forward pass of each PEER layer in the model.

1.  **Recording:** During each forward pass through a PEER layer, the hook records which experts were selected for the tokens in the batch.
2.  **Aggregation:** Usage counts are aggregated per layer and per expert index over the `log_freq` interval.
3.  **Processing:** At the end of each interval, the tracker calculates statistics:
    -   Total usages per layer.
    -   Number of unique experts used per layer.
    -   Usage coverage (percentage of experts used).
    -   Average usage per expert.
    -   Identification of "hot" experts based on the `usage_threshold`.
4.  **Logging (WandB):** If Weights & Biases integration is enabled (`wandb_config` is set), the tracker logs:
    -   Histograms showing the usage distribution of the most frequently used experts per layer.
    -   Line plots showing usage coverage over time.
    -   A count of currently identified "hot" experts.
    -   A table listing the identified "hot" experts (layer index, expert index).
    -   Heatmaps visualizing usage across top experts and layers.

## Interpreting the Logs

Monitoring these logs in WandB can help diagnose training issues:

-   **Low Coverage:** If only a small fraction of experts are ever used, it might indicate that the model isn't effectively leveraging its capacity or that the routing/retrieval mechanism isn't diverse enough.
-   **High Number of Hot Experts:** Consistently high numbers of hot experts suggest poor load balancing. This could be due to the retrieval mechanism, data distribution, or initialization.
-   **Persistent Hot Experts:** If the *same* experts remain hot throughout training, they might represent bottlenecks or overly specialized functions.

Based on these observations, you might consider adjustments to:

-   The PEER configuration (e.g., `num_heads`, `query_dim`).
-   The training data mix.
-   Learning rate or optimization strategy.
-   Initialization methods.
