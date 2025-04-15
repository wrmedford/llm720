# Model Analysis

Understanding the parameter distribution and efficiency of models, especially those using sparse expert layers like PEER, is crucial. This framework includes tools to analyze model parameter counts.

## Parameter Analysis Script

The `scripts/analyze_model.py` script calculates and displays the total parameters and estimated active parameters per token for the foundation model architecture.

**Usage:**

```bash
# Activate your virtual environment
source .venv/bin/activate

# Analyze using a configuration file (calculates based on config)
python scripts/analyze_model.py --config configs/config.yaml

# Analyze a specific checkpoint (loads weights for precise calculation)
python scripts/analyze_model.py --config configs/config.yaml --checkpoint ./output/final-model.safetensors

# Save visualization to a file
python scripts/analyze_model.py --config configs/config.yaml --output analysis_plot.png

# Save detailed results to JSON
python scripts/analyze_model.py --config configs/config.yaml --json analysis_results.json
```

**Output:**

The script provides:

-   **Total Parameters:** The overall number of parameters in the model.
-   **Active Parameters:** An estimation of the number of parameters involved in processing a single token (significantly lower for PEER models).
-   **Breakdown:** Parameter counts for different components (embeddings, attention, MLP/PEER layers).
-   **Efficiency Ratio:** The ratio of active parameters to total parameters.
-   **Visualizations (Optional):** Plots comparing total vs. active parameters.
-   **JSON Output (Optional):** Detailed parameter counts in a machine-readable format.

This analysis helps quantify the efficiency gains achieved by using sparse expert architectures like PEER compared to dense transformer models.
