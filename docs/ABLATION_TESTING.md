# Ablation Testing

This framework includes a meta-script (`scripts/run_ablations.py`) for running systematic ablation studies. This allows you to test the impact of different architectural choices and configurations on model performance.

## Ablation Axes

The script is designed to test combinations across several key configuration axes, defined in `llm/training/ablation.py`:

1.  **Data Mix (`data_mix`):** Different combinations and weightings of training datasets. You can define standard mixes in `ablation.py` or provide custom mixes via `configs/datasets.yaml`.
2.  **Expert Setup (`expert_setup`):** Variations in the total number of experts, the dimensionality of the product keys (`product_key_dim`), and the hidden size of each expert (`expert_hidden_size`).
3.  **Selection Heads (`selection_heads`):** Different configurations for the number of expert retrieval heads (`num_heads`) and the number of experts selected per token (`num_experts_per_tok`).

## Running Ablation Studies

The `run_ablations.py` script orchestrates the process of generating configurations, running training, evaluating checkpoints, and summarizing results.

**Basic Usage:**

```bash
# Activate your virtual environment
source .venv/bin/activate

# Run a full ablation study across all defined axes
# Uses the base config and default output directory
# Assumes 4 GPUs per experiment by default
python scripts/run_ablations.py --base-config configs/config.yaml --output-dir ablation_results --gpus 4
```

**Command-Line Arguments:**

-   `--base-config`: Path to the base YAML configuration file (e.g., `configs/config.yaml`). This provides the default settings that will be modified for each experiment.
-   `--output-dir`: Directory where all results, logs, configs, and checkpoints for the ablation run will be stored. A timestamped sub-directory will be created within this path.
-   `--gpus`: Number of GPUs to use *for each individual training experiment*. The script launches separate training runs sequentially.
-   `--axes`: (Optional) Specify which axes to ablate. Choose from `data_mix`, `expert_setup`, `selection_heads`. If omitted, all axes are combined.
    ```bash
    # Example: Ablate only data mix and expert setup
    python scripts/run_ablations.py --axes data_mix expert_setup ...
    ```
-   `--tokens`: (Optional) Limit the training duration for each experiment to a specific number of tokens. Useful for faster iterations.
    ```bash
    # Example: Limit each run to 5 Billion tokens
    python scripts/run_ablations.py --tokens 5000000000 ...
    ```
-   `--resume`: (Optional) Resume an interrupted ablation study from the most recent run in the `--output-dir`. Skips already completed experiments based on the `ablation_results.csv` file.
    ```bash
    python scripts/run_ablations.py --resume --output-dir ablation_results ...
    ```
-   `--custom-datasets`: (Optional) Path to a YAML file defining custom dataset mixes (see `configs/datasets.yaml` for format). These mixes will be added to or replace the default `data_mix` configurations.
    ```bash
    python scripts/run_ablations.py --custom-datasets configs/datasets.yaml ...
    ```

## Process Overview

1.  **Experiment Generation:** The script generates all combinations based on the selected `axes` and the configurations defined in `llm/training/ablation.py` (and optionally `configs/datasets.yaml`).
2.  **Configuration Generation:** For each combination, a unique YAML configuration file is created by modifying the `--base-config`.
3.  **Training:** The `scripts/train.sh` script is called to run training for the generated configuration, using the specified number of `--gpus`. Training duration can be limited by `--tokens`.
4.  **Evaluation:** After successful training, the final checkpoint is evaluated:
    -   Model parameter analysis (`scripts/analyze_model.py`) is run.
    -   Perplexity is calculated (`llm-eval perplexity`).
    -   Standard benchmarks (e.g., MMLU, MATH, AIME) are run (`llm-eval benchmark`).
5.  **Result Aggregation:** Key results (perplexity, benchmark scores, parameter counts, training time) are collected for each experiment.
6.  **Summary & Visualization:** After all experiments are completed (or when resuming), the results are saved to `ablation_results.csv`, and visualizations comparing performance across different configurations are generated in the `visualizations` sub-directory.

This systematic approach helps identify the most effective architectural choices and hyperparameter settings for the PEER and MLA based models.
