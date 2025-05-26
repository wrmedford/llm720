# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a foundation language model training framework implementing two core architectural innovations:
- **PEER (Parameter Efficient Expert Retrieval)**: Scalable MoE with millions of tiny experts using product key retrieval
- **MLA (Multi-Headed Latent Attention)**: Efficient attention based on DeepSeek V3 architecture

## Key Commands

### Building and Installation
```bash
# Install dependencies (with dev and eval extras)
uv pip install --no-build-isolation -e ".[dev,evals]"

# Build CUTLASS kernel for H100/A100 GPUs (optional, auto-builds when needed)
python setup.py build_ext --inplace

# Disable CUTLASS kernel building
export BUILD_CUTLASS_KERNEL=0
```

### Training
```bash
# Single GPU training
python scripts/train.py --config configs/config.yaml

# Distributed training (8 GPUs)
./scripts/train.sh --config configs/config.yaml --gpus-per-node 8

# Enable CUTLASS kernel for PEER
export USE_CUTLASS_KERNEL=1
python scripts/train.py --config configs/config.yaml
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_foundation_model.py

# Run with coverage
pytest --cov=llm
```

### Code Quality
```bash
# Format code
black llm/ scripts/ tests/
isort llm/ scripts/ tests/

# Lint code
flake8 llm/ scripts/ tests/

# Type checking
mypy llm/
```

### Evaluation
```bash
# Run evaluation script
python scripts/run_evaluation.py --checkpoint output/checkpoint-10000

# Run ablation tests
python scripts/run_ablations.py --config configs/ablation_config.yaml
```

## Architecture Overview

### Core Components
1. **llm/models/foundation.py**: Main model integrating PEER and MLA layers
2. **llm/models/experts.py**: PEER implementation with product key routing
3. **llm/models/attention.py**: MLA implementation with low-rank projections
4. **llm/models/kernels/**: CUTLASS optimized kernel for PEER on H100/A100

### Memory Hierarchy (for CUTLASS kernel)
- **SM Scratchpad**: Token slice + double-buffered expert weights
- **L2 Cache (40MB)**: Token chunks optimized for H100
- **HBM**: Hot expert cache
- **System RAM**: Full expert storage with UVA access

### Configuration System
- YAML-based configs in `configs/`
- Key files: `config.yaml` (main), `datasets.yaml` (data sources), `dry_run_config.yaml` (testing)
- Model params: hidden_size, num_layers, PEER/MLA specific settings
- Training params: batch_size, learning_rate, distributed settings

### Training Pipeline
- Distributed training with torchrun
- Checkpointing and resume support
- W&B integration for metrics and expert usage tracking
- Streaming datasets with interleaving support

## Important Notes

- PEER scales to millions of experts using O(sqrt(N)) product key selection
- MLA uses low-rank K/V projections (q_lora_rank ~0.25×hidden, kv_lora_rank ~0.083×hidden)
- Expert usage is tracked and logged to W&B for analysis
- CUTLASS kernel requires H100/A100 GPUs (compute capability 8.0+)
- Flash Attention 3 is required for MLA implementation
- FA3 must be installed from source: cd flash-attention/hopper && python setup.py install