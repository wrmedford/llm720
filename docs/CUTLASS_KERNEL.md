# CUTLASS PEER Kernel Documentation

## Overview

The CUTLASS PEER kernel provides an optimized implementation of the Parameter Efficient Expert Retrieval (PEER) module for NVIDIA GPUs, specifically targeting H100 and A100 architectures. This kernel offers significant performance improvements over the PyTorch implementation through:

- **Fused Operations**: Combines query projection, normalization, expert selection, and computation in a single kernel
- **Hierarchical Memory Management**: Efficient caching of expert weights across HBM and system RAM
- **Optimized Memory Access**: Uses Tensor Memory Accelerator (TMA) for efficient data movement
- **Double Buffering**: Overlaps computation with memory transfers

## Features

### Memory Hierarchy

The kernel implements a three-tier memory hierarchy:

1. **Shared Memory (SMEM)**: Holds active token chunks and currently-used expert weights
2. **High Bandwidth Memory (HBM)**: Caches frequently-accessed ("hot") experts
3. **System RAM**: Stores the full set of expert weights with UVA access

### Key Optimizations

- **L2 Cache Optimization**: Token chunks sized to fit in L2 cache (≤40MB)
- **Ping-Pong Buffering**: Double buffering pattern for expert weights
- **CLOCK-based Eviction**: O(1) cache eviction algorithm
- **Heat Tracking**: Tracks expert usage patterns for intelligent caching
- **WMMA Operations**: Uses Tensor Cores for matrix multiplication

## Usage

### Environment Variables

```bash
# Enable CUTLASS kernel (default: disabled)
export USE_CUTLASS_KERNEL=1

# Configure HBM cache size in MB (default: 16384)
export PEER_HBM_CACHE_MB=16384

# Use managed memory instead of pinned memory (default: 0)
export PEER_USE_MANAGED_MEMORY=0

# Control kernel building during setup (default: 1)
export BUILD_CUTLASS_KERNEL=1
```

### Python API

The CUTLASS kernel integrates seamlessly with the existing PEER module:

```python
import os
os.environ["USE_CUTLASS_KERNEL"] = "1"

from llm.models.experts import PEER

# Create PEER module - automatically uses CUTLASS kernel
model = PEER(
    input_dim=1024,
    output_dim=1024,
    num_experts=1048576,  # 1M experts
    num_experts_per_tok=16,
    num_heads=8,
    expert_hidden_size=1,
    query_dim=256,
).cuda()

# Use as normal
output = model(input_tensor)
```

### Building the Kernel

The kernel is built automatically when needed, or manually:

```bash
# Manual build
python setup.py build_ext --inplace

# Verify successful build
python -c "from llm.models.kernels.peer_cutlass import CUTLASS_AVAILABLE; print(f'CUTLASS available: {CUTLASS_AVAILABLE}')"
```

## Performance Considerations

### Hardware Requirements

- **GPU**: NVIDIA A100 (compute capability 8.0) or H100 (compute capability 9.0)
- **CUDA**: Version 11.0 or higher
- **Memory**: Sufficient system RAM for expert weights (e.g., 400GB for 1M experts)

### Tuning Parameters

1. **HBM Cache Size**: Adjust `PEER_HBM_CACHE_MB` based on available GPU memory
2. **Chunk Size**: Automatically computed based on L2 cache size
3. **Grid/Block Configuration**: Auto-tuned based on problem size

### Performance Tips

- Use pinned memory (default) for better PCIe transfer performance
- Ensure expert weights are 128-byte aligned for TMA
- Monitor cache hit rates via `print_cache_stats()`
- Consider reducing `num_experts_per_tok` if memory bandwidth limited

## Technical Details

### Kernel Configuration

The kernel uses compile-time configuration through the `PEERConfig` template:

```cpp
template<int MaxExperts_, int BlockM_, int BlockK_, int HiddenSize_, int OuterTiles_>
struct PEERConfig {
    static constexpr int MaxExperts = MaxExperts_;    // 1048576
    static constexpr int BlockM = BlockM_;            // 56
    static constexpr int BlockK = BlockK_;            // 128
    static constexpr int HiddenSize = HiddenSize_;    // 256
    static constexpr int OuterTiles = OuterTiles_;    // 64
};
```

### Shared Memory Layout

```
[Token Cache | U Buffer 0 | U Buffer 1 | V Buffer 0 | V Buffer 1 | Query | Hidden]
```

All sections are 64-byte aligned to prevent bank conflicts.

### Algorithm Flow

1. **Token Loading**: Load chunk of tokens into shared memory
2. **Query Projection**: Compute queries for all heads
3. **Product Key Routing**: Select top-k experts using product keys
4. **Expert Computation**: 
   - Double-buffered loading of expert weights
   - Fused GEMM + GELU activation
   - Weighted accumulation of outputs
5. **Output Writing**: Direct write to global memory

## Debugging

### Common Issues

1. **Kernel Launch Failure**: Check shared memory requirements
2. **Incorrect Results**: Verify weight alignment and dimensions
3. **Poor Performance**: Monitor cache hit rates and memory bandwidth

### Debug Output

Enable verbose output during kernel compilation:

```python
import torch.utils.cpp_extension as cpp_ext
cpp_ext.load(..., verbose=True)
```

### Cache Statistics

Monitor hierarchical cache performance:

```python
from llm.models.kernels import peer_cutlass_module
peer_cutlass_module.print_cache_stats()
```

## Comparison with Triton

While a Triton implementation was initially planned, the CUTLASS kernel was chosen due to:

- **TMA Support**: Direct access to H100's Tensor Memory Accelerator
- **Fine-grained Control**: Explicit shared memory management
- **Proven Performance**: CUTLASS is battle-tested for production workloads
- **Hardware Features**: Access to latest GPU features (e.g., asynchronous copies)

The Triton kernel stub remains in the codebase for potential future development.

## Future Improvements

- **FP8 Support**: Leverage H100's FP8 capabilities
- **Dynamic Shapes**: Runtime kernel selection based on problem size
- **Multi-GPU**: Distributed expert storage across GPUs
- **Quantization**: INT8/INT4 expert weights for memory efficiency