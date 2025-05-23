# Architecture Overview

This document describes the architectural components of the foundation language model, including PEER (Parameter Efficient Expert Retrieval) and MLA (Multi-head Latent Attention).

## PEER (Parameter Efficient Expert Retrieval)

PEER implements a highly scalable Mixture-of-Experts architecture that can efficiently handle millions of tiny experts. The key innovation is the use of product keys for O(sqrt(N)) expert selection complexity.

### Key Features

- **Product Key Routing**: Decomposes expert selection into multiple smaller searches
- **Tiny Experts**: Each expert has minimal parameters (default hidden_size=1)
- **Multi-head Design**: Separate expert pools per attention head
- **Flexible Activation**: Supports variable top-k expert selection

### Implementation Variants

1. **PyTorch Implementation**: Reference implementation with full flexibility
2. **CUTLASS Kernel**: Optimized CUDA kernel for H100/A100 GPUs (see [CUTLASS Kernel Documentation](CUTLASS_KERNEL.md))

## MLA (Multi-head Latent Attention)

MLA provides an efficient attention mechanism based on the DeepSeek V3 architecture, using low-rank projections and RoPE/NoPE decomposition.

### Key Features

- **Low-rank Key-Value Projection**: Reduces memory and computation
- **RoPE Integration**: Rotary position embeddings for better position modeling
- **Shared Key-Value Heads**: Parameter efficiency through head sharing

## Memory Hierarchy

The system implements a sophisticated memory hierarchy for expert weights:

1. **L1/Shared Memory**: Active tokens and current experts
2. **L2 Cache**: Token chunks optimized for 40MB L2 on H100
3. **HBM**: Hot expert cache (configurable size)
4. **System RAM**: Full expert storage with UVA access

For detailed memory management strategies, see [Memory Hierarchy Documentation](MEMORY_HIERARCHY.md).

## Integration

The foundation model combines PEER and MLA layers in a transformer architecture:

```python
class FoundationModel(nn.Module):
    def __init__(self, config):
        self.layers = nn.ModuleList([
            TransformerBlock(
                attention=MLA(config),
                ffn=PEER(config) if use_peer else FFN(config)
            )
            for _ in range(config.num_layers)
        ])
```

This architecture enables efficient scaling to billions of parameters while maintaining reasonable activation costs.