# Architecture Details

This project implements two core architectural innovations inspired by recent research from DeepSeek (e.g., DeepSeek-V2/V3, Mixture-of-A-Million-Experts):

1.  **Parameter Efficient Expert Retrieval (PEER):** An advanced Mixture-of-Experts (MoE) layer designed to scale to millions of tiny experts (often single neurons). Instead of traditional routing, PEER uses efficient "product key" retrieval to select a small subset of these experts for each token. This allows for a massive total parameter count while keeping the activated parameters per token low, aiming for high performance with reduced computational cost.
2.  **Multi-Headed Latent Attention (MLA):** An efficient attention mechanism based on the DeepSeek V3 architecture. It employs techniques like low-rank projections and RoPE/NoPE decomposition to optimize attention computation, particularly for inference, and integrates with optimized kernels like FlashAttention.

The goal is to build powerful LLMs that are efficient to train and run, leveraging sparse activation of a vast number of parameters.

## PEER (Parameter Efficient Expert Retrieval)

PEER uses product keys to efficiently select experts from a large pool.

**Key Concepts:**

-   **Product Keys:** Instead of a single large key space, PEER uses a Cartesian product of smaller sub-key spaces. An expert's key is formed by combining sub-keys from each dimension.
-   **Efficient Retrieval:** Queries are also split into sub-queries. Scores are computed independently within each dimension for the sub-queries against the sub-keys. An optimized combination strategy (avoiding full N-expert scoring) is used to find the top-k overall experts based on the combined sub-scores.
-   **Tiny Experts:** PEER is designed to work effectively with a very large number of simple experts, often just single neurons or small MLPs.

**Configuration:**

See `llm/models/experts.py` and the [Configuration Guide](CONFIGURATION.md) for details on parameters like:

-   `num_experts`: Total number of experts (must equal the product of `product_key_dim` sizes).
-   `product_key_dim`: List defining the size of each sub-key dimension (e.g., `[1024, 1024]` for 1M experts in 2D).
-   `num_experts_per_tok`: Number of experts to activate for each token.
-   `num_heads`: Number of independent retrieval heads.
-   `expert_hidden_size`: Size of the hidden layer within each expert MLP (can be 1).
-   `query_dim`: Dimension of the query vector used for retrieval.

## MLA (Multi-Headed Latent Attention)

MLA is an attention mechanism inspired by DeepSeek V3.

**Key Concepts:**

-   **Low-Rank Projections:** Key and Value states are often projected down to a lower-rank latent space (`kv_lora_rank`) before attention computation, reducing computational cost.
-   **RoPE/NoPE Decomposition:** Query and Key vectors are split into parts handled by Rotary Positional Embeddings (RoPE) and parts without positional encoding (NoPE).
-   **Separate Projections:** Distinct linear projections are used for Query, Key, and Value.
-   **Optimized Kernels:** Designed to leverage kernels like FlashAttention for performance. The implementation includes fallbacks to PyTorch's `scaled_dot_product_attention`.
-   **Decode Path Optimization:** Pre-computes certain weight transformations (`W_UK_T`, `W_UV`) to accelerate the attention calculation during single-token decoding steps.

**Configuration:**

See `llm/models/attention.py` and the [Configuration Guide](CONFIGURATION.md) for details on parameters like:

-   `q_lora_rank`: Rank for the optional query LoRA projection.
-   `kv_lora_rank`: Rank for the key-value latent projection.
-   `qk_rope_head_dim`: Dimension of the query/key part processed by RoPE.
-   `v_head_dim`: Dimension of the value head.
-   `qk_nope_head_dim`: Dimension of the query/key part *not* processed by RoPE.
