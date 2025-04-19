# Memory Hierarchy

![image](https://github.com/user-attachments/assets/bacc3962-95b2-4e46-8a62-9021b76c3c12)

| Level | Capacity (GH200)| What Lives Here | Access Pattern in Our Kernel |
|-------|----------|-----------------|------------------------------|
| **SM Scratchpad (SMEM)** | up to 228 KB / SM | *One* token slice (`TOKEN_BYTES`) **+** two expert‑weight buffers (`2 × EXPERT_BYTES`) for ping‑pong streaming | 1. Async‑copy token → SMEM (once per block)  <br>2. Double‑buffer expert tiles: prefetch next while dotting current |
| **L1 Cache (remainder of 256 KB)** | 256 KB − SMEM | Automatic cache for auxiliary loads (scale factors, metadata) | No manual management outside of cache hinting; hits are opportunistic |
| **L2 Cache** | 50 MB (shared) | *Token Block* ≤ 40 MB  <br>Spilled intermediate activations | First block that touches a token pulls it in; every other block hits in L2 |
| **HBM3 / VRAM** | 96 GB | Dense transformer weights  <br>Hot subset of expert weights  <br>K/V decode cache | Expert tiles streamed directly → SMEM via TMA (do **not** pollute L2) |
| **LPDDR5X (Grace RAM)** | ≥ 384 GB | Cold expert weights | Fetched on demand → VRAM via NVLink‑C2C |

### Execution Flow

1. **Choose token chunk** so `CHUNK_SIZE × TOKEN_BYTES ≤ 40 MB`; launch one grid for that chunk.  
2. **Per block**  
   1. Async‑copy its token slice → SMEM (shared across all experts in the block).  
   2. Loop over `BLOCK_EXPERTS`, double‑buffering weight tiles to hide HBM latency.  
3. When all blocks finish, move to the next token chunk if any remain.

> **Design rule:** reserve exactly `SMEM = TOKEN_BYTES + 2×EXPERT_BYTES`; leave the rest of the 256 KB array as true L1 to maximise incidental cache hits.
