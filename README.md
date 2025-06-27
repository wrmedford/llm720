# LLM720

LLM720 is a second generation Large Language Model that aims to be:

1. Open
2. Interpretable
3. Energy efficient

These goals are all accomplished through the same guiding design: Make a model that is as sparse as possible, so you can see exactly which weights are contributing to output while simultaneously not using more compute than necessary. 

We aim to accomplish this through a [fine grained Mixture of Experts architecture](https://arxiv.org/pdf/2407.04153) (He, 2024), while utilizing efficient attention mechanisms initially pioneered by DeepSeek with [Multi-Headed Latent Attention](https://arxiv.org/pdf/2412.19437) (Deepseek, 2024). This project is also exploring expanding model weights into system memory while maintaining performance with the hope of making frontier models less bound by GPU memory capacity by expanding on the Mixture of a Million Experts architecture originally developed by He. 

LLM720 gets its namesake from [LLM360](https://arxiv.org/pdf/2312.06550), where we intend to carry the torch of completely open-sourced model development (Liu et al, 2023). 

## Quick Start

- **Installation**: See [`docs/INSTALLATION.md`](docs/INSTALLATION.md)
- **Architecture**: See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- **Configuration**: See [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md)
- **Full Documentation**: Browse the [`docs/`](docs/) directory

## Join us on Discord

[![](https://dcbadge.limes.pink/api/server/https://discord.gg/5DgUmSB2uP)](https://discord.gg/https://discord.gg/5DgUmSB2uP)

## Special Thanks

Thank you to @lambdal for providing compute for this project.