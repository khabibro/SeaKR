# Setup

This project is prepared for SeaKR reproduction experiments.

Use a Python 3.10 environment. For local Mac work, use the environment only for repository inspection, dataset preparation, and retrieval smoke tests. Full SeaKR inference requires Linux with CUDA because the modified `vllm_uncertainty` package does not support macOS.

Recommended environment variables:

```bash
export DATA_DIR="${DATA_DIR:-data}"
export MODEL_DIR="${MODEL_DIR:-models}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
export HF_HOME="${HF_HOME:-.cache/huggingface}"
```

Do not place datasets, model weights, logs, or generated experiment outputs inside Obsidian.
