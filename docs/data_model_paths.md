# Data, Model, and Output Paths

SeaKR expects large artifacts to live outside the repository.

## Directories

- `DATA_DIR`
  - datasets
  - raw archives
  - processed retrieval data
  - Elasticsearch index data

- `MODEL_DIR`
  - Hugging Face checkpoints
  - local model directories

- `OUTPUT_DIR`
  - predictions
  - logs
  - metrics
  - figures
  - tables

- `HF_HOME`
  - Hugging Face cache
  - tokenizer cache
  - model download cache

## Recommended environment variables

```bash
export DATA_DIR="${DATA_DIR:-/path/to/datasets}"
export MODEL_DIR="${MODEL_DIR:-/path/to/models}"
export OUTPUT_DIR="${OUTPUT_DIR:-/path/to/outputs}"
export HF_HOME="${HF_HOME:-/path/to/huggingface-cache}"
```

## Policy

- Do not put datasets into the repo.
- Do not put model checkpoints into the repo.
- Do not put outputs into the repo.
- Do not commit logs from experiments.

## Default university layout

```text
$DATA_DIR
$MODEL_DIR
$OUTPUT_DIR
$HF_HOME
```

Keep these paths on the university machine or a mounted storage volume, not in GitHub.
