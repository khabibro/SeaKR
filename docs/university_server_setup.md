# University Server Setup

SeaKR is set up here for a Linux university machine with an NVIDIA CUDA GPU.
The Mac stays in the role of editor and GitHub client only.

## Workflow

1. Clone the repository on the university server.
2. Create the `seakr` conda environment.
3. Verify the GPU, CUDA runtime, CPU RAM, and disk space.
4. Install Python dependencies.
5. Install the modified `vllm_uncertainty` package in editable mode.
6. Install spaCy `en_core_web_sm`.
7. Log in to Hugging Face if the model is gated.
8. Download the model and datasets outside the repo.
9. Start Elasticsearch and verify the BM25 wiki index.
10. Run the smoke test.
11. Run the 20-example mini eval.
12. Run the 100-example eval.
13. Run the full reproduction only after the small runs are stable.

## Environment variables

Use these on the university server:

```bash
export DATA_DIR="${DATA_DIR:-/path/to/datasets}"
export MODEL_DIR="${MODEL_DIR:-/path/to/models}"
export OUTPUT_DIR="${OUTPUT_DIR:-/path/to/outputs}"
export HF_HOME="${HF_HOME:-/path/to/huggingface-cache}"
export ES_HOST="${ES_HOST:-localhost}"
export ES_PORT="${ES_PORT:-9201}"
```

## What stays unchanged

- prompts
- retrieval logic
- thresholds
- hidden-state extraction
- Gram determinant computation
- evaluation metrics
- pseudo-generation logic

## Recommended entry points

```bash
bash scripts/check_gpu_env.sh
bash scripts/setup_university_env.sh
bash scripts/run_smoke_test.sh
bash scripts/run_mini_eval.sh
```

## Notes

- Let `tensor_parallel_size` auto-detect for normal runs. Set `SEAKR_TENSOR_PARALLEL_SIZE=1` only on a single-GPU server, or `SEAKR_TENSOR_PARALLEL_SIZE=2` when reproducing on a two-GPU server.
- Do not store datasets, models, outputs, or logs in GitHub.
- Keep large artifacts on the university machine or mounted storage.
