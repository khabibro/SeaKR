# Reproduction Checklist

Follow this order on the university server:

1. Clone the repo.
2. Create the conda environment.
3. Install dependencies.
4. Verify the GPU.
5. Log in to Hugging Face.
6. Download `Llama-2-7b-chat-hf`.
7. Download datasets.
8. Start Elasticsearch.
9. Verify the BM25 wiki index.
10. Run the smoke test.
11. Run the 20-example mini eval.
12. Run the 100-example eval.
13. Run the full reproduction.

## Command outline

```bash
git clone <repo-url>
cd SeaKR
bash scripts/setup_university_env.sh
bash scripts/check_gpu_env.sh
huggingface-cli login
# download model into $MODEL_DIR
# download datasets into $DATA_DIR
# start Elasticsearch and build/verify the wiki index
bash scripts/run_smoke_test.sh
bash scripts/run_mini_eval.sh
# run the 100-example eval command you standardize locally
# run the full reproduction command you standardize locally
```

## Important constraints

- Do not store datasets, models, outputs, or logs in GitHub.
- Keep the SeaKR algorithm unchanged.
- Keep prompts, thresholds, retrieval logic, pseudo-generation logic, hidden-state extraction, Gram determinant computation, and evaluation metrics unchanged.
- Use `SEAKR_TENSOR_PARALLEL_SIZE=1` on a single-GPU server and `SEAKR_TENSOR_PARALLEL_SIZE=2` when running a two-GPU paper-style reproduction.
