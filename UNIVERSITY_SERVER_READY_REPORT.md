# University Server Ready Report

## Files created

- [docs/university_server_setup.md](file:///Users/admin/Downloads/Research/Repos/Anchor%20Paper/SeaKR/docs/university_server_setup.md)
- [docs/reproduction_checklist.md](file:///Users/admin/Downloads/Research/Repos/Anchor%20Paper/SeaKR/docs/reproduction_checklist.md)
- [docs/data_model_paths.md](file:///Users/admin/Downloads/Research/Repos/Anchor%20Paper/SeaKR/docs/data_model_paths.md)
- [configs/university.yaml](file:///Users/admin/Downloads/Research/Repos/Anchor%20Paper/SeaKR/configs/university.yaml)
- [scripts/check_gpu_env.sh](file:///Users/admin/Downloads/Research/Repos/Anchor%20Paper/SeaKR/scripts/check_gpu_env.sh)
- [scripts/setup_university_env.sh](file:///Users/admin/Downloads/Research/Repos/Anchor%20Paper/SeaKR/scripts/setup_university_env.sh)
- [scripts/run_paper_multihop.sh](file:///Users/admin/Downloads/Research/Repos/Anchor%20Paper/SeaKR/scripts/run_paper_multihop.sh)
- [UNIVERSITY_SERVER_READY_REPORT.md](file:///Users/admin/Downloads/Research/Repos/Anchor%20Paper/SeaKR/UNIVERSITY_SERVER_READY_REPORT.md)
- [PAPER_ALIGNMENT_REPORT.md](file:///Users/admin/Downloads/Research/Repos/Anchor%20Paper/SeaKR/PAPER_ALIGNMENT_REPORT.md)

## Files modified

- [.gitignore](file:///Users/admin/Downloads/Research/Repos/Anchor%20Paper/SeaKR/.gitignore)
- [scripts/run_smoke_test.sh](file:///Users/admin/Downloads/Research/Repos/Anchor%20Paper/SeaKR/scripts/run_smoke_test.sh)
- [scripts/run_mini_eval.sh](file:///Users/admin/Downloads/Research/Repos/Anchor%20Paper/SeaKR/scripts/run_mini_eval.sh)
- [scripts/seakr_one_gpu.py](file:///Users/admin/Downloads/Research/Repos/Anchor%20Paper/SeaKR/scripts/seakr_one_gpu.py)
- [main_multihop.py](file:///Users/admin/Downloads/Research/Repos/Anchor%20Paper/SeaKR/main_multihop.py)
- [main_simpleqa.py](file:///Users/admin/Downloads/Research/Repos/Anchor%20Paper/SeaKR/main_simpleqa.py)

## Current runtime state

- SeaKR entrypoints are GPU-count aware in `main_multihop.py` and `main_simpleqa.py`.
- Set `SEAKR_TENSOR_PARALLEL_SIZE=1` for one GPU or `SEAKR_TENSOR_PARALLEL_SIZE=2` for a two-GPU paper-style run.
- The SeaKR algorithm, prompts, retrieval logic, hidden-state extraction, Gram determinant logic, evaluation metrics, and pseudo-generation logic were not changed.
- Large artifacts remain outside GitHub by policy.

## Remaining manual steps

1. Clone or pull this repo on the university server.
2. Create the `seakr` conda environment.
3. Install dependencies.
4. Log in to Hugging Face.
5. Download `meta-llama/Llama-2-7b-chat-hf` into `$MODEL_DIR`.
6. Download the DPR Wikipedia dump and SeaKR datasets into `$DATA_DIR`.
7. Start Elasticsearch and build or verify the `wiki` index.
8. Run the GPU environment check.
9. Run the smoke test.
10. Run the 20-example mini eval.
11. Run the 100-example eval.
12. Run the full reproduction commands.

## Expected university machine requirements

- Linux
- NVIDIA CUDA GPU
- `nvidia-smi` available
- enough VRAM for Llama-2-7B-chat plus KV cache and vLLM overhead
- enough disk for model weights, datasets, Elasticsearch index data, and outputs
- conda or Miniforge available
- build toolchain for editable `vllm_uncertainty` install if wheels are not sufficient

## Risks

- Hugging Face gating for `Llama-2-7b-chat-hf`
- `vllm_uncertainty` build failures from missing CUDA toolchain or compiler packages
- GPU memory pressure on a 24 GB card during longer prompts or large cache usage
- Elasticsearch index build time and disk usage
- accidental storage of large artifacts inside GitHub

## Exact next commands

Run these on the university server after cloning the repo:

```bash
cd /path/to/SeaKR
bash scripts/setup_university_env.sh
huggingface-cli login
# download the model into $MODEL_DIR, and datasets into $DATA_DIR
# start Elasticsearch and verify the wiki index
bash scripts/check_gpu_env.sh
bash scripts/run_smoke_test.sh
bash scripts/run_mini_eval.sh
DATASET_NAME=hotpotqa bash scripts/run_paper_multihop.sh
python3 scripts/seakr_one_gpu.py run-mini-eval \
  --model-name-or-path "$MODEL_DIR/Llama-2-7b-chat-hf" \
  --dataset-name hotpotqa \
  --retriever-port 9201 \
  --output-dir "$OUTPUT_DIR/university_100" \
  --limit 100
# run the full HotpotQA / 2WikiHop / IIRC SeaKR commands from README.md
```

## Validation performed here

- Shell syntax checks passed for the new scripts.
- `python3 -m py_compile scripts/seakr_one_gpu.py` passed.
- No datasets or models were downloaded.
- No expensive SeaKR experiment was run.
