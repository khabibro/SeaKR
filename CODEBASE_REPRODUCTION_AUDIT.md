# SeaKR Codebase Reproduction Audit

Audit target: SeaKR repository for reproducing "SeaKR: Self-aware Knowledge Retrieval for Adaptive Retrieval-Augmented Generation" on a university Linux + NVIDIA CUDA GPU machine.

Audit date: 2026-07-08

No models or datasets were downloaded. No full experiments were run. This audit only inspected code, docs, scripts, configs, and local repository state.

## A. Overall Readiness Score

Status: Almost ready

The repository has the core SeaKR implementation, modified `vllm_uncertainty`, dataset loaders, BM25/Elasticsearch retrieval, paper-style prompts, uncertainty computation, smoke/mini scripts, and university setup documentation. It is ready to move to a university GPU server for setup and smoke testing.

It is not fully ready for final paper-score reproduction because dataset paths are still hardcoded under `./data/...`, Elasticsearch host is hardcoded to localhost in retrieval/index code, and final EM/F1 evaluation is still notebook-based rather than standardized as a reproducible command.

## B. What Is Already Ready

- Official SeaKR entrypoints exist:
  - `main_multihop.py`
  - `main_simpleqa.py`
- Modified vLLM source exists:
  - `vllm_uncertainty/`
- Dependency files exist:
  - `requirements.txt`
  - `requirements-seakr-lock.txt`
- Dataset loading code exists:
  - `SEAKR/dataset.py`
- Retrieval code exists:
  - `SEAKR/retriever.py`
  - `build_wiki_index.py`
- BM25 / Elasticsearch support exists:
  - BEIR `BM25Search`
  - BEIR `ElasticSearch`
  - Elasticsearch index builder
- Evaluation code exists, but only as notebooks:
  - `eval_multihop.ipynb`
  - `eval_singlehop.ipynb`
- Config, script, and docs directories exist:
  - `configs/`
  - `scripts/`
  - `docs/`
- University setup files exist:
  - `docs/university_server_setup.md`
  - `docs/reproduction_checklist.md`
  - `docs/data_model_paths.md`
  - `configs/university.yaml`
  - `scripts/check_gpu_env.sh`
  - `scripts/setup_university_env.sh`
  - `scripts/run_smoke_test.sh`
  - `scripts/run_mini_eval.sh`

## C. What Is Missing

- No scripted final EM/F1 evaluation command.
  - Evaluation logic exists in notebooks.
  - A faithful command-line evaluator would make final reproduction easier to rerun.
- No 100-example evaluation script.
  - Docs mention a 100-example eval, but only the 20-example mini script exists.
- Dataset paths are not externalized in official loaders.
  - Multihop and single-hop loaders default to paths under `./data/...`.
  - This conflicts with the desired policy that datasets live outside the repo through `DATA_DIR`.
- Elasticsearch host is not configurable in official retrieval/index code.
  - `SEAKR/retriever.py` uses `hostname={"host": "localhost", "port": port}`.
  - `build_wiki_index.py` also uses localhost.
  - `ES_HOST` exists in config/docs but is not consumed by the official code.
- Reproducibility metadata is incomplete.
  - Python version is documented as 3.10, but not pinned through an environment file.
  - Hardware info is checked by scripts, but not automatically saved into experiment output folders.
  - Exact full-run commands are in README, but not saved per dataset as scripts/manifests.
  - Random seed is set in vLLM sampling params, but there is no repository-level seed manifest.
- Resume support is partial.
  - `main_multihop.py` writes failed examples and pickled reasoning checkpoints on error.
  - There is no documented or scripted resume command that replays only failed/missing examples.

## D. What Is Risky

- `worker_use_ray=True` remains in `main_multihop.py`.
  - This follows the async/multihop vLLM path used by the codebase.
  - On a single GPU, Ray installation and initialization still need to be verified.
- Dataset artifacts exist locally under `data/` on this Mac.
  - Git tracks only placeholder README files under `data/`, `models/`, and `outputs`.
  - Local `data/` is about 30 GB and ignored by Git.
  - Before pushing, verify no large artifacts are staged.
- Evaluation is notebook-based.
  - This is acceptable for inspection, but final reproduction should have scripted evaluation or a documented notebook execution protocol.
- `requirements-seakr-lock.txt` appears to be captured from a local environment and includes at least one non-portable file URL for `packaging`.
  - Use it as a record, not as the primary installation source.

## E. What Might Change Paper Scores

These should remain fixed for final reproduction:

- Model checkpoint:
  - Use `meta-llama/Llama-2-7b-chat-hf` or the exact local equivalent.
  - Changing model family, chat template, or weights will affect EM/F1.
- Tokenizer:
  - Official code uses `AutoTokenizer.from_pretrained(args.model_name_or_path)`.
  - Tokenizer must match the model checkpoint.
- Prompts:
  - Prompt templates live in `SEAKR/dataset.py`.
  - Audit did not change prompts.
- Pseudo-generation count:
  - `n=20` appears in `SEAKR/reasoner.py` and `main_simpleqa.py`.
  - Audit did not change this.
- `eigen_threshold=-6.0`:
  - Present in `main_multihop.py` and configs.
  - Must be explicitly set in final commands if not relying on defaults.
- `selected_intermediate_layer=15`:
  - Present in official entrypoints and configs.
  - Changing it can affect hidden states, eigen score, retrieval triggering, and final answers.
- Retrieved passages per retrieval call:
  - `SEAKR/reasoner.py` retrieves `topk=3`.
  - This is unchanged.
- 10-shot prompting:
  - Dataset classes use `n_shot`; official parser default is 10.
- `max_reasoning_steps`:
  - README commands use 7.
  - `main_multihop.py` parser default is 10.
  - Final reproduction should pass `--max_reasoning_steps 7` explicitly if that is the target setting.
- `max_docs`:
  - README commands use 5.
  - `main_multihop.py` parser default is 5.
- Decoding parameters:
  - Greedy path: `n=1`, `temperature=0.0`, `top_p=1.0`, `max_tokens=100`, `seed=42`.
  - Sample path: `n=20`, `temperature=1.0`, `top_k=50`, `top_p=0.9`, `max_tokens=100`, `seed=42`.
  - Audit did not change these.
- Evaluation:
  - Official paper-score evaluation should use `eval_multihop.ipynb` / `eval_singlehop.ipynb` or a faithful scripted equivalent.
  - The 20-example mini script computes simple EM/F1 for smoke-level validation only; do not use it as final paper evaluation.

## F. Whether 1-GPU Mode Is Safe

1-GPU mode is likely safe for smoke tests and likely safe for algorithm-faithful reproduction if the model fits in VRAM.

Reasoning:

- Tensor parallelism changes how the model is partitioned, not SeaKR prompts, retrieval logic, pseudo-generation count, uncertainty formula, or evaluation metrics.
- The modified vLLM code computes uncertainty from output embeddings and log probabilities after generation.
- `tensor_parallel_size=1` should preserve the mathematical algorithm, but it may change low-level floating point behavior and throughput compared with multi-GPU tensor parallel execution.

Risk:

- A single 24 GB GPU may be tight for Llama-2-7B-chat with vLLM KV cache, `n=20` sampling, and long multihop prompts.
- Memory pressure can cause failures or force smaller concurrency, but reducing pseudo-generations or changing decoding would affect reproduction quality and should not be done for final scoring.

## G. Whether Multi-GPU Mode Is Supported

The bundled `vllm_uncertainty` supports tensor parallelism and distributed executors through vLLM.

Current official SeaKR entrypoints are GPU-count aware:

- `main_multihop.py` exposes `--tensor_parallel_size`.
- `main_simpleqa.py` exposes `--tensor_parallel_size`.
- If `SEAKR_TENSOR_PARALLEL_SIZE` is set, that value is used.
- Otherwise, the entrypoints default to `2` when at least two CUDA GPUs are visible and `1` when fewer than two are visible.
- `configs/university.yaml` records `tensor_parallel_size: "${SEAKR_TENSOR_PARALLEL_SIZE:-auto}"`.

Result:

- 1 GPU: supported by current code path, subject to VRAM.
- 2 GPUs: supported through `--tensor_parallel_size 2` or `SEAKR_TENSOR_PARALLEL_SIZE=2`.
- More than 2 GPUs: vLLM can support tensor parallel sizes greater than 2 if the model architecture supports the split; set `--tensor_parallel_size` intentionally after verification.

The tensor parallel fix is infrastructure-only: it forwards an existing vLLM parameter and does not change prompts, retrieval, uncertainty, or evaluation.

## H. Exact Commands To Run On The University Server

Set paths first:

```bash
export DATA_DIR=/path/to/datasets
export MODEL_DIR=/path/to/models
export OUTPUT_DIR=/path/to/outputs
export HF_HOME=/path/to/huggingface-cache
export ES_HOST=localhost
export ES_PORT=9201
export MODEL_NAME_OR_PATH="$MODEL_DIR/Llama-2-7b-chat-hf"
```

Clone and install:

```bash
git clone <repo-url>
cd SeaKR
bash scripts/setup_university_env.sh
conda activate seakr
huggingface-cli login
```

Download model and datasets manually outside Git:

```bash
# Download meta-llama/Llama-2-7b-chat-hf into $MODEL_DIR.
# Download SeaKR/DRAGIN multihop datasets into the expected dataset layout.
# Download DPR Wikipedia passages into $DATA_DIR/dpr or a chosen external path.
```

Start Elasticsearch and build the BM25 index:

```bash
ES_JAVA_OPTS="-Xms4g -Xmx4g" $DATA_DIR/elasticsearch-7.17.9/bin/elasticsearch
python build_wiki_index.py \
  --data_path "$DATA_DIR/dpr/psgs_w100.tsv" \
  --index_name wiki \
  --port "$ES_PORT"
```

Verify GPU/model/vLLM:

```bash
bash scripts/check_gpu_env.sh
```

Run one-example smoke test:

```bash
bash scripts/run_smoke_test.sh
```

Run 20-example mini eval:

```bash
bash scripts/run_mini_eval.sh
```

Run official multihop entrypoint after smoke tests pass:

```bash
python main_multihop.py \
  --n_shot 10 \
  --retriever_port "$ES_PORT" \
  --dataset_name hotpotqa \
  --eigen_threshold -6.0 \
  --save_dir "$OUTPUT_DIR/hotpotqa" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --served_model_name llama2-7b-chat \
  --selected_intermediate_layer 15 \
  --max_reasoning_steps 7 \
  --max_docs 5
```

Repeat with `--dataset_name twowikihop` and `--dataset_name iirc` after HotpotQA is stable. You can also use `scripts/run_paper_multihop.sh` with `DATASET_NAME` set to the target dataset.

## I. Required University Machine Specs

Minimum for smoke tests:

- Linux x86_64
- NVIDIA CUDA GPU
- CUDA-compatible PyTorch/vLLM stack
- Python 3.10 conda environment
- 24 GB VRAM for a realistic Llama-2-7B-chat single-GPU attempt
- 64 GB system RAM recommended
- 150 GB free disk minimum for model, datasets, Elasticsearch, and outputs
- `nvidia-smi`, `conda`, `gcc/g++`, `cmake`, and build tools

Recommended for paper-style reproduction:

- 2 or more NVIDIA GPUs if preserving original tensor parallel behavior
- At least 24 GB VRAM per GPU, more is better
- 128 GB system RAM preferred
- 250 GB or more free disk for datasets, indexes, model cache, logs, predictions, and repeated runs
- Stable local storage or mounted scratch storage for Elasticsearch

## J. Next Recommended Step

Move the repo to the university server and run setup plus GPU verification first.

Before final paper-score reproduction, add a safe CLI/config bridge for `tensor_parallel_size` so the same code can run:

- `tensor_parallel_size=1` on a one-GPU machine
- `tensor_parallel_size=2` on a two-GPU paper-style machine
- larger values only when intentionally verified

## Setup Script Audit

- GPU environment check: present as `scripts/check_gpu_env.sh`.
- Dependency installation: present as `scripts/setup_university_env.sh`.
- Smoke test: present as `scripts/run_smoke_test.sh`.
- Mini evaluation: present as `scripts/run_mini_eval.sh`.
- Full reproduction: missing as an executable script.

## Docs Audit

- University server setup: present.
- Dataset/model paths: present.
- Reproduction checklist: present.
- Elasticsearch setup: partially documented in README and setup docs.
- Hugging Face model access: documented at checklist level.
- How to resume experiments: missing.
- How to save outputs: partially documented through `OUTPUT_DIR` and README commands.

## Path Audit

- `DATA_DIR`: documented/configured, but official loaders still use hardcoded `./data/...` defaults.
- `MODEL_DIR`: documented/configured, used by helper scripts.
- `OUTPUT_DIR`: documented/configured, used by helper scripts.
- `HF_HOME`: documented/configured.
- `ES_HOST`: documented/configured, not consumed by official retrieval/index code.
- `ES_PORT`: documented/configured, used as `retriever_port` by official command line.
- Model path: official entrypoints require `--model_name_or_path`.
- Dataset path: official multihop loaders do not expose CLI dataset paths.
- Output path: official multihop uses `--save_dir`; simple QA uses `--output_dir`.

## .gitignore Audit

Required exclusions are present:

- `data/`
- `models/`
- `outputs/`
- `logs/`
- `.cache/`
- `*.pt`
- `*.bin`
- `*.safetensors`
- `*.ckpt`
- `*.gguf`
- `__pycache__/`
- `.env`

Git currently tracks only placeholder README files under `data/`, `models/`, and `outputs`. Large local files under `data/` are ignored and not tracked.

## Validation Performed

- Python syntax check passed:
  - `main_multihop.py`
  - `main_simpleqa.py`
  - `SEAKR/reasoner.py`
  - `SEAKR/dataset.py`
  - `SEAKR/retriever.py`
  - `scripts/seakr_one_gpu.py`
- Shell syntax check passed:
  - `scripts/check_gpu_env.sh`
  - `scripts/setup_university_env.sh`
  - `scripts/run_smoke_test.sh`
  - `scripts/run_mini_eval.sh`
  - `scripts/download_datasets.sh`

## SAFE INFRASTRUCTURE IMPROVEMENTS

Proposed, not applied in this audit:

- Add command-line EM/F1 evaluators equivalent to the notebooks.
  - Classification: SAFE if metrics exactly match the notebooks.
  - Expected score impact: none.
- Add `--data_dir` or environment-variable support for dataset paths.
  - Classification: SAFE if defaults remain unchanged.
  - Expected score impact: none if data contents are identical.
- Add `--es_host` support to `SEAKR/retriever.py` and `build_wiki_index.py`.
  - Classification: SAFE if default remains `localhost`.
  - Expected score impact: none.
- Add `scripts/run_full_reproduction.sh`.
  - Classification: SAFE if it only wraps documented commands.
  - Expected score impact: none.
- Add `scripts/run_100_eval.sh`.
  - Classification: SAFE if clearly labeled as limited evaluation and does not change final settings.
  - Expected score impact: none for final reproduction.
- Save `nvidia-smi`, Python/package versions, git commit, and command line into each output folder.
  - Classification: SAFE.
  - Expected score impact: none.
- Document a resume procedure using `results.jsonl`, `failed.jsonl`, and `reasoning_ckpt/`.
  - Classification: SAFE.
  - Expected score impact: none if resumed examples are handled consistently.

## Files Modified By This Audit

- `CODEBASE_REPRODUCTION_AUDIT.md`
- `PAPER_ALIGNMENT_REPORT.md`
- `main_multihop.py`
- `main_simpleqa.py`
- `configs/university.yaml`
- `scripts/run_paper_multihop.sh`
- `README.md`
- `docs/university_server_setup.md`
- `docs/reproduction_checklist.md`

The code changes are infrastructure-only. They expose tensor parallelism and correct CLI parsing for `eigen_alpha`; they do not alter SeaKR prompts, retrieval logic, pseudo-generation count, hidden-state extraction, Gram determinant computation, thresholds, or evaluation metrics.
