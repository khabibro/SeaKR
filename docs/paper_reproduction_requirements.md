# Strict Paper Reproduction Requirements

Use this file when the goal is to reproduce the SeaKR paper numbers, not just run a smoke test.

## Source Of Truth

- Paper: `arXiv:2406.19215`
- Official code repository: `THU-KEG/SeaKR`

## Required Model

- `meta-llama/Llama-2-7b-chat-hf`
- Local path should normally be:

```bash
export MODEL_NAME_OR_PATH="$MODEL_DIR/Llama-2-7b-chat-hf"
export SERVED_MODEL_NAME="llama2-7b-chat"
```

Do not substitute Llama-3, Mistral, quantized checkpoints, base Llama-2, or instruct models for paper-score reproduction.

## Required Data

Complex QA:

- 2WikiMultiHopQA official development set used by IRCoT/SeaKR
- HotpotQA official development set used by IRCoT/SeaKR
- IIRC answerable development subset

Simple QA:

- NaturalQuestions
- TriviaQA
- SQuAD
- DPR-style top-10 files provided by the official SeaKR workflow

Retriever corpus:

- English Wikipedia dump from December 20, 2018
- DPR `psgs_w100.tsv`
- BM25 index name: `wiki`
- Elasticsearch service version: `7.17.9`

The official code expects dataset files under `data/`. On a university server, keep the real data outside Git and use symlinks or mounted paths so the official relative paths resolve without committing data.

## Required Runtime

- Linux
- NVIDIA CUDA GPU
- Python `3.10`
- PyTorch `2.3.0`
- Modified vLLM source in `vllm_uncertainty/`
- Elasticsearch `7.17.9`
- spaCy `en_core_web_sm`

## Required SeaKR Settings

- `n_shot=10`
- `selected_intermediate_layer=15`
- `eigen_alpha=1e-3`
- `eigen_threshold=-6.0`
- retrieved passages `N=3`
- pseudo-generations `k=20`
- multihop `max_reasoning_steps=7`
- multihop `max_docs=5`
- metrics: EM and F1

## Tensor Parallelism

For a two-GPU paper-style run:

```bash
export SEAKR_TENSOR_PARALLEL_SIZE=2
```

For a one-GPU university run:

```bash
export SEAKR_TENSOR_PARALLEL_SIZE=1
```

Changing tensor parallelism is an infrastructure change. Do not change prompts, thresholds, retrieval logic, pseudo-generation count, hidden-state extraction, Gram determinant computation, or evaluation metrics.

## Required Preflight

Run:

```bash
bash scripts/check_paper_reproduction_inputs.sh
```

This script does not download anything and does not run experiments. It only checks whether the model, datasets, official paths, and runtime imports are consistent with the paper reproduction requirements.
