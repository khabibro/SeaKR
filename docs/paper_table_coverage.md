# SeaKR Paper Table Coverage

Source of truth:

- Paper: `arXiv:2406.19215`, "SeaKR: Self-aware Knowledge Retrieval for Adaptive Retrieval Augmented Generation".
- Official code release: `https://github.com/THU-KEG/SeaKR`.

## What This Repository Can Reproduce

The official release contains the core SeaKR implementation, modified vLLM uncertainty path, BM25/Elasticsearch indexing, multihop inference, simple-QA inference, and notebook evaluation.

This checkout adds command-line equivalents for the evaluation notebooks:

- `scripts/evaluate_predictions.py multihop`
- `scripts/evaluate_predictions.py singlehop`
- `scripts/run_paper_multihop.sh`
- `scripts/run_paper_singlehop.sh`

These commands are enough to reproduce SeaKR rows for:

- Table 1: complex QA SeaKR row on `twowikihop`, `hotpotqa`, `iirc`.
- Table 2: simple QA SeaKR row on `nq`, `tq`, `sq`.

They also support table-ready metrics for:

- `Final Answer`
- `Final Step Answer`
- `Final Read Answer`

## What Is Not Fully Covered By The Official Release

The official GitHub release does not include standalone runners for every baseline and analysis condition reported in the paper:

- Table 1 baselines: CoT, IRCoT, Self-RAG, FLARE, DRAGIN.
- Table 2 baselines: CoT, Self-RAG, FLARE, DRAGIN.
- Table 3 ablations: prompt/perplexity/LN-entropy/energy uncertainty, no self-aware retrieval, no self-aware reranking, rationales-only, knowledge-only.
- Table 4 backbone comparison: LLaMA-2 base/chat and LLaMA-3 base/instruct.
- Table 5 qualitative case study requires matching the sampled HotpotQA example and logging intermediate retrieval/reranking details.

Those tables require either re-implementing baseline/ablation variants or obtaining additional author code/configuration. The current repository should not claim complete paper table reproduction until those runners exist and are verified.

## Server Test Structure

Use these checks in order on a Linux CUDA server:

```bash
bash scripts/check_paper_reproduction_inputs.sh
bash scripts/run_smoke_test.sh
DATASET_NAME=hotpotqa bash scripts/run_mini_eval.sh
```

Then run full SeaKR rows:

```bash
DATASET_NAME=twowikihop SAVE_DIR=outputs/twowikihop_llama2 bash scripts/run_paper_multihop.sh
DATASET_NAME=hotpotqa SAVE_DIR=outputs/hotpotqa_llama2 bash scripts/run_paper_multihop.sh
DATASET_NAME=iirc SAVE_DIR=outputs/iirc_llama2 bash scripts/run_paper_multihop.sh

DATASET_NAME=nq RUN_DIR=outputs/nq_llama2_7b bash scripts/run_paper_singlehop.sh
DATASET_NAME=tq RUN_DIR=outputs/tq_llama2_7b bash scripts/run_paper_singlehop.sh
DATASET_NAME=sq RUN_DIR=outputs/sq_llama2_7b bash scripts/run_paper_singlehop.sh
```

Metrics are written under `outputs/tables` as JSON, CSV, and Markdown.
