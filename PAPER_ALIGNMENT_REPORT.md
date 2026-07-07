# SeaKR Paper Alignment Report

This report compares the codebase against the official SeaKR arXiv paper, `arXiv:2406.19215`.

## Paper Settings Checked

- Backbone LLM: LLaMA-2-chat with 7B parameters.
- Retriever: BM25 using Elasticsearch.
- Knowledge source: English Wikipedia dump from December 20, 2018.
- Complex QA datasets: 2WikiMultiHopQA, HotpotQA, answerable IIRC.
- Simple QA datasets: NaturalQuestions, TriviaQA, SQuAD.
- Retrieved passages per retrieval step: `N = 3`.
- Hidden-state samples for uncertainty: `k = 20`.
- In-context examples: `10`.
- Hidden-state layer: middle layer, `l = L / 2`; for LLaMA-2-7B this corresponds to layer `15` in the existing code path.
- Metrics: EM and F1.
- Runtime stack noted by the paper: modified vLLM `0.4.2`, PyTorch `2.3.0`, Elasticsearch `7.17.9`.

## Codebase Matches

- `SEAKR/reasoner.py` keeps `n=20` pseudo-generations for self-aware uncertainty.
- `SEAKR/reasoner.py` retrieves `topk=3` passages for self-aware re-ranking.
- `SEAKR/reasoner.py` uses the regularized Gram determinant path through modified vLLM uncertainty outputs.
- `main_multihop.py` and `main_simpleqa.py` default to `selected_intermediate_layer=15`.
- `main_multihop.py` defaults to `n_shot=10`.
- README multihop commands use `max_reasoning_steps=7`, `max_docs=5`, and `eigen_threshold=-6.0`.
- `requirements.txt` pins PyTorch `2.3.0` and Transformer/tokenizer versions compatible with the bundled modified vLLM.
- `scripts/setup_university_env.sh` installs Elasticsearch client `7.9.1`; docs and README specify Elasticsearch service `7.17.9`.

## Paper-Alignment Changes Applied

- `main_multihop.py`
  - Added `--tensor_parallel_size`.
  - Default is GPU-count aware: uses `2` when at least two CUDA GPUs are visible, otherwise `1`.
  - Supports override through `SEAKR_TENSOR_PARALLEL_SIZE`.
  - Changed `--eigen_alpha` parser type from `int` to `float` so the paper/code value `1e-3` can be passed explicitly.

- `main_simpleqa.py`
  - Added `--tensor_parallel_size`.
  - Default is GPU-count aware with the same override behavior.

- `configs/university.yaml`
  - Changed `tensor_parallel_size` from hardcoded `1` to `${SEAKR_TENSOR_PARALLEL_SIZE:-auto}`.

- `scripts/run_paper_multihop.sh`
  - Added a paper-setting wrapper around `main_multihop.py`.
  - It uses `n_shot=10`, `eigen_threshold=-6.0`, `selected_intermediate_layer=15`, `max_reasoning_steps=7`, and `max_docs=5`.
  - It does not alter prompts, retrieval logic, uncertainty computation, or evaluation.

## Still Manual

- Download LLaMA-2-7B-chat into `MODEL_DIR`.
- Download the exact benchmark datasets.
- Download/build the December 20, 2018 Wikipedia BM25 index.
- Start Elasticsearch `7.17.9`.
- Run official notebook evaluation or a faithful script equivalent for final EM/F1.

## Strict Guardrails Added

- `configs/paper.yaml` records the exact paper reproduction manifest.
- `docs/paper_reproduction_requirements.md` explains the required model, data, runtime, retrieval corpus, and settings.
- `scripts/check_paper_reproduction_inputs.sh` validates local model/data paths and runtime versions before paper runs.

Use the strict check before any claimed paper-score reproduction:

```bash
bash scripts/check_paper_reproduction_inputs.sh
```

## Do Not Use For Final Paper Scores

- `scripts/run_smoke_test.sh`: one-example infrastructure test only.
- `scripts/run_mini_eval.sh`: 20-example limited check only.
- `scripts/seakr_one_gpu.py`: verification helper, not the official full paper entrypoint.
