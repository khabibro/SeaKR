# Reproduction Checklist

Use this checklist for SeaKR reproduction.

## 1. Environment Setup

- Create a Python 3.10 environment.
- Install repository dependencies.
- Install modified `vllm_uncertainty` on Linux/CUDA.
- Verify `torch.cuda.is_available()`.
- Verify spaCy loads `en_core_web_sm`.

## 2. Model Access

- Confirm Hugging Face access to `meta-llama/Llama-2-7b-chat-hf`.
- Download the checkpoint into `$MODEL_DIR`.
- Verify tokenizer loading.

## 3. Dataset Download

- Download raw datasets into `$DATA_DIR/raw`.
- Extract or transform datasets into `$DATA_DIR/processed`.
- Build or verify the Elasticsearch BM25 Wikipedia index.

## 4. Small Test Run

- Start Elasticsearch.
- Run a tiny retrieval smoke test.
- Run a small SeaKR generation test.
- Verify JSONL output formatting.

## 5. Mini Reproduction Table

- Run limited examples for each target dataset.
- Save predictions under `$OUTPUT_DIR/predictions`.
- Save compact metrics under `$OUTPUT_DIR/tables`.
- Document any deviations from the paper.

## 6. Full Reproduction Run

- Run full HotpotQA/2WikiHop/IIRC experiments on Azure GPU.
- Preserve command lines and configs.
- Evaluate outputs with the repository notebooks or equivalent scripts.
- Report metrics without inventing or extrapolating results.
