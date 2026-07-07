#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${MODEL_DIR:-${ROOT}/models}/Llama-2-7b-chat-hf}"
DATASET_NAME="${DATASET_NAME:-hotpotqa}"
RETRIEVER_PORT="${RETRIEVER_PORT:-9201}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/mini}"

mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" "${ROOT}/scripts/seakr_one_gpu.py" run-mini-eval \
  --model-name-or-path "${MODEL_NAME_OR_PATH}" \
  --dataset-name "${DATASET_NAME}" \
  --retriever-port "${RETRIEVER_PORT}" \
  --output-dir "${OUTPUT_DIR}" \
  --limit 20

echo "mini eval outputs:"
ls -la "${OUTPUT_DIR}"
