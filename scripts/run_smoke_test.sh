#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${MODEL_DIR:-${ROOT}/models}/Llama-2-7b-chat-hf}"
DATASET_NAME="${DATASET_NAME:-hotpotqa}"
RETRIEVER_PORT="${RETRIEVER_PORT:-9201}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/smoke}"

mkdir -p "${OUTPUT_DIR}"

"${ROOT}/scripts/check_gpu_env.sh"

"${PYTHON_BIN}" "${ROOT}/scripts/seakr_one_gpu.py" run-smoke-test \
  --model-name-or-path "${MODEL_NAME_OR_PATH}" \
  --dataset-name "${DATASET_NAME}" \
  --retriever-port "${RETRIEVER_PORT}" \
  --output-dir "${OUTPUT_DIR}"

echo "smoke test outputs:"
ls -la "${OUTPUT_DIR}"
