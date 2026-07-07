#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${ROOT}/models/Llama-2-7b-chat-hf}"
DATASET_NAME="${DATASET_NAME:-hotpotqa}"
RETRIEVER_PORT="${RETRIEVER_PORT:-9201}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is missing. This script expects an NVIDIA GPU machine." >&2
  exit 1
fi

nvidia-smi

"${PYTHON_BIN}" "${ROOT}/scripts/seakr_one_gpu.py" check-environment \
  --model-name-or-path "${MODEL_NAME_OR_PATH}" \
  --dataset-name "${DATASET_NAME}" \
  --retriever-port "${RETRIEVER_PORT}"
