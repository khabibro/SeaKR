#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${MODEL_DIR:-${ROOT}/models}/Llama-2-7b-chat-hf}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-llama2-7b-chat}"
DATASET_NAME="${DATASET_NAME:-hotpotqa}"
RETRIEVER_PORT="${RETRIEVER_PORT:-${ES_PORT:-9201}}"
RETRIEVER_HOST="${RETRIEVER_HOST:-${ES_HOST:-localhost}}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs}"
SAVE_DIR="${SAVE_DIR:-${OUTPUT_DIR}/${DATASET_NAME}}"
TABLE_DIR="${TABLE_DIR:-${OUTPUT_DIR}/tables}"
TENSOR_PARALLEL_ARG=()

if [[ -n "${SEAKR_TENSOR_PARALLEL_SIZE:-}" ]]; then
  TENSOR_PARALLEL_ARG=(--tensor_parallel_size "${SEAKR_TENSOR_PARALLEL_SIZE}")
fi

if [[ -e "${SAVE_DIR}" ]]; then
  echo "SAVE_DIR already exists: ${SAVE_DIR}" >&2
  echo "Set SAVE_DIR to a new path so metrics are generated for the intended run." >&2
  exit 1
fi

"${PYTHON_BIN}" "${ROOT}/main_multihop.py" \
  --n_shot 10 \
  --retriever_host "${RETRIEVER_HOST}" \
  --retriever_port "${RETRIEVER_PORT}" \
  --dataset_name "${DATASET_NAME}" \
  --eigen_threshold -6.0 \
  --save_dir "${SAVE_DIR}" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --served_model_name "${SERVED_MODEL_NAME}" \
  --selected_intermediate_layer 15 \
  --max_reasoning_steps 7 \
  --max_docs 5 \
  "${TENSOR_PARALLEL_ARG[@]}"

"${PYTHON_BIN}" "${ROOT}/scripts/evaluate_predictions.py" multihop \
  --dataset-name "${DATASET_NAME}" \
  --predictions "${SAVE_DIR}/results.jsonl" \
  --output-dir "${TABLE_DIR}"
