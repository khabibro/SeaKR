#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${MODEL_DIR:-${ROOT}/models}/Llama-2-7b-chat-hf}"
DATASET_NAME="${DATASET_NAME:-nq}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs}"
RUN_DIR="${RUN_DIR:-${OUTPUT_DIR}/${DATASET_NAME}_singlehop}"
TABLE_DIR="${TABLE_DIR:-${OUTPUT_DIR}/tables}"
TENSOR_PARALLEL_ARG=()

case "${DATASET_NAME}" in
  nq|tq|sq) ;;
  *) echo "DATASET_NAME must be one of: nq, tq, sq" >&2; exit 1 ;;
esac

if [[ -n "${SEAKR_TENSOR_PARALLEL_SIZE:-}" ]]; then
  TENSOR_PARALLEL_ARG=(--tensor_parallel_size "${SEAKR_TENSOR_PARALLEL_SIZE}")
fi

if [[ -e "${RUN_DIR}" ]]; then
  echo "RUN_DIR already exists: ${RUN_DIR}" >&2
  echo "Set RUN_DIR to a new path so metrics are generated for the intended run." >&2
  exit 1
fi

"${PYTHON_BIN}" "${ROOT}/main_simpleqa.py" \
  --dataset_name "${DATASET_NAME}" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --selected_intermediate_layer 15 \
  --output_dir "${RUN_DIR}" \
  "${TENSOR_PARALLEL_ARG[@]}"

"${PYTHON_BIN}" "${ROOT}/scripts/evaluate_predictions.py" singlehop \
  --dataset-name "${DATASET_NAME}" \
  --ground-truth "${ROOT}/data/singlehop_data/processed_${DATASET_NAME}.json" \
  --direct-predictions "${RUN_DIR}/direct.jsonl" \
  --rag-predictions "${RUN_DIR}/rag.jsonl" \
  --output-dir "${TABLE_DIR}" \
  --eigen-threshold -6.0
