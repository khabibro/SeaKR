#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data}"
MODEL_DIR="${MODEL_DIR:-models}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
HF_HOME="${HF_HOME:-.cache/huggingface}"

export DATA_DIR MODEL_DIR OUTPUT_DIR HF_HOME

mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/predictions" "${OUTPUT_DIR}/tables"

cat <<'MSG'
Full evaluation placeholder.

Run full reproduction on a Linux CUDA GPU machine after mini checks pass.
Use configs/azure-full.yaml as the command manifest.
MSG
