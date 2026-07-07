#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data}"
MODEL_DIR="${MODEL_DIR:-models}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
HF_HOME="${HF_HOME:-.cache/huggingface}"

export DATA_DIR MODEL_DIR OUTPUT_DIR HF_HOME

mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/predictions" "${OUTPUT_DIR}/tables"

cat <<'MSG'
Mini evaluation placeholder.

Before running:
  1. Start Elasticsearch.
  2. Verify the wiki index exists.
  3. Verify the target model exists under $MODEL_DIR.

Use configs/local-mini.yaml as the command manifest.
MSG
