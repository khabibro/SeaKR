#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data}"

mkdir -p "${DATA_DIR}/raw" "${DATA_DIR}/processed"

cat <<'MSG'
Dataset download placeholder.

Download datasets directly on the machine that will run experiments.
Suggested locations:
  raw files:       $DATA_DIR/raw
  processed files: $DATA_DIR/processed

Do not place datasets in Obsidian and do not commit large datasets to Git.
MSG
