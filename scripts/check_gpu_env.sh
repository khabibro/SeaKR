#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${MODEL_DIR:-${ROOT}/models}/Llama-2-7b-chat-hf}"
DATASET_NAME="${DATASET_NAME:-hotpotqa}"
RETRIEVER_PORT="${RETRIEVER_PORT:-9201}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi: missing" >&2
  exit 1
fi

echo "nvidia-smi: ok"
nvidia-smi

echo "gpu summary:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "python version:"
"${PYTHON_BIN}" --version

echo "torch cuda:"
"${PYTHON_BIN}" - <<'PY'
import os
import shutil
import torch

print("torch_version:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_version:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu_name:", torch.cuda.get_device_name(0))
    print("vram_gb:", round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2))
else:
    raise SystemExit("torch.cuda.is_available() is False")

print("disk_space:")
usage = shutil.disk_usage(".")
print("  free_gb:", round(usage.free / (1024**3), 2))
print("ram:")
with open("/proc/meminfo", "r", encoding="utf-8") as fh:
    for line in fh:
        if line.startswith("MemTotal:") or line.startswith("MemAvailable:"):
            print(" ", line.strip())
PY

echo "disk space:"
df -h .

echo "ram:"
free -h

"${PYTHON_BIN}" "${ROOT}/scripts/seakr_one_gpu.py" check-environment \
  --model-name-or-path "${MODEL_NAME_OR_PATH}" \
  --dataset-name "${DATASET_NAME}" \
  --retriever-port "${RETRIEVER_PORT}"

echo "gpu env check complete"
