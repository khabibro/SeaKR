#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-seakr}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONDA_BIN="${CONDA_BIN:-conda}"

if ! command -v "${CONDA_BIN}" >/dev/null 2>&1; then
  echo "conda is missing" >&2
  exit 1
fi

source "$("${CONDA_BIN}" info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

conda activate "${ENV_NAME}"

"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
"${PYTHON_BIN}" -m pip install -r "${ROOT}/requirements.txt"
"${PYTHON_BIN}" -m pip install elasticsearch==7.9.1 pytrec_eval
"${PYTHON_BIN}" -m pip install -e "${ROOT}/vllm_uncertainty"
"${PYTHON_BIN}" -m spacy download en_core_web_sm

"${PYTHON_BIN}" - <<'PY'
import spacy
import torch
import transformers
import pandas
import elasticsearch
import faiss
from vllm import LLM, SamplingParams

nlp = spacy.load("en_core_web_sm")
print("spacy:", spacy.__version__, nlp.meta["version"])
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("pandas:", pandas.__version__)
print("elasticsearch:", elasticsearch.__version__)
print("faiss:", faiss.__version__)
print("vllm:", getattr(__import__("vllm"), "__version__", "imported"))
print("LLM and SamplingParams import: ok")
PY

echo "university env setup complete"
