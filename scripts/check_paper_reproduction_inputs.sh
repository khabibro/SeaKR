#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${MODEL_DIR:-${ROOT}/models}/Llama-2-7b-chat-hf}"
ES_PORT="${ES_PORT:-9201}"

fail() {
  echo "paper reproduction check failed: $*" >&2
  exit 1
}

check_file() {
  local path="$1"
  [[ -f "${path}" ]] || fail "missing file: ${path}"
}

check_dir() {
  local path="$1"
  [[ -d "${path}" ]] || fail "missing directory: ${path}"
}

echo "checking strict SeaKR paper reproduction inputs"

check_dir "${MODEL_NAME_OR_PATH}"
case "${MODEL_NAME_OR_PATH}" in
  *Llama-2-7b-chat-hf*) ;;
  *) fail "MODEL_NAME_OR_PATH must point to Llama-2-7b-chat-hf for paper reproduction: ${MODEL_NAME_OR_PATH}" ;;
esac

check_file "${MODEL_NAME_OR_PATH}/config.json"
check_file "${MODEL_NAME_OR_PATH}/tokenizer_config.json"

check_file "${ROOT}/data/dpr/psgs_w100.tsv"
check_file "${ROOT}/data/multihop_data/2wikimultihopqa/dev.json"
check_file "${ROOT}/data/multihop_data/hotpotqa/hotpotqa-dev.json"
check_file "${ROOT}/data/multihop_data/iirc/iirc_train_dev/dev.json"
check_file "${ROOT}/data/singlehop_data/processed_nq.json"
check_file "${ROOT}/data/singlehop_data/processed_tq.json"
check_file "${ROOT}/data/singlehop_data/processed_sq.json"
check_file "${ROOT}/data/singlehop_data/nq_top10.json"
check_file "${ROOT}/data/singlehop_data/tq_top10.json"
check_file "${ROOT}/data/singlehop_data/sq_top10.json"

"${PYTHON_BIN}" - <<PY
import json
import socket
import sys

required = {
    "torch": "2.3.0",
    "transformers": "4.40.2",
    "spacy": "3.7.2",
}

for module, expected in required.items():
    mod = __import__(module)
    actual = getattr(mod, "__version__", "")
    if actual != expected:
        raise SystemExit(f"{module} version must be {expected}, got {actual}")

import torch
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is False")

import spacy
spacy.load("en_core_web_sm")

from transformers import AutoTokenizer
AutoTokenizer.from_pretrained("${MODEL_NAME_OR_PATH}")

try:
    import vllm
    from vllm import LLM, SamplingParams
except Exception as exc:
    raise SystemExit(f"modified vLLM import failed: {exc}")

sock = socket.socket()
sock.settimeout(2)
try:
    sock.connect(("localhost", int("${ES_PORT}")))
except OSError as exc:
    raise SystemExit(f"Elasticsearch is not reachable on localhost:${ES_PORT}: {exc}")
finally:
    sock.close()

print("strict paper reproduction imports and paths: ok")
PY

echo "strict paper reproduction input check complete"
