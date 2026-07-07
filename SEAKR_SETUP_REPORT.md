# SeaKR Setup Report

Date: 2026-07-07

## Paper-Reproduction Target

- Paper: SeaKR: Self-aware Knowledge Retrieval for Adaptive Retrieval Augmented Generation, arXiv:2406.19215.
- Main model target: `meta-llama/Llama-2-7b-chat-hf`.
- SeaKR command model alias: `--served_model_name llama2-7b-chat`.
- Retrieval target: BM25 over Elasticsearch using DPR Wikipedia passages.
- Paper/code settings identified:
  - 10-shot prompting.
  - `N=3` retrieved passages.
  - `k=20` sampled responses for eigen/Gram uncertainty.
  - Retrieval trigger `eigen_threshold=-6.0`.
  - Intermediate layer: repository default `selected_intermediate_layer=15`; paper describes middle layer `l=L/2`.
  - HotpotQA command uses `max_reasoning_steps=7`, `max_docs=5`.

## Host Audit

- OS: macOS 26.5.1, arm64.
- Machine: MacBook Pro, Apple M3 Pro, 11 CPU cores, 18 GB RAM, 14-core Apple GPU.
- Disk after setup: 227 GiB free on the project volume.
- Homebrew: 6.0.8.
- Conda: 26.3.2.
- Git: 2.53.0.
- curl: 8.7.1.
- wget: installed 1.25.0.
- Java: OpenJDK 25.0.1 installed, but Elasticsearch uses its bundled JDK 19.0.2.
- CUDA/NVIDIA GPU: not present.

## Python Environment

- Conda env: `seakr`.
- Python: 3.10.20.
- pip dependency check: passed.
- Full package lockfile: `requirements-seakr-lock.txt`.
- Key packages:
  - `torch==2.3.0`
  - `transformers==4.40.2`
  - `tokenizers==0.19.1`
  - `beir==1.0.1`
  - `spacy==3.7.2`
  - `en-core-web-sm==3.7.1`
  - `faiss-cpu==1.14.3`
  - `pandas==2.3.3`
  - `sentence-transformers==2.7.0`
  - `elasticsearch==7.9.1`
  - `aiofiles==25.1.0`
  - `tenacity==9.1.4`
  - `gdown==6.1.0`

## vLLM Status

The modified `vllm_uncertainty` package was not installable on this host. Its `setup.py` fails immediately with:

```text
AssertionError: vLLM only supports Linux platform (including WSL).
```

This is a hard platform blocker for paper-faithful SeaKR generation on macOS. The paper reports experiments on a single NVIDIA 3090 24 GB GPU; this host has Apple Silicon/Metal, not CUDA.

## Downloaded Assets

- DPR Wikipedia passages:
  - `data/dpr/psgs_w100.tsv.gz`
  - `data/dpr/psgs_w100.tsv`
  - Extracted TSV line count: 21,015,325 including header.
- Elasticsearch:
  - Archive: `data/elasticsearch-7.17.9-darwin-aarch64.tar.gz`
  - Extracted: `data/elasticsearch-7.17.9`
  - Configured local endpoint: `127.0.0.1:9201`
  - Index: `wiki`
  - Final document count: 21,015,324
  - Final health: green after setting replicas to 0
- Multihop datasets:
  - `data/multihop_data/2wikimultihopqa/dev.json`
  - `data/multihop_data/hotpotqa/hotpotqa-dev.json`
  - `data/multihop_data/iirc/iirc_train_dev/dev.json`
  - Archive extracted as `multiphop_data`; renamed to expected `multihop_data`.
- Singlehop datasets:
  - `data/singlehop_data/nq_top10.json`
  - `data/singlehop_data/sq_top10.json`
  - `data/singlehop_data/tq_top10.json`
  - `data/singlehop_data/processed_nq.json`
  - `data/singlehop_data/processed_sq.json`
  - `data/singlehop_data/processed_tq.json`

## Disk Usage

- `data`: 30 GB
- `data/dpr`: 17 GB
- `data/elasticsearch-7.17.9`: 12 GB
- Elasticsearch index data: 12 GB
- `data/multihop_data`: 194 MB
- `data/singlehop_data`: 172 MB

## Verification

- `spacy.load("en_core_web_sm")`: passed.
- Python imports for torch, faiss, pandas, transformers, BEIR, Elasticsearch client, aiofiles, tenacity: passed.
- `pip check`: passed.
- Elasticsearch 7.17.9 startup: passed.
- BM25 retrieval via `SEAKR.retriever.BM25`: passed. Query returned Wikipedia passage ID `1` for an Aaron query.
- Dataset load checks:
  - HotpotQA: 7,405 rows.
  - 2WikiHop: 12,576 rows.
  - IIRC: 954 rows.
  - TriviaQA top10: 11,313 rows.
  - NQ top10: 3,610 rows.
  - SQuAD top10: 10,570 rows.

## Commands Needed To Resume

Start Elasticsearch:

```bash
cd "/Users/admin/Downloads/Research/Repos/Anchor Paper/SeaKR"
ES_JAVA_OPTS="-Xms4g -Xmx4g" data/elasticsearch-7.17.9/bin/elasticsearch
```

Verify index:

```bash
curl -sS "http://127.0.0.1:9201/_cat/count/wiki?v"
curl -sS "http://127.0.0.1:9201/_cluster/health/wiki?pretty"
```

Download the exact paper model after authenticating with Hugging Face:

```bash
conda activate seakr
huggingface-cli login
huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir models/Llama-2-7b-chat-hf
```

Full SeaKR inference requires Linux with CUDA and the modified vLLM installed.

## Final Component Status

- Python env: Ready.
- spaCy: Ready.
- Wikipedia DPR passages: Ready.
- Elasticsearch 7.17.9: Ready, stopped after verification.
- Wikipedia BM25 index: Ready.
- Multihop datasets: Ready.
- Singlehop datasets: Ready.
- Exact paper model: Needs Attention, gated Hugging Face access required.
- Modified vLLM: Failed on this macOS host; requires Linux/CUDA.
- HotpotQA full reproduction run: Not run because model and modified vLLM are unavailable on this host.
- Evaluation notebook: Not run because no generation outputs exist.

Overall status: Needs Attention. Retrieval/data environment is ready, but full paper reproduction is not possible on this Mac without an authenticated Llama-2 checkpoint and a Linux/CUDA machine for modified vLLM.
