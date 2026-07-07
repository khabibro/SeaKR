
  You are an expert ML engineer, MLOps engineer, Linux systems engineer, and AI research engineer.

  Your task is to finish the SeaKR paper reproduction setup on this Linux machine.

  Repository:

  /path/to/SeaKR

  Goal:

  Reproduce the SeaKR paper as faithfully as possible.

  Do not substitute models, datasets, software versions, or hyperparameters unless the original artifact is unavailable. If anything differs from the paper or
  repository, report it before proceeding.

  Important context from previous macOS setup:

  - The repo is SeaKR.
  - A setup report exists: SEAKR_SETUP_REPORT.md
  - A Python lockfile exists: requirements-seakr-lock.txt
  - Large generated files were intentionally not committed.
  - On macOS, the data/retrieval setup was completed, but modified vLLM could not run because vLLM requires Linux.
  - Linux should now complete the vLLM/model/inference/evaluation part.

  Paper-faithful target:

  - Paper: SeaKR: Self-aware Knowledge Retrieval for Adaptive Retrieval Augmented Generation
  - arXiv: 2406.19215
  - Exact model target: meta-llama/Llama-2-7b-chat-hf
  - SeaKR served model name: llama2-7b-chat
  - Retriever: BM25 over Elasticsearch
  - Elasticsearch version: 7.17.9
  - Wikipedia passages: DPR psgs_w100.tsv.gz
  - Index name: wiki
  - Default port: 9201
  - Main HotpotQA settings:
    - n_shot = 10
    - eigen_threshold = -6.0
    - max_reasoning_steps = 7
    - max_docs = 5
    - selected_intermediate_layer = 15 unless paper/code inspection shows otherwise
    - served_model_name = llama2-7b-chat

  Step 1: Inspect the repository

  Read:

  - README.md
  - SEAKR_SETUP_REPORT.md
  - requirements-seakr-lock.txt
  - build_wiki_index.py
  - main_multihop.py
  - main_simpleqa.py
  - SEAKR/*.py
  - vllm_uncertainty/setup.py
  - vllm_uncertainty/pyproject.toml
  - vllm_uncertainty/requirements-*.txt
  - evaluation notebooks

  Also read the SeaKR paper before making model or hyperparameter choices.

  Report any discrepancy between:
  - paper
  - README
  - code defaults
  - setup report

  Step 2: Audit the Linux machine

  Check and report:

  - OS and kernel
  - CPU
  - RAM
  - disk space
  - GPU model
  - NVIDIA driver
  - CUDA version
  - nvcc version
  - Python versions
  - Conda/Miniforge
  - pip
  - git
  - wget
  - curl
  - Java
  - Elasticsearch compatibility
  - Hugging Face login status

  Use status labels:

  - ✅ Installed
  - ⚠ Needs attention
  - ❌ Missing

  Step 3: Create or recreate the Conda environment

  If env `seakr` exists, inspect it first.

  Otherwise create:

  conda create -n seakr python=3.10 -y
  conda activate seakr

  Upgrade packaging tools:

  python -m pip install --upgrade pip setuptools wheel

  Install the repo/runtime dependencies.

  Prefer the versions in requirements-seakr-lock.txt where compatible with Linux/CUDA/vLLM.

  At minimum install:

  - beir==1.0.1
  - spacy==3.7.2
  - en_core_web_sm
  - aiofiles
  - tenacity
  - pandas
  - tqdm
  - transformers==4.40.2
  - tokenizers==0.19.1
  - torch==2.3.0 with the CUDA build appropriate for this machine
  - sentence-transformers==2.7.0
  - faiss-cpu unless GPU FAISS is explicitly needed
  - gdown

  Run:

  python -m pip check

  Verify imports:

  python - <<'PY'
  import spacy, torch, transformers, pandas, faiss, beir, elasticsearch
  nlp = spacy.load("en_core_web_sm")
  print("spacy", spacy.__version__, nlp.meta["version"])
  print("torch", torch.__version__, "cuda", torch.cuda.is_available())
  print("cuda version", torch.version.cuda)
  print("transformers", transformers.__version__)
  print("faiss", faiss.__version__)
  PY

  Step 4: Install modified vLLM

  Install the repository’s modified vLLM:

  cd vllm_uncertainty
  pip install -e .
  cd ..

  If it fails:

  - capture the exact error
  - identify whether it is CUDA, PyTorch, compiler, cmake, ninja, or dependency related
  - resolve safely
  - do not install upstream vLLM as a substitute unless explicitly explaining that it will not reproduce SeaKR uncertainty outputs

  Verify:

  python - <<'PY'
  import vllm
  from vllm import LLM, SamplingParams
  print("vllm", vllm.__version__ if hasattr(vllm, "__version__") else "imported")
  PY

  Step 5: Prepare data folders

  Expected structure:

  data/
    dpr/
      psgs_w100.tsv.gz
      psgs_w100.tsv
    elasticsearch-7.17.9/
    multihop_data/
      2wikimultihopqa/dev.json
      hotpotqa/hotpotqa-dev.json
      iirc/iirc_train_dev/dev.json
    singlehop_data/
      nq_top10.json
      sq_top10.json
      tq_top10.json
      processed_nq.json
      processed_sq.json
      processed_tq.json
  models/
    Llama-2-7b-chat-hf/

  If I copied archives from macOS, use them.

  If missing, download them.

  Step 6: Download DPR Wikipedia passages if missing

  If data/dpr/psgs_w100.tsv.gz does not exist:

  mkdir -p data/dpr
  wget -O data/dpr/psgs_w100.tsv.gz \
    https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz

  Verify gzip:

  gzip -t data/dpr/psgs_w100.tsv.gz

  Extract if TSV missing:

  gzip -dk data/dpr/psgs_w100.tsv.gz

  Verify:

  wc -l data/dpr/psgs_w100.tsv
  head -n 3 data/dpr/psgs_w100.tsv

  Expected line count:

  21015325 including header.

  Step 7: Download datasets if missing

  Install gdown if needed.

  Multihop:

  gdown 1xDqaPa8Kpnb95l7nHpwKWsBQUP9Ck7cn -O data/multihop_data.zip
  unzip -o data/multihop_data.zip -d data

  If it extracts as data/multiphop_data, rename:

  mv data/multiphop_data data/multihop_data

  Singlehop:

  gdown 1hn4Om_KkIGJpgG2wJjUu1mpPv9oq8M6G -O data/singlehop_data.zip
  unzip -o data/singlehop_data.zip -d data

  Verify loading:

  python - <<'PY'
  from SEAKR.dataset import HotpotQA, TwoWikiHop, IIRC
  import pandas as pd
  print("hotpot", len(HotpotQA(10).load_data()))
  print("twowiki", len(TwoWikiHop(10).load_data()))
  print("iirc", len(IIRC(10).load_data()))
  print("tq_top10", len(pd.read_json("data/singlehop_data/tq_top10.json")))
  print("nq_top10", len(pd.read_json("data/singlehop_data/nq_top10.json")))
  print("sq_top10", len(pd.read_json("data/singlehop_data/sq_top10.json")))
  PY

  Expected:
  - HotpotQA: 7405
  - 2WikiHop: 12576
  - IIRC: 954
  - TriviaQA top10: 11313
  - NQ top10: 3610
  - SQuAD top10: 10570

  Step 8: Install Elasticsearch 7.17.9 for Linux

  Use the Linux artifact, not the macOS artifact:

  cd data
  wget -O elasticsearch-7.17.9-linux-x86_64.tar.gz \
    https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.9-linux-x86_64.tar.gz

  tar -xzf elasticsearch-7.17.9-linux-x86_64.tar.gz
  cd ..

  Configure:

  Edit data/elasticsearch-7.17.9/config/elasticsearch.yml and append:

  cluster.name: seakr-local
  node.name: seakr-node-1
  path.data: data/es-data
  path.logs: logs
  network.host: 127.0.0.1
  http.port: 9201
  discovery.type: single-node
  xpack.security.enabled: false

  Start Elasticsearch:

  ES_JAVA_OPTS="-Xms4g -Xmx4g" data/elasticsearch-7.17.9/bin/elasticsearch

  Keep it running in one terminal/session.

  Verify from another terminal:

  curl -sS http://127.0.0.1:9201
  curl -sS "http://127.0.0.1:9201/_cluster/health?pretty"

  Step 9: Build or verify Wikipedia index

  If index `wiki` already exists, verify:

  curl -sS "http://127.0.0.1:9201/_cat/count/wiki?v"
  curl -sS "http://127.0.0.1:9201/_cluster/health/wiki?pretty"

  Expected count:

  21015324

  If missing or incomplete, rebuild:

  python build_wiki_index.py \
    --data_path data/dpr/psgs_w100.tsv \
    --index_name wiki \
    --port 9201

  After build:

  curl -sS -X POST "http://127.0.0.1:9201/wiki/_refresh"
  curl -sS -X PUT "http://127.0.0.1:9201/wiki/_settings" \
    -H "Content-Type: application/json" \
    -d '{"index":{"number_of_replicas":0}}'

  Verify count and health again.

  Test retriever:

  python - <<'PY'
  from SEAKR.retriever import BM25
  r = BM25(index_name="wiki", port=9201)
  ids, docs = r.retrieve(["Who was Aaron in the Bible?"], topk=3, disable_tqdm=True)
  print(ids.tolist())
  print(docs[0][0][:300].replace("\n", " "))
  PY

  Step 10: Download exact paper model

  Do not use Qwen, Mistral, Phi, Llama 3, quantized variants, or smaller substitutes.

  Use:

  meta-llama/Llama-2-7b-chat-hf

  First check auth:

  huggingface-cli whoami

  If not logged in:

  huggingface-cli login

  Then download:

  mkdir -p models
  huggingface-cli download meta-llama/Llama-2-7b-chat-hf \
    --local-dir models/Llama-2-7b-chat-hf

  Verify tokenizer/model files exist.

  Test tokenizer:

  python - <<'PY'
  from transformers import AutoTokenizer
  tok = AutoTokenizer.from_pretrained("models/Llama-2-7b-chat-hf")
  print(tok.__class__.__name__)
  print(tok.eos_token, tok.pad_token)
  PY

  Step 11: Patch only if necessary for Linux GPU topology

  Inspect main_multihop.py and main_simpleqa.py.

  They currently use:

  tensor_parallel_size=2
  worker_use_ray=True

  If the Linux machine has only one GPU, this will likely fail.

  Before patching, report the GPU count.

  If one GPU only, modify both scripts or add CLI flags so tensor_parallel_size can be set to 1 without changing paper logic.

  Do not change model, retrieval thresholds, prompts, or uncertainty method.

  Record any patch.

  Step 12: Run tiny end-to-end test

  Start Elasticsearch.

  Use the exact Llama-2-7B-Chat model.

  Run a tiny HotpotQA subset if possible. If the script does not support limiting examples, create a temporary test runner or minimally patch a `--limit`
  argument.

  Do not run the full dataset first.

  Verify:

  - model loads
  - modified vLLM produces `.uncertainty`
  - retrieval works
  - generation works
  - output JSONL is written
  - logs are created
  - no CUDA OOM occurs

  Step 13: Run HotpotQA reproduction

  Use the repository/paper command:

  python main_multihop.py \
    --n_shot 10 \
    --retriever_port 9201 \
    --dataset_name hotpotqa \
    --eigen_threshold -6.0 \
    --save_dir "outputs/hotpotqa" \
    --model_name_or_path models/Llama-2-7b-chat-hf \
    --served_model_name llama2-7b-chat \
    --max_reasoning_steps 7 \
    --max_docs 5

  If tensor_parallel_size needed adjustment for one GPU, use the adjusted CLI/config and document it.

  Save outputs under outputs/hotpotqa.

  Verify:

  - outputs/hotpotqa*/results.jsonl exists
  - outputs/hotpotqa*/failed.jsonl exists
  - args.txt exists
  - logs exist
  - JSONL rows are valid

  Step 14: Evaluation

  Open and inspect eval_multihop.ipynb.

  Run the evaluation for the generated HotpotQA results.

  If notebook execution needs conversion, use nbconvert or extract the evaluation logic into a script.

  Explain metrics:

  - Exact Match
  - F1
  - any dataset-specific metrics used by the notebook

  Show where the final numbers come from.

  Step 15: Final report

  Create or update:

  LINUX_REPRODUCTION_REPORT.md

  Include:

  - machine specs
  - GPU/CUDA/driver
  - exact Python env
  - exact vLLM install result
  - exact model checkpoint path
  - Hugging Face model revision if available
  - dataset verification
  - Elasticsearch version and index count
  - disk usage
  - commands executed
  - patches made
  - tiny test result
  - HotpotQA run result
  - evaluation result
  - deviations from paper, if any

  Final status per component:

  - ✅ Ready
  - ⚠ Needs Attention
  - ❌ Failed

  Do not claim full reproduction is complete unless the full HotpotQA run and evaluation completed successfully.