# Azure Readiness Report

Date: 2026-07-07

## Verdict

SeaKR is now prepared for a single-GPU Azure run at the code level.

The two SeaKR entrypoints are on `tensor_parallel_size=1`, and the new verification scripts use the direct one-GPU vLLM path. The SeaKR algorithm, prompts, retrieval logic, evaluation logic, and uncertainty math were not changed.

## Single GPU Compatibility

Yes, with one important condition:

- the Azure machine must have a working NVIDIA CUDA stack,
- the model checkpoint must already be present locally,
- Elasticsearch must be running for the multihop pipeline.

The vendored `vllm_uncertainty` fork still contains distributed/tensor-parallel infrastructure, but it already has `world_size == 1` bypasses and is designed to collapse to a single rank.

## Expected GPU Memory

Estimate only, based on code inspection:

- Llama-2-7B-chat weights in fp16/bf16: roughly 14 GB
- KV cache, activations, and vLLM runtime overhead: several additional GB
- expected peak on a smoke test: about 18-22 GB

That should fit on a 24 GB L4, but the headroom is not large. The exact peak depends on prompt length, cache allocation, and how many outputs are sampled per request.

## Expected Runtime

Estimate only:

- environment check: minutes
- one-example smoke test: tens of minutes, depending on model warmup and retrieval latency
- 20-example mini eval: several hours

The one-GPU setup will be slower than the original 2-GPU configuration, but it should be materially simpler to operate on Azure.

## Remaining Blockers

1. Model access
   - The checkpoint `meta-llama/Llama-2-7b-chat-hf` still needs to be available locally on Azure.

2. Retrieval service
   - The multihop path still requires Elasticsearch with the wiki index at `localhost:9201`.

3. Hardware headroom
   - If the L4 OOMs, the first likely cause is KV cache pressure, not SeaKR math.

4. Validation on the actual Azure VM
   - The repository is prepared, but the one-GPU pipeline still needs to be exercised on the target VM to confirm there are no host-specific CUDA or driver issues.

## Files Changed

- `main_multihop.py`
- `main_simpleqa.py`
- `MULTI_GPU_ANALYSIS.md`
- `AZURE_READINESS_REPORT.md`
- `scripts/seakr_one_gpu.py`
- `scripts/check_environment.sh`
- `scripts/run_smoke_test.sh`
- `scripts/run_mini_eval.sh`

## Safe Modifications

- `main_multihop.py`: `tensor_parallel_size=1`
- `main_simpleqa.py`: `tensor_parallel_size=1`
- `scripts/seakr_one_gpu.py`: single-GPU verification harnesses only
- `scripts/check_environment.sh`: environment and model-load checks only
- `scripts/run_smoke_test.sh`: one-example pipeline validation
- `scripts/run_mini_eval.sh`: 20-example mini evaluation wrapper

These changes do not alter prompts, retrieval logic, hyperparameters, uncertainty math, or evaluation definitions.

## Risky Modifications Not Applied

- changing prompts
- changing retrieval thresholds
- changing `eigen_alpha`
- changing `max_reasoning_steps`
- changing `max_docs`
- reducing pseudo-generation count
- quantizing the model
- rewriting the uncertainty computation
- changing the evaluation code

I also did not change the underlying vLLM distributed implementation. It already has a single-GPU path, so the safer move is to use it rather than rewrite it.

## Recommended Azure VM

Use a Linux GPU VM with:

- one NVIDIA L4 GPU
- 24 GB VRAM
- Ubuntu 22.04 or similar
- CUDA 12.x runtime compatibility
- enough disk for the model, datasets, and Elasticsearch index

On Azure, choose the closest current NCads / N-series SKU in your region that exposes a single L4 and 24 GB memory. The exact SKU name varies by region and inventory, so confirm it in the portal before provisioning.

## Practical Run Order

1. Run `scripts/check_environment.sh`
2. Start Elasticsearch
3. Run `scripts/run_smoke_test.sh`
4. Run `scripts/run_mini_eval.sh`
5. Only then move to the full SeaKR runs

