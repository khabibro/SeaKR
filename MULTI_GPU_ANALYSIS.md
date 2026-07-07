# Multi-GPU Analysis

Scope: inspect every place in the SeaKR repository that either hardcodes, documents, or structurally depends on more than one GPU. The goal is to separate true migration blockers from normal tensor-parallel infrastructure that already collapses to `world_size=1`.

## Summary

- The SeaKR entrypoints are already set to `tensor_parallel_size=1` in `main_multihop.py` and `main_simpleqa.py`.
- The remaining multi-GPU logic is mostly inside the vendored `vllm_uncertainty` fork. That code is tensor-parallel aware, but it already contains explicit `world_size == 1` bypasses and single-GPU paths.
- The real migration risk is not the SeaKR algorithm. It is accidental use of a Ray / distributed executor path or a test-only distributed case during validation.

## Findings

| File | Line(s) | Why it exists | Required? | Can safely become single GPU? | Confidence |
|---|---:|---|---|---|---|
| `main_multihop.py` | 109-120 | Builds `AsyncEngineArgs` for SeaKR multihop inference. `tensor_parallel_size=1` is already set, while `worker_use_ray=True` is a distributed-engine toggle. | Yes, for runtime. | Yes. `tp=1` is already correct. `worker_use_ray` is not needed on the one-GPU path. | High |
| `main_simpleqa.py` | 121-128 | Builds the single-hop `LLM` for the SeaKR direct/RAG evaluation path. `tensor_parallel_size=1` is already set. | Yes, for runtime. | Yes. It already matches one GPU. | High |
| `configs/local-mini.yaml` | 1-17 | Mini run manifest. The model config already says `tensor_parallel_size: 1`. | No, this is config documentation. | Yes. Already single GPU. | High |
| `configs/azure-full.yaml` | 1-17 | Azure run manifest. The model config already says `tensor_parallel_size: 1`. | No, this is config documentation. | Yes. Already single GPU. | High |
| `prompt.md` | 369-378 | Old guidance text that still mentions `tensor_parallel_size=2` and `worker_use_ray=True`. | No, documentation only. | Yes, but it should be treated as stale guidance rather than runtime truth. | High |
| `vllm_uncertainty/vllm/engine/arg_utils.py` | 231-252 | Exposes `--distributed-executor-backend`, `--worker-use-ray`, and `--tensor-parallel-size`. This is the main runtime knob set for vLLM. | Yes, but only as general infrastructure. | Yes. The parser already supports `tensor_parallel_size=1`. | High |
| `vllm_uncertainty/vllm/engine/async_llm_engine.py` | 313-390, 448-498, 507-525 | Selects Ray vs direct execution and has a special branch for `tensor_parallel_size == 1`. | Yes, for async inference. | Yes. The direct single-GPU branch is already present and does not require Ray. | High |
| `vllm_uncertainty/vllm/engine/llm_engine.py` | 166-172, 252-253, 340-360 | Logs `tensor_parallel_size`, stores usage stats, and selects the executor backend. | Yes, for engine startup. | Yes. Single-GPU execution is already supported through the default GPU executor. | High |
| `vllm_uncertainty/vllm/distributed/parallel_state.py` | 80-124, 128-252 | Initializes process groups, tensor-parallel groups, and pipeline-parallel groups. This is the core distributed runtime. | Yes, but only for the vLLM fork. | Yes, if `world_size=1` and `tensor_parallel_size=1`. The code is written to create 1-rank groups. | High |
| `vllm_uncertainty/vllm/distributed/communication_op.py` | 73-195, 227-317 | Implements collectives, gather, and broadcast. It explicitly bypasses work when `world_size == 1`. | Yes, as shared infrastructure. | Yes. This file is already safe for one GPU. | High |
| `vllm_uncertainty/vllm/worker/worker.py` | 25-116, 229-240 | GPU worker that initializes CUDA, distributed state, and model execution. | Yes, for GPU inference. | Yes, if the engine is launched with a single GPU. | Medium-High |
| `vllm_uncertainty/vllm/worker/cpu_worker.py` | 113-326 | CPU worker path that still initializes distributed environment and model-parallel groups. | No for Azure L4 SeaKR. | Yes, but this is not on the SeaKR GPU path. | Medium |
| `vllm_uncertainty/vllm/model_executor/layers/linear.py` | 223-237, 296, 331, 404, 501, 704 | Tensor-parallel-aware linear layers divide shapes by TP world size. | Yes, for model execution. | Yes. All these divisions reduce cleanly to a single partition when TP=1. | High |
| `vllm_uncertainty/vllm/model_executor/layers/activation.py` | 118 | Uses tensor-parallel world size for partition-aware activation handling. | Yes, for model execution. | Yes. TP=1 is valid. | High |
| `vllm_uncertainty/vllm/model_executor/layers/vocab_parallel_embedding.py` | 29-30, 65 | Computes partitioned vocab ranges from TP world size. | Yes, for model execution. | Yes. TP=1 yields an unpartitioned embedding. | High |
| `vllm_uncertainty/vllm/model_executor/layers/logits_processor.py` | 8 | Gathers logits across tensor-parallel ranks. | Yes, for model execution. | Yes. The gather path is a no-op under TP=1. | High |
| `vllm_uncertainty/vllm/model_executor/models/*.py` | Multiple, see notes below | Model classes compute per-partition head counts and KV head counts from TP world size. | Yes, but only as model executor infrastructure. | Yes, these classes are designed to work with TP=1. | High |
| `vllm_uncertainty/docs/source/serving/distributed_serving.rst` | 6-38 | vLLM docs explaining multi-GPU and multi-machine distributed serving. | No, docs only. | Yes, but not relevant to SeaKR runtime. | High |
| `vllm_uncertainty/benchmarks/kernels/benchmark_aqlm.py` | 14 | Pins `CUDA_VISIBLE_DEVICES='0'` for isolated benchmark runs. | No, benchmark-only. | Yes, but it is not part of SeaKR inference. | High |
| `vllm_uncertainty/benchmarks/kernels/benchmark_mixtral_moe.py` | 14, 30 | Pins `CUDA_VISIBLE_DEVICES='0'` and uses `tp_size = 2` for kernel tuning. | No, benchmark-only. | Yes, but it is unrelated to SeaKR reproduction. | High |
| `vllm_uncertainty/tests/distributed/test_basic_distributed_correctness.py` | 52 | Explicit `tensor_parallel_size=2` distributed correctness test. | No, test-only. | Yes, but it should remain a multi-GPU test. | High |
| `vllm_uncertainty/tests/distributed/test_chunked_prefill_distributed.py` | 55 | Explicit `tensor_parallel_size=2` distributed correctness test. | No, test-only. | Yes, but it should remain a multi-GPU test. | High |
| `vllm_uncertainty/tests/tensorizer_loader/test_tensorizer.py` | 256 | Uses `tensor_parallel_size=2` in a tensorizer loading test. | No, test-only. | Yes, but not needed for Azure single-GPU SeaKR. | High |
| `vllm_uncertainty/tests/lora/test_quant_model.py` | 172 | Uses `tensor_parallel_size=2` for LoRA + quantized model coverage. | No, test-only. | Yes, but not part of SeaKR runtime. | High |
| `vllm_uncertainty/tests/lora/test_llama.py` | 103, 115 | Uses `tensor_parallel_size=2` and `4` to compare LoRA behavior across TP sizes. | No, test-only. | Yes, but these are intended distributed tests. | High |
| `vllm_uncertainty/tests/lora/test_baichuan.py` | 87, 101 | Uses `tensor_parallel_size=2` and `4` in LoRA coverage. | No, test-only. | Yes, but keep as distributed tests. | High |
| `vllm_uncertainty/tests/lora/test_long_context.py` | 119 | Uses `tensor_parallel_size=4` in long-context coverage. | No, test-only. | Yes, but not relevant to SeaKR on Azure L4. | High |
| `vllm_uncertainty/tests/test_sharded_state_loader.py` | 56, 71, 81 | Uses `worker_use_ray=True` in sharded-state loading tests. | No, test-only. | Yes, but it should stay a distributed coverage case. | High |

## Notes on the vLLM fork

The `vllm_uncertainty` fork contains many more `get_tensor_model_parallel_world_size()` calls inside model definitions such as `llama.py`, `bloom.py`, `gpt_neox.py`, `qwen.py`, `mixtral.py`, `opt.py`, `gemma.py`, `mpt.py`, `baichuan.py`, and others. Those are not hard multi-GPU assumptions by themselves. They are the model-parallel implementation, and they already collapse correctly when TP=1.

I am not treating those as migration blockers because:

1. they are not hardcoded to more than one GPU,
2. the engine code already selects the single-GPU executor path when distributed execution is not requested,
3. the collectives in `communication_op.py` are explicitly no-ops at world size 1.

## Phase 2 migration plan

### SAFE

1. Keep `main_multihop.py` and `main_simpleqa.py` on `tensor_parallel_size=1`.
   - Why safe: this only changes how the model is partitioned, not the model, prompts, retrieval, or uncertainty math.
   - Impact: no expected change to retrieval logic, hidden-state extraction, Gram/logdet calculation, EM/F1 definitions, or prompt content. Inference speed will change, typically slower than 2-way tensor parallelism but simpler and more reliable on a single L4.

2. Use the direct single-GPU execution path in the new verification scripts.
   - Why safe: the vLLM engine already has a single-GPU path and the SeaKR code calls into the same generation and uncertainty logic.
   - Impact: no expected change to accuracy or evaluation semantics. Speed may be lower than a 2-GPU run, but the target hardware only has one GPU.

### LIKELY SAFE

1. Avoid Ray in the new Azure verification scripts.
   - Why likely safe: `worker_use_ray` is only needed for distributed execution. The single-GPU path does not need it.
   - Impact: no expected change to accuracy, EM/F1, retrieval, uncertainty estimation, hidden states, Gram determinant, or prompts. This mainly reduces process complexity and startup overhead.

2. Run the mini evaluation sequentially on one GPU.
   - Why likely safe: the script is just a harness around the existing reasoning loop.
   - Impact: no mathematical change. The only difference is execution scheduling and wall-clock time.

### NEEDS VERIFICATION

1. Any change to `max_model_len`, cache sizing, or generation count.
   - Why it needs verification: these can affect memory pressure and possibly output behavior.
   - Impact: may change runtime or output quality.

2. Any attempt to reduce model precision or quantize.
   - Why it needs verification: the user explicitly asked not to do this unless necessary.
   - Impact: could change accuracy and paper fidelity.

3. Any edit to retrieval thresholds, prompts, or uncertainty equations.
   - Why it needs verification: these are core to SeaKR behavior.
   - Impact: directly affects retrieval triggers, answer selection, and reported metrics.

