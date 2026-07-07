#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, LLM, SamplingParams

from SEAKR.dataset import get_dataset
from SEAKR.reasoner import MultiHopReasoner
from SEAKR.retriever import BM25
from SEAKR.utils import StepStatus


def _normalize_answer(text: Optional[str]) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _f1_score(prediction: Optional[str], gold: Optional[str]) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts: Dict[str, int] = {}
    gold_counts: Dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in gold_tokens:
        gold_counts[token] = gold_counts.get(token, 0) + 1
    common = 0
    for token, count in pred_counts.items():
        common += min(count, gold_counts.get(token, 0))
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _exact_match(prediction: Optional[str], gold: Optional[str]) -> float:
    return float(_normalize_answer(prediction) == _normalize_answer(gold))


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _build_tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _build_async_engine_args(args: argparse.Namespace) -> AsyncEngineArgs:
    return AsyncEngineArgs(
        model=args.model_name_or_path,
        served_model_name=args.served_model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        selected_intermediate_layer=args.selected_intermediate_layer,
        eigen_alpha=args.eigen_alpha,
        worker_use_ray=False,
        disable_log_requests=True,
        disable_log_stats=True,
        enable_prefix_caching=True,
        enforce_eager=True,
    )


def _build_retriever(tokenizer, retriever_port: int):
    return BM25(
        tokenizer=tokenizer,
        index_name="wiki",
        engine="elasticsearch",
        port=retriever_port,
    )


async def check_environment_async(args: argparse.Namespace) -> None:
    model_path = Path(args.model_name_or_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model path does not exist: {model_path}. "
            "Set MODEL_NAME_OR_PATH to a local model directory on Azure."
        )

    tokenizer = _build_tokenizer(args.model_name_or_path)
    print(f"tokenizer: {tokenizer.__class__.__name__}")

    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        selected_intermediate_layer=args.selected_intermediate_layer,
        eigen_alpha=args.eigen_alpha,
        enable_prefix_caching=True,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        n=20,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
        max_tokens=4,
        logprobs=0,
        seed=42,
        stop=["\n", "\n\n", "\nQuestion:", "\nContext"],
    )
    request_output = llm.generate(["Smoke test"], sampling_params)[0]
    uncertainty = getattr(request_output, "uncertainty", {})

    print("vllm_model_loaded: yes")
    print(f"hidden_state_path: {args.selected_intermediate_layer}")
    print(f"uncertainty_keys: {sorted(uncertainty.keys())}")
    if "eigen_score" not in uncertainty:
        raise RuntimeError(
            "Expected eigen_score to be computed during environment check."
        )
    if "perplexity" not in uncertainty and "ln_entropy" not in uncertainty:
        raise RuntimeError(
            "Expected a perplexity or entropy signal during environment check."
        )
    print(f"eigen_score: {uncertainty.get('eigen_score')}")
    print(f"perplexity: {uncertainty.get('perplexity')}")


async def run_smoke_test_async(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    _ensure_dir(output_dir)

    dataset_obj = get_dataset(args.dataset_name, args.n_shot)
    dataset_list = dataset_obj.load_data()
    if not dataset_list:
        raise RuntimeError("Dataset returned no examples.")
    entry = dataset_list[0]

    tokenizer = _build_tokenizer(args.model_name_or_path)
    retriever = _build_retriever(tokenizer, args.retriever_port)
    llm_engine = AsyncLLMEngine.from_engine_args(_build_async_engine_args(args))

    reasoner = MultiHopReasoner(
        qid=entry["qid"],
        question=entry["question"],
        dataset=dataset_obj,
        llm_engine=llm_engine,
        retriever=retriever,
        logger_dir=str(output_dir / "logs"),
        eigen_threshold=args.eigen_threshold,
        prob_threshold=args.prob_threshold,
    )

    _ensure_dir(output_dir / "logs")
    print(f"running smoke test for qid={entry['qid']}")

    direct_step = await reasoner.answer_direct()
    reasoner.running_steps.append(direct_step)
    if direct_step.status != StepStatus.DIRECT_FAILED:
        direct_step.status = StepStatus.DIRECT_FAILED

    rag_step = await reasoner.rag()
    if rag_step is None:
        raise RuntimeError("Retrieval did not return any docs during smoke test.")
    reasoner.running_steps.append(rag_step)

    await reasoner.read_all_steps()
    await reasoner.read_all_docs()
    reasoner.compare_answer()

    result = {
        "qid": entry["qid"],
        "question": entry["question"],
        "ground_truth": entry["answer"],
        "prediction": reasoner.final_answer,
        "retrieval_times": len(reasoner.doc_id_list),
        "llm_call_times": reasoner.llm_call_times,
        "final_step_score": reasoner.final_step_score,
        "final_read_score": reasoner.final_read_answer_score,
        "doc_ids": reasoner.doc_id_list,
        "direct_step": {
            "status": direct_step.status.name,
            "content": direct_step.content,
            "score": direct_step.score,
        },
        "rag_step": {
            "status": rag_step.status.name,
            "content": rag_step.content,
            "score": rag_step.score,
        },
        "steps": [{
            "status": step.status.name,
            "content": step.content,
            "score": step.score,
        } for step in reasoner.running_steps],
        "checks": {
            "retrieval_worked": bool(reasoner.doc_id_list),
            "uncertainty_worked": reasoner.final_step_score is not None
            and reasoner.final_read_answer_score is not None,
            "prediction_saved": True,
        },
    }

    if result["prediction"] is None:
        raise RuntimeError("Smoke test finished without a prediction.")
    if not result["checks"]["retrieval_worked"]:
        raise RuntimeError("Smoke test did not confirm retrieval.")
    if not result["checks"]["uncertainty_worked"]:
        raise RuntimeError("Smoke test did not confirm uncertainty scoring.")

    output_path = output_dir / "prediction.json"
    output_path.write_text(json.dumps(result, indent=2))
    print(f"prediction saved: {output_path}")
    print(json.dumps(result["checks"], indent=2))


async def run_mini_eval_async(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    _ensure_dir(output_dir)
    _ensure_dir(output_dir / "logs")

    dataset_obj = get_dataset(args.dataset_name, args.n_shot)
    dataset_list = dataset_obj.load_data()[: args.limit]
    if not dataset_list:
        raise RuntimeError("Dataset returned no examples.")

    tokenizer = _build_tokenizer(args.model_name_or_path)
    retriever = _build_retriever(tokenizer, args.retriever_port)
    llm_engine = AsyncLLMEngine.from_engine_args(_build_async_engine_args(args))

    predictions: List[Dict[str, Any]] = []
    exact_matches: List[float] = []
    f1_scores: List[float] = []
    retrieval_counts: List[int] = []
    llm_call_counts: List[int] = []

    for idx, entry in enumerate(dataset_list, start=1):
        print(f"[{idx}/{len(dataset_list)}] qid={entry['qid']}")
        reasoner = MultiHopReasoner(
            qid=entry["qid"],
            question=entry["question"],
            dataset=dataset_obj,
            llm_engine=llm_engine,
            retriever=retriever,
            logger_dir=str(output_dir / "logs"),
            eigen_threshold=args.eigen_threshold,
            prob_threshold=args.prob_threshold,
        )
        output = await reasoner.solve(
            max_reasoning_steps=args.max_reasoning_steps,
            max_docs=args.max_docs,
        )

        pred = output["Final Answer"]
        gold = entry["answer"]
        exact_matches.append(_exact_match(pred, gold))
        f1_scores.append(_f1_score(pred, gold))
        retrieval_counts.append(int(output["Retrieval Times"]))
        llm_call_counts.append(int(output["Call LLM Times"]))

        predictions.append({
            "qid": entry["qid"],
            "question": entry["question"],
            "ground_truth": gold,
            "prediction": pred,
            "retrieval_times": output["Retrieval Times"],
            "llm_call_times": output["Call LLM Times"],
            "final_step_answer": output["Final Step Answer"],
            "final_read_answer": output["Final Read Answer"],
            "final_step_score": getattr(reasoner, "final_step_score", None),
            "final_read_score": getattr(reasoner, "final_read_answer_score", None),
        })

    metrics = {
        "dataset_name": args.dataset_name,
        "limit": len(dataset_list),
        "exact_match": mean(exact_matches) if exact_matches else math.nan,
        "f1": mean(f1_scores) if f1_scores else math.nan,
        "avg_retrieval_times": mean(retrieval_counts) if retrieval_counts else math.nan,
        "avg_llm_call_times": mean(llm_call_counts) if llm_call_counts else math.nan,
    }

    (output_dir / "predictions.json").write_text(json.dumps(predictions, indent=2))
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"predictions saved: {output_dir / 'predictions.json'}")
    print(f"metrics saved: {output_dir / 'metrics.json'}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SeaKR single-GPU verification helper.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    env = subparsers.add_parser("check-environment", help="Verify the Azure GPU environment.")
    _add_common_args(env)

    smoke = subparsers.add_parser("run-smoke-test", help="Run one example through the SeaKR pipeline.")
    _add_common_args(smoke)
    smoke.add_argument("--output-dir", required=True)

    mini = subparsers.add_parser("run-mini-eval", help="Run a 20-example SeaKR mini evaluation.")
    _add_common_args(mini)
    mini.add_argument("--output-dir", required=True)
    mini.add_argument("--limit", type=int, default=20)

    return parser


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--served-model-name", default="llama2-7b-chat")
    parser.add_argument("--dataset-name", default="hotpotqa")
    parser.add_argument("--retriever-port", type=int, default=9201)
    parser.add_argument("--n-shot", type=int, default=10)
    parser.add_argument("--selected-intermediate-layer", type=int, default=15)
    parser.add_argument("--eigen-alpha", type=float, default=1e-3)
    parser.add_argument("--eigen-threshold", type=float, default=-6.0)
    parser.add_argument("--prob-threshold", type=float, default=0.1)
    parser.add_argument("--max-reasoning-steps", type=int, default=7)
    parser.add_argument("--max-docs", type=int, default=5)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "check-environment":
        asyncio.run(check_environment_async(args))
    elif args.command == "run-smoke-test":
        asyncio.run(run_smoke_test_async(args))
    elif args.command == "run-mini-eval":
        asyncio.run(run_mini_eval_async(args))
    else:
        raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
