#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def normalize_answer(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = value.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def as_gold_set(value: Any) -> set[str]:
    if isinstance(value, list):
        return {str(item) for item in value}
    return {str(value)}


def exact_match(prediction: Any, ground_truth: Any, aliases: Iterable[str] = ()) -> float:
    if not prediction:
        return 0.0
    golds = as_gold_set(ground_truth)
    golds.update(aliases)
    pred_norm = normalize_answer(prediction)
    return float(any(pred_norm == normalize_answer(gold) for gold in golds))


def f1_score(prediction: Any, ground_truth: Any, aliases: Iterable[str] = ()) -> float:
    if not prediction:
        return 0.0
    golds = as_gold_set(ground_truth)
    golds.update(aliases)
    best = 0.0
    prediction_norm = normalize_answer(prediction)
    for gold in golds:
        gold_norm = normalize_answer(gold)
        if prediction_norm in {"yes", "no", "noanswer"} and prediction_norm != gold_norm:
            continue
        if gold_norm in {"yes", "no", "noanswer"} and prediction_norm != gold_norm:
            continue
        pred_tokens = prediction_norm.split()
        gold_tokens = gold_norm.split()
        common = Counter(pred_tokens) & Counter(gold_tokens)
        same = sum(common.values())
        if same == 0 or not pred_tokens or not gold_tokens:
            continue
        precision = same / len(pred_tokens)
        recall = same / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_aliases(data_dir: Path) -> dict[str, list[str]]:
    alias_path = data_dir / "id_aliases.json"
    if not alias_path.exists():
        return {}
    aliases: dict[str, list[str]] = {}
    with alias_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            aliases[row["Q_id"]] = row.get("aliases", [])
    return aliases


def load_twowiki_answer_ids(data_dir: Path) -> dict[str, str]:
    dev_path = data_dir / "dev.json"
    if not dev_path.exists():
        return {}
    with dev_path.open() as handle:
        rows = json.load(handle)
    return {row["_id"]: row.get("answer_id") for row in rows}


def evaluate_multihop(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(Path(args.predictions))
    if not rows:
        raise SystemExit(f"No rows found in {args.predictions}")

    aliases_by_id: dict[str, list[str]] = {}
    answer_id_by_qid: dict[str, str] = {}
    if args.dataset_name == "twowikihop" and args.twh_data_dir:
        data_dir = Path(args.twh_data_dir)
        aliases_by_id = load_aliases(data_dir)
        answer_id_by_qid = load_twowiki_answer_ids(data_dir)

    per_column: dict[str, dict[str, float]] = {}
    for column in args.answer_columns:
        em_values: list[float] = []
        f1_values: list[float] = []
        for row in rows:
            answer_id = answer_id_by_qid.get(row.get("qid"))
            aliases = aliases_by_id.get(answer_id, []) if answer_id else []
            em_values.append(exact_match(row.get(column), row.get("ground_truth"), aliases))
            f1_values.append(f1_score(row.get(column), row.get("ground_truth"), aliases))
        per_column[column] = {
            "em": sum(em_values) / len(em_values),
            "f1": sum(f1_values) / len(f1_values),
        }

    return {
        "task_type": "multihop",
        "dataset_name": args.dataset_name,
        "num_examples": len(rows),
        "metrics": per_column,
    }


def evaluate_singlehop(args: argparse.Namespace) -> dict[str, Any]:
    data_path = Path(args.ground_truth)
    direct_path = Path(args.direct_predictions)
    rag_path = Path(args.rag_predictions)
    with data_path.open() as handle:
        gold_rows = json.load(handle)
    direct_rows = read_jsonl(direct_path)
    rag_rows = read_jsonl(rag_path)

    if len(direct_rows) != len(gold_rows):
        raise SystemExit(f"direct rows ({len(direct_rows)}) != gold rows ({len(gold_rows)})")
    docs_per_question = args.docs_per_question
    if len(rag_rows) != docs_per_question * len(gold_rows):
        raise SystemExit(
            f"rag rows ({len(rag_rows)}) != {docs_per_question} * gold rows ({len(gold_rows)})"
        )

    answers: list[Any] = []
    for idx, direct in enumerate(direct_rows):
        if direct.get("eigen_score", float("inf")) < args.eigen_threshold:
            answers.append(direct.get("answer"))
            continue
        batch = rag_rows[docs_per_question * idx: docs_per_question * (idx + 1)]
        best = min(batch, key=lambda row: row.get("eigen_score", float("inf")))
        answers.append(best.get("answer"))

    em_values = []
    f1_values = []
    for pred, gold_row in zip(answers, gold_rows):
        gold = gold_row.get("answer")
        em_values.append(exact_match(pred, gold))
        f1_values.append(f1_score(pred, gold))

    return {
        "task_type": "singlehop",
        "dataset_name": args.dataset_name,
        "num_examples": len(gold_rows),
        "eigen_threshold": args.eigen_threshold,
        "metrics": {
            "Final Answer": {
                "em": sum(em_values) / len(em_values),
                "f1": sum(f1_values) / len(f1_values),
            }
        },
    }


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{result['dataset_name']}_metrics.json").write_text(json.dumps(result, indent=2))
    csv_path = output_dir / f"{result['dataset_name']}_metrics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "answer_column", "em", "f1", "num_examples"])
        for column, metrics in result["metrics"].items():
            writer.writerow([
                result["dataset_name"],
                column,
                f"{100 * metrics['em']:.4f}",
                f"{100 * metrics['f1']:.4f}",
                result["num_examples"],
            ])
    md_lines = ["| Dataset | Answer Column | EM | F1 | N |", "|---|---:|---:|---:|---:|"]
    for column, metrics in result["metrics"].items():
        md_lines.append(
            f"| {result['dataset_name']} | {column} | {100 * metrics['em']:.2f} | "
            f"{100 * metrics['f1']:.2f} | {result['num_examples']} |"
        )
    (output_dir / f"{result['dataset_name']}_metrics.md").write_text("\n".join(md_lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Notebook-equivalent SeaKR EM/F1 evaluator.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    multihop = subparsers.add_parser("multihop")
    multihop.add_argument("--dataset-name", required=True, choices=["twowikihop", "hotpotqa", "iirc"])
    multihop.add_argument("--predictions", required=True)
    multihop.add_argument("--output-dir", required=True)
    multihop.add_argument("--twh-data-dir", default="data/multihop_data/2wikimultihopqa")
    multihop.add_argument(
        "--answer-columns",
        nargs="+",
        default=["Final Answer", "Final Step Answer", "Final Read Answer"],
    )

    singlehop = subparsers.add_parser("singlehop")
    singlehop.add_argument("--dataset-name", required=True, choices=["nq", "tq", "sq"])
    singlehop.add_argument("--ground-truth", required=True)
    singlehop.add_argument("--direct-predictions", required=True)
    singlehop.add_argument("--rag-predictions", required=True)
    singlehop.add_argument("--output-dir", required=True)
    singlehop.add_argument("--docs-per-question", type=int, default=10)
    singlehop.add_argument("--eigen-threshold", type=float, default=-6.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "multihop":
        result = evaluate_multihop(args)
    elif args.command == "singlehop":
        result = evaluate_singlehop(args)
    else:
        raise RuntimeError(args.command)
    write_outputs(result, Path(args.output_dir))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
