#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Aggregate repeated lm_eval sample files into a pass@1 summary."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate repeated lm_eval samples into DeepSeek-style pass@1.",
    )
    parser.add_argument(
        "root", type=Path, help="Root directory containing repeated eval run outputs"
    )
    parser.add_argument("--task", default="r1_aime24", help="Task name to aggregate")
    parser.add_argument(
        "--expected-runs", type=int, default=None, help="Expected samples per problem"
    )
    return parser.parse_args()


def load_samples(root: Path, task: str) -> list[dict]:
    pattern = f"**/samples_{task}_*.jsonl"
    rows: list[dict] = []
    for path in sorted(root.glob(pattern)):
        run_name = path.parent.parent.name
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                row["_sample_path"] = str(path)
                row["_run_name"] = run_name
                rows.append(row)
    return rows


def row_score(row: dict) -> float:
    value = row.get("exact_match")
    if value is None:
        raise KeyError(f"sample row has no exact_match: {row.get('_sample_path')}")
    return float(value)


def _extract_boxed(text: str) -> str | None:
    marker = text.rfind("boxed")
    if marker < 0:
        return None
    tail = text[marker + len("boxed") :].strip()
    if not tail:
        return None
    if tail[0] == "{":
        depth = 1
        answer = []
        for char in tail[1:]:
            if char == "{":
                depth += 1
                answer.append(char)
            elif char == "}":
                depth -= 1
                if depth == 0:
                    break
                answer.append(char)
            else:
                answer.append(char)
        return "".join(answer)
    return re.split(r"[\s.$]", tail, maxsplit=1)[0]


def normalize_answer(value) -> str:
    text = str(value).strip()
    text = text.replace("\\boxed", "boxed")
    boxed = _extract_boxed(text)
    if boxed is not None:
        text = boxed
    else:
        numbers = re.findall(r"-?\d*\.?\d+", text.replace(",", ""))
        if numbers:
            text = numbers[-1]
    text = text.strip().strip("$. ")
    text = re.sub(r"^0+(\d)$", r"\1", text)
    return text


def row_prediction(row: dict) -> str:
    if row_score(row) == 1.0:
        return normalize_answer(row.get("target", ""))
    filtered = row.get("filtered_resps")
    if isinstance(filtered, list) and filtered:
        return normalize_answer(filtered[0])
    resps = row.get("resps")
    if isinstance(resps, list) and resps:
        first = resps[0]
        if isinstance(first, list) and first:
            return normalize_answer(first[0])
        return normalize_answer(first)
    return ""


def summarize(rows: list[dict], task: str, expected_runs: int | None) -> dict:
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        doc = row.get("doc") or {}
        doc_key = str(doc.get("id", row.get("doc_id")))
        by_doc[doc_key].append(row)

    per_problem = []
    all_scores = []
    majority_scores = []
    for doc_key in sorted(
        by_doc, key=lambda value: int(value) if value.isdigit() else value
    ):
        doc_rows = by_doc[doc_key]
        scores = [row_score(row) for row in doc_rows]
        predictions = [row_prediction(row) for row in doc_rows]
        prediction_counts = Counter(predictions)
        majority_prediction, majority_count = prediction_counts.most_common(1)[0]
        all_scores.extend(scores)
        first = doc_rows[0]
        target = normalize_answer(first.get("target", ""))
        majority_correct = 1 if majority_prediction == target else 0
        majority_scores.append(majority_correct)
        per_problem.append(
            {
                "doc_key": doc_key,
                "doc_id": first.get("doc_id"),
                "aime_id": (first.get("doc") or {}).get("id"),
                "target": first.get("target"),
                "samples": len(scores),
                "correct": int(sum(scores)),
                "pass_at_1": mean(scores),
                "majority_prediction": majority_prediction,
                "majority_count": majority_count,
                "majority_correct": majority_correct,
            }
        )

    if not all_scores:
        raise SystemExit(f"No samples found for task {task}")

    missing = []
    if expected_runs is not None:
        missing = [item for item in per_problem if item["samples"] != expected_runs]

    return {
        "task": task,
        "root": None,
        "num_problems": len(per_problem),
        "num_samples": len(all_scores),
        "expected_runs": expected_runs,
        "pass_at_1": mean(all_scores),
        "pass_at_1_percent": 100.0 * mean(all_scores),
        "majority_at_1": mean(majority_scores),
        "majority_at_1_percent": 100.0 * mean(majority_scores),
        "sample_stddev": pstdev(all_scores) if len(all_scores) > 1 else 0.0,
        "missing_or_extra_sample_counts": missing,
        "per_problem": per_problem,
    }


def markdown(summary: dict) -> str:
    lines = [
        f"# pass@1 Summary: `{summary['task']}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Problems | {summary['num_problems']} |",
        f"| Samples | {summary['num_samples']} |",
        f"| pass@1 | {summary['pass_at_1']:.6f} |",
        f"| pass@1 (%) | {summary['pass_at_1_percent']:.2f} |",
        f"| majority@1 | {summary['majority_at_1']:.6f} |",
        f"| majority@1 (%) | {summary['majority_at_1_percent']:.2f} |",
        "",
        "| AIME ID | Target | Samples | Correct | pass@1 | Majority | majority@1 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["per_problem"]:
        lines.append(
            f"| {item['aime_id']} | {item['target']} | {item['samples']} | "
            f"{item['correct']} | {item['pass_at_1']:.4f} | "
            f"{item['majority_prediction']} | {item['majority_correct']} |"
        )

    if summary["missing_or_extra_sample_counts"]:
        lines.extend(["", "## Sample Count Warnings", ""])
        for item in summary["missing_or_extra_sample_counts"]:
            lines.append(
                f"- AIME ID {item['aime_id']}: {item['samples']} samples "
                f"(expected {summary['expected_runs']})"
            )

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    rows = load_samples(root, args.task)
    summary = summarize(rows, args.task, args.expected_runs)
    summary["root"] = str(root)

    json_path = root / f"{args.task}_pass1_summary.json"
    md_path = root / f"{args.task}_pass1_summary.md"
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    md_path.write_text(markdown(summary), encoding="utf-8")

    print(
        f"pass@1: {summary['pass_at_1_percent']:.2f}% ({summary['num_samples']} samples)"
    )
    print(
        f"majority@1: {summary['majority_at_1_percent']:.2f}% ({summary['num_problems']} problems)"
    )
    print(f"Summary JSON: {json_path}")
    print(f"Summary Markdown: {md_path}")
    if summary["missing_or_extra_sample_counts"]:
        print("Warning: some problems do not have the expected sample count.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
