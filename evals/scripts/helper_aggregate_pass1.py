#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Aggregate repeated lm_eval sample files into a pass@1 summary."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate repeated lm_eval samples into DeepSeek-style pass@1.",
    )
    parser.add_argument("root", type=Path, help="Root directory containing repeated eval run outputs")
    parser.add_argument("--task", default="r1_aime24", help="Task name to aggregate")
    parser.add_argument("--expected-runs", type=int, default=None, help="Expected samples per problem")
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


def summarize(rows: list[dict], task: str, expected_runs: int | None) -> dict:
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        doc = row.get("doc") or {}
        doc_key = str(doc.get("id", row.get("doc_id")))
        by_doc[doc_key].append(row)

    per_problem = []
    all_scores = []
    for doc_key in sorted(by_doc, key=lambda value: int(value) if value.isdigit() else value):
        doc_rows = by_doc[doc_key]
        scores = [row_score(row) for row in doc_rows]
        all_scores.extend(scores)
        first = doc_rows[0]
        per_problem.append(
            {
                "doc_key": doc_key,
                "doc_id": first.get("doc_id"),
                "aime_id": (first.get("doc") or {}).get("id"),
                "target": first.get("target"),
                "samples": len(scores),
                "correct": int(sum(scores)),
                "pass_at_1": mean(scores),
            }
        )

    if not all_scores:
        raise SystemExit(f"No samples found for task {task}")

    missing = []
    if expected_runs is not None:
        missing = [
            item
            for item in per_problem
            if item["samples"] != expected_runs
        ]

    return {
        "task": task,
        "root": None,
        "num_problems": len(per_problem),
        "num_samples": len(all_scores),
        "expected_runs": expected_runs,
        "pass_at_1": mean(all_scores),
        "pass_at_1_percent": 100.0 * mean(all_scores),
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
        "",
        "| AIME ID | Target | Samples | Correct | pass@1 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for item in summary["per_problem"]:
        lines.append(
            f"| {item['aime_id']} | {item['target']} | {item['samples']} | "
            f"{item['correct']} | {item['pass_at_1']:.4f} |"
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
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown(summary), encoding="utf-8")

    print(f"pass@1: {summary['pass_at_1_percent']:.2f}% ({summary['num_samples']} samples)")
    print(f"Summary JSON: {json_path}")
    print(f"Summary Markdown: {md_path}")
    if summary["missing_or_extra_sample_counts"]:
        print("Warning: some problems do not have the expected sample count.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
