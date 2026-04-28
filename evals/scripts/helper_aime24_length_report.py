#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Print an AIME24 generation-length report from lm_eval sample artifacts."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE = REPO_ROOT / "evals/reference/aime24_generation_length_reference.csv"
DEFAULT_REFERENCE_URL = "https://github.com/tenstorrent/tt-metal/issues/38446#issuecomment-4073998777"
DEFAULT_TOKENIZER = "deepseek-ai/DeepSeek-R1-0528"
SAMPLE_RE = re.compile(r"^samples_(?P<task>.+)_\d{4}-\d{2}-\d{2}T.*\.jsonl$")


@dataclass(frozen=True)
class ReferenceRow:
    aime_id: int
    reference_tt_tokens: int
    gpu_tokens_mean: float
    gpu_tokens_half_range: float
    gpu_correct: str

    @property
    def gpu_min(self) -> float:
        return self.gpu_tokens_mean - self.gpu_tokens_half_range

    @property
    def gpu_max(self) -> float:
        return self.gpu_tokens_mean + self.gpu_tokens_half_range


@dataclass
class SampleSummary:
    aime_id: int
    target: str
    token_counts: list[int]
    scores: list[float | None]

    @property
    def total_tokens(self) -> int:
        return sum(self.token_counts)

    @property
    def mean_tokens(self) -> float:
        return mean(self.token_counts)

    @property
    def half_range(self) -> float:
        return (max(self.token_counts) - min(self.token_counts)) / 2.0

    @property
    def correct_count(self) -> int:
        return sum(1 for score in self.scores if score == 1.0)

    @property
    def sample_count(self) -> int:
        return len(self.token_counts)


class MarkdownTable:
    def __init__(self, headers: Sequence[str], aligns: Sequence[str]) -> None:
        self.headers = list(headers)
        self.aligns = list(aligns)
        self.rows: list[list[str]] = []

    def add(self, row: Sequence[object]) -> None:
        self.rows.append([str(item) for item in row])

    def render(self) -> str:
        widths = [len(header) for header in self.headers]
        for row in self.rows:
            widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

        def cell(text: str, width: int, align: str) -> str:
            if align == "right":
                return text.rjust(width)
            if align == "center":
                return text.center(width)
            return text.ljust(width)

        def separator(width: int, align: str) -> str:
            if align == "right":
                return "-" * max(width - 1, 1) + ":"
            if align == "center":
                return ":" + "-" * max(width - 2, 1) + ":"
            return "-" * width

        lines = []
        lines.append("| " + " | ".join(cell(h, w, a) for h, w, a in zip(self.headers, widths, self.aligns)) + " |")
        lines.append("| " + " | ".join(separator(w, a) for w, a in zip(widths, self.aligns)) + " |")
        for row in self.rows:
            lines.append("| " + " | ".join(cell(c, w, a) for c, w, a in zip(row, widths, self.aligns)) + " |")
        return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an AIME24 generation-length table from lm_eval outputs.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="Eval output directory or samples_*.jsonl file. Defaults to the latest AIME24 output under eval_results/.",
    )
    parser.add_argument("--task", default=None, help="Task name to report. Defaults to r1_aime24 when present.")
    parser.add_argument("--tokenizer", default=None, help=f"Tokenizer repo. Defaults to run metadata or {DEFAULT_TOKENIZER}.")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE, help="Reference JSON file")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to also write the Markdown report")
    parser.add_argument("--no-color", action="store_true", help="Use ASCII status labels instead of emoji markers")
    parser.add_argument("--show-reference-tt", action="store_true", help="Include the historical reference TT-token column")
    return parser.parse_args()


def load_reference(path: Path) -> tuple[dict[int, ReferenceRow], dict]:
    text_lines = path.read_text(encoding="utf-8").splitlines()
    comments = [line[1:].strip() for line in text_lines if line.startswith("#")]
    csv_lines = [line for line in text_lines if line.strip() and not line.startswith("#")]
    data = list(csv.DictReader(csv_lines))
    rows = {
        int(row["aime_id"]): ReferenceRow(
            aime_id=int(row["aime_id"]),
            reference_tt_tokens=int(row["reference_tt_tokens"]),
            gpu_tokens_mean=float(row["gpu_tokens_mean"]),
            gpu_tokens_half_range=float(row["gpu_tokens_half_range"]),
            gpu_correct=str(row["gpu_correct"]),
        )
        for row in data
    }
    meta = {"source_url": DEFAULT_REFERENCE_URL, "comments": comments}
    for comment in comments:
        if ":" in comment:
            key, value = comment.split(":", 1)
            meta[key.strip()] = value.strip()
    return rows, meta


def sample_task(path: Path) -> str | None:
    match = SAMPLE_RE.match(path.name)
    if not match:
        return None
    return match.group("task")


def latest_aime_output() -> Path | None:
    root = REPO_ROOT / "eval_results"
    if not root.exists():
        return None
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and (path.name.startswith("r1_aime24_") or path.name.startswith("r1_aime24_short_"))
    ]
    return max(candidates, key=lambda candidate: candidate.stat().st_mtime) if candidates else None


def discover_sample_files(path: Path | None, task: str | None) -> tuple[Path, str, list[Path]]:
    if path is None:
        path = latest_aime_output()
        if path is None:
            raise SystemExit("No output directory supplied and no AIME24 outputs found under eval_results/.")
    path = path.expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"Path does not exist: {path}")

    if path.is_file():
        files = [path]
        inferred_tasks = {sample_task(path)} - {None}
        root = path.parent
    else:
        all_files = [candidate for candidate in sorted(path.glob("**/samples_*.jsonl")) if sample_task(candidate)]
        inferred_tasks = {sample_task(candidate) for candidate in all_files} - {None}
        root = path
        files = all_files

    if not files:
        raise SystemExit(f"No samples_*.jsonl files found under {path}")

    if task is None:
        if "r1_aime24" in inferred_tasks:
            task = "r1_aime24"
        elif "r1_aime24_short" in inferred_tasks:
            task = "r1_aime24_short"
        elif len(inferred_tasks) == 1:
            task = next(iter(inferred_tasks))
        else:
            tasks = ", ".join(sorted(inferred_tasks))
            raise SystemExit(f"Multiple sample tasks found ({tasks}); pass --task.")

    selected = [candidate for candidate in files if sample_task(candidate) == task]
    if not selected:
        tasks = ", ".join(sorted(inferred_tasks)) or "none"
        raise SystemExit(f"No sample files for task {task!r}; found tasks: {tasks}")
    return root, task, selected


def infer_tokenizer(root: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    for path in sorted(root.glob("**/results_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        model_args = data.get("config", {}).get("model_args", {})
        for key in ("tokenizer", "model"):
            value = model_args.get(key)
            if isinstance(value, str) and value:
                return value
        value = data.get("model_name")
        if isinstance(value, str) and value:
            return value
    return DEFAULT_TOKENIZER


@contextlib.contextmanager
def suppress_stderr() -> Iterable[None]:
    old_stderr = sys.stderr
    try:
        sys.stderr = io.StringIO()
        yield
    finally:
        sys.stderr = old_stderr


def load_tokenizer(name: str):
    try:
        with suppress_stderr():
            from transformers import AutoTokenizer, logging as transformers_logging
            transformers_logging.set_verbosity_error()
            return AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    except ImportError as exc:
        raise SystemExit(
            "Could not import transformers. Run one eval script first to bootstrap "
            ".workflow_venvs/.venv_evals_common, or invoke this through "
            "evals/scripts/run_aime24_length_report.sh."
        ) from exc
    except Exception as exc:  # noqa: BLE001 - include tokenizer name in the user-facing error.
        raise SystemExit(f"Could not load tokenizer {name!r}: {exc}") from exc


def response_text(row: dict) -> str:
    filtered = row.get("filtered_resps")
    if isinstance(filtered, list) and filtered:
        value = filtered[0]
        if isinstance(value, str):
            return value

    resps = row.get("resps")
    if isinstance(resps, list) and resps:
        value = resps[0]
        if isinstance(value, list) and value:
            value = value[0]
        if isinstance(value, str):
            return value
    return ""


def score(row: dict) -> float | None:
    value = row.get("exact_match")
    if value is None:
        metrics = row.get("metrics")
        if isinstance(metrics, dict):
            metric = metrics.get("exact_match")
            if isinstance(metric, dict):
                value = metric.get("value")
            else:
                value = metric
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def aime_id(row: dict) -> int | None:
    doc = row.get("doc")
    if isinstance(doc, dict) and doc.get("id") is not None:
        return int(doc["id"])
    doc_id = row.get("doc_id")
    if doc_id is not None:
        # AIME24 ids in HuggingFaceH4/aime_2024 are 60..89 in order.
        return int(doc_id) + 60
    return None


def load_samples(paths: Sequence[Path], tokenizer) -> dict[int, SampleSummary]:
    by_id: dict[int, SampleSummary] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                aid = aime_id(row)
                if aid is None:
                    raise SystemExit(f"Could not determine AIME id for {path}:{line_no}")
                text = response_text(row)
                token_count = len(tokenizer.encode(text, add_special_tokens=False))
                target = str(row.get("target", ""))
                if aid not in by_id:
                    by_id[aid] = SampleSummary(aime_id=aid, target=target, token_counts=[], scores=[])
                by_id[aid].token_counts.append(token_count)
                by_id[aid].scores.append(score(row))
    return by_id


def format_num(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value))}"
    return f"{value:.1f}"


def format_tokens(summary: SampleSummary) -> str:
    if summary.sample_count == 1:
        return f"{summary.token_counts[0]}"
    return f"{format_num(summary.mean_tokens)} ± {format_num(summary.half_range)}"


def format_gpu_tokens(row: ReferenceRow) -> str:
    return f"{format_num(row.gpu_tokens_mean)} ± {format_num(row.gpu_tokens_half_range)}"


def status(summary: SampleSummary, reference: ReferenceRow, ascii_only: bool) -> str:
    labels = {
        "below": "LOW" if ascii_only else "💪",
        "within": "OK" if ascii_only else "✅",
        "above": "HIGH" if ascii_only else "❌",
    }
    value = summary.mean_tokens
    if value < reference.gpu_min:
        return labels["below"]
    if value > reference.gpu_max:
        return labels["above"]
    return labels["within"]


def correct_status(summary: SampleSummary, ascii_only: bool) -> str:
    valid = [score for score in summary.scores if score is not None]
    if not valid:
        return "n/a"
    if summary.sample_count == 1:
        if valid[0] == 1.0:
            return "OK" if ascii_only else "✅"
        return "BAD" if ascii_only else "❌"
    return f"{summary.correct_count}/{summary.sample_count}"


def count_statuses(summaries: Iterable[SampleSummary], references: dict[int, ReferenceRow], ascii_only: bool) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for summary in summaries:
        ref = references.get(summary.aime_id)
        if ref is None:
            continue
        counts[status(summary, ref, ascii_only)] += 1
    return counts


def render_report(
    root: Path,
    task: str,
    sample_paths: Sequence[Path],
    tokenizer_name: str,
    references: dict[int, ReferenceRow],
    reference_meta: dict,
    summaries: dict[int, SampleSummary],
    ascii_only: bool,
    show_reference_tt: bool,
) -> str:
    present = [summaries[aid] for aid in sorted(summaries)]
    total_tokens = sum(summary.total_tokens for summary in present)
    total_samples = sum(summary.sample_count for summary in present)
    total_correct = sum(summary.correct_count for summary in present)
    valid_scores = sum(1 for summary in present for item in summary.scores if item is not None)
    status_counts = count_statuses(present, references, ascii_only)
    below_label = "LOW" if ascii_only else "💪"
    ok_label = "OK" if ascii_only else "✅"
    high_label = "HIGH" if ascii_only else "❌"

    lines = [
        f"# AIME24 Generation-Length Report: `{task}`",
        "",
        f"Output: `{root}`",
        f"Samples: {total_samples} generations across {len(present)} AIME problems from {len(sample_paths)} sample file(s)",
        f"Tokenizer: `{tokenizer_name}`",
        f"Reference: {reference_meta.get('source_url', DEFAULT_REFERENCE_URL)}",
        "",
    ]

    summary_table = MarkdownTable(["Metric", "Value"], ["left", "right"])
    summary_table.add(["Total generated tokens", f"{total_tokens:,}"])
    summary_table.add(["Mean tokens / generation", f"{(total_tokens / total_samples):,.1f}" if total_samples else "n/a"])
    summary_table.add(["Correct", f"{total_correct}/{valid_scores}" if valid_scores else "n/a"])
    summary_table.add([f"{below_label} below GPU min", status_counts.get(below_label, 0)])
    summary_table.add([f"{ok_label} within GPU range", status_counts.get(ok_label, 0)])
    summary_table.add([f"{high_label} above GPU max", status_counts.get(high_label, 0)])
    lines.append(summary_table.render())
    lines.extend([
        "",
        f"Legend: {below_label} below GPU min, {ok_label} within GPU min/max, {high_label} above GPU max.",
        "",
    ])

    headers = ["AIME ID", "TT Tokens", "GPU Tokens", "In range", "TT Correct", "GPU Correct"]
    aligns = ["right", "right", "right", "center", "center", "center"]
    if show_reference_tt:
        headers.insert(2, "Ref TT")
        aligns.insert(2, "right")
    table = MarkdownTable(headers, aligns)
    for aid in sorted(summaries):
        summary = summaries[aid]
        ref = references.get(aid)
        if ref is None:
            row = [aid, format_tokens(summary), "n/a", "n/a", correct_status(summary, ascii_only), "n/a"]
            if show_reference_tt:
                row.insert(2, "n/a")
        else:
            row = [
                aid,
                format_tokens(summary),
                format_gpu_tokens(ref),
                status(summary, ref, ascii_only),
                correct_status(summary, ascii_only),
                ref.gpu_correct,
            ]
            if show_reference_tt:
                row.insert(2, f"{ref.reference_tt_tokens}")
        table.add(row)
    lines.append(table.render())
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    references, reference_meta = load_reference(args.reference)
    root, task, sample_paths = discover_sample_files(args.path, args.task)
    tokenizer_name = infer_tokenizer(root, args.tokenizer)
    tokenizer = load_tokenizer(tokenizer_name)
    summaries = load_samples(sample_paths, tokenizer)
    report = render_report(
        root=root,
        task=task,
        sample_paths=sample_paths,
        tokenizer_name=tokenizer_name,
        references=references,
        reference_meta=reference_meta,
        summaries=summaries,
        ascii_only=args.no_color or os.environ.get("NO_COLOR") is not None,
        show_reference_tt=args.show_reference_tt,
    )
    print(report, end="")
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
        print(f"\nWrote Markdown report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
