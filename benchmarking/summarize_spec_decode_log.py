#!/usr/bin/env python3
"""Summarize a spec-decode benchmark log into three markdown tables.

Reads the per-run AIPerf CSV exports (``profile_export_aiperf.csv``) referenced
by the ``--artifact-dir`` argument in a run log, plus the
``[label] acceptance_rate=... for <slug>`` lines, then emits three markdown
tables where rows are metrics (plus acceptance_rate) and columns are the
per-configuration ``avg`` value:

  1. spec_bench: cols = osl-128 avg, osl-512 avg
  2. speed_bench (per category): cols = coding avg, humanities avg, ...
  3. speed_bench_throughput (per isl x concurrency): cols = 1k_maxcon-1 avg, ...

The CSV is the canonical, lossless export. We deliberately do *not* scrape the
rich-rendered console tables: when columns get narrow rich truncates the metric
name with an ellipsis (e.g. ``Output Token Throughp…``), which is ambiguous
between the two ``Output Token Throughput*`` metrics and silently drops cells.

Usage:
    python summarize_spec_decode_log.py LOG_PATH [-o OUT_PATH]
"""

import argparse
import csv
import re
import sys
from collections import OrderedDict
from pathlib import Path

ACCEPTANCE_RE = re.compile(r"\[[^\]]+\]\s+acceptance_rate=([\d.]+)\s+for\s+(\S+)")

# The full ``--artifact-dir <path>`` lives on the ``Running command:`` INFO line
# (unquoted, single line). The wrapped ``CLI Command`` echo spills the path onto
# the next line, so requiring a non-space token after the flag skips it.
ARTIFACT_DIR_RE = re.compile(r"--artifact-dir\s+(\S+)")

# Artifact dir names are ``<role>_<YYYY-MM-DD>_<HH-MM-SS>_<slug>``.
ARTIFACT_NAME_RE = re.compile(
    r"^[A-Za-z0-9]+_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_(.+)$"
)

# Canonical metric order for table rows.
METRIC_ORDER = [
    "Time to First Token (ms)",
    "Time to Second Token (ms)",
    "Time to First Output Token (ms)",
    "Request Latency (ms)",
    "Inter Token Latency (ms)",
    "Output Token Throughput Per User (tokens/sec/user)",
    "E2E Output Token Throughput (tokens/sec/user)",
    "Output Sequence Length (tokens)",
    "Input Sequence Length (tokens)",
    "Output Token Throughput (tokens/sec)",
    "Request Throughput (requests/sec)",
    "Request Count (requests)",
    "acceptance_rate",
]

SPEED_BENCH_CATEGORIES = [
    "coding",
    "humanities",
    "math",
    "multilingual",
    "qa",
    "rag",
    "reasoning",
    "roleplay",
    "stem",
    "summarization",
    "writing",
]

THROUGHPUT_ISLS = ["1k", "2k", "8k", "16k", "32k"]
THROUGHPUT_CONCURRENCIES = ["1", "16", "64"]


# CSV metric names that differ from the canonical row labels in METRIC_ORDER.
CSV_NAME_ALIASES = {
    "Request Count": "Request Count (requests)",
}


def _format_value(raw: str) -> str:
    """Format a raw CSV number to match the table style (2dp, thousands sep)."""
    try:
        return f"{float(raw):,.2f}"
    except (TypeError, ValueError):
        return raw


def parse_csv(csv_path):
    """Return {canonical_metric_name -> avg_str} from a profile_export_aiperf.csv.

    The export has two relevant sections: a per-metric statistics table with an
    ``avg`` column, and a single-value table with a ``Value`` column (where
    aggregate ``Output Token Throughput (tokens/sec)`` and friends live). Only
    rows whose (aliased) name is a known canonical metric are kept.
    """
    metrics = OrderedDict()
    value_idx = None
    for row in csv.reader(csv_path.read_text().splitlines()):
        if not row:
            value_idx = None  # blank line separates sections
            continue
        if row[0] == "Metric":
            if "avg" in row:
                value_idx = row.index("avg")
            elif "Value" in row:
                value_idx = row.index("Value")
            else:
                value_idx = None
            continue
        if value_idx is None or value_idx >= len(row):
            continue
        name = CSV_NAME_ALIASES.get(row[0].strip(), row[0].strip())
        if name in METRIC_ORDER:
            metrics[name] = _format_value(row[value_idx].strip())
    return metrics


def resolve_artifact_dir(logged_path, log_path):
    """Resolve a logged ``--artifact-dir`` path to an existing directory."""
    name = Path(logged_path).name
    for candidate in (
        Path(logged_path),
        log_path.parent.parent / logged_path,
        log_path.parent / "aiperf_artifacts" / name,
    ):
        if candidate.is_dir():
            return candidate
    return None


def extract_runs(lines, log_path):
    """Yield (metrics_dict, slug, acceptance_rate_str) for each run in the log.

    Metrics come from each run's ``profile_export_aiperf.csv``; the acceptance
    rate comes from the matching ``acceptance_rate=... for <slug>`` log line.
    """
    acceptance = {}
    artifact_dirs = OrderedDict()  # slug -> logged artifact-dir path (last wins)
    for line in lines:
        m = ACCEPTANCE_RE.search(line)
        if m:
            acceptance[m.group(2)] = m.group(1)
        m = ARTIFACT_DIR_RE.search(line)
        if m:
            name_match = ARTIFACT_NAME_RE.match(Path(m.group(1)).name)
            if name_match:
                artifact_dirs[name_match.group(1)] = m.group(1)

    for slug, logged_path in artifact_dirs.items():
        art_dir = resolve_artifact_dir(logged_path, log_path)
        if art_dir is None:
            print(f"warning: artifact dir not found for {slug}: {logged_path}",
                  file=sys.stderr)
            continue
        csv_path = art_dir / "profile_export_aiperf.csv"
        if not csv_path.is_file():
            print(f"warning: no CSV export for {slug}: {csv_path}", file=sys.stderr)
            continue
        yield parse_csv(csv_path), slug, acceptance.get(slug)


def classify(slug):
    """Map a run slug to (table_key, col_key); return (None, None) if unknown."""
    m = re.match(r"spec_bench_osl-(\d+)_", slug)
    if m:
        return "spec_bench", f"osl-{m.group(1)}"
    m = re.match(r"speed_bench_throughput_([0-9]+k)_(?:osl-\d+_)?maxcon-(\d+)", slug)
    if m:
        return "speed_bench_throughput", f"{m.group(1)}_maxcon-{m.group(2)}"
    m = re.match(r"speed_bench_([a-z]+)_(?:osl-\d+_)?maxcon-", slug)
    if m:
        return "speed_bench_categories", m.group(1)
    return None, None


def render_table(title, columns, rows):
    """Render a markdown table. ``rows[metric][col]`` is the cell value."""
    header = ["Metric"] + [f"{c} avg" for c in columns]
    out = [f"## {title}", ""]
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "|".join(["---"] * len(header)) + "|")
    for metric in METRIC_ORDER:
        row = [metric]
        for col in columns:
            row.append(rows.get(metric, {}).get(col, "—"))
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def build_view(table_data, columns):
    view = {}
    for metric in METRIC_ORDER:
        view[metric] = {}
        for col in columns:
            view[metric][col] = table_data.get(col, {}).get(metric, "—")
    return view


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log",
        type=Path,
        help="Path to the run log file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write output to this file instead of stdout.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    lines = args.log.read_text().splitlines()

    data = {
        "spec_bench": {},
        "speed_bench_categories": {},
        "speed_bench_throughput": {},
    }

    for metrics, slug, acceptance in extract_runs(lines, args.log):
        table_key, col_key = classify(slug)
        if table_key is None:
            continue
        col_data = data[table_key].setdefault(col_key, {})
        col_data.update(metrics)
        if acceptance is not None:
            col_data["acceptance_rate"] = acceptance

    spec_bench_cols = ["osl-128", "osl-512"]
    throughput_cols = [
        f"{isl}_maxcon-{c}" for isl in THROUGHPUT_ISLS for c in THROUGHPUT_CONCURRENCIES
    ]

    sections = [
        render_table(
            "spec_bench",
            spec_bench_cols,
            build_view(data["spec_bench"], spec_bench_cols),
        ),
        render_table(
            "speed_bench (all categories)",
            SPEED_BENCH_CATEGORIES,
            build_view(data["speed_bench_categories"], SPEED_BENCH_CATEGORIES),
        ),
        render_table(
            "speed_bench_throughput (all isl)",
            throughput_cols,
            build_view(data["speed_bench_throughput"], throughput_cols),
        ),
    ]

    text = "\n\n".join(sections) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
