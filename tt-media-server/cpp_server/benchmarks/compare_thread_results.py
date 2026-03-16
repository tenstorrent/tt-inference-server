#!/usr/bin/env python3
"""Compare benchmark results from two directories (e.g. 12 threads vs 24 threads).

Thread count is read from the directory name (e.g. 12_threads_default -> 12, 24_threads -> 24).
Only (ISL, OSL, conc) configs present in both directories are compared.
Δ% = (dir2 − dir1) / dir2 × 100 (positive = dir2 higher, negative = dir2 lower).

Usage:
  python compare_thread_results.py <dir1> <dir2> [--output table.md]
"""

import argparse
import json
import re
import sys
from pathlib import Path

BENCH_PATTERN = re.compile(
    r"bench_isl(?P<isl>\d+)_osl(?P<osl>\d+)_conc(?P<conc>\d+)_[\d-]+\.json"
)

METRICS = [
    ("request_throughput", "Req/s", ".1f"),
    ("mean_ttft_ms", "Mean TTFT (ms)", ".3f"),
    ("mean_tpot_ms", "Mean TPOT (ms)", ".4f"),
    ("output_throughput", "Out tok/s", ".0f"),
    ("p99_ttft_ms", "P99 TTFT (ms)", ".3f"),
    ("p99_tpot_ms", "P99 TPOT (ms)", ".4f"),
]


def thread_count_from_dirname(dir_path: Path) -> str:
    m = re.search(r"(\d+)_threads", dir_path.name)
    return m.group(1) if m else dir_path.name


def load_dir(results_dir: Path) -> dict[tuple[int, int, int], dict]:
    out = {}
    for f in sorted(results_dir.glob("bench_isl*_osl*_conc*_*.json")):
        m = BENCH_PATTERN.match(f.name)
        if not m:
            continue
        key = (int(m.group("isl")), int(m.group("osl")), int(m.group("conc")))
        with open(f) as fh:
            out[key] = json.load(fh)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Compare benchmark results from two thread-config directories."
    )
    parser.add_argument("dir1", type=Path, help="First results directory (e.g. bench_results/12_threads_default)")
    parser.add_argument("dir2", type=Path, help="Second results directory (e.g. bench_results/24_threads)")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Write Markdown table to this file")
    args = parser.parse_args()

    for d in (args.dir1, args.dir2):
        if not d.is_dir():
            print(f"Error: not a directory: {d}", file=sys.stderr)
            sys.exit(1)

    label1 = thread_count_from_dirname(args.dir1)
    label2 = thread_count_from_dirname(args.dir2)
    data1 = load_dir(args.dir1)
    data2 = load_dir(args.dir2)

    if not data1:
        print(f"No benchmark JSONs found in {args.dir1}", file=sys.stderr)
        sys.exit(1)
    if not data2:
        print(f"No benchmark JSONs found in {args.dir2}", file=sys.stderr)
        sys.exit(1)

    common = sorted(set(data1) & set(data2))
    if not common:
        print("No common (ISL, OSL, conc) configs between the two directories.", file=sys.stderr)
        sys.exit(1)

    def fmt(val, spec: str):
        if val is None:
            return "—"
        return format(float(val), spec)

    lines = []
    lines.append(f"# Benchmark comparison: {label1} threads vs {label2} threads")
    lines.append("")
    lines.append(f"| ISL | OSL | Conc | Metric | {label1}t | {label2}t | Δ% |")
    lines.append("| --- | --- | ---- | ------ | " + " | ".join(["---"] * 3) + " |")

    for (isl, osl, conc) in common:
        r1, r2 = data1[(isl, osl, conc)], data2[(isl, osl, conc)]
        first = True
        for key, label, spec in METRICS:
            v1 = r1.get(key)
            v2 = r2.get(key)
            isl_col = str(isl) if first else ""
            osl_col = str(osl) if first else ""
            conc_col = str(conc) if first else ""
            first = False
            if v1 is not None and v2 is not None and float(v2) != 0:
                pct = 100.0 * (float(v2) - float(v1)) / float(v2)
                delta = f"{pct:+.1f}%"
            else:
                delta = "—"
            lines.append(f"| {isl_col} | {osl_col} | {conc_col} | {label} | {fmt(v1, spec)} | {fmt(v2, spec)} | {delta} |")

    text = "\n".join(lines)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
    print(text)


if __name__ == "__main__":
    main()
