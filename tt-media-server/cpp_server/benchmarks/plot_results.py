#!/usr/bin/env python3
"""Visualize vLLM benchmark results produced by run_benchmarks.sh."""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

MIN_LATENCY_MS = (
    0.001  # 1µs minimum granularity: values are rounded to this before plotting
)


def round_latency_ms(value_ms: float) -> float:
    """Round latency to minimum granularity (1µs) so the chart is flat when values are effectively equal."""
    return round(value_ms / MIN_LATENCY_MS) * MIN_LATENCY_MS


FILENAME_PATTERN = re.compile(
    r"bench_isl(?P<isl>\d+)_osl(?P<osl>\d+)_conc(?P<conc>\d+)_(?P<ts>[\d-]+)\.json"
)


def load_results(results_dir: Path) -> list[dict]:
    records = []
    for f in sorted(results_dir.glob("bench_isl*_osl*_conc*_*.json")):
        m = FILENAME_PATTERN.match(f.name)
        if not m:
            continue
        with open(f) as fh:
            data = json.load(fh)
        data["_isl"] = int(m.group("isl"))
        data["_osl"] = int(m.group("osl"))
        data["_conc"] = int(m.group("conc"))
        records.append(data)
    return records


def filter_records(records, **kwargs):
    out = records
    for k, v in kwargs.items():
        out = [r for r in out if r.get(k) == v]
    return sorted(out, key=lambda r: (r["_isl"], r["_osl"], r["_conc"]))


def format_latency_ms(value_ms: float, decimals: int | None = None) -> str:
    """Format latency in ms with at least 1µs granularity (0.001 ms)."""
    if decimals is not None:
        return f"{value_ms:.{decimals}f}"
    if value_ms < 1 and value_ms >= MIN_LATENCY_MS:
        return f"{value_ms:.3f}"
    if value_ms < MIN_LATENCY_MS and value_ms > 0:
        return f"{value_ms:.3f}"
    return f"{value_ms:.2f}"


def setup_latency_axis(ax):
    """Set y-axis formatter for latency (ms)."""
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: format_latency_ms(x)))


def finalize_axes(axes):
    """Force all axes to start y-axis at 0 with headroom above the data for annotations."""
    for ax in axes:
        _, ymax = ax.get_ylim()
        ax.set_ylim(bottom=0, top=ymax * 1.15)


def _plot_latency_panel(ax, xs, ys, xlabel, title, marker, color):
    """Plot a latency metric panel with log-2 x-axis and ms annotations."""
    ax.plot(xs, ys, marker, color=color, linewidth=2, markersize=7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ms")
    ax.set_title(title)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs], rotation=45, ha="right", fontsize=7)
    setup_latency_axis(ax)
    ax.grid(True, alpha=0.3)
    for x, y in zip(xs, ys):
        ax.annotate(
            format_latency_ms(y), (x, y),
            textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8,
        )


def _plot_throughput_panel(ax, xs, ys, xlabel, ylabel, title, marker, color, fmt=".1f"):
    """Plot a throughput metric panel with log-2 x-axis and numeric annotations."""
    ax.plot(xs, ys, marker, color=color, linewidth=2, markersize=7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs], rotation=45, ha="right", fontsize=7)
    ax.grid(True, alpha=0.3)
    for x, y in zip(xs, ys):
        ax.annotate(
            f"{y:{fmt}}", (x, y),
            textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8,
        )


def _extract_metrics(data: list[dict]) -> dict[str, list[float]]:
    """Extract all plottable metric series from filtered records."""
    tpot_values = [r["mean_tpot_ms"] for r in data]
    return {
        "ttft": [round_latency_ms(r["mean_ttft_ms"]) for r in data],
        "tpot": [round_latency_ms(v) for v in tpot_values],
        "itl": [round_latency_ms(r.get("mean_itl_ms", 0.0)) for r in data],
        "req_throughput": [r["request_throughput"] for r in data],
        "tok_throughput": [r.get("output_throughput", 0.0) for r in data],
        "tpu": [1000.0 / v if v > 0 else 0.0 for v in tpot_values],
    }


def _plot_phase(xs, xlabel, suptitle, data, output_path):
    """Plot a 2x3 grid: latency row (TTFT, TPOT, ITL) and throughput row (req/s, tok/s, TPU)."""
    m = _extract_metrics(data)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(suptitle, fontsize=14)

    _plot_latency_panel(axes[0, 0], xs, m["ttft"], xlabel, "Mean TTFT", "o-", "tab:blue")
    _plot_latency_panel(axes[0, 1], xs, m["tpot"], xlabel, "Mean TPOT", "s-", "tab:orange")
    _plot_latency_panel(axes[0, 2], xs, m["itl"], xlabel, "Mean ITL", "^-", "tab:purple")

    _plot_throughput_panel(
        axes[1, 0], xs, m["req_throughput"], xlabel, "Requests/s",
        "Request Throughput", "D-", "tab:green", ".2f",
    )
    _plot_throughput_panel(
        axes[1, 1], xs, m["tok_throughput"], xlabel, "Tokens/s",
        "Output Throughput", "P-", "tab:red",
    )
    _plot_throughput_panel(
        axes[1, 2], xs, m["tpu"], xlabel, "Tokens/s/user",
        "Tokens Per Second Per User", "X-", "tab:cyan",
    )

    finalize_axes(axes.flat)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_phase1_isl(records, fixed_osl: int, fixed_conc: int, output_dir: Path):
    """Impact of increasing ISL on latency and throughput metrics."""
    data = filter_records(records, _osl=fixed_osl, _conc=fixed_conc)
    if not data:
        print(f"  No data for Phase 1 (OSL={fixed_osl}, conc={fixed_conc})")
        return
    isls = [r["_isl"] for r in data]
    _plot_phase(
        isls, "Input Sequence Length (ISL)",
        f"Impact of Input Sequence Length (OSL={fixed_osl}, concurrency={fixed_conc})",
        data, output_dir / "phase1_isl_impact.png",
    )


def plot_phase2_osl(records, fixed_isl: int, fixed_conc: int, output_dir: Path):
    """Impact of increasing OSL on latency and throughput metrics."""
    data = filter_records(records, _isl=fixed_isl, _conc=fixed_conc)
    if not data:
        print(f"  No data for Phase 2 (ISL={fixed_isl}, conc={fixed_conc})")
        return
    osls = [r["_osl"] for r in data]
    _plot_phase(
        osls, "Output Sequence Length (OSL)",
        f"Impact of Output Sequence Length (ISL={fixed_isl}, concurrency={fixed_conc})",
        data, output_dir / "phase2_osl_impact.png",
    )


def plot_phase3_concurrency(records, fixed_isl: int, fixed_osl: int, output_dir: Path):
    """Impact of increasing concurrency on latency and throughput metrics."""
    data = filter_records(records, _isl=fixed_isl, _osl=fixed_osl)
    if not data:
        print(f"  No data for Phase 3 (ISL={fixed_isl}, OSL={fixed_osl})")
        return
    concs = [r["_conc"] for r in data]
    _plot_phase(
        concs, "Max Concurrency",
        f"Impact of Concurrency (ISL={fixed_isl}, OSL={fixed_osl})",
        data, output_dir / "phase3_concurrency_impact.png",
    )


def main():
    parser = argparse.ArgumentParser(description="Plot vLLM benchmark results")
    parser.add_argument(
        "results_dir", type=Path, help="Directory containing benchmark JSON files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for plots (default: results_dir)",
    )
    parser.add_argument(
        "--phase1-osl",
        type=int,
        default=128,
        help="Fixed OSL used in Phase 1 (default: 128)",
    )
    parser.add_argument(
        "--phase1-conc",
        type=int,
        default=64,
        help="Fixed concurrency used in Phase 1 (default: 64)",
    )
    parser.add_argument(
        "--phase2-isl",
        type=int,
        default=512,
        help="Fixed ISL used in Phase 2 (default: 512)",
    )
    parser.add_argument(
        "--phase2-conc",
        type=int,
        default=64,
        help="Fixed concurrency used in Phase 2 (default: 64)",
    )
    parser.add_argument(
        "--phase3-isl",
        type=int,
        default=512,
        help="Fixed ISL used in Phase 3 (default: 512)",
    )
    parser.add_argument(
        "--phase3-osl",
        type=int,
        default=512,
        help="Fixed OSL used in Phase 3 (default: 512)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {args.results_dir} ...")
    records = load_results(args.results_dir)
    if not records:
        print("No matching benchmark result files found.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(records)} result(s).\n")

    print("Phase 1: ISL impact ...")
    plot_phase1_isl(records, args.phase1_osl, args.phase1_conc, output_dir)

    print("Phase 2: OSL impact ...")
    plot_phase2_osl(records, args.phase2_isl, args.phase2_conc, output_dir)

    print("Phase 3: Concurrency impact ...")
    plot_phase3_concurrency(records, args.phase3_isl, args.phase3_osl, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
