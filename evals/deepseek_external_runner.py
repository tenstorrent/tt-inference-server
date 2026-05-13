# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

"""Orchestrate DeepSeek-R1 endpoint eval suites via lm-eval.

This module deliberately keeps endpoint-specific orchestration small and leaves
dataset loading, prompt rendering, progress bars, sample logs, and scoring to
lm-evaluation-harness through evals/scripts/helper_external_lm_eval.sh.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "evals" / "scripts"
RUNNER = SCRIPTS_DIR / "helper_external_lm_eval.sh"
PASS1_RUNNER = SCRIPTS_DIR / "run_aime24_pass1x16_external.sh"
REPORT_REF_SOURCE = (
    "https://github.com/tenstorrent/bit_sculpt/blob/main/"
    "results/deepseek-r1-0528/reports/"
    "r1_0528_compression_campaign_2026-05.md"
)
AIME_SAMPLE_REF_SOURCE = (
    "https://github.com/tenstorrent/bit_sculpt/blob/main/"
    "results/deepseek-r1-0528/r1_30_archive/reference_evals/"
    "aime_sampled_baseline.json"
)


@dataclass(frozen=True)
class Reference:
    score: float
    source: str = REPORT_REF_SOURCE


@dataclass(frozen=True)
class Benchmark:
    key: str
    name: str
    kind: str
    reference: Reference
    task: str | None = None
    metric: str | None = None
    include_path: Path | None = None
    num_fewshot: int | None = None
    dependencies: tuple[str, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class ModeConfig:
    benchmarks: tuple[str, ...]
    limit: int | None = None
    mmlu_limit: int | None = None
    mmlu_pro_limit: int | None = None
    aime_samples: int = 1
    aime_limit: int | None = None
    aime_parallel_runs: int = 1
    max_concurrent: int | None = None
    max_gen_toks: int | None = None


@dataclass
class BenchmarkResult:
    benchmark: str
    measured: float | None
    reference: float | None
    output_dir: str
    status: str
    note: str = ""


BENCHMARKS: dict[str, Benchmark] = {
    "aime24_short_sanity": Benchmark(
        key="aime24_short_sanity",
        name="AIME24 short sanity",
        kind="lm_eval",
        task="r1_aime24_short",
        metric="exact_match,none",
        include_path=REPO_ROOT / "evals" / "custom_tasks" / "r1_aime24_short",
        reference=Reference(100.0, AIME_SAMPLE_REF_SOURCE),
        num_fewshot=0,
        note="Five selected low-token AIME24 prompts used as a gentle live-endpoint sanity check.",
    ),
    "aime24_pass1": Benchmark(
        key="aime24_pass1",
        name="AIME24 pass@1",
        kind="pass1",
        task="r1_aime24",
        reference=Reference(88.96, AIME_SAMPLE_REF_SOURCE),
        num_fewshot=0,
    ),
    "aime24_majority": Benchmark(
        key="aime24_majority",
        name="AIME24 majority@1",
        kind="pass1_derived",
        task="r1_aime24",
        metric="majority_at_1_percent",
        dependencies=("aime24_pass1",),
        reference=Reference(93.33, AIME_SAMPLE_REF_SOURCE),
    ),
    "aime25_pass1": Benchmark(
        key="aime25_pass1",
        name="AIME25 pass@1",
        kind="pass1",
        task="r1_aime25",
        include_path=REPO_ROOT / "evals" / "custom_tasks" / "r1_aime25",
        reference=Reference(89.17, AIME_SAMPLE_REF_SOURCE),
        num_fewshot=0,
    ),
    "aime25_majority": Benchmark(
        key="aime25_majority",
        name="AIME25 majority@1",
        kind="pass1_derived",
        task="r1_aime25",
        metric="majority_at_1_percent",
        dependencies=("aime25_pass1",),
        reference=Reference(93.33, AIME_SAMPLE_REF_SOURCE),
    ),
    "aime24_25_pass1": Benchmark(
        key="aime24_25_pass1",
        name="AIME24+25 pass@1",
        kind="combined_pass1",
        dependencies=("aime24_pass1", "aime25_pass1"),
        reference=Reference(88.44),
    ),
    "aime24_25_majority": Benchmark(
        key="aime24_25_majority",
        name="AIME24+25 majority@1",
        kind="combined_majority",
        dependencies=("aime24_pass1", "aime25_pass1"),
        reference=Reference(91.67),
    ),
    "math500": Benchmark(
        key="math500",
        name="MATH-500 pass@1",
        kind="lm_eval",
        task="r1_math500",
        metric="exact_match,none",
        reference=Reference(85.95),
        num_fewshot=0,
    ),
    "mmlu": Benchmark(
        key="mmlu",
        name="MMLU",
        kind="lm_eval",
        task="mmlu_generative",
        metric="exact_match,get_response",
        reference=Reference(85.01),
        note="Uses lm-eval's generative MMLU task for endpoint-only scoring.",
    ),
    "mmlu_pro": Benchmark(
        key="mmlu_pro",
        name="MMLU-Pro",
        kind="lm_eval",
        task="mmlu_pro",
        metric="exact_match,custom-extract",
        reference=Reference(77.04),
    ),
    "gsm8k": Benchmark(
        key="gsm8k",
        name="GSM8K strict",
        kind="lm_eval",
        task="gsm8k",
        metric="exact_match,strict-match",
        reference=Reference(95.00),
    ),
    "arc_challenge": Benchmark(
        key="arc_challenge",
        name="ARC-Challenge",
        kind="lm_eval",
        task="arc_challenge_chat",
        metric="exact_match,remove_whitespace",
        reference=Reference(64.51),
        note="Uses the generative chat ARC task because endpoint runs do not require logprobs.",
    ),
    "humaneval_instruct": Benchmark(
        key="humaneval_instruct",
        name="HumanEval-instruct",
        kind="lm_eval",
        task="humaneval_instruct",
        metric="pass_at_1,create_test",
        reference=Reference(82.32),
    ),
    "gpqa_diamond_cot": Benchmark(
        key="gpqa_diamond_cot",
        name="GPQA-Diamond CoT",
        kind="lm_eval",
        task="r1_gpqa_diamond",
        metric="exact_match,none",
        reference=Reference(82.83),
        num_fewshot=0,
    ),
    "mbpp_instruct": Benchmark(
        key="mbpp_instruct",
        name="MBPP-instruct",
        kind="lm_eval",
        task="mbpp_instruct",
        metric="pass_at_1,extract_code",
        reference=Reference(83.80),
    ),
}


FINAL_TABLE_BENCHMARKS = (
    "aime24_pass1",
    "aime24_majority",
    "aime25_pass1",
    "aime25_majority",
    "aime24_25_pass1",
    "aime24_25_majority",
    "math500",
    "mmlu",
    "mmlu_pro",
    "gsm8k",
    "arc_challenge",
    "humaneval_instruct",
    "gpqa_diamond_cot",
    "mbpp_instruct",
)


MODE_CONFIGS: dict[str, ModeConfig] = {
    "smoke": ModeConfig(
        benchmarks=FINAL_TABLE_BENCHMARKS,
        limit=2,
        mmlu_limit=1,
        mmlu_pro_limit=2,
        aime_samples=2,
        aime_limit=1,
        aime_parallel_runs=2,
    ),
    "quick": ModeConfig(
        benchmarks=FINAL_TABLE_BENCHMARKS,
        limit=16,
        mmlu_limit=1,
        mmlu_pro_limit=16,
        aime_samples=2,
        aime_limit=16,
        aime_parallel_runs=2,
    ),
    "full": ModeConfig(
        benchmarks=FINAL_TABLE_BENCHMARKS,
        aime_samples=16,
        aime_parallel_runs=1,
    ),
    "suite": ModeConfig(
        benchmarks=FINAL_TABLE_BENCHMARKS,
        aime_samples=16,
        aime_parallel_runs=1,
    ),
    "single": ModeConfig(
        benchmarks=("aime24_pass1", "aime24_majority"),
        aime_samples=1,
        aime_parallel_runs=1,
    ),
    "sanity": ModeConfig(
        benchmarks=("aime24_short_sanity",),
        max_concurrent=1,
        max_gen_toks=32768,
    ),
}


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _latest_result_json(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.glob("**/results_*.json"))
    return candidates[-1] if candidates else None


def _pass1_summary_path(output_dir: Path, task: str) -> Path:
    return output_dir / f"{task}_pass1_summary.json"


def _read_lm_eval_score(output_dir: Path, benchmark: Benchmark) -> float:
    result_path = _latest_result_json(output_dir)
    if result_path is None:
        raise FileNotFoundError(f"No lm-eval results_*.json found under {output_dir}")
    payload = _load_json(result_path)
    results = payload.get("results", {})
    task_results = results.get(benchmark.task)
    if not isinstance(task_results, dict):
        raise KeyError(f"Task {benchmark.task!r} not found in {result_path}")
    if benchmark.metric not in task_results:
        raise KeyError(
            f"Metric {benchmark.metric!r} not found for {benchmark.task!r} in {result_path}"
        )
    return 100.0 * float(task_results[benchmark.metric])


def _read_lm_eval_sample_count(output_dir: Path, benchmark: Benchmark) -> int | None:
    result_path = _latest_result_json(output_dir)
    if result_path is None:
        return None
    payload = _load_json(result_path)
    sample_counts = payload.get("n-samples", {})
    task_counts = sample_counts.get(benchmark.task or "")
    if not isinstance(task_counts, dict):
        return None
    effective = task_counts.get("effective")
    return int(effective) if effective is not None else None


def _read_pass1_score(output_dir: Path, task: str, metric: str = "pass_at_1_percent") -> float:
    summary_path = _pass1_summary_path(output_dir, task)
    payload = _load_json(summary_path)
    return float(payload[metric])


def _read_pass1_counts(output_dir: Path, task: str) -> tuple[int, int, int, int]:
    payload = _load_json(_pass1_summary_path(output_dir, task))
    num_samples = int(payload["num_samples"])
    correct_samples = int(round(float(payload["pass_at_1"]) * num_samples))
    num_problems = int(payload["num_problems"])
    correct_majority = int(round(float(payload["majority_at_1"]) * num_problems))
    return num_samples, correct_samples, num_problems, correct_majority


def _combined_score(root: Path, metric: str) -> float:
    a24 = root / "aime24_pass1"
    a25 = root / "aime25_pass1"
    if metric == "pass1":
        samples24, correct24, _, _ = _read_pass1_counts(a24, "r1_aime24")
        samples25, correct25, _, _ = _read_pass1_counts(a25, "r1_aime25")
        return 100.0 * (correct24 + correct25) / (samples24 + samples25)
    _, _, problems24, majority24 = _read_pass1_counts(a24, "r1_aime24")
    _, _, problems25, majority25 = _read_pass1_counts(a25, "r1_aime25")
    return 100.0 * (majority24 + majority25) / (problems24 + problems25)


def _benchmark_complete(root: Path, benchmark: Benchmark) -> bool:
    out = root / benchmark.key
    try:
        if benchmark.kind == "lm_eval":
            _read_lm_eval_score(out, benchmark)
            return True
        if benchmark.kind == "pass1":
            _read_pass1_score(out, benchmark.task or "")
            return True
        if benchmark.kind == "pass1_derived":
            dep = BENCHMARKS[benchmark.dependencies[0]]
            _read_pass1_score(root / dep.key, dep.task or "", benchmark.metric or "")
            return True
        if benchmark.kind == "combined_pass1":
            _combined_score(root, "pass1")
            return True
        if benchmark.kind == "combined_majority":
            _combined_score(root, "majority")
            return True
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return False
    return False


def _run(cmd: Sequence[str], env: dict[str, str]) -> None:
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run([str(part) for part in cmd], cwd=REPO_ROOT, env=env, check=True)


def _run_lm_eval(
    benchmark: Benchmark,
    output_dir: Path,
    limit: int | None,
    env: dict[str, str],
) -> None:
    cmd: list[str] = [
        str(RUNNER),
        "--task",
        benchmark.task or "",
        "--chat-api",
        "--output-dir",
        str(output_dir),
    ]
    if benchmark.num_fewshot is not None:
        cmd.extend(["--num-fewshot", str(benchmark.num_fewshot)])
    if benchmark.include_path is not None:
        cmd.extend(["--include-path", str(benchmark.include_path)])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    _run(cmd, env)


def _run_pass1(
    benchmark: Benchmark,
    output_dir: Path,
    mode_config: ModeConfig,
    env: dict[str, str],
) -> None:
    run_env = dict(env)
    run_env["AIME_PASS1_TASK"] = benchmark.task or ""
    run_env["AIME_PASS1_RUNS"] = str(mode_config.aime_samples)
    run_env["AIME_PASS1_PARALLEL_RUNS"] = str(mode_config.aime_parallel_runs)
    run_env["OUTPUT_DIR"] = str(output_dir)
    if benchmark.include_path is not None:
        run_env["AIME_PASS1_INCLUDE_PATH"] = str(benchmark.include_path)
    if mode_config.aime_limit is not None:
        run_env["AIME_PASS1_LIMIT"] = str(mode_config.aime_limit)
    _run([str(PASS1_RUNNER)], run_env)


def _limit_for(benchmark: Benchmark, mode_config: ModeConfig) -> int | None:
    if benchmark.key == "mmlu":
        return mode_config.mmlu_limit
    if benchmark.key == "mmlu_pro":
        return mode_config.mmlu_pro_limit
    return mode_config.limit


def _collect_result(root: Path, benchmark: Benchmark, status: str) -> BenchmarkResult:
    measured: float | None
    if benchmark.kind == "lm_eval":
        measured = _read_lm_eval_score(root / benchmark.key, benchmark)
    elif benchmark.kind == "pass1":
        measured = _read_pass1_score(root / benchmark.key, benchmark.task or "")
    elif benchmark.kind == "pass1_derived":
        dep = BENCHMARKS[benchmark.dependencies[0]]
        measured = _read_pass1_score(root / dep.key, dep.task or "", benchmark.metric or "")
    elif benchmark.kind == "combined_pass1":
        measured = _combined_score(root, "pass1")
    elif benchmark.kind == "combined_majority":
        measured = _combined_score(root, "majority")
    else:
        measured = None
    return BenchmarkResult(
        benchmark=benchmark.name,
        measured=measured,
        reference=benchmark.reference.score,
        output_dir=str(root / benchmark.key),
        status=status,
        note=benchmark.note,
    )


def _format_score(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.2f}"


def _summary_markdown(mode: str, output_dir: Path, results: Sequence[BenchmarkResult]) -> str:
    lines = [
        f"# DeepSeek External Eval Summary: `{mode}`",
        "",
        f"- Output dir: `{output_dir}`",
        f"- Updated: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "| Benchmark | Measured | Reference |",
        "|---|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.benchmark} | {_format_score(result.measured)} | "
            f"{_format_score(result.reference)} |"
        )
    notes = [result for result in results if result.note]
    if notes:
        lines.extend(["", "Notes:"])
        for result in notes:
            lines.append(f"- {result.benchmark}: {result.note}")
    source_by_name = {
        benchmark.name: benchmark.reference.source for benchmark in BENCHMARKS.values()
    }
    sources = sorted(
        {
            source_by_name[result.benchmark]
            for result in results
            if result.benchmark in source_by_name
        }
    )
    lines.extend(["", "Reference sources:"])
    for source in sources:
        lines.append(f"- {source}")
    lines.append("")
    return "\n".join(lines)


def _print_table(results: Sequence[BenchmarkResult]) -> None:
    rows = [
        ("Benchmark", "Measured", "Reference"),
        *[
            (result.benchmark, _format_score(result.measured), _format_score(result.reference))
            for result in results
        ],
    ]
    widths = [max(len(row[i]) for row in rows) for i in range(3)]
    measured_rule = "-" * max(1, widths[1] - 1) + ":"
    reference_rule = "-" * max(1, widths[2] - 1) + ":"
    print()
    print(f"| {rows[0][0]:<{widths[0]}} | {rows[0][1]:>{widths[1]}} | {rows[0][2]:>{widths[2]}} |")
    print(f"| {'-' * widths[0]} | {measured_rule:>{widths[1]}} | {reference_rule:>{widths[2]}} |")
    for row in rows[1:]:
        print(f"| {row[0]:<{widths[0]}} | {row[1]:>{widths[1]}} | {row[2]:>{widths[2]}} |")


def _resolve_benchmarks(args: argparse.Namespace, mode_config: ModeConfig) -> tuple[str, ...]:
    if not args.benchmarks:
        return mode_config.benchmarks
    keys = tuple(item.strip() for item in args.benchmarks.split(",") if item.strip())
    unknown = [key for key in keys if key not in BENCHMARKS]
    if unknown:
        raise SystemExit(f"Unknown benchmark(s): {', '.join(unknown)}")
    expanded: list[str] = []

    def add_with_dependencies(key: str) -> None:
        for dep in BENCHMARKS[key].dependencies:
            add_with_dependencies(dep)
        if key not in expanded:
            expanded.append(key)

    for key in keys:
        add_with_dependencies(key)
    return tuple(expanded)


def _prepare_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        root = Path(args.output_dir).expanduser()
    else:
        root = REPO_ROOT / "eval_results" / f"deepseek_external_{args.mode}_{_timestamp()}"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _write_run_state(
    root: Path,
    mode: str,
    benchmark_keys: Sequence[str],
    results: Sequence[BenchmarkResult],
) -> None:
    _write_json(
        root / "summary.json",
        {
            "mode": mode,
            "benchmarks": list(benchmark_keys),
            "updated": datetime.now(timezone.utc).isoformat(),
            "results": [result.__dict__ for result in results],
        },
    )
    (root / "summary.md").write_text(
        _summary_markdown(mode, root, results),
        encoding="utf-8",
    )


def _apply_mode_env(env: dict[str, str], mode_config: ModeConfig) -> dict[str, str]:
    run_env = dict(env)
    if mode_config.max_concurrent is not None:
        run_env["MAX_CONCURRENT"] = str(mode_config.max_concurrent)
    if mode_config.max_gen_toks is not None:
        run_env["MAX_GEN_TOKS"] = str(mode_config.max_gen_toks)
    return run_env


def _sanity_exit_code(output_dir: Path, results: Sequence[BenchmarkResult]) -> int:
    sanity = next(
        (result for result in results if result.benchmark == "AIME24 short sanity"),
        None,
    )
    if sanity is None or sanity.measured is None:
        print(
            "Sanity failed: no AIME24 short sanity result was produced.",
            file=sys.stderr,
        )
        print(f"Output dir: {output_dir}", file=sys.stderr)
        return 1

    benchmark = BENCHMARKS["aime24_short_sanity"]
    sample_count = _read_lm_eval_sample_count(output_dir / benchmark.key, benchmark) or 5
    correct = round((sanity.measured / 100.0) * sample_count)
    if sanity.measured >= 100.0:
        print(
            f"Sanity passed: {correct}/{sample_count} short AIME24 questions correct."
        )
        return 0

    print(
        "Sanity failed: expected all short AIME24 questions to be correct, "
        f"got {correct}/{sample_count} ({sanity.measured:.2f}%).",
        file=sys.stderr,
    )
    print(f"Output dir: {output_dir}", file=sys.stderr)
    return 1


def run_suite(args: argparse.Namespace) -> int:
    mode_config = MODE_CONFIGS[args.mode]
    benchmark_keys = _resolve_benchmarks(args, mode_config)
    output_dir = _prepare_output_dir(args)
    env = _apply_mode_env(os.environ.copy(), mode_config)

    force = args.force or args.mode == "sanity"
    if force and output_dir.exists():
        for key in benchmark_keys:
            target = output_dir / key
            if target.exists():
                shutil.rmtree(target)

    results: list[BenchmarkResult] = []
    print(f"Output dir: {output_dir}")
    print(f"Benchmarks: {', '.join(benchmark_keys)}")
    if mode_config.max_concurrent is not None:
        print(f"Max concurrent requests: {mode_config.max_concurrent}")
    if mode_config.max_gen_toks is not None:
        print(f"Max generation tokens: {mode_config.max_gen_toks}")

    for index, key in enumerate(benchmark_keys, start=1):
        benchmark = BENCHMARKS[key]
        if benchmark.kind in {"pass1_derived", "combined_pass1", "combined_majority"}:
            if not all(_benchmark_complete(output_dir, BENCHMARKS[dep]) for dep in benchmark.dependencies):
                print(f"==> {benchmark.name}: waiting for dependencies")
                continue
            result = _collect_result(output_dir, benchmark, "derived")
            results.append(result)
            _write_run_state(output_dir, args.mode, benchmark_keys, results)
            continue

        bench_output = output_dir / benchmark.key
        if not force and _benchmark_complete(output_dir, benchmark):
            print(f"==> [{index}/{len(benchmark_keys)}] {benchmark.name}: already complete")
            results.append(_collect_result(output_dir, benchmark, "skipped"))
            _write_run_state(output_dir, args.mode, benchmark_keys, results)
            continue

        print(f"==> [{index}/{len(benchmark_keys)}] {benchmark.name}")
        bench_output.mkdir(parents=True, exist_ok=True)
        if benchmark.kind == "lm_eval":
            _run_lm_eval(benchmark, bench_output, _limit_for(benchmark, mode_config), env)
        elif benchmark.kind == "pass1":
            _run_pass1(benchmark, bench_output, mode_config, env)
        else:
            raise RuntimeError(f"Unsupported benchmark kind: {benchmark.kind}")

        results.append(_collect_result(output_dir, benchmark, "completed"))
        _write_run_state(output_dir, args.mode, benchmark_keys, results)

    # Re-collect in requested order so resumed runs include prior completed rows.
    complete_results = []
    for key in benchmark_keys:
        benchmark = BENCHMARKS[key]
        if _benchmark_complete(output_dir, benchmark):
            complete_results.append(_collect_result(output_dir, benchmark, "complete"))
    _write_run_state(output_dir, args.mode, benchmark_keys, complete_results)
    _print_table(complete_results)
    print(f"\nSummary: {output_dir / 'summary.md'}")
    if args.mode == "sanity":
        return _sanity_exit_code(output_dir, complete_results)
    return 0


def list_benchmarks() -> None:
    for key, benchmark in BENCHMARKS.items():
        print(f"{key}\t{benchmark.name}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepSeek-R1 endpoint eval suites.")
    parser.add_argument("mode", nargs="?", choices=sorted(MODE_CONFIGS))
    parser.add_argument(
        "--benchmarks",
        help="Comma-separated benchmark keys. Dependencies are added automatically.",
    )
    parser.add_argument(
        "--output-dir",
        help="Result directory. Reusing a directory resumes completed benchmarks.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun selected benchmarks even when existing results are present.",
    )
    parser.add_argument(
        "--list-benchmarks",
        action="store_true",
        help="Print benchmark keys and exit.",
    )
    args = parser.parse_args(argv)
    if args.list_benchmarks:
        list_benchmarks()
        raise SystemExit(0)
    if args.mode is None:
        parser.error("the following arguments are required: mode")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    return run_suite(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
