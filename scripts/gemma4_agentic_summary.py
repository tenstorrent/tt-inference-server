#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Summarize retained Gemma4 agentic CI artifacts without copying task text."""

import argparse
import json
import re
import statistics
from datetime import datetime
from pathlib import Path

KV_SAMPLE = re.compile(
    r"Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%"
)
THROUGHPUT_SAMPLE = re.compile(
    r"Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s"
)
TRACE_WARM = re.compile(r"\d{4}-\d\d-\d\d (\d\d:\d\d:\d\d\.\d+).*Warming model trace")
TRACE_CAPTURE = re.compile(
    r"\d{4}-\d\d-\d\d (\d\d:\d\d:\d\d\.\d+).*Capturing model trace"
)


def seconds_between(start, end):
    if not start or not end:
        return None
    return (
        datetime.fromisoformat(end.replace("Z", "+00:00"))
        - datetime.fromisoformat(start.replace("Z", "+00:00"))
    ).total_seconds()


def time_of_day(value):
    hour, minute, second = value.split(":")
    return int(hour) * 3600 + int(minute) * 60 + float(second)


def summarize_server(root):
    logs = sorted(root.rglob("docker_server/*.log"))
    samples, throughput, warmups, captures = [], [], [], []
    seen = set()
    for path in logs:
        for line in path.open(errors="replace"):
            sample = KV_SAMPLE.search(line)
            if sample:
                # CI can append the same Docker log more than once.
                key = (line[:40], sample.group(0))
                if key not in seen:
                    seen.add(key)
                    samples.append((int(sample[1]), int(sample[2]), float(sample[3])))
                    rate = THROUGHPUT_SAMPLE.search(line)
                    if rate:
                        throughput.append((float(rate[1]), float(rate[2])))
            warm = TRACE_WARM.search(line)
            if warm:
                warmups.append(time_of_day(warm[1]))
            capture = TRACE_CAPTURE.search(line)
            if capture:
                captures.append(time_of_day(capture[1]))
    kv = sorted(s[2] for s in samples)
    running_counts = {
        str(count): sum(row[0] == count for row in samples)
        for count in sorted({row[0] for row in samples})
    }
    intervals = [b - a for a, b in zip(warmups, captures) if 0 <= b - a < 3600]
    return {
        "log_files": [str(p.relative_to(root)) for p in logs],
        "samples": len(samples),
        "running_peak": max((s[0] for s in samples), default=None),
        "running_counts": running_counts,
        "prompt_positive_fraction": sum(row[0] > 0 for row in throughput)
        / len(throughput)
        if throughput
        else None,
        "generation_positive_fraction": sum(row[1] > 0 for row in throughput)
        / len(throughput)
        if throughput
        else None,
        "both_positive_fraction": sum(row[0] > 0 and row[1] > 0 for row in throughput)
        / len(throughput)
        if throughput
        else None,
        "median_positive_generation_tps": statistics.median(
            row[1] for row in throughput if row[1] > 0
        )
        if any(row[1] > 0 for row in throughput)
        else None,
        "waiting_positive_fraction": sum(s[1] > 0 for s in samples) / len(samples)
        if samples
        else None,
        "kv_peak_pct": max(kv) if kv else None,
        "kv_p95_pct": kv[int(0.95 * (len(kv) - 1))] if kv else None,
        "trace_warmups": len(warmups),
        "trace_captures": len(captures),
        "warm_to_capture_total_s": sum(intervals) if intervals else None,
        "warm_to_capture_median_s": statistics.median(intervals) if intervals else None,
    }


def fit_request_latency(rows):
    """Fit API seconds to intercept, prompt kilotokens and output kilotokens."""
    if len(rows) < 4:
        return None
    normal = [[0.0] * 4 for _ in range(3)]
    for prompt, output, elapsed in rows:
        features = (1.0, prompt / 1000, output / 1000)
        for i in range(3):
            for j in range(3):
                normal[i][j] += features[i] * features[j]
            normal[i][3] += features[i] * elapsed
    for i in range(3):
        pivot = max(range(i, 3), key=lambda row: abs(normal[row][i]))
        normal[i], normal[pivot] = normal[pivot], normal[i]
        if abs(normal[i][i]) < 1e-10:
            return None
        factor = normal[i][i]
        normal[i] = [value / factor for value in normal[i]]
        for j in range(3):
            if j != i:
                factor = normal[j][i]
                normal[j] = [a - factor * b for a, b in zip(normal[j], normal[i])]
    intercept, prompt_s_per_ktok, output_s_per_ktok = (row[3] for row in normal)
    mean = sum(row[2] for row in rows) / len(rows)
    residual = sum(
        (
            elapsed
            - intercept
            - prompt_s_per_ktok * prompt / 1000
            - output_s_per_ktok * output / 1000
        )
        ** 2
        for prompt, output, elapsed in rows
    )
    variance = sum((row[2] - mean) ** 2 for row in rows)
    return {
        "matched_requests": len(rows),
        "input_tokens": sum(row[0] for row in rows),
        "output_tokens": sum(row[1] for row in rows),
        "api_s": sum(row[2] for row in rows),
        "intercept_s": intercept,
        "prompt_s_per_1000_tokens": prompt_s_per_ktok,
        "output_s_per_1000_tokens": output_s_per_ktok,
        "r_squared": 1 - residual / variance if variance else None,
    }


def summarize_task(path):
    summary = json.loads(path.read_text())
    stats = summary.get("stats") or {}
    evals = list((stats.get("evals") or {}).values())
    rewards = (
        ((evals[0].get("reward_stats") or {}).get("reward") or {}) if evals else {}
    )
    case_paths = sorted(p for p in path.parent.glob("*/result.json") if p != path)
    cases, request_rows, unmatched_cases = [], [], []
    for case_path in case_paths:
        item = json.loads(case_path.read_text())
        agent = item.get("agent_result") or {}
        request_ms = (agent.get("metadata") or {}).get("api_request_times_msec") or []
        wall = seconds_between(item.get("started_at"), item.get("finished_at"))
        api_s = sum(request_ms) / 1000 if request_ms else None
        reward = ((item.get("verifier_result") or {}).get("rewards") or {}).get(
            "reward"
        )
        cases.append(
            {
                "id": item.get("task_name", case_path.parent.name),
                "reward": reward,
                "wall_s": wall,
                "api_s": api_s,
                "non_api_s": wall - api_s
                if wall is not None and api_s is not None
                else None,
                "requests": len(request_ms) if request_ms else None,
                "input_tokens": agent.get("n_input_tokens"),
                "output_tokens": agent.get("n_output_tokens"),
                "exception": (item.get("exception_info") or {}).get("exception_type"),
            }
        )
        trajectory_path = case_path.parent / "agent" / "trajectory.json"
        if path.parent.name == "terminal_bench_2" and trajectory_path.exists():
            trajectory = json.loads(trajectory_path.read_text())
            metrics = [
                step["metrics"]
                for step in trajectory.get("steps") or []
                if isinstance(step.get("metrics"), dict)
                and "prompt_tokens" in step["metrics"]
                and "completion_tokens" in step["metrics"]
            ]
            if len(metrics) == len(request_ms):
                request_rows.extend(
                    (metric["prompt_tokens"], metric["completion_tokens"], ms / 1000)
                    for metric, ms in zip(metrics, request_ms)
                )
            else:
                unmatched_cases.append(item.get("task_name", case_path.parent.name))
    if path.parent.name == "swe_bench_verified":
        trajectories = path.parent / "mini_sweagent"
        for item in summary.get("trial_results") or []:
            case_id = item.get("task_name")
            trajectory = (
                trajectories / case_id / f"{case_id}.traj.json" if case_id else None
            )
            info = {}
            usages = []
            if trajectory and trajectory.exists():
                trajectory_data = json.loads(trajectory.read_text())
                info = trajectory_data.get("info") or {}
                usages = [
                    usage
                    for message in trajectory_data.get("messages") or []
                    if (
                        usage := (
                            ((message.get("extra") or {}).get("response") or {}).get(
                                "usage"
                            )
                            or {}
                        )
                    ).get("prompt_tokens")
                    is not None
                ]
            model_stats = info.get("model_stats") or {}
            cases.append(
                {
                    "id": case_id,
                    "reward": (
                        (item.get("verifier_result") or {}).get("rewards") or {}
                    ).get("reward"),
                    "api_calls": model_stats.get("api_calls"),
                    "exit_status": info.get("exit_status"),
                    "max_prompt_tokens": max(
                        (usage["prompt_tokens"] for usage in usages), default=None
                    ),
                    "max_total_tokens": max(
                        (usage.get("total_tokens") or 0 for usage in usages),
                        default=None,
                    ),
                    "output_tokens": sum(
                        usage.get("completion_tokens") or 0 for usage in usages
                    )
                    if usages
                    else None,
                }
            )
    result = {
        "task": path.parent.name,
        "result_path": str(path),
        "wall_s": seconds_between(
            summary.get("started_at"), summary.get("finished_at")
        ),
        "trials": stats.get(
            "n_completed_trials", evals[0].get("n_trials") if evals else None
        ),
        "errors": stats.get(
            "n_errored_trials", evals[0].get("n_errors") if evals else None
        ),
        "rewarded": len(rewards.get("1.0", [])),
        "zero_reward": len(rewards.get("0.0", [])),
        "input_tokens": stats.get("n_input_tokens"),
        "output_tokens": stats.get("n_output_tokens"),
        "cases": cases,
    }
    timed_cases = [case for case in cases if case.get("wall_s") is not None]
    result["total_case_wall_s"] = (
        sum(case["wall_s"] for case in timed_cases) if timed_cases else None
    )
    result["observed_case_parallelism"] = (
        result["total_case_wall_s"] / result["wall_s"]
        if result["wall_s"] and result["total_case_wall_s"] is not None
        else None
    )
    api_cases = [case for case in timed_cases if case.get("api_s") is not None]
    result["total_api_s"] = (
        sum(case["api_s"] for case in api_cases) if api_cases else None
    )
    result["total_non_api_s"] = (
        sum(case["non_api_s"] for case in api_cases) if api_cases else None
    )
    if path.parent.name == "terminal_bench_2":
        result["request_latency_fit"] = fit_request_latency(request_rows)
        result["unmatched_request_cases"] = unmatched_cases
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", type=Path, help="directory extracted by gh run download"
    )
    parser.add_argument("--output", type=Path, help="write compact JSON here")
    args = parser.parse_args()
    root = args.root.resolve()
    paths = sorted(
        p
        for p in root.rglob("result.json")
        if p.parent.name.startswith(("terminal_bench_2", "swe_bench_verified"))
        and "stats" in json.loads(p.read_text())
    )
    result = {
        "root": str(root),
        "server": summarize_server(root),
        "tasks": [summarize_task(p) for p in paths],
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
