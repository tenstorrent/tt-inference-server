# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def generate_agentic_report_data(model_spec: Any, eval_run_id: str) -> str:
    """Generate a file path pattern to locate agentic eval result files for the
    given evaluation run ID."""
    # TODO: discuss this -> should we enforce similar structure to other evals
    #  from llms-eval for example?
    return f"eval_{eval_run_id}/agentic/*/result.json"


def is_harbor_result(data: Any) -> bool:
    ## is this future proof though?
    if not isinstance(data, dict):
        return False
    stats = data.get("stats")
    return "trial_results" in data or (
        isinstance(stats, dict) and isinstance(stats.get("evals"), dict)
    )


def _first_numeric_reward(rewards: Any) -> Optional[float]:
    if not isinstance(rewards, dict):
        return None
    preferred_keys = ("reward", "score", "accuracy", "pass", "passed", "success")
    for key in preferred_keys:
        value = rewards.get(key)
        if isinstance(value, (int, float, bool)):
            return float(value)
    for value in rewards.values():
        if isinstance(value, (int, float, bool)):
            return float(value)
    return None


def _extract_harbor_trial_score(trial_result: dict) -> Optional[float]:
    verifier_result = trial_result.get("verifier_result") or {}
    rewards = verifier_result.get("rewards") or {}
    reward = _first_numeric_reward(rewards)
    if reward is None:
        return None
    return 1.0 if reward > 0 else 0.0


def _extract_harbor_dataset_name(data: dict) -> str:
    config = data.get("config", {})
    datasets = config.get("datasets", []) if isinstance(config, dict) else []
    if not datasets:
        return "N/A"
    dataset = datasets[0]
    if not isinstance(dataset, dict):
        return "N/A"
    name = dataset.get("name") or dataset.get("path") or "N/A"
    ref = dataset.get("ref") or dataset.get("version")
    return f"{name}@{ref}" if ref else str(name)


def _count_harbor_resolved_trials(eval_stats: dict) -> Optional[int]:
    reward_stats = eval_stats.get("reward_stats", {})
    rewards = reward_stats.get("reward", {}) if isinstance(reward_stats, dict) else {}
    if not isinstance(rewards, dict):
        return None

    n_resolved = 0
    has_reward_counts = False
    for reward, trial_names in rewards.items():
        try:
            reward_value = float(reward)
        except (TypeError, ValueError):
            continue
        if reward_value <= 0 or not isinstance(trial_names, list):
            continue
        n_resolved += len(trial_names)
        has_reward_counts = True
    return n_resolved if has_reward_counts else None


def _extract_harbor_summary_metrics(data: dict) -> dict:
    stats = data.get("stats", {})
    evals = stats.get("evals", {}) if isinstance(stats, dict) else {}
    if not evals:
        return {}

    eval_stats = next(iter(evals.values()))
    metrics_list = eval_stats.get("metrics", [])
    mean_metric = next(
        (
            metric.get("mean")
            for metric in metrics_list
            if isinstance(metric, dict) and isinstance(metric.get("mean"), (int, float))
        ),
        None,
    )
    n_trials = eval_stats.get("n_trials")
    n_resolved = _count_harbor_resolved_trials(eval_stats)

    metrics: dict = {}
    if mean_metric is not None:
        metrics["accuracy"] = mean_metric
        metrics["pass_at_1"] = mean_metric
    if isinstance(n_trials, int):
        metrics["n_trials"] = n_trials
    if n_resolved is not None:
        metrics["n_resolved"] = n_resolved
    return metrics


def _add_harbor_pass_at_metrics(data: dict, metrics: dict) -> None:
    stats = data.get("stats", {})
    evals = stats.get("evals", {}) if isinstance(stats, dict) else {}
    for eval_stats in evals.values():
        pass_at_k = eval_stats.get("pass_at_k", {})
        for key, value in pass_at_k.items():
            metrics[f"pass_at_{key}"] = value


def process_agentic_eval_files(agentic_files: list) -> tuple[dict, dict]:
    agentic_files = sorted(
        agentic_files, key=lambda f: Path(f).stat().st_mtime, reverse=True
    )
    results: dict = {}
    meta_data: dict = {}

    for filepath in agentic_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            job_dir = Path(filepath).parent
            task_name = job_dir.name
            trial_results = data.get("trial_results", [])
            trial_scores = [
                score
                for score in (
                    _extract_harbor_trial_score(trial_result)
                    for trial_result in trial_results
                )
                if score is not None
            ]

            if not trial_scores:
                metrics = _extract_harbor_summary_metrics(data)
                if not metrics:
                    logger.warning("No scored Harbor trials found in %s", filepath)
                    continue
            else:
                n_resolved = sum(trial_scores)
                n_trials = len(trial_scores)
                metrics = {
                    "accuracy": n_resolved / n_trials,
                    "pass_at_1": n_resolved / n_trials,
                    "n_trials": n_trials,
                    "n_resolved": n_resolved,
                }

            _add_harbor_pass_at_metrics(data, metrics)

            if task_name in results:
                continue

            results[task_name] = metrics
            meta_data[task_name] = {
                "task_name": task_name,
                "dataset_path": _extract_harbor_dataset_name(data),
                "job_result_path": str(filepath),
            }
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Could not process agentic eval file %s: %s", filepath, e)

    return results, meta_data
