# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Parser for Harbor-format agentic eval results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from report_module.schema import Block
from workflows.workflow_types import ReportCheckTypes

from .base import LLMResultParser


class AgenticEvalParser(LLMResultParser):
    kind = "evals"

    def __init__(
        self,
        *,
        task_name: str,
        score: Any = None,
        result_path: Optional[Path] = None,
    ) -> None:
        self.task_name = task_name
        self.score = score
        self.result_path = result_path

    def parse(self, raw: Mapping[str, Any], *, device: str = "") -> Block:
        metrics = extract_harbor_metrics(raw)
        targets: Dict[str, Any] = {"task_name": self.task_name}
        if device:
            targets["device"] = device
        if self.score is not None:
            targets.update(
                {
                    "tolerance": self.score.tolerance,
                    "published_score": self.score.published_score,
                    "published_score_ref": self.score.published_score_ref,
                }
            )
        if self.result_path is not None:
            targets["job_result_path"] = str(self.result_path)

        return Block(
            kind=self.kind,
            task_type="llm",
            title=f"Agentic Eval - {self.task_name}",
            id=_block_id(self.task_name, device),
            targets=targets,
            data={
                "success": True,
                "accuracy_check": compute_accuracy_check(metrics, self.score),
                **metrics,
            },
        )

    def failure_block(self, *, return_code: int, device: str = "") -> Block:
        targets: Dict[str, Any] = {"task_name": self.task_name}
        if device:
            targets["device"] = device
        return Block(
            kind=self.kind,
            task_type="llm",
            title=f"Agentic Eval - {self.task_name} (FAILED)",
            id=_block_id(self.task_name, device),
            targets=targets,
            data={"success": False, "accuracy_check": 3, "subprocess_rc": return_code},
        )


def extract_harbor_metrics(raw: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = _extract_harbor_summary_metrics(raw)
    _add_harbor_pass_at_metrics(raw, metrics)
    return metrics


def compute_accuracy_check(metrics: Mapping[str, Any], score: Any = None) -> int:
    """Map accuracy to the report check convention: NA=1, PASS=2, FAIL=3."""

    accuracy = metrics.get("accuracy")
    if accuracy is None or score is None:
        return ReportCheckTypes.NA
    accuracy = _normalize_accuracy_to_percent(accuracy)
    target = score.gpu_reference_score or score.published_score
    tol = score.tolerance or 0.05
    if accuracy >= target * (1 - tol):
        return ReportCheckTypes.PASS
    return ReportCheckTypes.FAIL


def _normalize_accuracy_to_percent(accuracy: Any) -> Any:
    if isinstance(accuracy, (int, float)) and 0 <= accuracy <= 1:
        return accuracy * 100
    return accuracy


def _extract_harbor_summary_metrics(raw: Mapping[str, Any]) -> Dict[str, Any]:
    stats = raw.get("stats", {})
    evals = stats.get("evals", {}) if isinstance(stats, Mapping) else {}
    if not evals:
        return {}

    eval_stats = next(iter(evals.values()))
    if not isinstance(eval_stats, Mapping):
        return {}

    metrics_list = eval_stats.get("metrics", [])
    mean_metric = next(
        (
            metric.get("mean")
            for metric in metrics_list
            if isinstance(metric, Mapping)
            and isinstance(metric.get("mean"), (int, float))
        ),
        None,
    )
    n_trials = eval_stats.get("n_trials")
    n_resolved = _count_harbor_resolved_trials(eval_stats)

    metrics: Dict[str, Any] = {}
    if mean_metric is not None:
        metrics["accuracy"] = mean_metric
        metrics["pass_at_1"] = mean_metric
    if isinstance(n_trials, int):
        metrics["n_trials"] = n_trials
    if n_resolved is not None:
        metrics["n_resolved"] = n_resolved
    return metrics


def _add_harbor_pass_at_metrics(
    raw: Mapping[str, Any],
    metrics: Dict[str, Any],
) -> None:
    stats = raw.get("stats", {})
    evals = stats.get("evals", {}) if isinstance(stats, Mapping) else {}
    for eval_stats in evals.values():
        if not isinstance(eval_stats, Mapping):
            continue
        pass_at_k = eval_stats.get("pass_at_k", {})
        if not isinstance(pass_at_k, Mapping):
            continue
        for key, value in pass_at_k.items():
            metrics[f"pass_at_{key}"] = value


def _count_harbor_resolved_trials(eval_stats: Mapping[str, Any]) -> Optional[int]:
    reward_stats = eval_stats.get("reward_stats", {})
    rewards = (
        reward_stats.get("reward", {}) if isinstance(reward_stats, Mapping) else {}
    )
    if not isinstance(rewards, Mapping):
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


def _block_id(task_name: str, device: str) -> str:
    parts = [p for p in (task_name, device) if p]
    return "_".join(parts).replace("/", "__").replace("\\", "__").replace(" ", "_")
