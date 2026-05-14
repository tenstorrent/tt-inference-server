# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SWEbenchRunConfig:
    task_name: str
    dataset_name: str
    dataset_split: str
    sweagent_subset: str
    agent_backend: str
    model_name: str
    api_base: str
    output_dir: Path
    sweagent_config: str
    mini_config: str
    mini_model_class: str
    mini_environment_class: str
    n_concurrent_trials: int
    max_workers: int
    n_tasks: Optional[int]
    temperature: float
    top_p: float
    max_input_tokens: int
    max_output_tokens: Optional[int]
    completion_kwargs: dict[str, Any]
    swebench_timeout_sec: Optional[int]
    shuffle: bool
    random_delay_multiplier: float
    score_existing_predictions: bool


def _run_command(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    logger.info("Running command: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _write_swebench_container_name_patch(output_dir: Path) -> Path:
    patch_dir = output_dir / "swebench_harness_patch"
    patch_dir.mkdir(parents=True, exist_ok=True)
    patch_path = patch_dir / "sitecustomize.py"
    patch_path.write_text(
        """
import re

from swebench.harness.test_spec import TestSpec

_ORIGINAL_GET_INSTANCE_CONTAINER_NAME = TestSpec.get_instance_container_name


def _get_safe_instance_container_name(self, run_id=None):
    container_name = _ORIGINAL_GET_INSTANCE_CONTAINER_NAME(self, run_id)
    container_name = re.sub(r"[^a-zA-Z0-9_.-]", "-", container_name)
    container_name = re.sub(r"^[^a-zA-Z0-9]+", "", container_name)
    return container_name or f"eval.{self.instance_id}"


TestSpec.get_instance_container_name = _get_safe_instance_container_name
""".lstrip(),
        encoding="utf-8",
    )
    return patch_dir


def _add_swebench_container_name_patch_to_env(
    output_dir: Path, env: dict[str, str]
) -> dict[str, str]:
    patched_env = dict(env)
    patch_dir = _write_swebench_container_name_patch(output_dir)
    python_path = patched_env.get("PYTHONPATH")
    patched_env["PYTHONPATH"] = (
        str(patch_dir) if not python_path else f"{patch_dir}{os.pathsep}{python_path}"
    )
    return patched_env


def _write_sweagent_model_config(config: SWEbenchRunConfig) -> Path:
    model_config: dict[str, Any] = {
        "agent": {
            "model": {
                "name": config.model_name,
                "api_base": config.api_base,
                "api_key": "$OPENAI_API_KEY",
                "temperature": config.temperature,
                "top_p": config.top_p,
                "per_instance_cost_limit": 0.0,
                "total_cost_limit": 0.0,
                "max_input_tokens": config.max_input_tokens,
            }
        }
    }
    if config.max_output_tokens is not None:
        model_config["agent"]["model"]["max_output_tokens"] = config.max_output_tokens
    if config.completion_kwargs:
        model_config["agent"]["model"]["completion_kwargs"] = config.completion_kwargs

    config_path = config.output_dir / "sweagent_model_config.yaml"
    config_path.write_text(json.dumps(model_config, indent=2), encoding="utf-8")
    return config_path


def _write_mini_sweagent_model_config(config: SWEbenchRunConfig) -> Path:
    model_kwargs: dict[str, Any] = {
        "api_base": config.api_base,
        "api_key": os.environ.get("OPENAI_API_KEY", "EMPTY"),
        "drop_params": True,
        "temperature": config.temperature,
        "top_p": config.top_p,
    }
    if config.max_output_tokens is not None:
        model_kwargs["max_tokens"] = config.max_output_tokens
    if config.completion_kwargs:
        model_kwargs.update(config.completion_kwargs)

    model_config = {
        "model": {
            "model_name": config.model_name,
            "model_class": config.mini_model_class,
            "cost_tracking": "ignore_errors",
            "model_kwargs": model_kwargs,
        }
    }
    config_path = config.output_dir / "mini_sweagent_model_config.yaml"
    config_path.write_text(json.dumps(model_config, indent=2), encoding="utf-8")
    return config_path


def _get_sweagent_source_dir() -> Optional[Path]:
    source_dir = Path(sys.executable).parent.parent / "SWE-agent"
    return source_dir if source_dir.exists() else None


def _resolve_sweagent_config_path(
    config_path: str, sweagent_source_dir: Optional[Path]
) -> str:
    path = Path(config_path)
    if path.is_absolute() or sweagent_source_dir is None:
        return str(path)

    if path.parts and path.parts[0] == "config":
        relative_path = Path(*path.parts[1:])
        return str(sweagent_source_dir / "config" / relative_path)
    return str(sweagent_source_dir / path)


def build_sweagent_command(
    config: SWEbenchRunConfig,
    sweagent_config_path: Path,
    sweagent_output_dir: Path,
) -> list[str]:
    sweagent_exec = Path(sys.executable).parent / "sweagent"
    sweagent_source_dir = _get_sweagent_source_dir()
    base_config_path = _resolve_sweagent_config_path(
        config.sweagent_config, sweagent_source_dir
    )
    cmd = [
        str(sweagent_exec),
        "run-batch",
        "--config",
        base_config_path,
        "--config",
        str(sweagent_config_path),
        "--output_dir",
        str(sweagent_output_dir),
        "--num_workers",
        str(config.n_concurrent_trials),
        "--random_delay_multiplier",
        str(config.random_delay_multiplier),
        "--instances.type",
        "swe_bench",
        "--instances.subset",
        config.sweagent_subset,
        "--instances.split",
        config.dataset_split,
        f"--instances.shuffle={str(config.shuffle).lower()}",
    ]
    if config.n_tasks is not None:
        cmd.extend(["--instances.slice", f":{config.n_tasks}"])
    return cmd


def build_mini_sweagent_command(
    config: SWEbenchRunConfig,
    mini_config_path: Path,
    mini_output_dir: Path,
) -> list[str]:
    mini_exec = Path(sys.executable).parent / "mini-extra"
    cmd = [
        str(mini_exec),
        "swebench",
        "--model",
        config.model_name,
        "--subset",
        config.sweagent_subset,
        "--split",
        config.dataset_split,
        "--workers",
        str(config.n_concurrent_trials),
        "--output",
        str(mini_output_dir),
        "--config",
        config.mini_config,
        "--config",
        str(mini_config_path),
        "--environment-class",
        config.mini_environment_class,
    ]
    if config.shuffle:
        cmd.append("--shuffle")
    if config.n_tasks is not None:
        cmd.extend(["--slice", f":{config.n_tasks}"])
    return cmd


def _find_sweagent_preds(sweagent_output_dir: Path) -> Path:
    pred_files = sorted(
        sweagent_output_dir.rglob("preds.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not pred_files:
        raise FileNotFoundError(
            f"No SWE-agent preds.json found under {sweagent_output_dir}"
        )
    return pred_files[0]


def convert_sweagent_preds_to_jsonl(
    preds_path: Path, predictions_path: Path, model_name: str
) -> list[dict[str, Any]]:
    preds = json.loads(preds_path.read_text(encoding="utf-8"))
    records = []
    if isinstance(preds, dict):
        iterable = preds.items()
    elif isinstance(preds, list):
        iterable = ((record.get("instance_id"), record) for record in preds)
    else:
        raise ValueError(f"Unsupported SWE-agent predictions format in {preds_path}")

    for instance_id, prediction in iterable:
        if not instance_id or not isinstance(prediction, dict):
            continue
        record = dict(prediction)
        record["instance_id"] = instance_id
        record.setdefault("model_name_or_path", model_name)
        if "model_patch" not in record:
            record["model_patch"] = record.get("patch", "")
        records.append(record)

    predictions_path.write_text(
        "\n".join(json.dumps(record) for record in records) + ("\n" if records else ""),
        encoding="utf-8",
    )
    return records


def build_swebench_harness_command(
    config: SWEbenchRunConfig,
    predictions_path: Path,
    run_id: str,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        config.dataset_name,
        "--split",
        config.dataset_split,
        "--predictions_path",
        str(predictions_path),
        "--max_workers",
        str(config.max_workers),
        "--run_id",
        run_id,
    ]
    if config.swebench_timeout_sec is not None:
        cmd.extend(["--timeout", str(config.swebench_timeout_sec)])
    return cmd


def _find_harness_report(output_dir: Path, model_name: str, run_id: str) -> Path:
    expected_path = output_dir / f"{model_name.replace('/', '__')}.{run_id}.json"
    if expected_path.exists():
        return expected_path

    report_files = sorted(
        output_dir.rglob(f"*.{run_id}.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not report_files:
        raise FileNotFoundError(f"No SWE-bench report found for run_id={run_id}")
    return report_files[0]


def normalize_swebench_report(
    harness_report_path: Path,
    result_path: Path,
    config: SWEbenchRunConfig,
    predictions_path: Path,
) -> dict[str, Any]:
    report = json.loads(harness_report_path.read_text(encoding="utf-8"))
    submitted_ids = set(report.get("submitted_ids", []))
    resolved_ids = set(report.get("resolved_ids", []))

    if not submitted_ids:
        submitted_count = int(report.get("submitted_instances", 0))
        resolved_count = int(report.get("resolved_instances", 0))
        unresolved_ids = set()
    else:
        submitted_count = len(submitted_ids)
        resolved_count = len(resolved_ids)
        unresolved_ids = submitted_ids - resolved_ids

    accuracy = resolved_count / submitted_count if submitted_count else 0.0
    trial_results = [
        {
            "task_name": instance_id,
            "verifier_result": {
                "rewards": {
                    "reward": 1.0 if instance_id in resolved_ids else 0.0,
                    "resolved": instance_id in resolved_ids,
                }
            },
        }
        for instance_id in sorted(submitted_ids)
    ]

    eval_key = f"{config.agent_backend}__{config.model_name}__{config.dataset_name}"
    normalized = {
        "config": {
            "datasets": [
                {
                    "name": config.dataset_name,
                    "split": config.dataset_split,
                }
            ],
            "agents": [
                {
                    "name": config.agent_backend,
                    "model_name": config.model_name,
                }
            ],
            "predictions_path": str(predictions_path),
            "swebench_report_path": str(harness_report_path),
        },
        "stats": {
            "evals": {
                eval_key: {
                    "n_trials": submitted_count,
                    "metrics": [
                        {
                            "name": "accuracy",
                            "mean": accuracy,
                        }
                    ],
                    "pass_at_k": {
                        "1": accuracy,
                    },
                    "reward_stats": {
                        "reward": {
                            "0.0": sorted(unresolved_ids),
                            "1.0": sorted(resolved_ids),
                        }
                    },
                }
            }
        },
        "trial_results": trial_results,
    }
    result_path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    return normalized


def run(config: SWEbenchRunConfig) -> int:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("OPENAI_API_KEY", "EMPTY")
    env.setdefault("OPENAI_BASE_URL", config.api_base)
    env.setdefault("OPENAI_API_BASE", config.api_base)
    env.setdefault("SWE_AGENT_LOG_STREAM_LEVEL", "INFO")
    env.setdefault("MSWEA_COST_TRACKING", "ignore_errors")
    sweagent_source_dir = _get_sweagent_source_dir()
    if sweagent_source_dir is not None:
        env.setdefault("SWE_AGENT_CONFIG_DIR", str(sweagent_source_dir / "config"))
        env.setdefault("SWE_AGENT_TOOLS_DIR", str(sweagent_source_dir / "tools"))
        env.setdefault(
            "SWE_AGENT_TRAJECTORY_DIR",
            str(sweagent_source_dir / "trajectories"),
        )

    predictions_path = config.output_dir / "predictions.jsonl"
    result_path = config.output_dir / "result.json"
    run_id = config.task_name

    if config.score_existing_predictions:
        if not predictions_path.exists():
            raise FileNotFoundError(
                f"Cannot score existing predictions; missing {predictions_path}"
            )
        logger.info("Scoring existing predictions from %s", predictions_path)
    elif config.agent_backend == "swe-agent":
        sweagent_config_path = _write_sweagent_model_config(config)
        sweagent_output_dir = config.output_dir / "sweagent"
        sweagent_cmd = build_sweagent_command(
            config, sweagent_config_path, sweagent_output_dir
        )
        _run_command(sweagent_cmd, cwd=config.output_dir, env=env)
        preds_path = _find_sweagent_preds(sweagent_output_dir)
    elif config.agent_backend == "mini-swe-agent":
        mini_config_path = _write_mini_sweagent_model_config(config)
        mini_output_dir = config.output_dir / "mini_sweagent"
        mini_cmd = build_mini_sweagent_command(
            config, mini_config_path, mini_output_dir
        )
        _run_command(mini_cmd, cwd=config.output_dir, env=env)
        preds_path = _find_sweagent_preds(mini_output_dir)
    else:
        raise ValueError(f"Unsupported SWE-bench agent backend: {config.agent_backend}")

    if not config.score_existing_predictions:
        convert_sweagent_preds_to_jsonl(preds_path, predictions_path, config.model_name)

    harness_cmd = build_swebench_harness_command(config, predictions_path, run_id)
    env = _add_swebench_container_name_patch_to_env(config.output_dir, env)
    _run_command(harness_cmd, cwd=config.output_dir, env=env)

    harness_report_path = _find_harness_report(
        config.output_dir, config.model_name, run_id
    )
    normalize_swebench_report(
        harness_report_path, result_path, config, predictions_path
    )
    logger.info("Wrote SWE-bench normalized result to %s", result_path)
    return 0
