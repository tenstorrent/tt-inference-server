# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import re
import time

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from llm_module.agentic.progress import (
    TIMEOUT_EXIT_CODE,
    make_swebench_probe,
    run_with_progress,
    worst_case_ceiling_s,
)

logger = logging.getLogger(__name__)

# The SWE-bench harness builds/pulls Docker images (a shared base image plus
# per-instance images) from ghcr.io. Those transfers can fail transiently
# mid-stream (e.g. ``ChunkedEncodingError: Response ended prematurely`` while
# pulling ``ghcr.io/epoch-research/swe-bench.base.x86_64``). Retry a few times
# before giving up; both counts are env-tunable for CI.
_HARNESS_MAX_ATTEMPTS = 3
_HARNESS_RETRY_DELAY_SEC = 30

# litellm's OpenAI-compatible path defaults to httpx.Timeout(None) -- an
# infinite read timeout -- so without an explicit value a never-answered
# request blocks the worker thread forever.
DEFAULT_LLM_TIMEOUT_SEC = 10 * 60
# Per-task budget B for the wave-aware deadline model. mini-swe-agent runs each
# instance in its own container started with ``sleep <this>``; once it exits no
# further agent action can succeed, so this is the authoritative wall-clock
# ceiling for a single instance.
DEFAULT_MINI_CONTAINER_TIMEOUT_SEC = 2 * 60 * 60
# Grace for dataset load + image pulls before the first wave can start.
DEFAULT_STARTUP_GRACE_SEC = 10 * 60
# If no instance completes for ``B + stall_grace`` the run is wedged (every
# in-flight instance is necessarily past its own budget); kill it.
DEFAULT_STALL_GRACE_SEC = 5 * 60
# How often the progress watchdog logs elapsed / percent / max-allowed time.
DEFAULT_PROGRESS_LOG_INTERVAL_SEC = 5 * 60


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
    # Per-request LLM timeout (seconds) written into the agent model config;
    # ``None`` keeps the client default (infinite for litellm's OpenAI path).
    llm_timeout_sec: Optional[int] = DEFAULT_LLM_TIMEOUT_SEC
    # Wave-aware deadline model (see progress.py). ``mini_container_timeout_sec``
    # is the per-task budget B, also written into the mini config as
    # ``environment.container_timeout``.
    mini_container_timeout_sec: int = DEFAULT_MINI_CONTAINER_TIMEOUT_SEC
    startup_grace_sec: int = DEFAULT_STARTUP_GRACE_SEC
    stall_grace_sec: int = DEFAULT_STALL_GRACE_SEC
    progress_log_interval_sec: int = DEFAULT_PROGRESS_LOG_INTERVAL_SEC
    # Explicit flat wall-clock kill for the agent subprocess. ``None`` uses the
    # wave-aware ceiling derived from the fields above; set to override.
    agent_subprocess_timeout_sec: Optional[int] = None
    # When False the progress watchdog logs deadlines but never kills the agent
    # subprocess, letting it run to completion. Killing early leaves unfinished
    # instances out of ``preds.json``, and the score is resolved/submitted over
    # exactly those predictions, so an early kill inflates accuracy.
    enforce_agent_deadline: bool = False
    instance_ids: list[str] = field(default_factory=list)
    # Interpreter whose bin/ holds the ``sweagent`` / ``mini-extra`` CLIs and
    # whose ``-m swebench`` is importable. ``None`` uses the current interpreter
    # (standalone ``run_agentic.py`` re-execs into the EVALS_AGENTIC venv); set
    # on the release path where the harness runs as a child of the engine.
    venv_python: Optional[Path] = None


def _interpreter(config: SWEbenchRunConfig) -> Path:
    return Path(config.venv_python) if config.venv_python else Path(sys.executable)


def _mini_output_dir(config: SWEbenchRunConfig) -> Path:
    """Where ``mini-extra swebench`` writes preds.json, logs and per-instance dirs."""
    return config.output_dir / "mini_sweagent"


def _resolve_total(config: SWEbenchRunConfig) -> Optional[int]:
    """Instance count known up front: explicit ids win, else ``n_tasks``."""
    if config.instance_ids:
        return len(config.instance_ids)
    return config.n_tasks


def _flat_agent_timeout(config: SWEbenchRunConfig) -> Optional[float]:
    """Flat wall-clock bound for backends without a progress probe.

    Explicit ``agent_subprocess_timeout_sec`` wins; otherwise derive the
    wave-aware worst-case ceiling when the total is known, else ``None``.
    """
    if config.agent_subprocess_timeout_sec is not None:
        return config.agent_subprocess_timeout_sec
    total = _resolve_total(config)
    if total:
        return worst_case_ceiling_s(
            total,
            config.n_concurrent_trials,
            config.mini_container_timeout_sec,
        )
    return None


def _run_command(
    cmd: list[str], cwd: Path, env: dict[str, str], timeout_s: Optional[float] = None
) -> int:
    logger.info("Running command: %s", " ".join(cmd))
    try:
        return subprocess.run(cmd, cwd=cwd, env=env, timeout=timeout_s).returncode
    except subprocess.TimeoutExpired:
        # 124 matches /usr/bin/timeout and llm_module.drivers._subprocess so
        # callers treat a timed-out command like any other nonzero exit.
        logger.error(
            "Command exceeded timeout of %.0fs and was killed: %s",
            timeout_s,
            " ".join(cmd),
        )
        return 124


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, TypeError, ValueError):
        return default


def _run_command_with_retries(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    *,
    max_attempts: int = _HARNESS_MAX_ATTEMPTS,
    retry_delay_sec: float = _HARNESS_RETRY_DELAY_SEC,
) -> int:
    """Run ``cmd``, retrying on non-zero exit up to ``max_attempts`` times.

    Intended for the SWE-bench harness step, whose Docker image builds/pulls can
    fail on transient network/registry errors. The harness caches images that
    already built and re-scores against the existing predictions, so a retry
    only rebuilds what failed -- cheap and idempotent, unlike re-running the
    agent (which would re-invoke the LLM).
    """
    attempt = 1
    while True:
        rc = _run_command(cmd, cwd=cwd, env=env)
        if rc == 0 or attempt >= max_attempts:
            return rc
        logger.warning(
            "Command failed (rc=%s) on attempt %s/%s; retrying in %ss: %s",
            rc,
            attempt,
            max_attempts,
            retry_delay_sec,
            " ".join(cmd),
        )
        time.sleep(retry_delay_sec)
        attempt += 1


def _write_swebench_harness_patch(output_dir: Path) -> Path:
    patch_dir = output_dir / "swebench_harness_patch"
    patch_dir.mkdir(parents=True, exist_ok=True)
    patch_path = patch_dir / "sitecustomize.py"
    patch_path.write_text(
        """
import logging
import re

from swebench.harness.test_spec import TestSpec

_ORIGINAL_GET_INSTANCE_CONTAINER_NAME = TestSpec.get_instance_container_name


def _get_safe_instance_container_name(self, run_id=None):
    container_name = _ORIGINAL_GET_INSTANCE_CONTAINER_NAME(self, run_id)
    container_name = re.sub(r"[^a-zA-Z0-9_.-]", "-", container_name)
    container_name = re.sub(r"^[^a-zA-Z0-9]+", "", container_name)
    return container_name or f"eval.{self.instance_id}"


TestSpec.get_instance_container_name = _get_safe_instance_container_name


# The epoch-research SWE-bench fork's build_image() pushes every freshly built
# image to ghcr.io/epoch-research/... as a shared cache. We have no push
# credentials for that namespace, so the push fails -- often mid-stream with
# ``ChunkedEncodingError: Response ended prematurely`` -- and, because it runs
# inside build_image()'s try block, turns a successful local build into a fatal
# BuildImageError. Make the push best-effort so local builds are kept and used.
try:
    from docker.models.images import ImageCollection

    _ORIGINAL_PUSH = ImageCollection.push

    def _best_effort_push(self, *args, **kwargs):
        try:
            return _ORIGINAL_PUSH(self, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            logging.getLogger("swebench.harness.docker_build").warning(
                "Ignoring non-fatal SWE-bench image push failure: %s", exc
            )
            return ""

    ImageCollection.push = _best_effort_push
except Exception:  # noqa: BLE001
    pass
""".lstrip(),
        encoding="utf-8",
    )
    return patch_dir


def _add_swebench_harness_patch_to_env(
    output_dir: Path, env: dict[str, str]
) -> dict[str, str]:
    patched_env = dict(env)
    patch_dir = _write_swebench_harness_patch(output_dir)
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
    completion_kwargs = dict(config.completion_kwargs or {})
    if config.llm_timeout_sec is not None:
        completion_kwargs.setdefault("timeout", config.llm_timeout_sec)
    if completion_kwargs:
        model_config["agent"]["model"]["completion_kwargs"] = completion_kwargs

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
    if config.llm_timeout_sec is not None:
        model_kwargs.setdefault("timeout", config.llm_timeout_sec)

    model_section: dict[str, Any] = {
        "model_name": config.model_name,
        "model_class": config.mini_model_class,
        "cost_tracking": "ignore_errors",
        "model_kwargs": model_kwargs,
    }
    model_config: dict[str, Any] = {"model": model_section}
    # Pin the container lifetime to our per-task budget B rather than inheriting
    # mini-swe-agent's default ("2h"). ``container_timeout`` is passed verbatim
    # to ``docker run ... sleep <value>``; use an explicit seconds suffix.
    if config.mini_container_timeout_sec is not None:
        model_config["environment"] = {
            "container_timeout": f"{int(config.mini_container_timeout_sec)}s"
        }
    config_path = config.output_dir / "mini_sweagent_model_config.yaml"
    config_path.write_text(json.dumps(model_config, indent=2), encoding="utf-8")
    return config_path


def _get_sweagent_source_dir(interpreter: Optional[Path] = None) -> Optional[Path]:
    base = Path(interpreter) if interpreter else Path(sys.executable)
    source_dir = base.parent.parent / "SWE-agent"
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
    interpreter = _interpreter(config)
    sweagent_exec = interpreter.parent / "sweagent"
    sweagent_source_dir = _get_sweagent_source_dir(interpreter)
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
    mini_exec = _interpreter(config).parent / "mini-extra"
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
    if config.instance_ids:
        regex = "^(" + "|".join(re.escape(iid) for iid in config.instance_ids) + ")$"
        cmd.extend(["--filter", regex])
    elif config.n_tasks is not None:
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
        str(_interpreter(config)),
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


# Leading timestamp of an agent log line, e.g.
# ``2026-07-02 02:41:45,703 - minisweagent - INFO - ...``. Also matches the
# ISO ``T`` separator and dot-milliseconds so both agent backends parse.
_LOG_TIMESTAMP_RE = re.compile(r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}[.,]\d+)")


def _parse_log_timestamp(line: str) -> Optional[datetime]:
    match = _LOG_TIMESTAMP_RE.search(line)
    if match is None:
        return None
    try:
        return datetime.fromisoformat(
            match.group(1).replace(",", ".").replace("T", " ")
        )
    except ValueError:
        return None


def _agent_log_time_window(
    output_dir: Path,
) -> tuple[Optional[datetime], Optional[datetime]]:
    """Earliest and latest timestamps across the agent log(s) under output_dir.

    SWE-bench's normalized report has no timing, but the mini-swe-agent /
    SWE-agent logs are timestamped per line (e.g.
    ``.../mini_sweagent/minisweagent.log``). We take the first timestamp of the
    run as the start and the last as the finish so the report can derive a mean
    time per task, mirroring terminal-bench's Harbor timing fields.
    """
    earliest: Optional[datetime] = None
    latest: Optional[datetime] = None
    for log_path in sorted(Path(output_dir).rglob("*.log")):
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            ts = _parse_log_timestamp(line)
            if ts is not None:
                if earliest is None or ts < earliest:
                    earliest = ts
                break
        for line in reversed(lines):
            ts = _parse_log_timestamp(line)
            if ts is not None:
                if latest is None or ts > latest:
                    latest = ts
                break
    return earliest, latest


def normalize_swebench_report(
    harness_report_path: Path,
    result_path: Path,
    config: SWEbenchRunConfig,
    predictions_path: Path,
) -> dict[str, Any]:
    """Normalize the SWE-bench harness report into harbor-format result.json."""
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

    # The harness only knows the instances that reached preds.json, so grading
    # against that count would score an interrupted run as resolved/finished and
    # inflate accuracy. Grade against the instance count we asked for; anything
    # missing never produced a patch and is by definition unresolved.
    missing_ids: set[str] = set()
    if config.instance_ids:
        missing_ids = set(config.instance_ids) - submitted_ids
    graded_count = submitted_count + len(missing_ids)
    expected_total = _resolve_total(config)
    if expected_total is not None:
        graded_count = max(graded_count, expected_total)
    if graded_count > submitted_count + len(missing_ids):
        # n_tasks told us how many to expect but not which ones, so the gap can
        # only widen the denominator, not name the absent instances.
        logger.warning(
            "Grading %s instances but only %s reached the report; counting the "
            "%s unaccounted instance(s) as unresolved.",
            graded_count,
            submitted_count,
            graded_count - submitted_count,
        )
    elif missing_ids:
        logger.warning(
            "Counting %s requested instance(s) with no prediction as unresolved: %s",
            len(missing_ids),
            ", ".join(sorted(missing_ids)),
        )

    unresolved_ids |= missing_ids
    accuracy = resolved_count / graded_count if graded_count else 0.0
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
        for instance_id in sorted(submitted_ids | missing_ids)
    ]

    eval_key = f"{config.agent_backend}__{config.model_name}__{config.dataset_name}"
    normalized = {
        "_result_format": "harbor",
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
                    "n_trials": graded_count,
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

    # Inject start/finish/n_total_trials so the report can compute a mean time
    # per task, matching terminal-bench's Harbor timing fields. SWE-bench's own
    # report carries no timing, so derive the window from the agent log.
    started, finished = _agent_log_time_window(config.output_dir)
    if started is not None and finished is not None:
        normalized["started_at"] = started.isoformat()
        normalized["finished_at"] = finished.isoformat()
    normalized["n_total_trials"] = graded_count

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
    sweagent_source_dir = _get_sweagent_source_dir(_interpreter(config))
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
        # swe-agent's output layout differs from mini-swe-agent's, so no
        # progress probe; bound it with the explicit override or the derived
        # worst-case ceiling. Honour enforce_agent_deadline here too, otherwise
        # this backend would keep killing while mini-swe-agent does not.
        rc = _run_command(
            sweagent_cmd,
            cwd=config.output_dir,
            env=env,
            timeout_s=(
                _flat_agent_timeout(config) if config.enforce_agent_deadline else None
            ),
        )
        # A timeout kill (124) still leaves partial predictions worth grading;
        # only a genuine agent error aborts early.
        if rc != 0 and rc != TIMEOUT_EXIT_CODE:
            return rc
        try:
            preds_path = _find_sweagent_preds(sweagent_output_dir)
        except FileNotFoundError:
            logger.error(
                "Agent timed out before writing any predictions; nothing to grade."
            )
            return rc if rc != 0 else TIMEOUT_EXIT_CODE
        if rc == TIMEOUT_EXIT_CODE:
            logger.warning(
                "Agent hit the deadline; grading the partial predictions in %s.",
                preds_path,
            )
    elif config.agent_backend == "mini-swe-agent":
        mini_config_path = _write_mini_sweagent_model_config(config)
        mini_output_dir = _mini_output_dir(config)
        mini_cmd = build_mini_sweagent_command(
            config, mini_config_path, mini_output_dir
        )
        rc = run_with_progress(
            mini_cmd,
            cwd=config.output_dir,
            env=env,
            probe=make_swebench_probe(mini_output_dir, _resolve_total(config)),
            label=config.task_name,
            per_task_budget_s=config.mini_container_timeout_sec,
            concurrency=config.n_concurrent_trials,
            startup_grace_s=config.startup_grace_sec,
            stall_grace_s=config.stall_grace_sec,
            log_interval_s=config.progress_log_interval_sec,
            hard_timeout_s=config.agent_subprocess_timeout_sec,
            enforce_deadlines=config.enforce_agent_deadline,
            log=logger,
        )
        # A watchdog timeout (124) still leaves partial predictions in
        # preds.json worth grading; only a genuine agent error aborts early.
        if rc != 0 and rc != TIMEOUT_EXIT_CODE:
            return rc
        try:
            preds_path = _find_sweagent_preds(mini_output_dir)
        except FileNotFoundError:
            # Killed before any instance finished -> nothing to grade.
            logger.error(
                "Agent timed out before writing any predictions; nothing to grade."
            )
            return rc if rc != 0 else TIMEOUT_EXIT_CODE
        if rc == TIMEOUT_EXIT_CODE:
            logger.warning(
                "Agent hit the deadline; grading the partial predictions in %s.",
                preds_path,
            )
    else:
        raise ValueError(f"Unsupported SWE-bench agent backend: {config.agent_backend}")

    if not config.score_existing_predictions:
        convert_sweagent_preds_to_jsonl(preds_path, predictions_path, config.model_name)

    harness_cmd = build_swebench_harness_command(config, predictions_path, run_id)
    env = _add_swebench_harness_patch_to_env(config.output_dir, env)
    rc = _run_command_with_retries(
        harness_cmd,
        cwd=config.output_dir,
        env=env,
        max_attempts=_env_int("SWEBENCH_HARNESS_MAX_ATTEMPTS", _HARNESS_MAX_ATTEMPTS),
        retry_delay_sec=_env_int(
            "SWEBENCH_HARNESS_RETRY_DELAY_SEC", _HARNESS_RETRY_DELAY_SEC
        ),
    )
    if rc != 0:
        return rc

    harness_report_path = _find_harness_report(
        config.output_dir, config.model_name, run_id
    )
    normalize_swebench_report(
        harness_report_path, result_path, config, predictions_path
    )
    logger.info("Wrote SWE-bench normalized result to %s", result_path)
    return 0
