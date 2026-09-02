# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from __future__ import annotations

import hashlib
import json
import logging
import os
import signal
import subprocess
import sys
import re
import tempfile
import time

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_TOKEN_BUDGET_MODEL_CLASS = (
    "llm_module.agentic.mini_swe_token_budget.TokenBudgetLitellmModel"
)

_MINI_SWE_BUDGET_DISCIPLINE_CORE = """\
You are a software engineering agent with a bounded interaction and context budget.
Use repository evidence only; do not assume task-specific implementation details.
Work in this order: reproduce the failure with the smallest useful check, locate the
causal source, edit the source, run focused verification, inspect the diff, and submit.
Do not repeat an identical or substantially overlapping read/search unless the prior
result was truncated and the next command narrows the missing region. Prefer targeted
searches and line ranges over full-file reads.
"""


def render_budget_discipline_template(step_limit: Optional[int] = None) -> str:
    """Render the shared mini-swe system template with the run's step budget.

    mini-swe-agent 2.2.8 gives the model no step feedback of its own, so
    schedule discipline is only actionable when the concrete numbers are
    rendered into the prompt at config-write time. ``step_limit=None`` keeps
    the budget-agnostic core text and makes no numeric promises.
    """
    if step_limit is None:
        return _MINI_SWE_BUDGET_DISCIPLINE_CORE
    if (
        not isinstance(step_limit, int)
        or isinstance(step_limit, bool)
        or step_limit <= 0
    ):
        raise ValueError("step_limit must be a positive integer or None")
    act_by = (step_limit + 1) // 2  # ceil(step_limit / 2)
    reserve_from = max(1, step_limit - 3)
    reserve_count = step_limit - reserve_from + 1
    return _MINI_SWE_BUDGET_DISCIPLINE_CORE + (
        f"You have exactly {step_limit} interaction steps total; each command "
        "consumes one step and the run stops when they are spent.\n"
        "Select the best-supported causal location and begin your smallest "
        f"general source change no later than step {act_by}.\n"
        f"Reserve steps {reserve_from}–{step_limit} (the final "
        f"{reserve_count}) for mutation, verification, diff inspection, and "
        "submission.\n"
    )


# The SWE-bench harness builds/pulls Docker images (a shared base image plus
# per-instance images) from ghcr.io. Those transfers can fail transiently
# mid-stream (e.g. ``ChunkedEncodingError: Response ended prematurely`` while
# pulling ``ghcr.io/epoch-research/swe-bench.base.x86_64``). Retry a few times
# before giving up; both counts are env-tunable for CI.
_HARNESS_MAX_ATTEMPTS = 3
_HARNESS_RETRY_DELAY_SEC = 30


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
    agent_generation_timeout_sec: Optional[int]
    shuffle: bool
    random_delay_multiplier: float
    score_existing_predictions: bool
    instance_ids: list[str] = field(default_factory=list)
    mini_agent_kwargs: dict[str, Any] = field(default_factory=dict)
    # Maximum command-output characters retained in model-visible history.
    # The exact tokenizer still counts the complete retained request and
    # remains the authoritative input-envelope gate.
    mini_observation_chars: Optional[int] = None
    # Qualification identity copied from the catalogue into the detached
    # verifier result. Local/report-only runs remain runnable, but only a
    # models_ci_graded result is eligible to support release promotion.
    qualification_claim: str = "local_behavioral_only"
    selection_policy: Optional[str] = None
    instance_selection_provenance: Optional[str] = None
    dataset_revision: Optional[str] = None
    ordered_instance_ids_sha256: Optional[str] = None
    selected_instances_sha256: Optional[str] = None
    eval_limit_mode: Optional[str] = None
    # Exact Hugging Face tokenizer used to render/count each mini-swe request.
    # Kept separate from model_name, which includes LiteLLM's provider prefix.
    tokenizer_name: Optional[str] = None
    # Interpreter whose bin/ holds the ``sweagent`` / ``mini-extra`` CLIs and
    # whose ``-m swebench`` is importable. ``None`` uses the current interpreter
    # (standalone ``run_agentic.py`` re-execs into the EVALS_AGENTIC venv); set
    # on the release path where the harness runs as a child of the engine.
    venv_python: Optional[Path] = None


def _interpreter(config: SWEbenchRunConfig) -> Path:
    return Path(config.venv_python) if config.venv_python else Path(sys.executable)


def _canonical_selected_rows(rows: list[dict[str, Any]]) -> bytes:
    return json.dumps(
        rows, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _prepare_pinned_swebench_dataset(
    config: SWEbenchRunConfig,
) -> tuple[str, Optional[str]]:
    """Materialize one revision-pinned, content-bound local dataset snapshot.

    Both mini-swe-agent and the isolated verifier receive this same directory,
    so neither independently resolves a mutable Hub alias. Legacy/local rows
    without a declared revision keep their existing dataset source, while a
    graded run fails closed unless the selected row bytes match the catalogue.
    """
    graded = config.qualification_claim == "models_ci_graded"
    if not config.dataset_revision:
        if graded:
            raise RuntimeError(
                "graded SWE-bench requires an exact dataset revision and "
                "content-bound local snapshot"
            )
        return config.dataset_name, None
    if not config.instance_ids:
        raise RuntimeError(
            "revision-pinned SWE-bench snapshot requires exact selected instance IDs"
        )
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", config.dataset_split):
        raise RuntimeError("SWE-bench dataset split is not a safe local name")

    from datasets import load_dataset

    source = load_dataset(
        config.dataset_name,
        split=config.dataset_split,
        revision=config.dataset_revision,
    )
    by_id: dict[str, dict[str, Any]] = {}
    for raw in source:
        row = dict(raw)
        instance_id = row.get("instance_id")
        if instance_id in config.instance_ids:
            if instance_id in by_id:
                raise RuntimeError(
                    f"pinned SWE-bench dataset repeats instance {instance_id!r}"
                )
            by_id[instance_id] = row
    missing = [
        instance_id for instance_id in config.instance_ids if instance_id not in by_id
    ]
    if missing:
        raise RuntimeError(
            f"pinned SWE-bench dataset is missing selected instances: {missing}"
        )
    ordered_rows = [by_id[instance_id] for instance_id in config.instance_ids]
    content_sha256 = hashlib.sha256(_canonical_selected_rows(ordered_rows)).hexdigest()
    if graded and config.selected_instances_sha256 != content_sha256:
        raise RuntimeError(
            "graded SWE-bench selected-instance content digest mismatch: "
            f"expected={config.selected_instances_sha256!r}, actual={content_sha256!r}"
        )

    snapshot_dir = config.output_dir / "pinned_swebench_dataset"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / f"{config.dataset_split}.jsonl"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=snapshot_dir,
        prefix=".dataset.",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        for row in ordered_rows:
            stream.write(
                json.dumps(
                    row,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, snapshot_path)

    # Exercise the same dataset loader the agent and verifier will use, but do
    # not hash its inferred Python values.  Recent datasets releases infer ISO
    # date strings in JSONL as ``datetime`` objects, which is neither JSON
    # serializable nor byte-identical to the selected Hub rows.  The immutable
    # JSONL bytes are the content boundary; the loader check only proves that
    # it preserves the exact ordered instance identity.
    reloaded = [
        dict(row) for row in load_dataset(str(snapshot_dir), split=config.dataset_split)
    ]
    if [row.get("instance_id") for row in reloaded] != config.instance_ids:
        raise RuntimeError("local SWE-bench snapshot changed instance identity/order")
    snapshot_rows = [
        json.loads(line)
        for line in snapshot_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    reloaded_sha256 = hashlib.sha256(
        _canonical_selected_rows(snapshot_rows)
    ).hexdigest()
    if reloaded_sha256 != content_sha256:
        raise RuntimeError("local SWE-bench snapshot changed during materialization")
    receipt = {
        "schema": "ttis.pinned-swebench-dataset/v1",
        "dataset_name": config.dataset_name,
        "dataset_split": config.dataset_split,
        "dataset_revision": config.dataset_revision,
        "ordered_instance_ids": config.instance_ids,
        "ordered_instance_ids_sha256": config.ordered_instance_ids_sha256,
        "selected_instances_sha256": content_sha256,
        "snapshot_dir": str(snapshot_dir),
    }
    # Keep the receipt outside the dataset directory. Hugging Face treats every
    # JSON/JSONL file in a local dataset directory as data; placing metadata
    # beside ``test.jsonl`` would make the agent and verifier load different
    # schemas on their second open.
    (config.output_dir / "pinned_swebench_dataset_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return str(snapshot_dir), content_sha256


def _run_command(cmd: list[str], cwd: Path, env: dict[str, str]) -> int:
    logger.info("Running command: %s", " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, env=env).returncode


def _run_bounded_process_group(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    *,
    timeout_sec: float,
    terminate_grace_sec: float = 20.0,
    cleanup_container_label: Optional[str] = None,
) -> int:
    """Run a harness subprocess with a hard wall-clock and group cleanup."""
    if timeout_sec <= 0:
        raise ValueError(f"timeout_sec must be positive, got {timeout_sec!r}")
    logger.info(
        "Running bounded command (timeout %.0fs): %s", timeout_sec, " ".join(cmd)
    )
    process = subprocess.Popen(cmd, cwd=cwd, env=env, start_new_session=True)
    try:
        try:
            return process.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            logger.error(
                "Command exceeded %.0fs; terminating process group: %s",
                timeout_sec,
                " ".join(cmd),
            )
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=terminate_grace_sec)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            return 124
    finally:
        if cleanup_container_label:
            _cleanup_labeled_containers(cleanup_container_label, env)


def _agentic_container_label(config: SWEbenchRunConfig) -> str:
    identity = hashlib.sha256(str(config.output_dir.resolve()).encode()).hexdigest()[
        :16
    ]
    return f"ttis.agentic_run={identity}"


def _cleanup_labeled_containers(label: str, env: dict[str, str]) -> None:
    """Synchronously remove only containers carrying this run's unique label."""
    executable = env.get("MSWEA_DOCKER_EXECUTABLE", "docker")
    try:
        listed = subprocess.run(
            [executable, "ps", "-aq", "--filter", f"label={label}"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=env,
        )
        if listed.returncode != 0:
            raise RuntimeError(
                listed.stderr.strip() or f"docker ps exited {listed.returncode}"
            )
        container_ids = listed.stdout.split()
        if not container_ids:
            return
        removed = subprocess.run(
            [executable, "rm", "-f", *container_ids],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            env=env,
        )
        if removed.returncode != 0:
            raise RuntimeError(
                removed.stderr.strip() or f"docker rm exited {removed.returncode}"
            )
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        raise RuntimeError(
            f"failed to clean agent containers for label {label!r}: {exc}"
        ) from exc


def _atomic_write_json(path: Path, value: object) -> None:
    """Durably replace a JSON state file without exposing partial contents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


def _valid_prediction(record: object) -> bool:
    return (
        isinstance(record, dict)
        and isinstance(record.get("model_patch"), str)
        and bool(record["model_patch"].strip())
    )


def _load_successful_predictions(
    path: Path, expected_ids: list[str]
) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"resume state is unreadable at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"resume state at {path} is not a JSON object")
    expected = set(expected_ids)
    unknown = sorted(set(value) - expected)
    if unknown:
        raise RuntimeError(f"resume state contains unexpected instance IDs: {unknown}")
    invalid = sorted(
        instance_id for instance_id, row in value.items() if not _valid_prediction(row)
    )
    if invalid:
        raise RuntimeError(f"resume state contains failed/empty samples: {invalid}")
    return value


def _validate_prediction_file(path: Path, expected_ids: list[str]) -> dict[str, dict]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"predictions are unreadable at {path}: {exc}") from exc
    if not isinstance(value, dict) or not value:
        raise RuntimeError(f"predictions at {path} are empty or not an object")
    invalid = sorted(
        instance_id for instance_id, row in value.items() if not _valid_prediction(row)
    )
    if invalid:
        raise RuntimeError(f"predictions contain failed/empty samples: {invalid}")
    if expected_ids:
        missing = sorted(set(expected_ids) - set(value))
        extra = sorted(set(value) - set(expected_ids))
        if missing or extra:
            raise RuntimeError(
                f"prediction IDs differ from fixed selection: missing={missing}, extra={extra}"
            )
    return value


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
    if config.completion_kwargs:
        model_config["agent"]["model"]["completion_kwargs"] = config.completion_kwargs

    config_path = config.output_dir / "sweagent_model_config.yaml"
    config_path.write_text(json.dumps(model_config, indent=2), encoding="utf-8")
    return config_path


def _write_mini_sweagent_model_config(config: SWEbenchRunConfig) -> Path:
    if config.mini_model_class not in ("litellm", _TOKEN_BUDGET_MODEL_CLASS):
        raise ValueError(
            "mini-swe-agent input-budget enforcement currently requires the "
            f"LiteLLM model path, got {config.mini_model_class!r}"
        )
    if not config.tokenizer_name:
        raise ValueError(
            "mini-swe-agent input-budget enforcement requires tokenizer_name"
        )
    max_input_tokens = config.max_input_tokens
    max_output_tokens = config.max_output_tokens
    if (
        not isinstance(max_input_tokens, int)
        or isinstance(max_input_tokens, bool)
        or max_input_tokens <= 0
    ):
        raise ValueError("max_input_tokens must be a positive integer")
    if max_output_tokens is not None:
        if (
            not isinstance(max_output_tokens, int)
            or isinstance(max_output_tokens, bool)
            or max_output_tokens <= 0
        ):
            raise ValueError("max_output_tokens must be a positive integer")
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

    model_config: dict[str, Any] = {
        "model": {
            "model_name": config.model_name,
            "model_class": _TOKEN_BUDGET_MODEL_CLASS,
            "cost_tracking": "ignore_errors",
            "model_kwargs": model_kwargs,
            "tokenizer_name": config.tokenizer_name,
            "max_input_tokens": config.max_input_tokens,
            "token_count_log": str(
                config.output_dir / "mini_sweagent_token_counts.jsonl"
            ),
        }
    }
    max_observation_chars = getattr(config, "mini_observation_chars", None)
    if max_observation_chars is not None:
        if (
            not isinstance(max_observation_chars, int)
            or isinstance(max_observation_chars, bool)
            or max_observation_chars < 2
        ):
            raise ValueError("mini_observation_chars must be an integer >= 2")
        model_config["model"]["observation_retained_payload_chars"] = (
            max_observation_chars
        )
        half = max_observation_chars // 2
        model_config["model"]["observation_template"] = (
            "{% if output.exception_info -%}\n"
            "{% set observation_type = 'exception_and_output' -%}\n"
            "{% set payload = 'exception: ' ~ output.exception_info ~ "
            "'\\noutput: ' ~ output.output -%}\n"
            "{% else -%}\n"
            "{% set observation_type = 'command_output' -%}\n"
            "{% set payload = output.output -%}\n"
            "{% endif -%}\n"
            "<returncode>{{output.returncode}}</returncode>\n"
            '<observation type="{{ observation_type }}">\n'
            f"{{% if payload | length <= {max_observation_chars} -%}}\n"
            "<payload>\n{{ payload -}}\n</payload>\n"
            "{%- else -%}\n"
            "<warning>Observation payload exceeded the declared budget; "
            "an equal head/tail sample follows.</warning>\n"
            f"<payload_head>\n{{{{ payload[:{half}] }}}}\n</payload_head>\n"
            "<elided_chars>{{ payload | length - "
            f"{max_observation_chars} }}}}</elided_chars>\n"
            f"<payload_tail>\n{{{{ payload[-{half}:] }}}}\n</payload_tail>\n"
            "{%- endif -%}\n"
            "</observation>"
            # Per-turn step feedback. mini-swe-agent 2.2.8 renders the
            # observation template with the agent's template vars
            # (minisweagent/agents/default.py: execute_actions passes
            # get_template_vars(), which merges AgentConfig.model_dump() —
            # step_limit — and n_model_calls, the number of model calls made
            # so far, i.e. the step this observation concludes). AgentConfig
            # defaults step_limit to 0 (unlimited), so guard the line rather
            # than render "step k of 0".
            "{% if step_limit %}\n[step {{ n_model_calls }} of "
            "{{ step_limit }}]{% endif %}"
        )
    if config.mini_environment_class == "docker":
        model_config["environment"] = {
            "run_args": ["--rm", "--label", _agentic_container_label(config)]
        }
    step_limit = None
    if config.mini_agent_kwargs:
        if not isinstance(config.mini_agent_kwargs, dict):
            raise ValueError("mini_agent_kwargs must be a dictionary")
        prompt_override_keys = {
            "instance_template",
            "system_template",
        }.intersection(config.mini_agent_kwargs)
        if prompt_override_keys:
            raise ValueError(
                "mini_agent_kwargs cannot override shared mini-swe prompt templates: "
                f"{sorted(prompt_override_keys)}"
            )
        step_limit = config.mini_agent_kwargs.get("step_limit")
        if step_limit is not None and (
            not isinstance(step_limit, int)
            or isinstance(step_limit, bool)
            or step_limit <= 0
        ):
            raise ValueError("mini_agent_kwargs.step_limit must be a positive integer")
        model_config["agent"] = dict(config.mini_agent_kwargs)
    model_config.setdefault("agent", {})["system_template"] = (
        render_budget_discipline_template(step_limit)
    )
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
    *,
    instance_ids: Optional[list[str]] = None,
    dataset_source: Optional[str] = None,
) -> list[str]:
    mini_exec = _interpreter(config).parent / "mini-extra"
    cmd = [
        str(mini_exec),
        "swebench",
        "--model",
        config.model_name,
        "--subset",
        dataset_source or config.sweagent_subset,
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
    selected_ids = config.instance_ids if instance_ids is None else instance_ids
    if selected_ids:
        regex = "^(" + "|".join(re.escape(iid) for iid in selected_ids) + ")$"
        cmd.extend(["--filter", regex])
    elif config.n_tasks is not None:
        cmd.extend(["--slice", f":{config.n_tasks}"])
    return cmd


def _run_fixed_mini_sweagent_samples(
    config: SWEbenchRunConfig,
    mini_config_path: Path,
    env: dict[str, str],
    dataset_source: Optional[str] = None,
) -> tuple[int, Path]:
    """Run fixed IDs one at a time with atomic successful-sample resume state."""
    if not config.instance_ids:
        raise ValueError("fixed-sample runner requires instance_ids")
    timeout = config.agent_generation_timeout_sec
    if timeout is None or timeout <= 0:
        raise ValueError("fixed-sample runner requires a positive generation timeout")

    output_dir = config.output_dir / "mini_sweagent"
    container_label = (
        _agentic_container_label(config)
        if config.mini_environment_class == "docker"
        else None
    )
    resume_path = output_dir / "successful_samples.json"
    predictions = _load_successful_predictions(resume_path, config.instance_ids)
    started = time.monotonic()
    for instance_id in config.instance_ids:
        if instance_id in predictions:
            logger.info("Resuming successful SWE-bench sample %s", instance_id)
            continue
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            return 124, output_dir / "preds.json"
        sample_dir = output_dir / "samples" / instance_id
        sample_predictions_path = sample_dir / "preds.json"
        if sample_predictions_path.exists():
            try:
                previous = json.loads(
                    sample_predictions_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                previous = None
            previous_record = (
                previous.get(instance_id) if isinstance(previous, dict) else None
            )
            if _valid_prediction(previous_record):
                # The sample completed before a crash but the consolidated
                # resume state did not. Recover it without another model call.
                predictions[instance_id] = previous_record
                _atomic_write_json(resume_path, predictions)
                _atomic_write_json(output_dir / "preds.json", predictions)
                continue
            failed_path = sample_dir / f"preds.failed.{int(time.time())}.json"
            os.replace(sample_predictions_path, failed_path)
        command = build_mini_sweagent_command(
            config,
            mini_config_path,
            sample_dir,
            instance_ids=[instance_id],
            dataset_source=dataset_source,
        )
        rc = _run_bounded_process_group(
            command,
            cwd=config.output_dir,
            env=env,
            timeout_sec=remaining,
            cleanup_container_label=container_label,
        )
        if rc != 0:
            return rc, output_dir / "preds.json"
        try:
            sample_predictions = json.loads(
                sample_predictions_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            logger.error(
                "Sample %s produced unreadable predictions: %s", instance_id, exc
            )
            return 65, output_dir / "preds.json"
        record = (
            sample_predictions.get(instance_id)
            if isinstance(sample_predictions, dict)
            else None
        )
        if not _valid_prediction(record):
            logger.error(
                "Sample %s produced a failed/empty patch sentinel", instance_id
            )
            return 65, output_dir / "preds.json"
        predictions[instance_id] = record
        _atomic_write_json(resume_path, predictions)
        _atomic_write_json(output_dir / "preds.json", predictions)

    missing = [
        instance_id
        for instance_id in config.instance_ids
        if instance_id not in predictions
    ]
    if missing:
        logger.error("Fixed SWE-bench run is missing predictions: %s", missing)
        return 65, output_dir / "preds.json"
    _atomic_write_json(output_dir / "preds.json", predictions)
    return 0, output_dir / "preds.json"


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
    dataset_source: Optional[str] = None,
) -> list[str]:
    cmd = [
        str(_interpreter(config)),
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_source or config.dataset_name,
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
    dataset_content_sha256: Optional[str] = None,
) -> dict[str, Any]:
    """Normalize the SWE-bench harness report into harbor-format result.json."""
    report = json.loads(harness_report_path.read_text(encoding="utf-8"))
    submitted_rows = report.get("submitted_ids", [])
    resolved_rows = report.get("resolved_ids", [])
    if not isinstance(submitted_rows, list) or not isinstance(resolved_rows, list):
        raise RuntimeError("SWE-bench verifier ID fields must be lists")
    submitted_ids = set(submitted_rows)
    resolved_ids = set(resolved_rows)

    graded = config.qualification_claim == "models_ci_graded"
    expected_ids = set(config.instance_ids)
    if graded and not submitted_ids:
        raise RuntimeError(
            "graded SWE-bench verifier report must contain nonempty submitted_ids; "
            "aggregate-only counts cannot support release promotion"
        )
    if graded and (
        len(submitted_ids) != len(submitted_rows)
        or not all(
            isinstance(instance_id, str) and instance_id
            for instance_id in submitted_rows
        )
    ):
        raise RuntimeError(
            "graded SWE-bench verifier submitted_ids must be unique nonempty strings"
        )
    if graded and expected_ids and submitted_ids != expected_ids:
        raise RuntimeError(
            "graded SWE-bench verifier submitted_ids differ from the exact "
            "predeclared run set: "
            f"expected={sorted(expected_ids)}, submitted={sorted(submitted_ids)}"
        )

    if not submitted_ids:
        submitted_count = int(report.get("submitted_instances", 0))
        resolved_count = int(report.get("resolved_instances", 0))
        unresolved_ids = set()
    else:
        submitted_count = len(submitted_ids)
        resolved_count = len(resolved_ids)
        unresolved_ids = submitted_ids - resolved_ids

    if submitted_count <= 0:
        raise RuntimeError(
            "SWE-bench verifier report contains zero submitted instances; "
            "refusing to normalize an unverified run as 0% success"
        )
    unexpected_resolved = resolved_ids - submitted_ids
    if unexpected_resolved:
        raise RuntimeError(
            "SWE-bench verifier report resolved instances that were not "
            f"submitted: {sorted(unexpected_resolved)}"
        )
    if resolved_count < 0 or resolved_count > submitted_count:
        raise RuntimeError(
            "SWE-bench verifier report has an invalid resolved/submitted count: "
            f"resolved={resolved_count}, submitted={submitted_count}"
        )

    accuracy = resolved_count / submitted_count
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
            "qualification_contract": {
                "qualification_claim": config.qualification_claim,
                "selection_policy": config.selection_policy,
                "instance_selection_provenance": (config.instance_selection_provenance),
                "dataset_revision": config.dataset_revision,
                "ordered_instance_ids_sha256": (config.ordered_instance_ids_sha256),
                "selected_instances_sha256": dataset_content_sha256,
                "predeclared_instance_ids": sorted(expected_ids),
                "eval_limit_mode": config.eval_limit_mode,
                "mini_observation_chars": config.mini_observation_chars,
                "max_input_tokens": config.max_input_tokens,
                "max_output_tokens": config.max_output_tokens,
            },
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

    # Inject start/finish/n_total_trials so the report can compute a mean time
    # per task, matching terminal-bench's Harbor timing fields. SWE-bench's own
    # report carries no timing, so derive the window from the agent log.
    started, finished = _agent_log_time_window(config.output_dir)
    if started is not None and finished is not None:
        normalized["started_at"] = started.isoformat()
        normalized["finished_at"] = finished.isoformat()
    normalized["n_total_trials"] = submitted_count

    result_path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    return normalized


def run(config: SWEbenchRunConfig) -> int:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_source, dataset_content_sha256 = _prepare_pinned_swebench_dataset(config)

    env = os.environ.copy()
    env.setdefault("OPENAI_API_KEY", "EMPTY")
    env.setdefault("OPENAI_BASE_URL", config.api_base)
    env.setdefault("OPENAI_API_BASE", config.api_base)
    env.setdefault("SWE_AGENT_LOG_STREAM_LEVEL", "INFO")
    env.setdefault("MSWEA_COST_TRACKING", "ignore_errors")
    # mini-extra runs from the eval venv and output directory. Make this source
    # tree importable so its fail-closed model class is the class that dispatches
    # every request; do not rely on the caller's current working directory.
    repo_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = repo_root + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
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
        rc = _run_command(sweagent_cmd, cwd=config.output_dir, env=env)
        if rc != 0:
            return rc
        preds_path = _find_sweagent_preds(sweagent_output_dir)
    elif config.agent_backend == "mini-swe-agent":
        mini_config_path = _write_mini_sweagent_model_config(config)
        mini_output_dir = config.output_dir / "mini_sweagent"
        if config.instance_ids:
            rc, preds_path = _run_fixed_mini_sweagent_samples(
                config, mini_config_path, env, dataset_source
            )
        else:
            if (
                config.agent_generation_timeout_sec is None
                or config.agent_generation_timeout_sec <= 0
            ):
                raise ValueError(
                    "mini-swe-agent requires a positive agent generation timeout"
                )
            mini_cmd = build_mini_sweagent_command(
                config,
                mini_config_path,
                mini_output_dir,
                dataset_source=dataset_source,
            )
            rc = _run_bounded_process_group(
                mini_cmd,
                cwd=config.output_dir,
                env=env,
                timeout_sec=config.agent_generation_timeout_sec,
                cleanup_container_label=(
                    _agentic_container_label(config)
                    if config.mini_environment_class == "docker"
                    else None
                ),
            )
            preds_path = mini_output_dir / "preds.json"
        if rc != 0:
            return rc
        if not preds_path.exists():
            raise FileNotFoundError(f"No mini-swe-agent predictions at {preds_path}")
        try:
            _validate_prediction_file(preds_path, config.instance_ids)
        except RuntimeError as exc:
            logger.error(
                "Refusing to score invalid mini-swe-agent predictions: %s", exc
            )
            return 65
    else:
        raise ValueError(f"Unsupported SWE-bench agent backend: {config.agent_backend}")

    if not config.score_existing_predictions:
        convert_sweagent_preds_to_jsonl(preds_path, predictions_path, config.model_name)

    harness_cmd = build_swebench_harness_command(
        config,
        predictions_path,
        run_id,
        dataset_source=dataset_source,
    )
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
        harness_report_path,
        result_path,
        config,
        predictions_path,
        dataset_content_sha256,
    )
    logger.info("Wrote SWE-bench normalized result to %s", result_path)
    return 0
