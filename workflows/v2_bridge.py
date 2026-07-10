# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from __future__ import annotations

import json
import logging
import os
import shlex
import sys
from pathlib import Path
from typing import List

from workflows.run_workflows import WorkflowResult
from workflows.utils import (
    ensure_readwriteable_dir,
    get_default_workflow_root_log_dir,
    run_command,
)
from workflows.workflow_types import (
    ModelType,
    WorkflowType,
    WorkflowVenvType,
)
from workflows.workflow_venvs import VENV_CONFIGS

logger = logging.getLogger("run_log")

_V2_DIR_NAME = "tt-inference-server-v2"

_V2_WORKFLOW_NAMES = {
    WorkflowType.BENCHMARKS: "benchmarks",
    WorkflowType.EVALS: "evals",
    WorkflowType.SPEC_TESTS: "spec_tests",
    WorkflowType.RELEASE: "release",
    WorkflowType.AGENTIC: "agentic",
    WorkflowType.SERVING_BENCH: "serving_bench",
}

_V2_EVAL_WORKFLOWS = frozenset({WorkflowType.EVALS, WorkflowType.RELEASE})


_V2_EVAL_VENV_BY_MODEL_TYPE = {
    ModelType.AUDIO: WorkflowVenvType.EVALS_AUDIO,
}

# Model types fully onboarded to v2's media engine. Every model of these types
# routes to v2 by model_type
_V2_MEDIA_MODEL_TYPES = frozenset(
    {
        ModelType.IMAGE,
        ModelType.VIDEO,
        ModelType.AUDIO,
        ModelType.TEXT_TO_SPEECH,
    }
)


def is_v2_routed_model(model_spec) -> bool:
    """True if the model routes to v2 purely by its model_type."""
    return model_spec.model_type in _V2_MEDIA_MODEL_TYPES


def _is_prefix_cache_run(wf, runtime_config) -> bool:
    return wf == WorkflowType.BENCHMARKS and getattr(
        runtime_config, "prefix_cache", False
    )


def _is_spec_decode_run(wf, runtime_config) -> bool:
    return wf == WorkflowType.BENCHMARKS and getattr(
        runtime_config, "spec_decode", False
    )


def _is_llm_benchmark_run(wf, model_spec, runtime_config) -> bool:
    """Any LLM model + ``--workflow benchmarks`` routes to v2's ``llm_module``;
    the ``--tools`` value selects the driver. The prefix-cache and spec-decode
    variants have their own dispatch and are handled separately.
    """
    return (
        wf == WorkflowType.BENCHMARKS
        and model_spec.model_type == ModelType.LLM
        and not _is_prefix_cache_run(wf, runtime_config)
        and not _is_spec_decode_run(wf, runtime_config)
    )


def _llm_release_includes_agentic(model_spec) -> bool:
    """True if an LLM release should also run agentic evals.

    Agentic evals (Terminal-Bench-2 / SWE-bench Verified) now run in-process as
    a child of the v2 release engine: the harness binaries are resolved from the
    EVALS_AGENTIC venv explicitly (not from ``sys.executable``), so their Blocks
    land in the single release report. This predicate gates only the up-front
    provisioning of the EVALS_AGENTIC venv; the release engine itself decides
    whether to run the agentic child (see ReleaseWorkflow._llm_children).
    """
    if model_spec.model_type != ModelType.LLM:
        return False
    try:
        from evals.eval_config import EVAL_CONFIGS
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Could not import EVAL_CONFIGS (%s); skipping agentic.", e)
        return False
    cfg = EVAL_CONFIGS.get(model_spec.model_name)
    if cfg is None:
        return False
    return any(
        task.workflow_venv_type == WorkflowVenvType.EVALS_AGENTIC for task in cfg.tasks
    )


def _is_llm_eval_run(wf, model_spec) -> bool:
    """LLM ``--workflow evals`` / ``--workflow release`` route to v2.

    Standard evals run lm-eval / lmms-eval through ``EvalsWorkflow``; release
    additionally runs the perf benchmark. Both go through the generic run.py
    branch (no launcher) — the eval subprocess uses the per-task venv binary.
    """
    return model_spec.model_type == ModelType.LLM and wf in (
        WorkflowType.EVALS,
        WorkflowType.RELEASE,
    )


def can_route_to_v2(model_spec, runtime_config) -> bool:
    wf = WorkflowType.from_string(runtime_config.workflow)
    # Agentic evals, serving-bench benchmark suites, and the prefix-cache /
    # spec-decode benchmarks are v2-only features with no v1 driver. They route
    # to v2 for ANY model, regardless of model_type.
    if (
        wf in (WorkflowType.AGENTIC, WorkflowType.SERVING_BENCH)
        or _is_prefix_cache_run(wf, runtime_config)
        or _is_spec_decode_run(wf, runtime_config)
    ):
        return True
    if _is_llm_benchmark_run(wf, model_spec, runtime_config):
        return True
    if _is_llm_eval_run(wf, model_spec):
        return True
    # IMAGE / VIDEO / AUDIO / TEXT_TO_SPEECH are fully onboarded to v2, so every
    # model of those types routes by model_type — no per-name allowlist, so new
    # media models (e.g. Qwen-Image) are picked up automatically.
    if is_v2_routed_model(model_spec):
        return wf in _V2_WORKFLOW_NAMES
    return False


def run_v2_workflows(model_spec, runtime_config, json_fpath) -> List[WorkflowResult]:
    wf = WorkflowType.from_string(runtime_config.workflow)
    v2_workflow = _V2_WORKFLOW_NAMES.get(wf)
    if v2_workflow is None:
        raise ValueError(
            f"v2 bridge does not handle workflow {wf.name!r}. "
            f"Supported: {sorted(_V2_WORKFLOW_NAMES.values())}"
        )

    repo_root = Path(__file__).resolve().parent.parent
    v2_dir = repo_root / _V2_DIR_NAME

    output_dir = get_default_workflow_root_log_dir() / "reports_output" / v2_workflow
    ensure_readwriteable_dir(output_dir)

    # Agentic, prefix-cache, and spec-decode go through their dedicated venv
    # launchers (run_agentic.py / run_prefix_cache.py / run_spec_decode.py),
    # which materialize the EVALS_AGENTIC / V2_PREFIX_CACHE / V2_SPEC_DECODE
    # venv and re-exec run.py inside it. They run from this interpreter
    # (sys.executable) — the launchers import only the lightweight workflows.*
    # helpers before re-execing.
    if wf == WorkflowType.AGENTIC:
        cmd = _build_agentic_cmd(
            v2_dir, model_spec, runtime_config, json_fpath, output_dir
        )
        delegate_desc = "agentic (run_agentic.py)"
    elif _is_prefix_cache_run(wf, runtime_config):
        cmd = _build_prefix_cache_cmd(
            v2_dir, model_spec, runtime_config, json_fpath, output_dir
        )
        delegate_desc = "prefix-cache (run_prefix_cache.py)"
    elif _is_spec_decode_run(wf, runtime_config):
        cmd = _build_spec_decode_cmd(
            v2_dir, model_spec, runtime_config, json_fpath, output_dir
        )
        delegate_desc = "spec-decode (run_spec_decode.py)"
    elif _is_llm_benchmark_run(wf, model_spec, runtime_config):
        return [run_v2_llm_benchmark_workflow(model_spec, runtime_config, json_fpath)]
    else:
        v2_run_py = v2_dir / "run.py"
        if not v2_run_py.is_file():
            raise FileNotFoundError(
                f"v2 entry point not found at {v2_run_py}. "
                "The tt-inference-server-v2/ directory is required for image-model workflows."
            )
        venv_python = _ensure_v2_venv(model_spec)
        _ensure_v2_dependency_venvs(model_spec, wf, runtime_config)

        cmd = [
            str(venv_python),
            str(v2_run_py),
            "--model",
            model_spec.model_name,
            "--workflow",
            v2_workflow,
            "--device",
            runtime_config.device,
            "--service-port",
            str(runtime_config.service_port),
            "--runtime-model-spec-json",
            str(json_fpath),
            "--output-dir",
            str(output_dir),
        ]
        if runtime_config.docker_server:
            cmd.append("--docker-server")
        _extend_if_set(cmd, "--server-url", getattr(runtime_config, "server_url", None))
        if wf == WorkflowType.SERVING_BENCH:
            _extend_if_set(
                cmd, "--serving-bench-suites", runtime_config.serving_bench_suites
            )
        elif _is_llm_eval_run(wf, model_spec):
            # Standard evals (and release) need the bearer token to reach a
            # JWT-protected server; run.py mints it from --jwt-secret/$JWT_SECRET.
            _forward_jwt(cmd, runtime_config)
            if wf == WorkflowType.RELEASE:
                _forward_prefix_cache(cmd, runtime_config)
                _forward_spec_decode(cmd, runtime_config)
        else:
            sdxl_n = getattr(runtime_config, "sdxl_num_prompts", None)
            if sdxl_n not in (None, "", "0"):
                cmd.extend(["--num-prompts", str(sdxl_n)])
        _warn_on_unsupported_args(runtime_config)
        delegate_desc = "run.py"

    env = os.environ.copy()
    env["TT_V1_RUN_COMMAND"] = "python " + shlex.join(sys.argv)

    logger.info(
        "Delegating workflow %r to v2 engine via %s.", v2_workflow, delegate_desc
    )
    return_code = run_command(cmd, logger=logger, env=env)
    if return_code != 0:
        logger.error(
            f"⛔ v2 workflow: {v2_workflow}, failed with return code: {return_code}"
        )
    else:
        logger.info(f"✅ Completed v2 workflow: {v2_workflow}")
    # Agentic evals run in-process as a release child (their Blocks are already
    # in the report the engine wrote). Parameter tests will be added here once
    # they are available on main as a v2 workflow.
    return [WorkflowResult(workflow_name=v2_workflow, return_code=return_code)]


def run_v2_llm_benchmark_workflow(
    model_spec, runtime_config, json_fpath
) -> WorkflowResult:
    """Run LLM benchmarks through v2's ``run_llm_bench.py`` launcher.

    Only reached for ``--workflow benchmarks`` (see ``_is_llm_benchmark_run``).
    Release does not call this: its perf benchmark runs inside the v2 release
    engine via the generic run.py path in :func:`run_v2_workflows`.
    """
    repo_root = Path(__file__).resolve().parent.parent
    v2_dir = repo_root / _V2_DIR_NAME
    output_dir = get_default_workflow_root_log_dir() / "reports_output" / "benchmarks"
    ensure_readwriteable_dir(output_dir)

    cmd = _build_llm_bench_cmd(
        v2_dir, model_spec, runtime_config, json_fpath, output_dir
    )
    env = os.environ.copy()
    env["TT_V1_RUN_COMMAND"] = "python " + shlex.join(sys.argv)

    logger.info(
        "Delegating LLM benchmarks to v2 engine via llm-bench (run_llm_bench.py)."
    )
    return_code = run_command(cmd, logger=logger, env=env)
    if return_code != 0:
        logger.error(
            "⛔ v2 LLM benchmarks workflow failed with return code: %s", return_code
        )
    else:
        logger.info("✅ Completed v2 LLM benchmarks workflow")
    return WorkflowResult(workflow_name="benchmarks", return_code=return_code)


def _base_v2_cmd(
    launcher, model_spec, runtime_config, json_fpath, output_dir, v2_workflow
):
    cmd = [
        sys.executable,
        str(launcher),
        "--model",
        model_spec.model_name,
        "--workflow",
        v2_workflow,
        "--device",
        runtime_config.device,
        "--service-port",
        str(runtime_config.service_port),
        "--runtime-model-spec-json",
        str(json_fpath),
        "--output-dir",
        str(output_dir),
    ]
    if runtime_config.docker_server:
        cmd.append("--docker-server")
    if getattr(runtime_config, "server_url", None):
        cmd.extend(["--server-url", runtime_config.server_url])
    return cmd


def _resolve_launcher(v2_dir, filename, label):
    launcher = v2_dir / filename
    if not launcher.is_file():
        raise FileNotFoundError(f"v2 {label} launcher not found at {launcher}.")
    return launcher


def _forward_jwt(cmd, runtime_config) -> None:
    # run.py reads $JWT_SECRET when --jwt-secret is omitted; only forward an
    # explicit value so the env fallback still works.
    _extend_if_set(cmd, "--jwt-secret", runtime_config.jwt_secret)


def _forward_prefix_cache(cmd, runtime_config) -> None:
    if not getattr(runtime_config, "prefix_cache", False):
        return
    cmd.append("--prefix-cache")
    cmd.extend(["--prefix-cache-preset", runtime_config.prefix_cache_preset])
    _extend_if_set(
        cmd, "--prefix-cache-scenarios", runtime_config.prefix_cache_scenarios
    )
    _extend_if_set(cmd, "--prefix-cache-arrival", runtime_config.prefix_cache_arrival)
    _extend_if_set(
        cmd, "--prefix-cache-request-rate", runtime_config.prefix_cache_request_rate
    )
    _extend_if_set(
        cmd, "--prefix-cache-scenarios-json", runtime_config.prefix_cache_scenarios_json
    )
    _extend_if_set(cmd, "--prefix-cache-trace", runtime_config.prefix_cache_trace)
    _extend_if_set(
        cmd,
        "--prefix-cache-goodput",
        getattr(runtime_config, "prefix_cache_goodput", None),
    )
    # --prefix-cache-metrics-url is action="append" (a list); emit one flag
    # per URL rather than stringifying the whole list, which would forward a
    # bogus "['https://...']" URL and leave the hit-rate column null.
    for metrics_url in getattr(runtime_config, "prefix_cache_metrics_url", None) or []:
        _extend_if_set(cmd, "--prefix-cache-metrics-url", metrics_url)


def _forward_spec_decode(cmd, runtime_config) -> None:
    if not getattr(runtime_config, "spec_decode", False):
        return
    cmd.append("--spec-decode")
    cmd.extend(["--spec-decode-preset", runtime_config.spec_decode_preset])
    _extend_if_set(
        cmd, "--spec-decode-warmup-requests", runtime_config.spec_decode_warmup_requests
    )


def _build_agentic_cmd(v2_dir, model_spec, runtime_config, json_fpath, output_dir):
    launcher = _resolve_launcher(v2_dir, "run_agentic.py", "agentic")
    cmd = _base_v2_cmd(
        launcher, model_spec, runtime_config, json_fpath, output_dir, "agentic"
    )
    _forward_jwt(cmd, runtime_config)
    return cmd


def _build_prefix_cache_cmd(v2_dir, model_spec, runtime_config, json_fpath, output_dir):
    launcher = _resolve_launcher(v2_dir, "run_prefix_cache.py", "prefix-cache")
    cmd = _base_v2_cmd(
        launcher, model_spec, runtime_config, json_fpath, output_dir, "benchmarks"
    )
    cmd.append("--prefix-cache")
    cmd.extend(["--prefix-cache-preset", runtime_config.prefix_cache_preset])
    _extend_if_set(
        cmd, "--prefix-cache-scenarios", runtime_config.prefix_cache_scenarios
    )
    _extend_if_set(cmd, "--prefix-cache-arrival", runtime_config.prefix_cache_arrival)
    _extend_if_set(
        cmd, "--prefix-cache-request-rate", runtime_config.prefix_cache_request_rate
    )
    _extend_if_set(
        cmd, "--prefix-cache-scenarios-json", runtime_config.prefix_cache_scenarios_json
    )
    _extend_if_set(cmd, "--prefix-cache-trace", runtime_config.prefix_cache_trace)
    _extend_if_set(
        cmd,
        "--prefix-cache-goodput",
        getattr(runtime_config, "prefix_cache_goodput", None),
    )
    for metrics_url in getattr(runtime_config, "prefix_cache_metrics_url", None) or []:
        _extend_if_set(cmd, "--prefix-cache-metrics-url", metrics_url)
    _forward_jwt(cmd, runtime_config)
    return cmd


def _build_llm_bench_cmd(v2_dir, model_spec, runtime_config, json_fpath, output_dir):
    launcher = _resolve_launcher(v2_dir, "run_llm_bench.py", "llm-bench")
    cmd = _base_v2_cmd(
        launcher, model_spec, runtime_config, json_fpath, output_dir, "benchmarks"
    )
    _extend_if_set(cmd, "--tools", runtime_config.tools)
    _extend_if_set(cmd, "--goodput", getattr(runtime_config, "goodput", None))
    _forward_jwt(cmd, runtime_config)
    return cmd


def _build_spec_decode_cmd(v2_dir, model_spec, runtime_config, json_fpath, output_dir):
    launcher = _resolve_launcher(v2_dir, "run_spec_decode.py", "spec-decode")
    cmd = _base_v2_cmd(
        launcher, model_spec, runtime_config, json_fpath, output_dir, "benchmarks"
    )
    cmd.append("--spec-decode")
    cmd.extend(["--spec-decode-preset", runtime_config.spec_decode_preset])
    _extend_if_set(
        cmd, "--spec-decode-warmup-requests", runtime_config.spec_decode_warmup_requests
    )
    _forward_jwt(cmd, runtime_config)
    return cmd


def _extend_if_set(cmd, flag, value) -> None:
    if value not in (None, ""):
        cmd.extend([flag, str(value)])


def _ensure_v2_venv(model_spec) -> Path:
    """Materialize the V2_RUN_SCRIPT venv and return its interpreter path.

    Mirrors the per-venv body of ``WorkflowSetup.create_required_venvs``
    (workflows/run_workflows.py) but skips the v1 task-config expansion:
    v2 owns its own sub-workflow dispatch and has no v1 ``task.workflow_venv_type``
    entries to pull in. ``VenvConfig.setup`` is idempotent, so calling
    this on every dispatch is cheap once the venv exists.
    """
    venv_config = VENV_CONFIGS[WorkflowVenvType.V2_RUN_SCRIPT]
    setup_completed = venv_config.setup(model_spec=model_spec)
    assert setup_completed, "Failed to setup venv: V2_RUN_SCRIPT"
    return venv_config.venv_python


# Standard LLM/VLM eval backends (mirrors llm_module.eval_configs). EVALS_AGENTIC
# is provisioned by the agentic launcher, not here.
_V2_LLM_STANDARD_EVAL_VENVS = frozenset(
    {
        WorkflowVenvType.EVALS_COMMON,
        WorkflowVenvType.EVALS_META,
        WorkflowVenvType.EVALS_VISION,
    }
)


def _eval_samples_task_names(runtime_config):
    """Task names selected by --eval-samples (JSON string or file), or None."""
    raw = getattr(runtime_config, "eval_samples", None)
    if not raw:
        return None
    try:
        mapping = json.loads(raw)
    except (TypeError, ValueError):
        try:
            mapping = json.loads(Path(raw).read_text())
        except Exception:
            return None
    return set(mapping) if isinstance(mapping, dict) else None


def _is_smoke_mode(runtime_config) -> bool:
    mode = getattr(runtime_config, "limit_samples_mode", None)
    if not mode:
        return False
    from workflows.workflow_types import EvalLimitMode

    try:
        return EvalLimitMode.from_string(mode) == EvalLimitMode.SMOKE_TEST
    except Exception:
        return False


def _selected_eval_tasks(tasks, runtime_config):
    """Apply the same selection ``get_llm_eval_tasks`` does, so we provision
    only the venvs the run will actually use (--eval-samples / smoke-test).
    Falls back to all tasks when nothing narrows them (over-provision is safe;
    under-provision would break a task the run still tries to execute)."""
    names = _eval_samples_task_names(runtime_config)
    if names:
        sel = [t for t in tasks if t.task_name in names]
        if sel:
            return sel
    if _is_smoke_mode(runtime_config) and tasks:
        return [tasks[0]]
    return tasks


def _llm_eval_venv_types(model_spec, runtime_config=None) -> List[WorkflowVenvType]:
    """Standard eval venvs the run will actually use (from EVAL_CONFIGS).

    Honors --eval-samples / smoke-test so a single-task run doesn't provision
    the (heavy) venvs of tasks it won't execute.
    """
    try:
        from evals.eval_config import EVAL_CONFIGS
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Could not import EVAL_CONFIGS (%s); skipping eval venvs.", e)
        return []
    cfg = EVAL_CONFIGS.get(model_spec.model_name)
    if cfg is None:
        return []
    tasks = _selected_eval_tasks(cfg.tasks, runtime_config)
    seen = {
        t.workflow_venv_type
        for t in tasks
        if t.workflow_venv_type in _V2_LLM_STANDARD_EVAL_VENVS
    }
    return sorted(seen, key=lambda v: v.name)


def _v2_dependency_venv_types(
    model_spec, wf, runtime_config=None
) -> List[WorkflowVenvType]:
    venv_types: List[WorkflowVenvType] = []
    if wf in _V2_EVAL_WORKFLOWS:
        eval_venv = _V2_EVAL_VENV_BY_MODEL_TYPE.get(model_spec.model_type)
        if eval_venv is not None:
            venv_types.append(eval_venv)
        if model_spec.model_type == ModelType.LLM:
            venv_types.extend(_llm_eval_venv_types(model_spec, runtime_config))
    # The release benchmark child runs the default perf tool (vllm) in-process
    # under V2_RUN_SCRIPT, so its tool venv must exist up front.
    if wf == WorkflowType.RELEASE and model_spec.model_type == ModelType.LLM:
        venv_types.append(WorkflowVenvType.V2_LLM_VLLM)
        if getattr(runtime_config, "prefix_cache", False):
            venv_types.append(WorkflowVenvType.V2_PREFIX_CACHE)
        if getattr(runtime_config, "spec_decode", False):
            venv_types.append(WorkflowVenvType.V2_SPEC_DECODE)
        # The agentic release child resolves harbor/sweagent from the
        # EVALS_AGENTIC venv, so it must exist before the engine subprocess runs.
        if _llm_release_includes_agentic(model_spec):
            venv_types.append(WorkflowVenvType.EVALS_AGENTIC)
    return venv_types


def _ensure_v2_dependency_venvs(model_spec, wf, runtime_config=None) -> None:
    for venv_type in _v2_dependency_venv_types(model_spec, wf, runtime_config):
        venv_config = VENV_CONFIGS[venv_type]
        logger.info("Provisioning v2 dependency venv: %s", venv_type.name)
        setup_completed = venv_config.setup(model_spec=model_spec)
        assert setup_completed, f"Failed to setup venv: {venv_type.name}"


def _warn_on_unsupported_args(runtime_config) -> None:
    unsupported = []
    if getattr(runtime_config, "markers", None):
        unsupported.append("--markers")
    if getattr(runtime_config, "match_all_markers", False):
        unsupported.append("--match-all-markers")
    if getattr(runtime_config, "exclude_markers", None):
        unsupported.append("--exclude-markers")
    if getattr(runtime_config, "test_name", None):
        unsupported.append("--test-name")
    if unsupported:
        logger.warning(
            "v2 engine does not honor these v1 flags for image models; "
            "they will be ignored: %s",
            ", ".join(unsupported),
        )
