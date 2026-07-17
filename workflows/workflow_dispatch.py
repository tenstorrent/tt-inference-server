# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from __future__ import annotations

import json
import logging
import shlex
import sys
from pathlib import Path
from typing import List

from workflows.workflow_result import WorkflowResult
from workflows.utils import (
    ensure_readwriteable_dir,
    get_default_workflow_root_log_dir,
)
from workflows.workflow_types import (
    ModelType,
    WorkflowType,
    WorkflowVenvType,
)

from workflow_module import VenvCommand, WorkflowRunner

logger = logging.getLogger("run_log")

_V2_WORKFLOW_NAMES = {
    WorkflowType.BENCHMARKS: "benchmarks",
    WorkflowType.EVALS: "evals",
    WorkflowType.SPEC_TESTS: "spec_tests",
    WorkflowType.STRESS_TESTS: "stress_tests",
    WorkflowType.RELEASE: "release",
    WorkflowType.AGENTIC: "agentic",
    WorkflowType.SERVING_BENCH: "serving_bench",
}

_V2_EVAL_WORKFLOWS = frozenset({WorkflowType.EVALS, WorkflowType.RELEASE})


_V2_EVAL_VENV_BY_MODEL_TYPE = {
    ModelType.AUDIO: WorkflowVenvType.EVALS_AUDIO,
    ModelType.EMBEDDING: WorkflowVenvType.EVALS_EMBEDDING,
}

# Model types that share LLM code path rather than a media runner.
_LLM_LIKE_TYPES = frozenset({ModelType.LLM, ModelType.VLM})

# Model types fully onboarded to v2. Every model of these types routes to v2 by
# model_type — no per-name allowlist, so new models are picked up automatically.
_V2_ROUTED_MODEL_TYPES = frozenset(
    {
        ModelType.IMAGE,
        ModelType.VIDEO,
        ModelType.AUDIO,
        ModelType.TEXT_TO_SPEECH,
        ModelType.CNN,
        ModelType.EMBEDDING,
    }
)


def is_engine_routed_model(model_spec) -> bool:
    """True if the model routes to v2 purely by its model_type."""
    return model_spec.model_type in _V2_ROUTED_MODEL_TYPES


def _is_prefix_cache_run(wf, runtime_config) -> bool:
    return wf == WorkflowType.BENCHMARKS and getattr(
        runtime_config, "prefix_cache", False
    )


def _is_spec_decode_run(wf, runtime_config) -> bool:
    return wf == WorkflowType.BENCHMARKS and getattr(
        runtime_config, "spec_decode", False
    )


def _is_llm_benchmark_run(wf, model_spec, runtime_config) -> bool:
    """Any LLM/VLM model + ``--workflow benchmarks`` routes to v2's ``llm_module``;
    the ``--tools`` value selects the driver. The prefix-cache and spec-decode
    variants have their own dispatch and are handled separately.
    """
    return (
        wf == WorkflowType.BENCHMARKS
        and model_spec.model_type in _LLM_LIKE_TYPES
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
    if model_spec.model_type not in _LLM_LIKE_TYPES:
        return False
    try:
        from reference_config.evals.eval_config import EVAL_CONFIGS
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
    """LLM/VLM ``--workflow evals`` / ``--workflow release`` route to v2.

    Standard evals run lm-eval / lmms-eval through ``EvalsWorkflow`` (VLMs use
    the lmms-eval / EVALS_VISION tasks); release additionally runs the perf
    benchmark. Both go through the generic run.py branch (no launcher) — the
    eval subprocess uses the per-task venv binary.
    """
    return model_spec.model_type in _LLM_LIKE_TYPES and wf in (
        WorkflowType.EVALS,
        WorkflowType.RELEASE,
    )


def _is_llm_spec_test_run(wf, model_spec) -> bool:
    """LLM/VLM ``--workflow spec_tests`` routes to parameter-conformance
    suite (``test_module/llm_tests/vllm_param_conformance_test.py``), registered
    under the ``spec_tests`` workflow via ``test_suites/llm.json``.
    """
    return model_spec.model_type in _LLM_LIKE_TYPES and wf == WorkflowType.SPEC_TESTS


def can_dispatch_to_engine(model_spec, runtime_config) -> bool:
    wf = WorkflowType.from_string(runtime_config.workflow)
    # Agentic evals, serving-bench benchmark suites, and the prefix-cache /
    # spec-decode benchmarks are v2-only features with no v1 driver. They route
    # to v2 for ANY model, regardless of model_type.
    if (
        wf
        in (
            WorkflowType.AGENTIC,
            WorkflowType.SERVING_BENCH,
            WorkflowType.STRESS_TESTS,
        )
        or _is_prefix_cache_run(wf, runtime_config)
        or _is_spec_decode_run(wf, runtime_config)
    ):
        return True
    if _is_llm_benchmark_run(wf, model_spec, runtime_config):
        return True
    if _is_llm_eval_run(wf, model_spec):
        return True
    if _is_llm_spec_test_run(wf, model_spec):
        return True
    # IMAGE / VIDEO / AUDIO / TEXT_TO_SPEECH / CNN / EMBEDDING are fully
    # onboarded to v2, so every model of those types routes by model_type — no
    # per-name allowlist, so new models (e.g. Qwen-Image) are picked up
    # automatically. The v1 eval/benchmark paths for these types are retired.
    if is_engine_routed_model(model_spec):
        return wf in _V2_WORKFLOW_NAMES
    return False


def build_engine_commands(model_spec, runtime_config, json_fpath) -> list:
    """Build the command(s) that run the requested workflow through the engine.

    Pure builder: returns a list of :class:`VenvCommand`s with no subprocess and
    no venv provisioning (VenvCommand provisions on ``execute()``). ``run.py``
    prepends a ``ServerCommand`` and drives the combined list with a single
    WorkflowRunner; 

    Command shapes:
    - Agentic / prefix-cache / spec-decode / llm-bench run their launcher in the
      current interpreter (``venv_type=None``); the launchers re-exec into their
      own venv (EVALS_AGENTIC / V2_PREFIX_CACHE / V2_SPEC_DECODE / tool venv).
    - stress-tests runs in the STRESS_TESTS_RUN_SCRIPT venv.
    - Everything else runs ``run_workflows.py`` in V2_RUN_SCRIPT (plus its
      dependency venvs, provisioned by the command).
    """
    wf = WorkflowType.from_string(runtime_config.workflow)
    v2_workflow = _V2_WORKFLOW_NAMES.get(wf)
    if v2_workflow is None:
        raise ValueError(
            f"v2 bridge does not handle workflow {wf.name!r}. "
            f"Supported: {sorted(_V2_WORKFLOW_NAMES.values())}"
        )

    repo_root = Path(__file__).resolve().parent.parent
    output_dir = get_default_workflow_root_log_dir() / "reports_output" / v2_workflow
    ensure_readwriteable_dir(output_dir)

    if wf == WorkflowType.AGENTIC:
        return [
            VenvCommand(
                None,
                _build_agentic_cmd(
                    repo_root, model_spec, runtime_config, json_fpath, output_dir
                ),
                env=_engine_env(),
                label=v2_workflow,
            )
        ]
    if _is_prefix_cache_run(wf, runtime_config):
        return [
            VenvCommand(
                None,
                _build_prefix_cache_cmd(
                    repo_root, model_spec, runtime_config, json_fpath, output_dir
                ),
                env=_engine_env(),
                label=v2_workflow,
            )
        ]
    if _is_spec_decode_run(wf, runtime_config):
        return [
            VenvCommand(
                None,
                _build_spec_decode_cmd(
                    repo_root, model_spec, runtime_config, json_fpath, output_dir
                ),
                env=_engine_env(),
                label=v2_workflow,
            )
        ]
    if _is_llm_benchmark_run(wf, model_spec, runtime_config):
        return [
            VenvCommand(
                None,
                _build_llm_bench_cmd(
                    repo_root, model_spec, runtime_config, json_fpath, output_dir
                ),
                env=_engine_env(),
                label=v2_workflow,
            )
        ]
    if wf == WorkflowType.STRESS_TESTS:
        return [
            VenvCommand(
                WorkflowVenvType.STRESS_TESTS_RUN_SCRIPT,
                _stress_argv(repo_root, model_spec, runtime_config, json_fpath),
                model_spec=model_spec,
                env=_engine_env(),
                label=v2_workflow,
            )
        ]

    # Generic engine path: run_workflows.py in V2_RUN_SCRIPT + its dependency venvs.
    v2_run_py = repo_root / "run_workflows.py"
    if not v2_run_py.is_file():
        raise FileNotFoundError(
            f"Workflow entry point not found at {v2_run_py}. "
            "run_workflows.py is required for image-model workflows."
        )
    _warn_on_unsupported_args(runtime_config)
    return [
        VenvCommand(
            WorkflowVenvType.V2_RUN_SCRIPT,
            _engine_run_argv(
                v2_run_py, model_spec, runtime_config, json_fpath, v2_workflow,
                output_dir, wf,
            ),
            model_spec=model_spec,
            env=_engine_env(),
            label=v2_workflow,
            dependency_venvs=_v2_dependency_venv_types(model_spec, wf, runtime_config),
        )
    ]


def dispatch_workflows(model_spec, runtime_config, json_fpath) -> List[WorkflowResult]:
    """Build the engine command(s) and run them via a WorkflowRunner.

    ``run.py`` builds a combined ``[ServerCommand, *engine]`` list and drives it
    directly; this thin wrapper remains for standalone callers that want to run
    the engine command(s) on their own.
    """
    wf = WorkflowType.from_string(runtime_config.workflow)
    v2_workflow = _V2_WORKFLOW_NAMES.get(wf, runtime_config.workflow)
    commands = build_engine_commands(model_spec, runtime_config, json_fpath)
    logger.info("Delegating workflow %r to v2 engine.", v2_workflow)
    return_code = WorkflowRunner(commands).run()
    if return_code != 0:
        logger.error(
            f"⛔ v2 workflow: {v2_workflow}, failed with return code: {return_code}"
        )
    else:
        logger.info(f"✅ Completed v2 workflow: {v2_workflow}")
    return [WorkflowResult(workflow_name=v2_workflow, return_code=return_code)]


def _engine_env() -> dict:
    """Env overrides forwarded to every engine subprocess (VenvCommand merges
    these over ``os.environ``)."""
    return {"TT_V1_RUN_COMMAND": "python " + shlex.join(sys.argv)}


def _engine_run_argv(
    v2_run_py, model_spec, runtime_config, json_fpath, v2_workflow, output_dir, wf
) -> List[str]:
    """Build the ``run_workflows.py`` argv (interpreter-agnostic).

    :class:`VenvCommand` prepends the V2_RUN_SCRIPT interpreter, so this returns
    everything *after* it — the script path plus flags, mirroring what the old
    hand-rolled ``cmd`` carried minus its leading ``venv_python``.
    """
    argv = [
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
        argv.append("--docker-server")
    _extend_if_set(argv, "--server-url", getattr(runtime_config, "server_url", None))
    if wf == WorkflowType.SERVING_BENCH:
        _extend_if_set(
            argv, "--serving-bench-suites", runtime_config.serving_bench_suites
        )
    elif _is_llm_eval_run(wf, model_spec) or _is_llm_spec_test_run(wf, model_spec):
        # Standard evals/release and LLM/VLM parameter-conformance (spec_tests)
        # need the bearer token to reach a JWT-protected server; run.py mints it
        # from --jwt-secret/$JWT_SECRET.
        _forward_jwt(argv, runtime_config)
        if wf == WorkflowType.RELEASE:
            _forward_prefix_cache(argv, runtime_config)
            _forward_spec_decode(argv, runtime_config)
    else:
        sdxl_n = getattr(runtime_config, "sdxl_num_prompts", None)
        if sdxl_n not in (None, "", "0"):
            argv.extend(["--num-prompts", str(sdxl_n)])
    return argv


def _base_engine_argv(
    launcher, model_spec, runtime_config, json_fpath, output_dir, v2_workflow
):
    """Common launcher argv (interpreter-agnostic; VenvCommand prepends python)."""
    argv = [
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
        argv.append("--docker-server")
    if getattr(runtime_config, "server_url", None):
        argv.extend(["--server-url", runtime_config.server_url])
    return argv


def _resolve_launcher(repo_root, filename, label):
    launcher = repo_root / "launchers" / filename
    if not launcher.is_file():
        raise FileNotFoundError(f"{label} launcher not found at {launcher}.")
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


def _stress_argv(repo_root, model_spec, runtime_config, json_fpath):
    """Argv for the stress-tests script (VenvCommand runs it in, and provisions,
    the STRESS_TESTS_RUN_SCRIPT venv)."""
    script = repo_root / "test_module" / "stress_tests" / "run_stress_tests.py"
    output_path = get_default_workflow_root_log_dir() / "stress_tests_output"
    ensure_readwriteable_dir(output_path)
    return [
        str(script),
        "--runtime-model-spec-json",
        str(json_fpath),
        "--output-path",
        str(output_path),
        "--model",
        model_spec.model_name,
        "--device",
        runtime_config.device,
    ]


def _build_agentic_cmd(repo_root, model_spec, runtime_config, json_fpath, output_dir):
    launcher = _resolve_launcher(repo_root, "run_agentic.py", "agentic")
    cmd = _base_engine_argv(
        launcher, model_spec, runtime_config, json_fpath, output_dir, "agentic"
    )
    _forward_jwt(cmd, runtime_config)
    return cmd


def _build_prefix_cache_cmd(repo_root, model_spec, runtime_config, json_fpath, output_dir):
    launcher = _resolve_launcher(repo_root, "run_prefix_cache.py", "prefix-cache")
    cmd = _base_engine_argv(
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


def _build_llm_bench_cmd(repo_root, model_spec, runtime_config, json_fpath, output_dir):
    launcher = _resolve_launcher(repo_root, "run_llm_bench.py", "llm-bench")
    cmd = _base_engine_argv(
        launcher, model_spec, runtime_config, json_fpath, output_dir, "benchmarks"
    )
    _extend_if_set(cmd, "--tools", runtime_config.tools)
    _extend_if_set(cmd, "--goodput", getattr(runtime_config, "goodput", None))
    _forward_jwt(cmd, runtime_config)
    return cmd


def _build_spec_decode_cmd(repo_root, model_spec, runtime_config, json_fpath, output_dir):
    launcher = _resolve_launcher(repo_root, "run_spec_decode.py", "spec-decode")
    cmd = _base_engine_argv(
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
        from reference_config.evals.eval_config import EVAL_CONFIGS
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
        if model_spec.model_type in _LLM_LIKE_TYPES:
            venv_types.extend(_llm_eval_venv_types(model_spec, runtime_config))
    # The release benchmark child runs the default perf tool (vllm) in-process
    # under V2_RUN_SCRIPT, so its tool venv must exist up front.
    if wf == WorkflowType.RELEASE and model_spec.model_type in _LLM_LIKE_TYPES:
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
