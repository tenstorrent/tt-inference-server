# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from __future__ import annotations

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

_V2_WORKFLOW_NAMES = {
    WorkflowType.BENCHMARKS: "benchmarks",
    WorkflowType.EVALS: "evals",
    WorkflowType.SPEC_TESTS: "spec_tests",
    WorkflowType.RELEASE: "release",
    WorkflowType.AGENTIC: "agentic",
}

_V2_EVAL_WORKFLOWS = frozenset({WorkflowType.EVALS, WorkflowType.RELEASE})


_V2_EVAL_VENV_BY_MODEL_TYPE = {
    ModelType.AUDIO: WorkflowVenvType.EVALS_AUDIO,
}

# Only models actually validated end-to-end against v2's engine are routed here.
_V2_ROUTED_MODELS = frozenset(
    {
        "stable-diffusion-xl-base-1.0",
        "stable-diffusion-xl-base-1.0-img-2-img",
        "stable-diffusion-xl-1.0-inpainting-0.1",
        "whisper-large-v3",
        "distil-large-v3",
        "Z-Image-Turbo",
    }
)


def is_v2_routed_model(model_spec) -> bool:
    return model_spec.model_name in _V2_ROUTED_MODELS


def _is_prefix_cache_run(wf, runtime_config) -> bool:
    return wf == WorkflowType.BENCHMARKS and getattr(
        runtime_config, "prefix_cache", False
    )


def can_route_to_v2(model_spec, runtime_config) -> bool:
    wf = WorkflowType.from_string(runtime_config.workflow)
    # Agentic evals and the prefix-cache benchmark are v2-only features with no
    # v1 driver. They route to v2 for ANY model (not just the image/audio set in
    # _V2_ROUTED_MODELS), launched through their dedicated venv launchers.
    if wf == WorkflowType.AGENTIC:
        return True
    if _is_prefix_cache_run(wf, runtime_config):
        return True
    if not is_v2_routed_model(model_spec):
        return False
    return wf in _V2_WORKFLOW_NAMES


def run_v2_workflows(model_spec, runtime_config, json_fpath) -> List[WorkflowResult]:
    wf = WorkflowType.from_string(runtime_config.workflow)
    v2_workflow = _V2_WORKFLOW_NAMES.get(wf)
    if v2_workflow is None:
        raise ValueError(
            f"v2 bridge does not handle workflow {wf.name!r}. "
            f"Supported: {sorted(_V2_WORKFLOW_NAMES.values())}"
        )

    repo_root = Path(__file__).resolve().parent.parent
    v2_dir = repo_root / "tt-inference-server-v2"

    output_dir = get_default_workflow_root_log_dir() / "reports_output" / v2_workflow
    ensure_readwriteable_dir(output_dir)

    # Agentic and prefix-cache go through their dedicated venv launchers
    # (run_agentic.py / run_prefix_cache.py), which materialize the
    # EVALS_AGENTIC / V2_PREFIX_CACHE venv and re-exec run.py inside it. They
    # run from this interpreter (sys.executable) — the launchers import only
    # the lightweight workflows.* helpers before re-execing.
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
    else:
        v2_run_py = v2_dir / "run.py"
        if not v2_run_py.is_file():
            raise FileNotFoundError(
                f"v2 entry point not found at {v2_run_py}. "
                "The tt-inference-server-v2/ directory is required for image-model workflows."
            )
        venv_python = _ensure_v2_venv(model_spec)
        _ensure_v2_dependency_venvs(model_spec, wf)

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
    return [WorkflowResult(workflow_name=v2_workflow, return_code=return_code)]


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
    return cmd


def _build_agentic_cmd(v2_dir, model_spec, runtime_config, json_fpath, output_dir):
    launcher = v2_dir / "run_agentic.py"
    if not launcher.is_file():
        raise FileNotFoundError(f"v2 agentic launcher not found at {launcher}.")
    return _base_v2_cmd(
        launcher, model_spec, runtime_config, json_fpath, output_dir, "agentic"
    )


def _build_prefix_cache_cmd(v2_dir, model_spec, runtime_config, json_fpath, output_dir):
    launcher = v2_dir / "run_prefix_cache.py"
    if not launcher.is_file():
        raise FileNotFoundError(f"v2 prefix-cache launcher not found at {launcher}.")
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
    # run.py reads $JWT_SECRET when --jwt-secret is omitted; only forward an
    # explicit value so the env fallback still works.
    _extend_if_set(cmd, "--jwt-secret", runtime_config.jwt_secret)
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


def _v2_dependency_venv_types(model_spec, wf) -> List[WorkflowVenvType]:
    venv_types: List[WorkflowVenvType] = []
    if wf in _V2_EVAL_WORKFLOWS:
        eval_venv = _V2_EVAL_VENV_BY_MODEL_TYPE.get(model_spec.model_type)
        if eval_venv is not None:
            venv_types.append(eval_venv)
    return venv_types


def _ensure_v2_dependency_venvs(model_spec, wf) -> None:
    for venv_type in _v2_dependency_venv_types(model_spec, wf):
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
