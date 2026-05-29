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
from workflows.workflow_types import WorkflowType

logger = logging.getLogger("run_log")

_V2_WORKFLOW_NAMES = {
    WorkflowType.BENCHMARKS: "benchmarks",
    WorkflowType.EVALS: "evals",
    WorkflowType.SPEC_TESTS: "spec_tests",
    WorkflowType.RELEASE: "release",
}

# Only sdxlmodels actually validated end-to-end against v2's engine are routed here.
_V2_ROUTED_MODELS = frozenset(
    {
        "stable-diffusion-xl-base-1.0",
        "stable-diffusion-xl-base-1.0-img-2-img",
        "stable-diffusion-xl-1.0-inpainting-0.1",
    }
)


def is_v2_routed_model(model_spec) -> bool:
    return model_spec.model_name in _V2_ROUTED_MODELS


def can_route_to_v2(model_spec, runtime_config) -> bool:
    if not is_v2_routed_model(model_spec):
        return False
    return WorkflowType.from_string(runtime_config.workflow) in _V2_WORKFLOW_NAMES


def run_v2_workflows(model_spec, runtime_config, json_fpath) -> List[WorkflowResult]:
    wf = WorkflowType.from_string(runtime_config.workflow)
    v2_workflow = _V2_WORKFLOW_NAMES.get(wf)
    if v2_workflow is None:
        raise ValueError(
            f"v2 bridge does not handle workflow {wf.name!r}. "
            f"Supported: {sorted(_V2_WORKFLOW_NAMES.values())}"
        )

    repo_root = Path(__file__).resolve().parent.parent
    v2_run_py = repo_root / "tt-inference-server-v2" / "workflow_runner.py"
    if not v2_run_py.is_file():
        raise FileNotFoundError(
            f"v2 entry point not found at {v2_run_py}. "
            "The tt-inference-server-v2/ directory is required for image-model workflows."
        )

    output_dir = get_default_workflow_root_log_dir() / "reports_output" / v2_workflow
    ensure_readwriteable_dir(output_dir)

    cmd = [
        sys.executable,
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

    env = os.environ.copy()
    env["TT_V1_RUN_COMMAND"] = "python " + shlex.join(sys.argv)

    logger.info("Delegating image-model workflow %r to v2 engine.", v2_workflow)
    return_code = run_command(cmd, logger=logger, env=env)
    if return_code != 0:
        logger.error(
            f"⛔ v2 workflow: {v2_workflow}, failed with return code: {return_code}"
        )
    else:
        logger.info(f"✅ Completed v2 workflow: {v2_workflow}")
    return [WorkflowResult(workflow_name=v2_workflow, return_code=return_code)]


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
