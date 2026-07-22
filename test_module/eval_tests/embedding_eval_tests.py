# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC


from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from workflows.workflow_types import WorkflowVenvType
from workflows.workflow_venvs import VENV_CONFIGS

from report_module.schema import Block

from .._test_common import ReportCheckTypes, block_id
from ..context import HardwareRequirement, MediaContext, require_health

logger = logging.getLogger(__name__)


OPENAI_API_KEY = "your-secret-key"
MTEB_TASKS = ["STS12"]
EMBEDDING_DIMENSIONS = 1000

MTEB_RESULT_START = "===MTEB_RESULT_JSON_START==="
MTEB_RESULT_END = "===MTEB_RESULT_JSON_END==="


def _embedding_model_config(ctx: MediaContext) -> tuple[str, int, int]:
    """Return (hf_model_repo, isl, dimensions) derived from model_spec env vars."""
    env = ctx.model_spec.device_model_spec.env_vars
    return (
        ctx.model_spec.hf_model_repo,
        int(env.get("VLLM__MAX_MODEL_LENGTH", 1024)),
        EMBEDDING_DIMENSIONS,
    )


def _parse_embedding_evals_output(stdout: str) -> dict:
    """Extract the JSON metrics block emitted by _mteb_eval_runner.py."""
    if MTEB_RESULT_START not in stdout or MTEB_RESULT_END not in stdout:
        logger.error("MTEB result markers not found in runner output.")
        raise ValueError("MTEB runner produced no result block")
    section = stdout.split(MTEB_RESULT_START, 1)[1].split(MTEB_RESULT_END, 1)[0].strip()
    report_data = json.loads(section)
    logger.info(f"Parsed evaluation results: {report_data}")
    return report_data


def _run_embedding_mteb_eval(ctx: MediaContext) -> dict:
    """Run the MTEB eval inside the EVALS_EMBEDDING venv and return metrics."""
    model_name, isl, dimensions = _embedding_model_config(ctx)

    venv_config = VENV_CONFIGS.get(WorkflowVenvType.EVALS_EMBEDDING)
    venv_python = venv_config.venv_path / "bin" / "python"
    if not venv_python.is_file():
        raise FileNotFoundError(
            f"EVALS_EMBEDDING venv python not found at {venv_python}; "
            "venv not provisioned."
        )

    runner = Path(__file__).with_name("_mteb_eval_runner.py")
    cmd = [
        str(venv_python),
        str(runner),
        "--base-url",
        ctx.base_url,
        "--model",
        model_name,
        "--isl",
        str(isl),
        "--dimensions",
        str(dimensions),
        "--api-key",
        OPENAI_API_KEY,
        "--tasks",
        *MTEB_TASKS,
    ]

    logger.info("Running embedding MTEB eval via %s: tasks=%s", venv_python, MTEB_TASKS)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return _parse_embedding_evals_output(proc.stdout)


def run_embedding_eval(ctx: MediaContext) -> Block:
    """Run evaluations for an embedding model."""
    logger.info(
        f"Running evals for model: {ctx.model_spec.model_name} on device: {ctx.device.name}"
    )
    require_health(ctx, HardwareRequirement.ANY_CHIP)

    try:
        logger.info("Running embedding eval...")
        metrics = _run_embedding_mteb_eval(ctx)
    except Exception as e:
        logger.error(f"Eval execution encountered an error: {e}")
        raise

    logger.info("Generating evals report...")
    return Block(
        kind="evals",
        task_type="embedding",
        title="Embedding Eval",
        id=block_id(ctx) or None,
        targets={"task_name": ctx.all_params.tasks[0].task_name},
        data={
            "task_name": ctx.all_params.tasks[0].task_name,
            # MTEB produces correlation metrics but no reference score to grade
            # against, so accuracy is Not Applicable (non-blocking).
            "accuracy_check": ReportCheckTypes.NA,
            **metrics,
        },
    )


__all__ = ["run_embedding_eval"]
