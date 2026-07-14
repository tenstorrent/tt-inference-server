# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Thin v2 bridge for agentic eval drivers."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

from llm_module import (
    DriverContext,
    HttpServerController,
    LLMRunConfig,
    RemoteOpenAIController,
    ServerConnection,
    make_agentic_driver,
)
from report_module.schema import Block
from workflows.workflow_types import WorkflowVenvType
from workflow_module import accept_blocks

from .._test_common import sweep_envelope
from ..context import MediaContext

logger = logging.getLogger(__name__)


def _select_agentic_tasks(ctx: MediaContext) -> list:
    """Return the EVALS_AGENTIC tasks for this model.

    Standard (lm-eval) tasks in the same EvalConfig are owned by
    ``--workflow evals`` (which conversely filters out agentic tasks), so the
    agentic runner simply selects the agentic tasks and skips the rest. Mixed
    configs are a normal pattern (e.g. a model with GPQA + Terminal-Bench +
    SWE-bench), and ``--eval-samples`` cannot be combined with ``--ci-mode``
    (they are mutually exclusive in run.py), so failing hard on mixed configs
    would leave no way to run agentic evals in CI.
    """
    tasks = getattr(ctx.all_params, "tasks", []) or []
    agentic = [
        t for t in tasks if t.workflow_venv_type == WorkflowVenvType.EVALS_AGENTIC
    ]
    non_agentic = [
        t for t in tasks if t.workflow_venv_type != WorkflowVenvType.EVALS_AGENTIC
    ]
    if non_agentic:
        logger.info(
            "Skipping %d non-agentic task(s) under --workflow agentic "
            "(run them via --workflow evals): %s",
            len(non_agentic),
            [t.task_name for t in non_agentic],
        )
    return agentic


def _server_connection(ctx: MediaContext) -> ServerConnection:
    return ServerConnection(
        base_url=ctx.server_host,
        service_port=ctx.server_port,
        model=ctx.model_spec.hf_model_repo,
    )


def _driver_context(ctx: MediaContext) -> DriverContext:
    device = ctx.device.name if hasattr(ctx.device, "name") else str(ctx.device)
    # In a `release` run the agentic driver shares the run directory with the
    # LLM benchmark ("llm/") and prefix-cache ("prefix_cache/") outputs, so
    # group agentic results under a top-level "agentic/" dir (mirroring the
    # LLM layout) via ``agentic_release_layout``. The standalone `agentic`
    # workflow keeps its existing eval_<hf>/agentic/<task> layout.
    release_layout = getattr(ctx.runtime_config, "workflow", None) == "release"
    return DriverContext(
        output_dir=Path(ctx.output_path),
        device=device,
        agentic_release_layout=release_layout,
    )


def _configure_openai_env(ctx: MediaContext) -> None:
    base_url = f"{ctx.base_url}/v1"
    os.environ.setdefault("OPENAI_API_KEY", os.getenv("API_KEY", "EMPTY"))
    os.environ.setdefault("OPENAI_BASE_URL", base_url)
    os.environ.setdefault("OPENAI_API_BASE", base_url)
    logger.info("OpenAI-compatible environment configured for agentic evals.")


def _require_openai_server(ctx: MediaContext) -> None:
    """Block until the inference server is ready for the agentic harnesses.

    Reuses the same readiness controllers as the LLM eval/benchmark paths
    instead of a single-shot probe, so a server that is still coming up (or
    hits a transient blip) is retried rather than failing the run instantly:

    * local ``--docker-server`` (``--net host``) deployments are polled on
      vLLM's ``/health`` via :class:`HttpServerController`;
    * remote OpenAI-compatible endpoints, which do not expose ``/health``, are
      polled on ``/v1/models`` via :class:`RemoteOpenAIController`.
    """
    auth_token = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or ""
    if ctx.remote_server:
        controller = RemoteOpenAIController(
            base_url=ctx.server_url,
            auth_token=auth_token,
        )
    else:
        controller = HttpServerController(
            base_url=ctx.server_host,
            service_port=ctx.server_port,
            auth_token=auth_token,
        )

    endpoint = getattr(controller, "health_url", None) or getattr(
        controller, "models_url", ""
    )
    if not controller.wait_for_healthy():
        raise RuntimeError(f"Inference server health check failed at {endpoint}")
    logger.info("Inference server health check passed via %s", endpoint)


def run_llm_agentic_eval(ctx: MediaContext) -> List[Block]:
    """Run every EVALS_AGENTIC task for this model; return one Block per task."""
    _configure_openai_env(ctx)
    _require_openai_server(ctx)

    agentic_tasks = _select_agentic_tasks(ctx)
    if not agentic_tasks:
        raise RuntimeError(
            f"No EVALS_AGENTIC tasks configured for {ctx.model_spec.model_name!r}. "
            "Check evals/eval_config.py."
        )

    runtime_config = getattr(ctx, "runtime_config", None)
    server = _server_connection(ctx)
    driver_context = _driver_context(ctx)
    placeholder_config = LLMRunConfig(isl=0, osl=0, max_concurrency=0, num_prompts=0)

    blocks: List[Block] = []
    for task in agentic_tasks:
        driver = make_agentic_driver(task, runtime_config=runtime_config)
        logger.info("Running %s task: %s", driver.name, task.task_name)
        outcome = driver.run(placeholder_config, server, driver_context)

        if outcome.return_code != 0:
            logger.error(
                "Task %s exited with rc=%d",
                task.task_name,
                outcome.return_code,
            )
            blocks.append(
                driver.failure_block(
                    return_code=outcome.return_code,
                    device=driver_context.device,
                )
            )
            continue
        if outcome.raw is None:
            continue

        blocks.append(driver.parse(outcome.raw, device=driver_context.device))
        logger.info(
            "Task %s done: accuracy=%s",
            task.task_name,
            blocks[-1].data.get("accuracy"),
        )

    accept_blocks(blocks, envelope=sweep_envelope(ctx))
    return blocks


__all__ = ["run_llm_agentic_eval"]
