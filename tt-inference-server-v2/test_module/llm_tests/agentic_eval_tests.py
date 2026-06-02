# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Thin v2 bridge for agentic eval drivers."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List
from urllib.error import URLError
from urllib.request import urlopen

from llm_module import (
    DriverContext,
    LLMRunConfig,
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
    """Return EVALS_AGENTIC tasks; raise loudly if mixed with non-agentic."""
    tasks = getattr(ctx.all_params, "tasks", []) or []
    agentic = [
        t for t in tasks if t.workflow_venv_type == WorkflowVenvType.EVALS_AGENTIC
    ]
    non_agentic = [
        t for t in tasks if t.workflow_venv_type != WorkflowVenvType.EVALS_AGENTIC
    ]
    if agentic and non_agentic:
        raise RuntimeError(
            f"v2 agentic runner only supports EVALS_AGENTIC tasks. "
            f"Got non-agentic tasks: {[t.task_name for t in non_agentic]}. "
            f"Either port those to v2, remove {ctx.model_spec.model_name!r} from "
            f"_V2_ROUTED_MODELS, or use --eval-samples to select agentic tasks only."
        )
    return agentic


def _server_connection(ctx: MediaContext) -> ServerConnection:
    return ServerConnection(
        base_url="http://127.0.0.1",
        service_port=ctx.service_port,
        model=ctx.model_spec.hf_model_repo,
    )


def _driver_context(ctx: MediaContext) -> DriverContext:
    device = ctx.device.name if hasattr(ctx.device, "name") else str(ctx.device)
    return DriverContext(output_dir=Path(ctx.output_path), device=device)


def _configure_openai_env(ctx: MediaContext) -> None:
    base_url = f"http://127.0.0.1:{ctx.service_port}/v1"
    os.environ.setdefault("OPENAI_API_KEY", os.getenv("API_KEY", "EMPTY"))
    os.environ.setdefault("OPENAI_BASE_URL", base_url)
    os.environ.setdefault("OPENAI_API_BASE", base_url)
    logger.info("OpenAI-compatible environment configured for agentic evals.")


def _require_openai_server(ctx: MediaContext) -> None:
    """Check the OpenAI-compatible server path used by agentic harnesses."""

    url = f"http://127.0.0.1:{ctx.service_port}/v1/models"
    try:
        with urlopen(url, timeout=30) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"Expected status 200 from {url}, got {response.status}"
                )
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"OpenAI-compatible server health check failed: {url}"
        ) from exc

    model_ids = [
        item.get("id")
        for item in payload.get("data", [])
        if isinstance(item, dict) and item.get("id")
    ]
    expected = ctx.model_spec.hf_model_repo
    if expected not in model_ids:
        logger.warning(
            "OpenAI server is healthy but %s was not listed by /v1/models: %s",
            expected,
            model_ids,
        )
    logger.info("OpenAI-compatible server health check passed via %s", url)


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
