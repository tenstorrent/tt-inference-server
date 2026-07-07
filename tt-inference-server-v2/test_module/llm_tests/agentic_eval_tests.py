# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Thin v2 bridge for agentic eval drivers."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from llm_module import (
    DriverContext,
    HttpServerController,
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

# Fallback health-wait budget when the model spec doesn't set one (mirrors
# llm_eval_tests; the per-model value comes from tensor_cache_timeout).
_DEFAULT_WAIT_HEALTHY_TIMEOUT_S = 3600.0


def _select_agentic_tasks(ctx: MediaContext) -> list:
    """Return the EVALS_AGENTIC subset of the model's eval tasks.

    Non-agentic tasks are simply ignored: standard (lm-eval / lmms-eval)
    tasks belong to the evals workflow, and a release run drives both
    children off the same task list.
    """
    tasks = getattr(ctx.all_params, "tasks", []) or []
    agentic = [
        t for t in tasks if t.workflow_venv_type == WorkflowVenvType.EVALS_AGENTIC
    ]
    non_agentic = [
        t.task_name
        for t in tasks
        if t.workflow_venv_type != WorkflowVenvType.EVALS_AGENTIC
    ]
    if agentic and non_agentic:
        logger.info(
            "Ignoring non-agentic eval tasks (the standard evals workflow "
            "runs them): %s",
            non_agentic,
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
    return DriverContext(output_dir=Path(ctx.output_path), device=device)


def _configure_openai_env(ctx: MediaContext, auth_token: str = "") -> None:
    base_url = f"{ctx.base_url}/v1"
    if auth_token:
        # The agentic harnesses (harbor / sweagent / mini-extra subprocesses)
        # read the bearer token from OPENAI_API_KEY.
        os.environ["OPENAI_API_KEY"] = auth_token
    else:
        os.environ.setdefault("OPENAI_API_KEY", os.getenv("API_KEY", "EMPTY"))
    os.environ.setdefault("OPENAI_BASE_URL", base_url)
    os.environ.setdefault("OPENAI_API_BASE", base_url)
    logger.info("OpenAI-compatible environment configured for agentic evals.")


def _wait_for_healthy(ctx: MediaContext, auth_token: str = "") -> bool:
    """Poll /health until the server is up (a cold server otherwise fails the
    single /v1/models probe below with connection-refused)."""
    server = HttpServerController(
        base_url=ctx.server_host,
        service_port=ctx.server_port,
        auth_token=auth_token,
    )
    timeout = (
        getattr(
            getattr(ctx.model_spec, "device_model_spec", None),
            "tensor_cache_timeout",
            None,
        )
        or _DEFAULT_WAIT_HEALTHY_TIMEOUT_S
    )
    return server.wait_for_healthy(timeout=timeout)


def _require_openai_server(ctx: MediaContext, auth_token: str = "") -> None:
    """Check the OpenAI-compatible server path used by agentic harnesses."""

    url = f"{ctx.base_url}/v1/models"
    headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
    try:
        with urlopen(Request(url, headers=headers), timeout=30) as response:
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


def run_llm_agentic_eval(
    ctx: MediaContext,
    *,
    auth_token: str = "",
    venv_python: Optional[str] = None,
) -> List[Block]:
    """Run every EVALS_AGENTIC task for this model; return one Block per task.

    ``auth_token`` is the bearer token sent to a JWT/API-key-protected server;
    ``venv_python`` pins the EVALS_AGENTIC interpreter for the release path
    (``None`` resolves the harness binaries from ``sys.executable``).
    """
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

    _configure_openai_env(ctx, auth_token)
    if not _wait_for_healthy(ctx, auth_token):
        # Mirror run_llm_eval: emit FAIL blocks so the report shows the tasks
        # that never ran (acceptance fails on them) instead of dropping them.
        logger.error("⛔ inference server not healthy; aborting agentic evals.")
        blocks = [
            make_agentic_driver(
                task, runtime_config=runtime_config, venv_python=venv_python
            ).failure_block(return_code=1, device=driver_context.device)
            for task in agentic_tasks
        ]
        accept_blocks(blocks, envelope=sweep_envelope(ctx))
        return blocks
    _require_openai_server(ctx, auth_token)

    blocks: List[Block] = []
    for task in agentic_tasks:
        driver = make_agentic_driver(
            task, runtime_config=runtime_config, venv_python=venv_python
        )
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
