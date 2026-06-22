# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Orchestrator for the AIPerf speculative-decoding benchmark sweep.

Bridges ``test_module`` to ``llm_module``: builds the sweep plan from
``llm_module.spec_decode``, runs one :class:`AIPerfSpecDecodeDriver`
invocation per :class:`SpecDecodeRun`, converts each driver payload into
a :class:`report_module.schema.Block` via :class:`AIPerfSpecDecodeParser`,
and forwards the Blocks to ``workflow_module.accept_blocks`` so the
unified report generator picks them up alongside any other workflow
output.

Designed for **sequential, one-server-at-a-time** use (same as v1's
``benchmarking/run_spec_decode_benchmarks.py``): one invocation runs the
sweep against whatever server it is pointed at. Server-side speculative
config is out of scope — it belongs to whoever launched the server,
before the benchmark starts.
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from llm_module import ServerConnection
from llm_module.config import DriverContext
from llm_module.drivers.aiperf_spec_decode import (
    AIPerfSpecDecodeDriver,
    SpecDecodeDriverResult,
)
from llm_module.parsers.aiperf_spec_decode import AIPerfSpecDecodeParser
from llm_module.spec_decode import (
    build_runs as build_spec_decode_runs,
    summarize_runs as summarize_spec_decode_runs,
)
from report_module.schema import Block
from workflow_module import accept_blocks

from ..context import MediaContext

logger = logging.getLogger(__name__)

DEFAULT_WARMUP_REQUESTS = 4


def run_spec_decode(
    ctx: MediaContext,
    *,
    preset: str = "full",
    warmup_requests: int = DEFAULT_WARMUP_REQUESTS,
    auth_token: str = "",
    venv_python: Optional[Path] = None,
    output_subdir: str = "spec_decode",
    inter_run_sleep_s: float = 2.0,
    health_timeout_s: float = 600.0,
) -> List[Block]:
    """Run the spec-decode sweep end-to-end.

    Parameters
    ----------
    ctx:
        v2 :class:`MediaContext` (same one ``run.py`` uses for the media
        workflows). Provides ``model_spec``, ``device``, ``service_port``,
        and ``output_path``.
    preset:
        Sweep preset (``--spec-decode-preset``). ``full`` (default) runs
        every SPEED-Bench qualitative category plus the whole throughput
        ISL x concurrency grid; ``ci`` runs only the ``coding``
        qualitative category plus ``speed_bench_throughput_32k`` at
        concurrency 1/16/64.
    warmup_requests:
        Short chat-completion requests sent before the sweep (matches v1
        behavior; 0 disables).
    auth_token:
        Bearer token sent to the inference server (JWT, OPENAI_API_KEY).
        Empty string disables auth.
    venv_python:
        Python interpreter that has ``aiperf`` installed. Falls back to
        ``sys.executable``.
    output_subdir:
        Sub-directory of ``ctx.output_path`` where the per-run JSONs and
        AIPerf artifacts go.
    inter_run_sleep_s:
        Seconds to sleep between runs so ``/metrics`` ticks settle
        (matches v1 behavior).

    Returns
    -------
    list[Block]
        One Block per successful run (kind ``aiperf_spec_decode``). The
        same Blocks are also forwarded to
        :func:`workflow_module.accept_blocks` so the unified report
        generator picks them up.
    """
    runs = build_spec_decode_runs(preset)
    if not runs:
        logger.error("Spec-decode sweep is empty (preset=%s); nothing to run.", preset)
        return []
    logger.info(summarize_spec_decode_runs(runs))

    output_root = Path(ctx.output_path) / output_subdir
    artifact_root = output_root / "aiperf_artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    spec = ctx.model_spec
    model_repo = getattr(spec, "hf_model_repo", "") or ""
    model_id = getattr(spec, "model_id", "") or model_repo
    device_label = ctx.device.name if hasattr(ctx.device, "name") else str(ctx.device)

    driver = AIPerfSpecDecodeDriver(
        venv_python=Path(venv_python) if venv_python else Path(sys.executable),
        artifact_root=artifact_root,
        model_repo=model_repo,
        model_id=model_id,
        tokenizer=model_repo,
        output_dir=output_root,
    )

    server = ServerConnection(
        base_url="http://localhost",
        service_port=ctx.service_port,
        model=model_repo,
        tokenizer=model_repo,
        auth_token=auth_token,
    )
    context = DriverContext(output_dir=output_root, device=device_label)

    url = server.url_with_port
    if not _wait_for_url_healthy(url, auth_token=auth_token, timeout=health_timeout_s):
        logger.error("[spec-decode] endpoint not healthy at %s; aborting sweep.", url)
        return []

    _warmup_endpoint(
        url, model_repo, auth_token=auth_token, num_requests=warmup_requests
    )

    parser = AIPerfSpecDecodeParser()
    blocks: List[Block] = []
    for i, run in enumerate(runs, 1):
        logger.info("[spec-decode] Running %d/%d: %s", i, len(runs), run.slug)
        if i > 1 and inter_run_sleep_s:
            time.sleep(inter_run_sleep_s)

        outcome: SpecDecodeDriverResult = driver.run(run, server, context)
        if outcome.return_code != 0 or outcome.payload is None:
            logger.error(
                "[spec-decode] %s failed (rc=%d); continuing.",
                run.slug,
                outcome.return_code,
            )
            continue

        blocks.append(parser.parse(outcome.payload, device=device_label))

    if not blocks:
        logger.error("[spec-decode] No blocks produced -- sweep had zero successes.")
        return []

    accept_blocks(
        blocks,
        envelope={
            "model_name": getattr(spec, "model_name", "") or model_repo,
            "device": device_label,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )
    logger.info(
        "[spec-decode] Sweep complete: %d successful run(s) / %d planned",
        len(blocks),
        len(runs),
    )
    return blocks


def _wait_for_url_healthy(
    base_url: str,
    *,
    auth_token: str = "",
    timeout: float = 600.0,
    interval: float = 5.0,
) -> bool:
    """Poll ``{base_url}/health`` until it returns 200 or the deadline expires."""
    import requests

    headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
    health_url = base_url.rstrip("/") + "/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = requests.get(health_url, headers=headers, timeout=10)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException as exc:
            logger.debug("health probe to %s failed: %s", health_url, exc)
        time.sleep(interval)
    return False


def _warmup_endpoint(
    base_url: str,
    hf_model_repo: str,
    *,
    auth_token: str = "",
    num_requests: int = DEFAULT_WARMUP_REQUESTS,
    max_tokens: int = 32,
    timeout: float = 120.0,
) -> int:
    """Send ``num_requests`` identical short chat-completion requests.

    Returns the number of successful warmup requests (0 to num_requests).
    """
    import requests

    if num_requests <= 0:
        return 0
    headers = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": hf_model_repo,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Briefly summarize the following sentence: "
                    "Speculative decoding lets a small draft model propose "
                    "tokens that the larger target model verifies in parallel."
                ),
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    successes = 0
    for i in range(num_requests):
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=timeout
            )
            response.raise_for_status()
            successes += 1
        except requests.exceptions.RequestException as exc:
            logger.warning("warmup request %d/%d failed: %s", i + 1, num_requests, exc)
    logger.info(
        "warmup: %d/%d requests succeeded at %s", successes, num_requests, base_url
    )
    return successes


__all__ = ["run_spec_decode"]
