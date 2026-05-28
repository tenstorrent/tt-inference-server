# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Speculative-decoding benchmark runner.

Drives one ``AIPerfSpecDecodeDriver`` over the configured sweep
(:data:`SPEC_DECODE_PROFILES`) twice — once with the inference server in
baseline mode, once with spec-decode enabled. Both phases run inside a
single v2 invocation; an ``input()`` prompt sits between them while the
user manually restarts the server with the toggled spec-decode config
(out-of-band — :class:`ServerController` does not own that lifecycle).

The runner emits one :class:`Block` per (sweep point × phase) tagged with
``phase="baseline"`` / ``phase="spec"``, plus a final ``baseline ↔ spec``
pairing step handled in the workflow's ``format_results`` override via
:func:`report_module.spec_decode_pairing.pair_baseline_spec`.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import requests

from llm_module import (
    AIPerfSpecDecodeDriver,
    DriverContext,
    ServerConnection,
    SpecDecodeRunConfig,
)
from report_module.schema import Block
from workflow_module import accept_blocks

from ..context import MediaContext
from .spec_decode_config import SPEC_DECODE_PROFILES

logger = logging.getLogger(__name__)

PHASE_BASELINE = "baseline"
PHASE_SPEC = "spec"

DEFAULT_WARMUP_REQUESTS = 4
SPEC_DECODE_PHASE_PAUSE_ENV = "SPEC_DECODE_PHASE_PAUSE"


def run_llm_spec_decode_benchmark(ctx: MediaContext) -> Tuple[List[Block], List[int]]:
    """Run the two-phase spec-decode sweep against ``ctx``'s endpoint.

    Returns ``(blocks, return_codes)`` so the dispatcher can surface
    non-zero aiperf exits while still leaving the Blocks it did produce
    in the accumulator.
    """
    profile_name = (ctx.spec_decode_profile or "full").strip().lower()
    if profile_name not in SPEC_DECODE_PROFILES:
        raise ValueError(
            f"Unknown spec-decode profile {profile_name!r}. "
            f"Available: {sorted(SPEC_DECODE_PROFILES)}"
        )
    configs = SPEC_DECODE_PROFILES[profile_name]
    if not configs:
        raise ValueError(f"Spec-decode profile {profile_name!r} is empty")

    driver = AIPerfSpecDecodeDriver(venv_python=ctx.spec_decode_venv_python)
    server = ServerConnection(
        base_url="http://localhost",
        service_port=ctx.service_port,
        model=ctx.model_spec.hf_model_repo,
    )
    device_label = ctx.device.name if hasattr(ctx.device, "name") else str(ctx.device)
    driver_context = DriverContext(
        output_dir=Path(ctx.output_path) / "spec_decode",
        device=device_label,
    )

    warmup_requests = (
        ctx.spec_decode_warmup_requests
        if ctx.spec_decode_warmup_requests is not None
        else DEFAULT_WARMUP_REQUESTS
    )
    envelope = {
        "model_name": server.model,
        "device": device_label,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    all_blocks: List[Block] = []
    all_rcs: List[int] = []

    logger.info(
        "Spec-decode sweep: profile=%s configs=%d url=%s",
        profile_name,
        len(configs),
        server.url_with_port,
    )

    # Phase 1: baseline (server is presumed to be running WITHOUT spec-decode).
    rcs_baseline, blocks_baseline = _run_phase(
        phase=PHASE_BASELINE,
        configs=configs,
        driver=driver,
        server=server,
        driver_context=driver_context,
        device_label=device_label,
        warmup_requests=warmup_requests,
    )
    all_rcs.extend(rcs_baseline)
    all_blocks.extend(blocks_baseline)
    if blocks_baseline:
        accept_blocks(blocks_baseline, envelope=envelope)

    # Pause and wait for the user to restart the server with spec-decode on.
    if not _phase_pause():
        logger.warning("Spec-decode phase pause skipped by user; aborting sweep.")
        return all_blocks, all_rcs

    # Phase 2: spec (server is now presumed to be running WITH spec-decode on).
    rcs_spec, blocks_spec = _run_phase(
        phase=PHASE_SPEC,
        configs=configs,
        driver=driver,
        server=server,
        driver_context=driver_context,
        device_label=device_label,
        warmup_requests=warmup_requests,
    )
    all_rcs.extend(rcs_spec)
    all_blocks.extend(blocks_spec)
    if blocks_spec:
        accept_blocks(blocks_spec, envelope=envelope)

    return all_blocks, all_rcs


def _run_phase(
    *,
    phase: str,
    configs: Sequence[SpecDecodeRunConfig],
    driver: AIPerfSpecDecodeDriver,
    server: ServerConnection,
    driver_context: DriverContext,
    device_label: str,
    warmup_requests: int,
) -> Tuple[List[int], List[Block]]:
    """Drive ``driver`` over the sweep for a single phase.

    Steps: poll health, warmup, then loop config × driver.run() →
    driver.parse_with_phase(). A failed health check skips the phase
    (returning a single rc=1). Per-config aiperf failures are recorded
    but don't abort the rest of the sweep.
    """
    logger.info("=== Phase: %s ===", phase)
    url = server.url_with_port
    auth_token = server.auth_token

    if not _wait_for_url_healthy(url, jwt_token=auth_token):
        logger.error("[%s] endpoint not healthy at %s; aborting phase.", phase, url)
        return [1], []

    _warmup_endpoint(
        url,
        server.model,
        jwt_token=auth_token,
        num_requests=warmup_requests,
    )

    blocks: List[Block] = []
    rcs: List[int] = []
    for i, config in enumerate(configs, 1):
        logger.info("[%s %d/%d] running %s", phase, i, len(configs), config.slug)
        time.sleep(2)  # let /metrics counters settle between runs
        outcome = driver.run(config, server, driver_context)
        rcs.append(outcome.return_code)
        if outcome.return_code != 0:
            logger.error(
                "[%s] aiperf exited %d on %s", phase, outcome.return_code, config.slug
            )
            continue
        if outcome.raw is None:
            logger.error(
                "[%s] aiperf produced no parseable raw result for %s",
                phase,
                config.slug,
            )
            continue
        block = driver.parse_with_phase(
            outcome.raw, device=device_label, phase=phase
        )
        blocks.append(block)
        acceptance = (block.data or {}).get("Spec Decode", {}).get("acceptance_rate")
        logger.info(
            "[%s] %s acceptance_rate=%.3f",
            phase,
            config.slug,
            acceptance if acceptance is not None else float("nan"),
        )
    return rcs, blocks


def _phase_pause() -> bool:
    """Block until the user signals the server has been restarted.

    Returns ``True`` to proceed to the spec phase, ``False`` to abort.
    The env var ``SPEC_DECODE_PHASE_PAUSE`` overrides the default
    ``input()`` prompt for non-TTY use:

      - ``skip`` / ``continue`` / ``yes``: skip the prompt, proceed to
        the spec phase (assumes external orchestration handled the
        restart and is ready).
      - ``abort`` / ``no``: skip the prompt, abort the spec phase.
      - anything else (or unset): show the interactive prompt.
    """
    override = os.environ.get(SPEC_DECODE_PHASE_PAUSE_ENV, "").strip().lower()
    if override in {"skip", "continue", "yes"}:
        logger.info(
            "%s=%s → proceeding to spec phase without prompt",
            SPEC_DECODE_PHASE_PAUSE_ENV,
            override,
        )
        return True
    if override in {"abort", "no"}:
        logger.info(
            "%s=%s → aborting spec phase without prompt",
            SPEC_DECODE_PHASE_PAUSE_ENV,
            override,
        )
        return False

    msg = (
        "\n"
        "================================================================\n"
        " Baseline phase complete.\n"
        " Restart the inference server with spec-decode ENABLED, wait for\n"
        " it to be healthy at the same URL, then press Enter to continue.\n"
        " (Type 'abort' + Enter to skip the spec phase and exit.)\n"
        "================================================================\n"
        "> "
    )
    try:
        reply = input(msg).strip().lower()
    except EOFError:
        logger.error("EOF while waiting for phase-pause input; aborting spec phase.")
        return False
    if reply == "abort":
        return False
    return True


def _wait_for_url_healthy(
    base_url: str,
    *,
    jwt_token: str = "",
    timeout: float = 600.0,
    interval: float = 5.0,
) -> bool:
    """Poll ``{base_url}/health`` until it returns 200 or ``timeout`` elapses."""
    headers = {"Authorization": f"Bearer {jwt_token}"} if jwt_token else {}
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
    jwt_token: str = "",
    num_requests: int = DEFAULT_WARMUP_REQUESTS,
    max_tokens: int = 32,
    timeout: float = 120.0,
) -> int:
    """Fire ``num_requests`` identical short chat-completion requests.

    Warms CUDA/HBM kernel caches, autotune passes, and KV-cache machinery
    so the first measured benchmark request doesn't pay cold-start cost.
    Identical payload across baseline and spec phases so neither side is
    "more warmed" than the other. Returns the count of successful warmup
    requests.
    """
    if num_requests <= 0:
        return 0
    headers = {"Content-Type": "application/json"}
    if jwt_token:
        headers["Authorization"] = f"Bearer {jwt_token}"
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


__all__ = [
    "PHASE_BASELINE",
    "PHASE_SPEC",
    "SPEC_DECODE_PHASE_PAUSE_ENV",
    "run_llm_spec_decode_benchmark",
]
