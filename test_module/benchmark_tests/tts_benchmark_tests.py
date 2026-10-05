# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Workflow adapter for the chunk-aware TTS load harness.

The load generator remains usable as a standalone CLI. This module supplies
the workflow-facing policy (the CI sweep shape), translates its aggregate rows
to the common report schema, and grades the six FC/SC/TC percentile SLOs.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import aiohttp
from report_module.schema import Block
from workflow_module.context_helpers import get_num_calls

from .._test_common import (
    MetricSpec,
    ReportCheckTypes,
    block_id,
    run_tiered_check,
)
from ..context import MediaContext, require_health
from ..test_status import TtsTestStatus
from . import tts_load_harness

logger = logging.getLogger(__name__)

DEFAULT_TTS_TEXT = "Hello, this is a test of the text to speech system."
TT_TTS_IMPL = "tt-tts"


# QUAD validation shape from docs/TTS_SHARED_MEMORY_DECODERS.md. Five levels
# exercise the decoder scaling curve while keeping the default CI run bounded.
TTS_CI_CONCURRENCY = (1, 2, 4, 8, 16)
TTS_CI_DURATION_SECONDS = 60.0
TTS_CI_WARMUP_SECONDS = 20.0
TTS_CI_SETTLE_SECONDS = 5.0
TTS_CI_TEXT_TOKENS = 1024
TTS_CI_CHARS_PER_TOKEN = 3.46
TTS_CI_TIMEOUT_SECONDS = 300.0
TTS_RAW_RESULTS_FILENAME = "tts_load_results.json"


_LATENCY_SPECS = (
    ("FC P50", "fc_p50_ms", "fc_p50_ms"),
    ("FC P95", "fc_p95_ms", "fc_p95_ms"),
    ("SC P50", "sc_p50_ms", "sc_p50_ms"),
    ("SC P95", "sc_p95_ms", "sc_p95_ms"),
    ("TC P50", "tc_p50_ms", "tc_p50_ms"),
    ("TC P95", "tc_p95_ms", "tc_p95_ms"),
)


def _finite(value):
    """Return a JSON-safe finite float, or ``None`` for a missing sample."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _report_rows(rows: list[dict]) -> list[dict]:
    """Rename harness internals to the public FC/SC/TC report vocabulary."""
    report_rows = []
    for row in rows:
        report_rows.append(
            {
                "concurrency": row["conc"],
                "average_concurrency": _finite(row["C"]),
                "num_successful": row["n_ok"],
                "error_request_count": row["n_err"],
                "throughput_rps": _finite(row["rps"]),
                "million_chars_per_hour": _finite(row["mchar_h"]),
                "average_chunks": _finite(row["chunks"]),
                "fc_p50_ms": _finite(row["ttfb_p50"]),
                "fc_p95_ms": _finite(row["ttfb_p95"]),
                "sc_p50_ms": _finite(row["ttfs_p50"]),
                "sc_p95_ms": _finite(row["ttfs_p95"]),
                "tc_p50_ms": _finite(row["ttft_p50"]),
                "tc_p95_ms": _finite(row["ttft_p95"]),
            }
        )
    return report_rows


def _worst_latency_values(rows: list[dict]) -> dict[str, float | None]:
    """Worst measured percentile across the entire concurrency ladder.

    Grading the maximum makes the single CI verdict equivalent to requiring
    every measured concurrency point to satisfy every latency SLO.
    """
    return {
        field: max(
            (row[field] for row in rows if row.get(field) is not None),
            default=None,
        )
        for _, field, _ in _LATENCY_SPECS
    }


def _results_are_complete(rows: list[dict]) -> bool:
    """Reject partial sweeps and latency-only success in the presence of errors."""
    expected_concurrency = set(TTS_CI_CONCURRENCY)
    measured_concurrency = {row.get("concurrency") for row in rows}
    return measured_concurrency == expected_concurrency and all(
        row.get("num_successful", 0) > 0
        and row.get("error_request_count", 0) == 0
        and all(row.get(field) is not None for _, field, _ in _LATENCY_SPECS)
        for row in rows
    )


def _target_checks(ctx: MediaContext, rows: list[dict]):
    worst = _worst_latency_values(rows)
    target_checks, target_check = run_tiered_check(
        ctx,
        [
            MetricSpec(
                name,
                worst[field],
                target_attr,
                lower_is_better=True,
                field_name=field,
                inclusive=True,
            )
            for name, field, target_attr in _LATENCY_SPECS
        ],
    )
    integrity_check = (
        ReportCheckTypes.PASS if _results_are_complete(rows) else ReportCheckTypes.FAIL
    )
    for tier in target_checks.values():
        tier["benchmark_integrity"] = 1
        tier["benchmark_integrity_ratio"] = (
            1.0 if integrity_check == ReportCheckTypes.PASS else 0.0
        )
        tier["benchmark_integrity_check"] = integrity_check
    if integrity_check == ReportCheckTypes.FAIL:
        target_check = ReportCheckTypes.FAIL
    return target_checks, target_check


def _sweep_args(output_path: Path) -> argparse.Namespace:
    """Arguments consumed by :func:`tts_load_harness.sweep`."""
    return argparse.Namespace(
        concurrency=",".join(str(value) for value in TTS_CI_CONCURRENCY),
        duration=TTS_CI_DURATION_SECONDS,
        warmup=TTS_CI_WARMUP_SECONDS,
        settle=TTS_CI_SETTLE_SECONDS,
        arrival="closed",
        rate=None,
        max_inflight=0,
        on_full="shed",
        out=str(output_path / TTS_RAW_RESULTS_FILENAME),
    )


def _target(ctx: MediaContext) -> tts_load_harness.Target:
    parsed = urlparse(ctx.base_url)
    if not parsed.hostname:
        raise ValueError(
            f"TTS benchmark could not resolve a host from {ctx.base_url!r}"
        )
    nchars = round(TTS_CI_TEXT_TOKENS * TTS_CI_CHARS_PER_TOKEN)
    return tts_load_harness.Target(
        host=parsed.hostname,
        port=ctx.server_port,
        timeout=TTS_CI_TIMEOUT_SECONDS,
        text=tts_load_harness.build_text(nchars),
    )


def run_tts_load_benchmark(ctx: MediaContext) -> Block:
    """Run the chunk-aware FC/SC/TC concurrency sweep for a TTS model."""
    require_health(ctx)
    output_path = Path(ctx.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    document = tts_load_harness.sweep(_sweep_args(output_path), _target(ctx))
    rows, chars_per_request = tts_load_harness.aggregate(document)
    report_rows = _report_rows(rows)
    target_checks, target_check = _target_checks(ctx, report_rows)

    num_successful = sum(row["num_successful"] for row in report_rows)
    error_request_count = sum(row["error_request_count"] for row in report_rows)
    return Block(
        kind="benchmarks",
        task_type="text_to_speech",
        title="Text-to-Speech Load Benchmark",
        id=block_id(ctx) or None,
        targets={
            "concurrency": list(TTS_CI_CONCURRENCY),
            "text_tokens": TTS_CI_TEXT_TOKENS,
        },
        data={
            "Benchmarks": {
                "num_requests": num_successful + error_request_count,
                "num_successful": num_successful,
                "error_request_count": error_request_count,
                "chars_per_request": chars_per_request,
                "arrival_model": "closed",
                "raw_results": TTS_RAW_RESULTS_FILENAME,
                "target_check": target_check,
                "target_checks": target_checks,
            },
            "Latency by Concurrency": report_rows,
        },
    )


def _tts_num_calls(ctx: MediaContext, is_eval: bool = False) -> int:
    base = get_num_calls(ctx)
    if base != 2:
        logger.info(f"Using configured num_eval_runs: {base} calls")
        return base
    tts_default = 5 if is_eval else 10
    workflow_type = "eval" if is_eval else "benchmark"
    logger.info(
        f"Using TTS-specific {workflow_type} default: {tts_default} calls (was {base})"
    )
    return tts_default


def _tts_test_text(ctx: MediaContext) -> str:
    if (
        not isinstance(ctx.all_params, (list, tuple))
        and hasattr(ctx.all_params, "tasks")
        and len(ctx.all_params.tasks) > 0
    ):
        task = ctx.all_params.tasks[0]
        if hasattr(task, "text"):
            return task.text
        if hasattr(task, "task_name"):
            return task.task_name
    return DEFAULT_TTS_TEXT


async def _generate_speech(
    ctx: MediaContext,
) -> tuple[bool, float, Optional[float], Optional[float], Optional[float]]:
    logger.info("🔊 Calling TTS /v1/audio/speech endpoint")
    text = _tts_test_text(ctx)

    headers = {
        "accept": "application/json",
        "Authorization": "Bearer your-secret-key",
        "Content-Type": "application/json",
    }
    payload = {"text": text, "response_format": "json"}

    url = f"{ctx.base_url}/v1/audio/speech"
    start_time = time.monotonic()
    ttft_ms: Optional[float] = None
    audio_duration: Optional[float] = None

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(
                        f"TTS request failed with status {response.status}: {error_text}"
                    )
                    return False, 0.0, None, None, None

                content_type = response.headers.get("Content-Type", "").lower()
                logger.debug(f"Response Content-Type: {content_type}")
                if "audio" in content_type or "wav" in content_type:
                    logger.error(
                        f"Received audio/wav response instead of JSON. "
                        f"Make sure response_format='json' is set in request. "
                        f"Content-Type: {content_type}. Request payload was: {payload}"
                    )
                    return False, 0.0, None, None, None

                response_start = time.monotonic()
                response_data = await response.json()

                ttft_ms = (response_start - start_time) * 1000

                audio_duration = response_data.get("duration")
                if audio_duration is None:
                    logger.warning("Duration not found in response data")
                else:
                    logger.info(f"Audio duration: {audio_duration}s")

                audio_base64 = response_data.get("audio")
                if not audio_base64:
                    logger.error("Audio data not found in response")
                    return False, 0.0, None, None, None

                logger.info(f"Received audio data (base64 length: {len(audio_base64)})")

        total_duration = time.monotonic() - start_time

        rtr = None
        if audio_duration is not None and total_duration > 0:
            rtr = audio_duration / total_duration
            logger.info(
                f"Calculated RTR: {rtr:.2f} (audio_duration={audio_duration}s, "
                f"processing_time={total_duration:.2f}s)"
            )
        else:
            logger.warning(
                "Could not calculate RTR: missing duration or invalid processing time"
            )

        rtr_str = f"{rtr:.2f}" if rtr is not None else "N/A"
        logger.info(
            f"✅ Done in {total_duration:.2f}s | TTFT={ttft_ms:.2f}ms | RTR={rtr_str}"
        )
        return True, total_duration, ttft_ms, rtr, audio_duration

    except Exception as e:
        logger.error(f"TTS generation failed: {type(e).__name__}: {e}")
        return False, 0.0, None, None, None


def _run_tts_benchmark(ctx: MediaContext, num_calls: int) -> list[TtsTestStatus]:
    logger.info(f"Running TTS benchmark with {num_calls} calls.")
    status_list: list[TtsTestStatus] = []
    test_text = _tts_test_text(ctx)

    for i in range(num_calls):
        logger.info(f"Generating speech {i + 1}/{num_calls}...")
        status, elapsed, ttft_ms, rtr, audio_duration = asyncio.run(
            _generate_speech(ctx)
        )
        logger.debug(f"Generated speech in {elapsed:.2f} seconds.")
        status_list.append(
            TtsTestStatus(
                status=status,
                elapsed=elapsed,
                ttft_ms=ttft_ms,
                rtr=rtr,
                text=test_text,
                audio_duration=audio_duration,
            )
        )
    return status_list


def _tts_avg(status_list: list[TtsTestStatus], attr: str) -> Optional[float]:
    valid = [getattr(s, attr) for s in status_list if getattr(s, attr) is not None]
    return sum(valid) / len(valid) if valid else None


def _tts_target_checks(
    ctx: MediaContext, ttft_ms_value: Optional[float], rtr_value: Optional[float]
) -> tuple[dict, ReportCheckTypes]:
    logger.info("Computing 3-tier target checks for TTFT, RTR")
    return run_tiered_check(
        ctx,
        [
            MetricSpec(
                "TTFT",
                ttft_ms_value,
                "ttft_ms",
                lower_is_better=True,
                field_name="ttft",
            ),
            MetricSpec(
                "RTR", rtr_value, "rtr", lower_is_better=False, field_name="rtr"
            ),
        ],
    )


def _tts_ttft_percentiles(
    status_list: list[TtsTestStatus],
) -> tuple[float, float, float]:
    """Return (P50, P90, P95) of TTFT in ms across successful requests."""
    logger.info("Calculating TTFT percentiles (P50, P90, P95)")
    valid = [s.ttft_ms for s in status_list if s.ttft_ms is not None]
    if not valid:
        return 0.0, 0.0, 0.0
    sorted_ttft = sorted(valid)
    n = len(sorted_ttft)

    def _percentile(fraction: float) -> float:
        index = min(math.ceil(n * fraction) - 1, n - 1)
        return sorted_ttft[max(index, 0)]

    return _percentile(0.5), _percentile(0.9), _percentile(0.95)


def _tts_throughput_rps(
    status_list: list[TtsTestStatus], wall_seconds: float
) -> Optional[float]:
    """Requests-per-second over the benchmark wall-clock (informational).

    The TTS benchmark issues requests sequentially, so this reflects
    serial end-to-end throughput, not peak concurrent throughput. There is
    no throughput target for TTS, so this metric is display-only.
    """
    successful = sum(1 for s in status_list if s.status)
    if wall_seconds <= 0 or successful == 0:
        return None
    return successful / wall_seconds


def _run_legacy_tts_benchmark(ctx: MediaContext) -> Block:
    """Preserve the pre-load-harness behavior for SpeechT5 implementations."""
    logger.info(
        f"Running benchmarks for model: {ctx.model_spec.model_name} on device: {ctx.device.name}"
    )
    require_health(ctx)

    try:
        num_calls = _tts_num_calls(ctx, is_eval=False)
        bench_start = time.monotonic()
        status_list = _run_tts_benchmark(ctx, num_calls)
        wall_seconds = time.monotonic() - bench_start
    except Exception as e:
        logger.error(f"Benchmark execution encountered an error: {e}")
        raise

    logger.info("Generating benchmark report...")
    ttft_value = _tts_avg(status_list, "ttft_ms")
    rtr_value = _tts_avg(status_list, "rtr")
    p50_ttft, p90_ttft, p95_ttft = _tts_ttft_percentiles(status_list)
    throughput_rps = _tts_throughput_rps(status_list, wall_seconds)
    target_checks, target_check = _tts_target_checks(ctx, ttft_value, rtr_value)

    return Block(
        kind="benchmarks",
        task_type="text_to_speech",
        title="Text-to-Speech Benchmark",
        id=block_id(ctx) or None,
        targets={"num_prompts": len(status_list)},
        data={
            "Benchmarks": {
                "num_requests": len(status_list),
                "ttft": ttft_value / 1000 if ttft_value is not None else None,
                "rtr": rtr_value,
                # ttft_p50 and throughput_rps are informational only: TTS has
                # no P50/throughput targets, so they carry no target check.
                "ttft_p50": p50_ttft / 1000,
                "ttft_p90": p90_ttft / 1000,
                "ttft_p95": p95_ttft / 1000,
                "throughput_rps": throughput_rps,
                "target_check": target_check,
                "target_checks": target_checks,
            },
        },
    )


def run_tts_benchmark(ctx: MediaContext) -> Block:
    """Dispatch TTS benchmarks by implementation without changing legacy models."""
    impl_name = getattr(getattr(ctx.model_spec, "impl", None), "impl_name", None)
    if impl_name == TT_TTS_IMPL:
        return run_tts_load_benchmark(ctx)
    return _run_legacy_tts_benchmark(ctx)


__all__ = ["run_tts_benchmark"]
