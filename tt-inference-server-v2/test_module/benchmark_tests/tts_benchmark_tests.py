# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC


from __future__ import annotations

import asyncio
import logging
import math
import sys
import time
from pathlib import Path
from typing import Optional

import aiohttp

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from workflows.utils import get_num_calls

from ..context import MediaContext, common_report_metadata, require_health
from ..test_status import TtsTestStatus

logger = logging.getLogger(__name__)


DEFAULT_TTS_TEXT = "Hello, this is a test of the text to speech system."


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


def _tts_avg(status_list: list[TtsTestStatus], attr: str) -> float:
    if not status_list:
        return 0.0
    valid = [getattr(s, attr) for s in status_list if getattr(s, attr) is not None]
    return sum(valid) / len(valid) if valid else 0.0


def _tts_tail_latency(status_list: list[TtsTestStatus]) -> tuple[float, float]:
    logger.info("Calculating tail latency (P90, P95)")
    if not status_list:
        return 0.0, 0.0
    valid = [s.ttft_ms for s in status_list if s.ttft_ms is not None]
    if not valid:
        return 0.0, 0.0
    sorted_ttft = sorted(valid)
    n = len(sorted_ttft)
    p90_index = min(math.ceil(n * 0.9) - 1, n - 1)
    p95_index = min(math.ceil(n * 0.95) - 1, n - 1)
    return sorted_ttft[p90_index], sorted_ttft[p95_index]


def run_tts_benchmark(ctx: MediaContext) -> dict:
    """Run benchmarks for a TTS model (SpeechT5, etc.)."""
    logger.info(
        f"Running benchmarks for model: {ctx.model_spec.model_name} on device: {ctx.device.name}"
    )
    require_health(ctx)

    try:
        num_calls = _tts_num_calls(ctx, is_eval=False)
        status_list = _run_tts_benchmark(ctx, num_calls)
    except Exception as e:
        logger.error(f"Benchmark execution encountered an error: {e}")
        raise

    logger.info("Generating benchmark report...")
    ttft_value = _tts_avg(status_list, "ttft_ms")
    rtr_value = _tts_avg(status_list, "rtr")
    p90_ttft, p95_ttft = _tts_tail_latency(status_list)

    report_data = common_report_metadata(ctx, "tts")
    report_data["benchmarks"] = {
        "num_requests": len(status_list),
        "ttft": ttft_value / 1000,
        "rtr": rtr_value,
        "ttft_p90": p90_ttft / 1000,
        "ttft_p95": p95_ttft / 1000,
    }

    return report_data


__all__ = ["run_tts_benchmark"]
