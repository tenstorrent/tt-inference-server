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
from workflows.utils_report import get_performance_targets
from workflows.workflow_types import ReportCheckTypes

from ..context import MediaContext, common_eval_metadata, require_health
from ..test_status import TtsTestStatus

logger = logging.getLogger(__name__)


DEFAULT_TTS_TEXT = "Hello, this is a test of the text to speech system."


def _tts_ttft(status_list: list[TtsTestStatus]) -> float:
    valid = [s.ttft_ms for s in status_list if s.ttft_ms is not None]
    return sum(valid) / len(valid) if valid else 0


def _tts_rtr(status_list: list[TtsTestStatus]) -> float:
    valid = [s.rtr for s in status_list if s.rtr is not None]
    return sum(valid) / len(valid) if valid else 0


def _tts_tail_latency(status_list: list[TtsTestStatus]) -> tuple[float, float]:
    valid = [s.ttft_ms for s in status_list if s.ttft_ms is not None]
    if not valid:
        return 0.0, 0.0
    sorted_ttft = sorted(valid)
    n = len(sorted_ttft)
    p90_index = min(math.ceil(n * 0.9) - 1, n - 1)
    p95_index = min(math.ceil(n * 0.95) - 1, n - 1)
    return sorted_ttft[p90_index], sorted_ttft[p95_index]


def _tts_num_calls(ctx: MediaContext, is_eval: bool) -> int:
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


def _tts_performance_check(
    ctx: MediaContext,
    ttft_value: Optional[float],
    rtr_value: Optional[float],
) -> ReportCheckTypes:
    logger.info("Calculating performance check based on TTFT, RTR targets")

    device_str = ctx.model_spec.cli_args.get("device")
    targets = get_performance_targets(
        ctx.model_spec.model_name,
        device_str,
        model_type=ctx.model_spec.model_type.name,
    )
    logger.info(f"Performance targets: {targets}")

    if not targets.ttft_ms:
        logger.warning("⚠️ No TTFT target found, skipping performance check")
        return ReportCheckTypes.NA

    tolerance = targets.tolerance if targets.tolerance else 0.05
    logger.info(f"Using tolerance: {tolerance * 100:.2f}%")

    checks_passed = 0
    checks_total = 0

    if targets.ttft_ms is not None:
        checks_total += 1
        threshold = targets.ttft_ms * (1 + tolerance)
        if ttft_value <= threshold:
            logger.info(f"✅ TTFT PASSED: {ttft_value:.2f}ms <= {threshold:.2f}ms")
            checks_passed += 1
        else:
            logger.warning(f"❌ TTFT FAILED: {ttft_value:.2f}ms > {threshold:.2f}ms")

    if targets.rtr is not None:
        checks_total += 1
        threshold = targets.rtr * (1 - tolerance)
        if rtr_value >= threshold:
            logger.info(f"✅ RTR PASSED: {rtr_value:.2f} >= {threshold:.2f}")
            checks_passed += 1
        else:
            logger.warning(f"❌ RTR FAILED: {rtr_value:.2f} < {threshold:.2f}")

    if checks_total == 0:
        logger.warning("⚠️ No metrics available for validation")
        return ReportCheckTypes.NA
    if checks_passed == checks_total:
        logger.info(f"✅ All {checks_total} performance checks passed")
        return ReportCheckTypes.PASS

    logger.warning(
        f"❌ {checks_total - checks_passed}/{checks_total} performance checks failed"
    )
    return ReportCheckTypes.FAIL


def run_tts_eval(ctx: MediaContext) -> dict:
    """Run evaluations for a TTS model (SpeechT5, etc.)."""
    logger.info(
        f"Running evals for model: {ctx.model_spec.model_name} on device: {ctx.device.name}"
    )
    require_health(ctx)

    try:
        num_calls = _tts_num_calls(ctx, is_eval=True)
        status_list = _run_tts_benchmark(ctx, num_calls)
    except Exception as e:
        logger.error(f"Eval execution encountered an error: {e}")
        raise

    logger.info("Generating eval report...")
    ttft_value = _tts_ttft(status_list)
    rtr_value = _tts_rtr(status_list)
    p90_ttft, p95_ttft = _tts_tail_latency(status_list)
    logger.info(f"Extracted TTFT value: {ttft_value:.2f}ms")
    logger.info(f"Extracted RTR value: {rtr_value:.2f}")
    logger.info(f"Extracted P90 TTFT: {p90_ttft:.2f}ms, P95 TTFT: {p95_ttft:.2f}ms")

    benchmark_data = common_eval_metadata(ctx, "text_to_speech")
    benchmark_data["device"] = ctx.device.name
    benchmark_data["published_score"] = ctx.all_params.tasks[0].score.published_score
    benchmark_data["score"] = ttft_value
    benchmark_data["published_score_ref"] = ctx.all_params.tasks[
        0
    ].score.published_score_ref
    benchmark_data["rtr"] = rtr_value
    benchmark_data["p90_ttft"] = p90_ttft
    benchmark_data["p95_ttft"] = p95_ttft
    benchmark_data["performance_check"] = _tts_performance_check(
        ctx, ttft_value, rtr_value
    )
    # No quality metric implemented yet for TTS; always reports N/A.
    benchmark_data["accuracy_check"] = ReportCheckTypes.NA

    return benchmark_data


__all__ = ["run_tts_eval"]
