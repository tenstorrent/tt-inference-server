# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC


from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import aiohttp
import requests

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from workflows.utils import (
    get_num_calls,
    is_preprocessing_enabled_for_whisper,
    is_streaming_enabled_for_whisper,
)
from workflows.utils_report import get_performance_targets
from workflows.workflow_types import ReportCheckTypes

from ..context import MediaContext, common_report_metadata, count_tokens, require_health
from ..test_status import AudioTestStatus

logger = logging.getLogger(__name__)


def _audio_avg(status_list: list[AudioTestStatus], attr: str) -> float:
    if not status_list:
        return 0.0
    valid = [getattr(s, attr) for s in status_list if getattr(s, attr) is not None]
    return sum(valid) / len(valid) if valid else 0.0


def _transcribe_audio_streaming_off(
    ctx: MediaContext, is_preprocessing_enabled: bool
) -> tuple[bool, float, Optional[float], Optional[float], Optional[float]]:
    logger.info("Transcribing audio without streaming")
    with open(f"{ctx.test_payloads_path}/image_client_audio_payload", "r") as f:
        audio_file = json.load(f)

    headers = {
        "accept": "application/json",
        "Authorization": "Bearer your-secret-key",
        "Content-Type": "application/json",
    }
    payload = {
        "file": audio_file["file"],
        "stream": False,
        "is_preprocessing_enabled": is_preprocessing_enabled,
    }

    start_time = time.time()
    response = requests.post(
        f"{ctx.base_url}/v1/audio/transcriptions",
        json=payload,
        headers=headers,
        timeout=90,
    )
    elapsed = time.time() - start_time
    ttft = elapsed
    tsu = None

    rtr = None
    if response.status_code == 200:
        try:
            response_data = response.json()
            audio_duration = response_data.get("duration")
            if audio_duration is not None:
                rtr = audio_duration / elapsed
                logger.info(
                    f"Calculated RTR: {rtr:.2f} (audio_duration={audio_duration}s, "
                    f"processing_time={elapsed:.2f}s)"
                )
            else:
                logger.warning("Duration not found in response data")
        except Exception as e:
            logger.error(f"Failed to calculate RTR: {e}")

    return (response.status_code == 200), elapsed, ttft, tsu, rtr


async def _transcribe_audio_streaming_on(
    ctx: MediaContext, is_preprocessing_enabled: bool
) -> tuple[bool, float, Optional[float], Optional[float], Optional[float]]:
    logger.info("Transcribing audio with streaming enabled")

    with open(
        f"{ctx.test_payloads_path}/image_client_audio_streaming_payload", "r"
    ) as f:
        audio_file = json.load(f)

    hf_model_repo = ctx.model_spec.hf_model_repo
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer your-secret-key",
        "Content-Type": "application/json",
    }
    payload = {
        "file": audio_file["file"],
        "stream": True,
        "is_preprocessing_enabled": is_preprocessing_enabled,
    }

    url = f"{ctx.base_url}/v1/audio/transcriptions"
    start_time = time.monotonic()
    ttft: Optional[float] = None
    total_text = ""
    total_tokens = 0
    chunk_texts: list[str] = []
    audio_duration: Optional[float] = None

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=90),
            ) as response:
                if response.status != 200:
                    return False, 0.0, None, None, None

                async for line in response.content:
                    if not line.strip():
                        continue
                    try:
                        line_str = line.decode("utf-8").strip()
                        if not line_str:
                            continue
                        result = json.loads(line_str)
                        logger.info(f"Received chunk: {result}")
                    except (UnicodeDecodeError, json.JSONDecodeError) as e:
                        logger.error(f"Failed to parse chunk: {e}")
                        continue

                    text = result.get("text", "")
                    chunk_id = result.get("chunk_id", "final")

                    if "duration" in result:
                        audio_duration = result.get("duration")
                        logger.info(f"Found audio duration in chunk: {audio_duration}s")

                    chunk_tokens = count_tokens(hf_model_repo, text)

                    if text.strip():
                        total_text += text + " "
                        chunk_texts.append(text)
                        total_tokens += chunk_tokens

                    is_speaker_marker = text.strip().startswith(
                        "[SPEAKER_"
                    ) and text.strip().endswith("]")
                    now = time.monotonic()
                    if ttft is None and chunk_tokens > 0 and not is_speaker_marker:
                        ttft = now - start_time
                        logger.info(
                            f"🎯 TTFT set at {ttft:.2f}s for first meaningful content: {text!r}"
                        )

                    elapsed = now - start_time
                    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
                    tokens_per_user_per_sec = tokens_per_sec / 1
                    logger.info(
                        f"[{elapsed:.2f}s] chunk={chunk_id} chunk_tokens={chunk_tokens} "
                        f"total_tokens={total_tokens} tps={tokens_per_sec:.2f} "
                        f"t/s/u={tokens_per_user_per_sec:.2f} text={text!r}"
                    )

        end_time = time.monotonic()
        total_duration = end_time - start_time
        content_streaming_time = total_duration - (ttft if ttft is not None else 0)
        final_tokens = total_tokens
        final_tps = (
            final_tokens / content_streaming_time if content_streaming_time > 0 else 0
        )
        final_tokens_per_user_per_sec = final_tps / 1

        rtr = None
        if audio_duration is not None:
            rtr = audio_duration / total_duration
            logger.info(
                f"Calculated RTR: {rtr:.2f} (audio_duration={audio_duration}s, "
                f"processing_time={total_duration:.2f}s)"
            )
        else:
            logger.warning("Audio duration not found in streaming response")

        final_ttft = ttft if ttft is not None else 0.0
        rtr_display = f"{rtr:.2f}" if rtr is not None else "N/A"
        logger.info(
            f"\n✅ Done in {total_duration:.2f}s | TTFT={final_ttft:.2f}s | "
            f"Total tokens={final_tokens} | TPS={final_tps:.2f} | "
            f"T/S/U={final_tokens_per_user_per_sec:.2f} | RTR={rtr_display}"
        )

        return True, total_duration, final_ttft, final_tokens_per_user_per_sec, rtr

    except Exception as e:
        logger.error(f"Streaming transcription failed: {e}")
        return False, 0.0, None, None, None


async def _transcribe_audio(
    ctx: MediaContext,
) -> tuple[bool, float, Optional[float], Optional[float], Optional[float]]:
    logger.info("🔈 Calling whisper")
    is_preprocessing_enabled = is_preprocessing_enabled_for_whisper(ctx)
    logging.info(f"Preprocessing enabled: {is_preprocessing_enabled}")

    if is_streaming_enabled_for_whisper(ctx):
        return await _transcribe_audio_streaming_on(ctx, is_preprocessing_enabled)

    return _transcribe_audio_streaming_off(ctx, is_preprocessing_enabled)


def _run_audio_transcription_benchmark(
    ctx: MediaContext, num_calls: int
) -> list[AudioTestStatus]:
    logger.info(f"Running audio transcription benchmark with {num_calls} calls.")
    status_list: list[AudioTestStatus] = []
    for i in range(num_calls):
        logger.info(f"Transcribing audio {i + 1}/{num_calls}...")
        status, elapsed, ttft, tsu, rtr = asyncio.run(_transcribe_audio(ctx))
        logger.info(f"Transcribed audio in {elapsed:.2f} seconds.")
        status_list.append(
            AudioTestStatus(status=status, elapsed=elapsed, ttft=ttft, tsu=tsu, rtr=rtr)
        )
    return status_list


def _audio_accuracy_check(
    ctx: MediaContext,
    ttft_value: float,
    tsu_value: float,
    rtr_value: float,
) -> ReportCheckTypes:
    logger.info("Calculating accuracy check based on TTFT, RTR, T/S/U targets")

    device_str = ctx.model_spec.cli_args.get("device")
    targets = get_performance_targets(
        ctx.model_spec.model_name,
        device_str,
        model_type=ctx.model_spec.model_type.name,
    )
    logger.info(f"Performance targets: {targets}")

    if not targets.ttft_ms:
        logger.warning("⚠️ No TTFT target found, skipping accuracy check")
        return ReportCheckTypes.NA

    available_metrics = [
        "TTFT" if targets.ttft_ms else None,
        "T/S/U" if targets.tput_user and tsu_value else None,
        "RTR" if targets.rtr and rtr_value else None,
    ]
    logger.info(
        f"Available metrics for validation: {[m for m in available_metrics if m]}"
    )

    tolerance = targets.tolerance if targets.tolerance else 0.05
    logger.info(f"Using tolerance: {tolerance * 100:.2f}%")

    checks_passed = 0
    checks_total = 0

    if targets.ttft_ms is not None:  # pragma: no branch
        checks_total += 1
        ttft_threshold = targets.ttft_ms * (1 + tolerance)
        if ttft_value <= ttft_threshold:
            logger.info(f"✅ TTFT PASSED: {ttft_value:.2f}ms <= {ttft_threshold:.2f}ms")
            checks_passed += 1
        else:
            logger.warning(
                f"❌ TTFT FAILED: {ttft_value:.2f}ms > {ttft_threshold:.2f}ms"
            )

    if targets.tput_user is not None and tsu_value is not None:
        checks_total += 1
        tsu_threshold = targets.tput_user * (1 - tolerance)
        if tsu_value >= tsu_threshold:
            logger.info(f"✅ T/S/U PASSED: {tsu_value:.2f} >= {tsu_threshold:.2f}")
            checks_passed += 1
        else:
            logger.warning(f"❌ T/S/U FAILED: {tsu_value:.2f} < {tsu_threshold:.2f}")

    if targets.rtr is not None and rtr_value is not None:
        checks_total += 1
        rtr_threshold = targets.rtr * (1 - tolerance)
        if rtr_value >= rtr_threshold:
            logger.info(f"✅ RTR PASSED: {rtr_value:.2f} >= {rtr_threshold:.2f}")
            checks_passed += 1
        else:
            logger.warning(f"❌ RTR FAILED: {rtr_value:.2f} < {rtr_threshold:.2f}")

    if checks_total == 0:  # pragma: no cover
        logger.warning("No targets available for accuracy check")
        return ReportCheckTypes.NA
    if checks_passed == checks_total:
        logger.info(f"🎉 ALL CHECKS PASSED ({checks_passed}/{checks_total})")
        return ReportCheckTypes.PASS

    logger.warning(f"⛔️ SOME CHECKS FAILED ({checks_passed}/{checks_total} passed)")
    return ReportCheckTypes.FAIL


def run_audio_benchmark(ctx: MediaContext) -> dict:
    """Run benchmarks for an audio model (Whisper, etc.)."""
    logger.info(
        f"Running benchmarks for model: {ctx.model_spec.model_name} on device: {ctx.device.name}"
    )
    require_health(ctx)

    try:
        num_calls = get_num_calls(ctx)
        status_list = _run_audio_transcription_benchmark(ctx, num_calls)
    except Exception as e:
        logger.error(f"Benchmark execution encountered an error: {e}")
        raise

    logger.info("Generating benchmark report...")
    ttft_value = _audio_avg(status_list, "ttft")
    rtr_value = _audio_avg(status_list, "rtr")
    tsu_value = _audio_avg(status_list, "tsu")
    accuracy_check = _audio_accuracy_check(ctx, ttft_value, tsu_value, rtr_value)

    report_data = common_report_metadata(ctx, "audio")
    report_data["benchmarks"] = {
        "num_requests": len(status_list),
        "num_inference_steps": 0,
        "ttft": ttft_value,
        "inference_steps_per_second": 0,
        "t/s/u": tsu_value,
        "rtr": rtr_value,
        "accuracy_check": accuracy_check,
    }
    report_data["streaming_enabled"] = is_streaming_enabled_for_whisper(ctx)
    report_data["preprocessing_enabled"] = is_preprocessing_enabled_for_whisper(ctx)

    return report_data


__all__ = ["run_audio_benchmark"]
