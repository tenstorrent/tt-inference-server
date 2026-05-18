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

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from workflows.utils import (
    get_num_calls,
    is_preprocessing_enabled_for_whisper,
    is_streaming_enabled_for_whisper,
)
from report_module.schema import Block

from .._test_common import ReportCheckTypes, block_id, block_targets
from ..context import MediaContext, count_tokens, require_health
from ..test_status import AudioTestStatus

logger = logging.getLogger(__name__)


def _audio_ttft(status_list: list[AudioTestStatus]) -> float:
    valid = [s.ttft for s in status_list if s.ttft is not None]
    return sum(valid) / len(valid) if valid else 0


def _audio_rtr(status_list: list[AudioTestStatus]) -> float:
    valid = [s.rtr for s in status_list if s.rtr is not None]
    return sum(valid) / len(valid) if valid else 0


def _audio_tsu(status_list: list[AudioTestStatus]) -> float:
    valid = [s.tsu for s in status_list if s.tsu is not None]
    return sum(valid) / len(valid) if valid else 0


async def _transcribe_audio_streaming_off(
    ctx: MediaContext, is_preprocessing_enabled: bool
) -> tuple[bool, float, Optional[float], Optional[float], Optional[float]]:
    logger.info("Transcribing audio without streaming")
    with open(Path(ctx.test_payloads_path) / "image_client_audio_payload", "r") as f:
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
    status_ok = False
    response_data = None
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{ctx.base_url}/v1/audio/transcriptions",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=90),
        ) as response:
            status_ok = response.status == 200
            if status_ok:
                response_data = await response.json()
    elapsed = time.time() - start_time
    ttft = elapsed
    tsu = None

    rtr = None
    if status_ok and response_data is not None:
        try:
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

    return status_ok, elapsed, ttft, tsu, rtr


async def _transcribe_audio_streaming_on(
    ctx: MediaContext, is_preprocessing_enabled: bool
) -> tuple[bool, float, Optional[float], Optional[float], Optional[float]]:
    """Streaming transcription. Measures TTFT excluding speaker markers."""
    logger.info("Transcribing audio with streaming enabled")

    with open(
        Path(ctx.test_payloads_path) / "image_client_audio_streaming_payload", "r"
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
                    logger.info(
                        f"[{elapsed:.2f}s] chunk={chunk_id} chunk_tokens={chunk_tokens} "
                        f"total_tokens={total_tokens} tps={tokens_per_sec:.2f} "
                        f"t/s/u={tokens_per_sec:.2f} text={text!r}"
                    )

        end_time = time.monotonic()
        total_duration = end_time - start_time
        content_streaming_time = total_duration - (ttft if ttft is not None else 0)
        final_tokens = total_tokens
        final_tps = (
            final_tokens / content_streaming_time if content_streaming_time > 0 else 0
        )
        final_tokens_per_user_per_sec = final_tps

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
    logger.info(f"Preprocessing enabled: {is_preprocessing_enabled}")

    if is_streaming_enabled_for_whisper(ctx):
        return await _transcribe_audio_streaming_on(ctx, is_preprocessing_enabled)

    return await _transcribe_audio_streaming_off(ctx, is_preprocessing_enabled)


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


def _is_whisper(ctx: MediaContext) -> bool:
    impl = getattr(ctx.model_spec, "impl", None)
    return getattr(impl, "impl_name", None) == "whisper"


def _run_whisper_lmms_eval(ctx: MediaContext) -> Block:
    """Run the real lmms-eval librispeech_test_other WER eval via WhisperEvalTest."""
    from .._test_common import TestConfig
    from .whisper_eval_test import WhisperEvalTest

    task = ctx.all_params.tasks[0]
    timeout_s = 1800  # full LibriSpeech run on n150 is ~10 min; pad for safety

    test = WhisperEvalTest(
        config=TestConfig.create_default(timeout=timeout_s),
        targets={},
    )
    test.hf_model_repo = ctx.model_spec.hf_model_repo
    test.base_url = ctx.base_url
    test.task_name = task.task_name
    test.test_limit = None
    test.debug_mode = False
    test.mock_mode = False

    logger.info(
        "Invoking WhisperEvalTest (lmms-eval): task=%s base_url=%s hf_model_repo=%s",
        test.task_name,
        test.base_url,
        test.hf_model_repo,
    )

    results = asyncio.run(
        asyncio.wait_for(test._run_specific_test_async(), timeout=timeout_s)
    )

    wer = None
    metrics = results.get("metrics") or {}
    for key in (f"{task.task_name}_wer", "wer"):
        val = metrics.get(key)
        if isinstance(val, (int, float)):
            wer = float(val)
            break

    if wer is None:
        detailed = results.get("detailed_results") or {}
        task_block = detailed.get(task.task_name) or {}
        raw = task_block.get("wer,none")
        if isinstance(raw, (int, float)):
            wer = float(raw)

    if wer is None:
        logger.error(
            "Could not parse WER from lmms-eval results; emitting FAIL block. "
            "results keys=%s",
            list(results.keys()),
        )
        return Block(
            kind="evals",
            task_type="audio",
            title="Audio Eval",
            id=block_id(ctx) or None,
            targets={
                "task_name": task.task_name,
                "tolerance": task.score.tolerance,
                "published_score": task.score.published_score,
                "published_score_ref": task.score.published_score_ref,
            },
            data={
                "task_name": task.task_name,
                "tolerance": task.score.tolerance,
                "published_score": task.score.published_score,
                "score": None,
                "published_score_ref": task.score.published_score_ref,
                "accuracy_check": ReportCheckTypes.FAIL,
                "wer": None,
                "error": "WER not found in lmms-eval results",
            },
        )

    score = 100.0 - wer
    published = task.score.published_score
    reference = getattr(task.score, "gpu_reference_score", None) or published
    tolerance = task.score.tolerance if task.score.tolerance is not None else 0.05
    if reference:
        ratio = score / reference
        accuracy_check = (
            ReportCheckTypes.PASS
            if ratio >= (1.0 - tolerance)
            else ReportCheckTypes.FAIL
        )
    else:
        accuracy_check = ReportCheckTypes.NA

    logger.info(
        "WhisperEvalTest WER=%.4f score=%.4f published=%s tolerance=%s check=%s",
        wer,
        score,
        published,
        tolerance,
        accuracy_check,
    )

    return Block(
        kind="evals",
        task_type="audio",
        title="Audio Eval",
        id=block_id(ctx) or None,
        targets={
            "task_name": task.task_name,
            "tolerance": tolerance,
            "published_score": published,
            "published_score_ref": task.score.published_score_ref,
        },
        data={
            "task_name": task.task_name,
            "tolerance": tolerance,
            "published_score": published,
            "score": score,
            "wer": wer,
            "published_score_ref": task.score.published_score_ref,
            "accuracy_check": accuracy_check,
        },
    )


def run_audio_eval(ctx: MediaContext) -> Block:
    """Run evaluations for an audio model (Whisper, etc.)."""
    logger.info(
        f"Running evals for model: {ctx.model_spec.model_name} on device: {ctx.device.name}"
    )
    require_health(ctx)

    if _is_whisper(ctx):
        return _run_whisper_lmms_eval(ctx)

    try:
        num_calls = get_num_calls(ctx)
        status_list = _run_audio_transcription_benchmark(ctx, num_calls)
    except Exception as e:
        logger.error(f"Eval execution encountered an error: {e}")
        raise

    logger.info("Generating eval report...")
    ttft_value = _audio_ttft(status_list)
    rtr_value = _audio_rtr(status_list)
    tsu_value = _audio_tsu(status_list)
    logger.info(f"Extracted TTFT value: {ttft_value}")
    logger.info(f"Extracted RTR value: {rtr_value}")
    logger.info(f"Extracted T/S/U value: {tsu_value}")

    task = ctx.all_params.tasks[0]
    return Block(
        kind="audio_eval",
        id=block_id(ctx) or None,
        targets=block_targets(ctx, task_type="audio"),
        data={
            "task_name": task.task_name,
            "tolerance": task.score.tolerance,
            "published_score": task.score.published_score,
            "score": ttft_value,
            "published_score_ref": task.score.published_score_ref,
            # TODO: replace hardcoded PASS with a real accuracy evaluation.
            "accuracy_check": ReportCheckTypes.PASS,
            "t/s/u": tsu_value,
            "rtr": rtr_value,
        },
    )


__all__ = ["run_audio_eval"]
