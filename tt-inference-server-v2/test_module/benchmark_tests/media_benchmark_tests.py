# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC


from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

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
from workflows.workflow_types import ReportCheckTypes, WorkflowVenvType
from workflows.workflow_venvs import VENV_CONFIGS

from ..context import (
    MediaContext,
    common_report_metadata,
    count_tokens,
    require_health,
)
from ..test_status import (
    AudioTestStatus,
    CnnGenerationTestStatus,
    ImageGenerationTestStatus,
    TtsTestStatus,
    VideoGenerationTestStatus,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Image
# =============================================================================

SDXL_SD35_BENCHMARK_NUM_PROMPTS = 20
SDXL_BENCHMARK_NUM_PROMPTS = 100
SDXL_SD35_INFERENCE_STEPS = 20
SDXL_INPAINTING_INFERENCE_STEPS = 20
FLUX_MOTIF_INFERENCE_STEPS = 28
FLUX_1_SCHNELL_INFERENCE_STEPS = 4

SDXL_IMG2IMG_INFERENCE_STEPS = 30
GUIDANCE_SCALE_IMG2IMG = 7.5
SEED_IMG2IMG = 0
STRENGTH_IMG2IMG = 0.6

GUIDANCE_SCALE_INPAINTING = 8.0
SEED_INPAINTING = 0
STRENGTH_INPAINTING = 0.99


def _generate_image(
    ctx: MediaContext, num_inference_steps: int = 20
) -> tuple[bool, float]:
    logger.info("🌅 Generating image")
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer your-secret-key",
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": "Rabbit",
        "seed": 0,
        "guidance_scale": 3.0,
        "number_of_images": 1,
        "num_inference_steps": num_inference_steps,
    }
    start_time = time.time()
    response = requests.post(
        f"{ctx.base_url}/v1/images/generations",
        json=payload,
        headers=headers,
        timeout=90,
    )
    elapsed = time.time() - start_time

    if response.status_code != 200:
        logger.error(f"❌ Image generation failed with status {response.status_code}")
        try:
            logger.error(f"Error details: {response.json()}")
        except Exception as e:
            logger.error(f"Could not parse error response: {e}")
            logger.error(f"Raw response: {response.text[:500]}")
        raise RuntimeError(
            f"Image generation failed with status {response.status_code}"
        )

    logger.info(f"✅ Image generation successful in {elapsed:.2f}s")
    return True, elapsed


def _generate_image_img2img(
    ctx: MediaContext, num_inference_steps: int = SDXL_IMG2IMG_INFERENCE_STEPS
) -> tuple[bool, float]:
    logger.info("🌆 Generating image with img2img")
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer your-secret-key",
        "Content-Type": "application/json",
    }
    with open(f"{ctx.test_payloads_path}/image_client_img2img_payload", "r") as f:
        image_data = json.load(f)

    payload = {
        "prompt": "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k",
        "image": image_data["file"],
        "seed": SEED_IMG2IMG,
        "guidance_scale": GUIDANCE_SCALE_IMG2IMG,
        "number_of_images": 1,
        "strength": STRENGTH_IMG2IMG,
        "num_inference_steps": num_inference_steps,
    }
    start_time = time.time()
    response = requests.post(
        f"{ctx.base_url}/v1/images/image-to-image",
        json=payload,
        headers=headers,
        timeout=90,
    )
    elapsed = time.time() - start_time

    if response.status_code != 200:
        logger.error(
            f"❌ Image-to-image generation failed with status {response.status_code}"
        )
        try:
            logger.error(f"Error details: {response.json()}")
        except Exception as e:
            logger.error(f"Could not parse error response: {e}")
            logger.error(f"Raw response: {response.text[:500]}")
        raise RuntimeError(
            f"Image-to-image generation failed with status {response.status_code}"
        )

    logger.info(f"✅ Image-to-image generation successful in {elapsed:.2f}s")
    return True, elapsed


def _generate_image_inpainting(
    ctx: MediaContext, num_inference_steps: int = SDXL_INPAINTING_INFERENCE_STEPS
) -> tuple[bool, float]:
    logger.info("🏞️ Generating image with inpainting")
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer your-secret-key",
        "Content-Type": "application/json",
    }
    with open(f"{ctx.test_payloads_path}/image_client_inpainting_payload", "r") as f:
        payload_data = json.load(f)
        inpaint_image = payload_data["inpaint_image"]
        inpaint_mask = payload_data["inpaint_mask"]

    payload = {
        "prompt": "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k",
        "image": inpaint_image,
        "mask": inpaint_mask,
        "seed": SEED_INPAINTING,
        "guidance_scale": GUIDANCE_SCALE_INPAINTING,
        "number_of_images": 1,
        "strength": STRENGTH_INPAINTING,
        "num_inference_steps": num_inference_steps,
    }
    start_time = time.time()
    response = requests.post(
        f"{ctx.base_url}/v1/images/edits",
        json=payload,
        headers=headers,
        timeout=90,
    )
    elapsed = time.time() - start_time

    if response.status_code != 200:
        logger.error(
            f"❌ Inpainting generation failed with status {response.status_code}"
        )
        try:
            logger.error(f"Error details: {response.json()}")
        except Exception as e:
            logger.error(f"Could not parse error response: {e}")
            logger.error(f"Raw response: {response.text[:500]}")
        raise RuntimeError(
            f"Inpainting generation failed with status {response.status_code}"
        )

    logger.info(f"✅ Inpainting generation successful in {elapsed:.2f}s")
    return True, elapsed


def _build_image_status_list(
    ctx: MediaContext,
    num_calls: int,
    inference_steps: int,
    generator: Callable[..., tuple[bool, float]],
    generator_steps_kwarg: bool = False,
) -> list[ImageGenerationTestStatus]:
    status_list: list[ImageGenerationTestStatus] = []
    for i in range(num_calls):
        logger.info(f"Generating image {i + 1}/{num_calls}...")
        if generator_steps_kwarg:
            status, elapsed = generator(ctx, inference_steps)
        else:
            status, elapsed = generator(ctx)
        inference_steps_per_second = inference_steps / elapsed if elapsed > 0 else 0
        logger.info(
            f"Generated image with {inference_steps} steps in {elapsed:.2f} seconds."
        )
        status_list.append(
            ImageGenerationTestStatus(
                status=status,
                elapsed=elapsed,
                num_inference_steps=inference_steps,
                inference_steps_per_second=inference_steps_per_second,
            )
        )
    return status_list


def _run_sdxl_image_generation_benchmark(
    ctx: MediaContext, num_calls: int
) -> list[ImageGenerationTestStatus]:
    logger.info("Running image generation benchmark.")
    return _build_image_status_list(
        ctx, num_calls, SDXL_SD35_INFERENCE_STEPS, _generate_image
    )


def _run_img2img_generation_benchmark(
    ctx: MediaContext, num_calls: int
) -> list[ImageGenerationTestStatus]:
    logger.info("Running image-to-image generation benchmark.")
    return _build_image_status_list(
        ctx, num_calls, SDXL_IMG2IMG_INFERENCE_STEPS, _generate_image_img2img
    )


def _run_inpainting_generation_benchmark(
    ctx: MediaContext, num_calls: int
) -> list[ImageGenerationTestStatus]:
    logger.info("Running inpainting generation benchmark.")
    return _build_image_status_list(
        ctx, num_calls, SDXL_INPAINTING_INFERENCE_STEPS, _generate_image_inpainting
    )


def _run_flux_1_dev_benchmark(
    ctx: MediaContext, num_calls: int
) -> list[ImageGenerationTestStatus]:
    logger.info("Running Flux 1 Dev or Schnell benchmark.")
    return _build_image_status_list(
        ctx,
        num_calls,
        FLUX_MOTIF_INFERENCE_STEPS,
        _generate_image,
        generator_steps_kwarg=True,
    )


def _run_flux_1_schnell_benchmark(
    ctx: MediaContext, num_calls: int
) -> list[ImageGenerationTestStatus]:
    logger.info("Running Flux 1 Schnell benchmark.")
    return _build_image_status_list(
        ctx,
        num_calls,
        FLUX_1_SCHNELL_INFERENCE_STEPS,
        _generate_image,
        generator_steps_kwarg=True,
    )


def _run_motif_image_6b_preview_benchmark(
    ctx: MediaContext, num_calls: int
) -> list[ImageGenerationTestStatus]:
    logger.info("Running Motif Image 6B Preview benchmark.")
    return _build_image_status_list(
        ctx,
        num_calls,
        FLUX_MOTIF_INFERENCE_STEPS,
        _generate_image,
        generator_steps_kwarg=True,
    )


IMAGE_BENCHMARK_DISPATCH: dict[
    str, Callable[[MediaContext, int], list[ImageGenerationTestStatus]]
] = {
    "tt-sdxl-trace": _run_sdxl_image_generation_benchmark,
    "tt-sdxl-image-to-image": _run_img2img_generation_benchmark,
    "tt-sdxl-edit": _run_inpainting_generation_benchmark,
    "tt-sd3.5": _run_sdxl_image_generation_benchmark,
    "tt-flux.1-dev": _run_flux_1_dev_benchmark,
    "tt-flux.1-schnell": _run_flux_1_schnell_benchmark,
    "tt-motif-image-6b-preview": _run_motif_image_6b_preview_benchmark,
}


def _image_ttft(status_list: list[ImageGenerationTestStatus]) -> float:
    logger.info("Calculating TTFT value")
    return sum(s.elapsed for s in status_list) / len(status_list) if status_list else 0


def run_image_benchmark(ctx: MediaContext) -> dict:
    """Run benchmarks for an image model (SDXL, SD3.5, Flux, Motif)."""
    logger.info(
        f"Running benchmarks for model: {ctx.model_spec.model_name} on device: {ctx.device.name}"
    )

    runner_in_use = require_health(ctx)

    try:
        num_calls = get_num_calls(ctx)
        if runner_in_use == "tt-sdxl-trace":
            logger.info(
                f"Overriding num_calls for SDXL trace model to {SDXL_BENCHMARK_NUM_PROMPTS} prompts"
            )
            num_calls = SDXL_BENCHMARK_NUM_PROMPTS

        benchmark_fn = IMAGE_BENCHMARK_DISPATCH.get(
            runner_in_use, _run_sdxl_image_generation_benchmark
        )
        status_list = benchmark_fn(ctx, num_calls)
    except Exception as e:
        logger.error(f"Benchmark execution encountered an error: {e}")
        raise

    logger.info("Generating benchmark report...")
    ttft_value = _image_ttft(status_list)
    report_data = common_report_metadata(ctx, "image")
    report_data["benchmarks"] = {
        "num_requests": len(status_list),
        "num_inference_steps": status_list[0].num_inference_steps if status_list else 0,
        "ttft": ttft_value,
        "inference_steps_per_second": (
            sum(s.inference_steps_per_second for s in status_list) / len(status_list)
            if status_list
            else 0
        ),
    }

    return report_data


# =============================================================================
# Audio
# =============================================================================


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


# =============================================================================
# CNN
# =============================================================================


def _analyze_image(ctx: MediaContext) -> tuple[bool, float]:
    logger.info("🔍 Analyzing image")
    with open(f"{ctx.test_payloads_path}/image_client_image_payload", "r") as f:
        image_payload = f.read()

    headers = {
        "accept": "application/json",
        "Authorization": "Bearer your-secret-key",
        "Content-Type": "application/json",
    }
    payload = {"prompt": image_payload}
    start_time = time.time()
    response = requests.post(
        f"{ctx.base_url}/v1/cnn/search-image",
        json=payload,
        headers=headers,
        timeout=90,
    )
    elapsed = time.time() - start_time
    return (response.status_code == 200), elapsed


def _run_image_analysis_benchmark(
    ctx: MediaContext, num_calls: int
) -> list[CnnGenerationTestStatus]:
    logger.info("Running image analysis benchmark.")
    status_list: list[CnnGenerationTestStatus] = []
    for i in range(num_calls):
        logger.info(f"Analyzing image {i + 1}/{num_calls}...")
        status, elapsed = _analyze_image(ctx)
        logger.info(f"Analyzed image with 50 steps in {elapsed:.2f} seconds.")
        status_list.append(CnnGenerationTestStatus(status=status, elapsed=elapsed))
    return status_list


def _cnn_ttft(status_list: list[CnnGenerationTestStatus]) -> float:
    logger.info("Calculating TTFT value")
    return sum(s.elapsed for s in status_list) / len(status_list) if status_list else 0


def run_cnn_benchmark(ctx: MediaContext) -> dict:
    """Run benchmarks for a CNN model (MobileNetV2, ResNet, etc.)."""
    logger.info(
        f"Running benchmarks for model: {ctx.model_spec.model_name} on device: {ctx.device.name}"
    )
    require_health(ctx)

    try:
        num_calls = get_num_calls(ctx)
        status_list = _run_image_analysis_benchmark(ctx, num_calls)
    except Exception as e:
        logger.error(f"Benchmark execution encountered an error: {e}")
        raise

    logger.info("Generating benchmark report...")
    ttft_value = _cnn_ttft(status_list)
    report_data = common_report_metadata(ctx, "cnn")
    report_data["benchmarks"] = {
        "num_requests": len(status_list),
        "num_inference_steps": status_list[0].num_inference_steps if status_list else 0,
        "ttft": ttft_value,
        "inference_steps_per_second": (
            sum(s.inference_steps_per_second for s in status_list) / len(status_list)
            if status_list
            else 0
        ),
    }

    return report_data


# =============================================================================
# Embedding
# =============================================================================

BENCHMARK_RESULT_START = "============ Serving Benchmark Result ============"
BENCHMARK_RESULT_END = "=================================================="
OPENAI_API_KEY = "your-secret-key"


def _embedding_params(ctx: MediaContext) -> tuple[str, int, int, int]:
    """Return (model, isl, num_calls, concurrency)."""
    env = ctx.model_spec.device_model_spec.env_vars
    return (
        ctx.model_spec.hf_model_repo,
        int(env.get("VLLM__MAX_MODEL_LENGTH", 1024)),
        1000,
        int(env.get("VLLM__MAX_NUM_SEQS", 1)),
    )


def _parse_embedding_benchmark_output(output: str) -> dict:
    if BENCHMARK_RESULT_START not in output:
        logger.warning("Benchmark result section not found in output.")
        return {}

    section = output.split(BENCHMARK_RESULT_START, 1)[1]
    if BENCHMARK_RESULT_END in section:
        section = section.split(BENCHMARK_RESULT_END, 1)[0]
    section = section.strip()

    if not section:
        logger.warning("Benchmark result section is empty after parsing.")
        return {}

    metrics: dict = {}
    for line in section.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            key_clean = re.sub(r"\s*\([^)]*\)", "", key).strip()
            metrics[key_clean] = value.strip()
    logger.info(f"Parsed benchmark metrics: {metrics}")
    return metrics


def _run_embedding_transcription_benchmark(ctx: MediaContext) -> dict:
    model, isl, num_calls, _concurrency = _embedding_params(ctx)

    venv_config = VENV_CONFIGS.get(WorkflowVenvType.BENCHMARKS_VLLM)
    vllm_exec = venv_config.venv_path / "bin" / "vllm"

    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    cmd = [
        str(vllm_exec),
        "bench",
        "serve",
        "--model",
        model,
        "--random-input-len",
        str(isl),
        "--num-prompts",
        str(num_calls),
        "--backend",
        "openai-embeddings",
        "--endpoint",
        "/v1/embeddings",
        "--dataset-name",
        "random",
        "--save-result",
        "--result-dir",
        "benchmark",
    ]

    logger.info(f"Running embedding benchmark with {num_calls} calls...")
    output = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return _parse_embedding_benchmark_output(output)


def run_embedding_benchmark(ctx: MediaContext) -> dict:
    """Run benchmarks for an embedding model."""
    logger.info(
        f"Running benchmarks for model: {ctx.model_spec.model_name} on device: {ctx.device.name}"
    )
    require_health(ctx)

    try:
        metrics = _run_embedding_transcription_benchmark(ctx)
    except Exception as e:
        logger.error(f"Benchmark execution encountered an error: {e}")
        raise

    logger.info("Generating benchmark report...")
    _model, isl, _num_calls, concurrency = _embedding_params(ctx)

    total_input_tokens = float(metrics.get("Total input tokens", 0))
    benchmark_duration = float(metrics.get("Benchmark duration", 1.0))
    successful_requests = int(metrics.get("Successful requests", 0))
    failed_requests = int(metrics.get("Failed requests", 0))
    mean_e2el = float(metrics.get("Mean E2EL", 0.0))
    req_tput = float(metrics.get("Request throughput", 0.0))

    tput_prefill = (
        total_input_tokens / benchmark_duration if benchmark_duration else 0.0
    )

    report_data = common_report_metadata(ctx, "embedding")
    report_data["benchmarks"] = {
        "isl": isl,
        "concurrency": concurrency,
        "num_requests": successful_requests + failed_requests,
        "tput_user": tput_prefill / float(concurrency) if concurrency else 0.0,
        "tput_prefill": tput_prefill,
        "e2el": mean_e2el,
        "req_tput": req_tput,
    }

    return report_data


# =============================================================================
# TTS
# =============================================================================

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


# =============================================================================
# Video
# =============================================================================

DEFAULT_VIDEO_POLLING_INTERVAL_SECONDS = 5
DEFAULT_VIDEO_TIMEOUT_SECONDS = 1200
VIDEO_INFERENCE_STEPS = {"mochi-1-preview": 50, "Wan2.2-T2V-A14B-Diffusers": 40}
VIDEO_JOB_STATUS_COMPLETED = "completed"
VIDEO_JOB_STATUS_FAILED = "failed"
VIDEO_JOB_STATUS_CANCELLED = "cancelled"


def _download_video(ctx: MediaContext, job_id: str, headers: dict) -> str:
    logger.info(f"Downloading video for job: {job_id}")
    try:
        output_dir = Path("/tmp/videos")
        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = output_dir / f"{job_id}.mp4"

        response = requests.get(
            f"{ctx.base_url}/v1/videos/generations/{job_id}/download",
            headers=headers,
            timeout=300,
            stream=True,
        )
        if response.status_code != 200:
            logger.error(f"Failed to download video: {response.status_code}")
            return ""

        with open(video_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Video downloaded: {video_path}")
        return str(video_path)
    except Exception as e:
        logger.error(f"Error downloading video: {e}")
        return ""


def _poll_video_completion(
    ctx: MediaContext,
    job_id: str,
    headers: dict,
    polling_interval: int = DEFAULT_VIDEO_POLLING_INTERVAL_SECONDS,
    timeout: int = DEFAULT_VIDEO_TIMEOUT_SECONDS,
) -> str:
    logger.info(f"Polling video job: {job_id}")
    logger.info(f"Polling interval: {polling_interval} seconds")
    logger.info(f"Timeout: {timeout} seconds")
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(
                f"{ctx.base_url}/v1/videos/generations/{job_id}",
                headers=headers,
                timeout=30,
            )
            if response.status_code != 200:
                logger.warning(f"Failed to get job status: {response.status_code}")
                time.sleep(polling_interval)
                continue

            job_data = response.json()
            status = job_data.get("status")
            logger.info(f"Job {job_id} status: {status}")

            if status == VIDEO_JOB_STATUS_COMPLETED:
                return _download_video(ctx, job_id, headers)
            if status in (VIDEO_JOB_STATUS_FAILED, VIDEO_JOB_STATUS_CANCELLED):
                logger.error(f"Video generation {status}: {job_id}")
                return ""

            logger.info(
                f"Still processing, waiting {polling_interval} seconds and polling again"
            )
            time.sleep(polling_interval)
        except Exception as e:
            logger.error(f"Error polling job status: {e}")
            time.sleep(polling_interval)

    logger.error(f"Video generation timed out after {timeout}s")
    return ""


def _generate_video(
    ctx: MediaContext, prompt: str, num_inference_steps: int = 20
) -> tuple[bool, float, str, str]:
    logger.info(f"🎬 Generating video with prompt: {prompt}")
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer your-secret-key",
        "Content-Type": "application/json",
    }
    payload = {"prompt": prompt, "num_inference_steps": num_inference_steps}
    logger.info(f"Payload: {payload}")

    start_time = time.time()
    try:
        response = requests.post(
            f"{ctx.base_url}/v1/videos/generations",
            json=payload,
            headers=headers,
            timeout=90,
        )
        if response.status_code != 202:
            logger.error(
                f"Failed to submit video generation job: {response.status_code}"
            )
            return False, time.time() - start_time, "", ""

        job_data = response.json()
        job_id = job_data.get("id")
        logger.info(f"Video generation job submitted: {job_id}")

        video_path = _poll_video_completion(ctx, job_id, headers)
        elapsed = time.time() - start_time

        if video_path:
            logger.info(f"✅ Video generated successfully: {video_path}")
            return True, elapsed, job_id, video_path

        logger.error(f"❌ Video generation failed for job: {job_id}")
        return False, elapsed, job_id, ""
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Video generation error: {e}")
        return False, elapsed, "", ""


def _run_video_generation_benchmark(
    ctx: MediaContext, num_calls: int
) -> list[VideoGenerationTestStatus]:
    logger.info("Running video generation benchmark.")
    inference_steps = VIDEO_INFERENCE_STEPS[ctx.model_spec.model_name]
    logger.info(f"Inference steps: {inference_steps}")

    status_list: list[VideoGenerationTestStatus] = []
    for i in range(num_calls):
        logger.info(f"Generating video {i + 1}/{num_calls}...")
        status, elapsed, job_id, video_path = _generate_video(
            ctx,
            prompt=f"Test video generation {i + 1}",
            num_inference_steps=inference_steps,
        )
        logger.info(f"Generated video in {elapsed:.2f} seconds.")

        inference_steps_per_second = inference_steps / elapsed if elapsed > 0 else 0
        status_list.append(
            VideoGenerationTestStatus(
                status=status,
                elapsed=elapsed,
                num_inference_steps=inference_steps,
                inference_steps_per_second=inference_steps_per_second,
                job_id=job_id,
                video_path=video_path,
                prompt=f"Test video generation {i + 1}",
            )
        )
    return status_list


def _video_ttft(status_list: list[VideoGenerationTestStatus]) -> float:
    logger.info("Calculating TTFT value")
    return sum(s.elapsed for s in status_list) / len(status_list) if status_list else 0


def run_video_benchmark(ctx: MediaContext) -> dict:
    """Run benchmarks for a video model (Mochi, WAN, etc.)."""
    logger.info(
        f"Running benchmarks for model: {ctx.model_spec.model_name} on device: {ctx.device.name}"
    )
    require_health(ctx)

    try:
        num_calls = get_num_calls(ctx)
        status_list = _run_video_generation_benchmark(ctx, num_calls)
    except Exception as e:
        logger.error(f"Benchmark execution encountered an error: {e}")
        raise

    logger.info("Generating benchmark report...")
    ttft_value = _video_ttft(status_list)
    report_data = common_report_metadata(ctx, "video")
    report_data["benchmarks"] = {
        "num_requests": len(status_list),
        "num_inference_steps": status_list[0].num_inference_steps if status_list else 0,
        "ttft": ttft_value,
        "inference_steps_per_second": (
            sum(s.inference_steps_per_second for s in status_list) / len(status_list)
            if status_list
            else 0
        ),
    }

    return report_data


__all__ = [
    "run_image_benchmark",
    "run_audio_benchmark",
    "run_cnn_benchmark",
    "run_embedding_benchmark",
    "run_tts_benchmark",
    "run_video_benchmark",
    "IMAGE_BENCHMARK_DISPATCH",
]
