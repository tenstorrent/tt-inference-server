# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Unified eval tests for every media type (function-based).

Public entry points:

* ``run_image_eval(ctx)``
* ``run_audio_eval(ctx)``
* ``run_cnn_eval(ctx)``
* ``run_embedding_eval(ctx)``
* ``run_tts_eval(ctx)``
* ``run_video_eval(ctx)``

Each takes a :class:`MediaContext` and performs the same work the legacy
``utils/media_clients/<media>_client.py`` classes did in ``run_eval()``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import aiohttp
import requests

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from server_tests.test_cases.image_generation_eval_test import (
    ImageGenerationEvalsTest,
)
from server_tests.test_classes import TestConfig as ServerTestConfig
from utils.sdxl_accuracy_utils.sdxl_accuracy_utils import (
    calculate_accuracy_check,
    calculate_metrics,
    sdxl_get_prompts,
)
from workflows.utils import (
    get_num_calls,
    is_preprocessing_enabled_for_whisper,
    is_sdxl_num_prompts_enabled,
    is_streaming_enabled_for_whisper,
)
from workflows.utils_report import get_performance_targets
from workflows.workflow_types import ReportCheckTypes

from .context import MediaContext, count_tokens, get_health
from .test_status import (
    AudioTestStatus,
    CnnGenerationTestStatus,
    ImageGenerationTestStatus,
    TtsTestStatus,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Shared helpers
# =============================================================================


def _common_eval_metadata(ctx: MediaContext, task_type: str) -> dict:
    return {
        "model": ctx.model_spec.model_name,
        "device": ctx.device.name.lower(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "task_type": task_type,
        "task_name": ctx.all_params.tasks[0].task_name,
        "tolerance": ctx.all_params.tasks[0].score.tolerance,
    }


def _require_health(ctx: MediaContext) -> str:
    health_status, runner_in_use = get_health(ctx)
    if not health_status:
        logger.error("Health check failed.")
        raise RuntimeError("Health check failed")
    logger.info(f"Health check passed. Runner in use: {runner_in_use}")
    return runner_in_use


# =============================================================================
# Image
# =============================================================================

SDXL_SD35_INFERENCE_STEPS = 20
IMAGE_FORMAT_FOR_EVALS = "PNG"
IMAGE_QUALITY_FOR_EVALS = 100
SDXL_INPAINTING_INFERENCE_STEPS = 20
NEGATIVE_PROMPT = (
    "normal quality, low quality, worst quality, low res, blurry, nsfw, nude"
)
GUIDANCE_SCALE = 8
NUM_INFERENCE_STEPS = 20
FLUX_MOTIF_INFERENCE_STEPS = 28
FLUX_1_SCHNELL_INFERENCE_STEPS = 4

SDXL_IMG2IMG_INFERENCE_STEPS = 30
GUIDANCE_SCALE_IMG2IMG = 7.5
SEED_IMG2IMG = 0
STRENGTH_IMG2IMG = 0.6

GUIDANCE_SCALE_INPAINTING = 8.0
SEED_INPAINTING = 0
STRENGTH_INPAINTING = 0.99


def _image_ttft(status_list: list[ImageGenerationTestStatus]) -> float:
    return (
        sum(s.elapsed for s in status_list) / len(status_list) if status_list else 0
    )


async def _generate_image_eval_async(
    ctx: MediaContext,
    session: aiohttp.ClientSession,
    prompt: str,
    num_inference_steps: int = NUM_INFERENCE_STEPS,
) -> tuple[bool, float, Optional[str]]:
    logger.info(f"🌅 Generating image for prompt: {prompt}")
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer your-secret-key",
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": prompt,
        "negative_prompt": NEGATIVE_PROMPT,
        "num_inference_steps": num_inference_steps,
        "seed": 0,
        "guidance_scale": GUIDANCE_SCALE,
        "image_return_format": IMAGE_FORMAT_FOR_EVALS,
        "image_quality": IMAGE_QUALITY_FOR_EVALS,
        "number_of_images": 1,
    }

    start_time = time.time()
    try:
        async with session.post(
            f"{ctx.base_url}/v1/images/generations",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=25000),
        ) as response:
            elapsed = time.time() - start_time
            if response.status != 200:
                logger.error(
                    f"❌ Image generation for eval failed with status: {response.status}"
                )
                return False, elapsed, None
            response_data = await response.json()
            images = response_data.get("images", [])
            base64image = images[0] if images else None
            logger.info(f"✅ Image generation for eval succeeded in {elapsed:.2f}s")
            return True, elapsed, base64image
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"❌ Image generation for eval failed: {e}")
        return False, elapsed, None


async def _run_image_generation_eval(
    ctx: MediaContext, runner: Optional[str] = None
) -> tuple[list[ImageGenerationTestStatus], float]:
    logger.info("Running image generation eval.")
    num_prompts = is_sdxl_num_prompts_enabled(ctx)
    logger.info(f"Number of prompts set to: {num_prompts}")

    prompts = sdxl_get_prompts(0, num_prompts)
    logger.info(f"Retrieved {len(prompts)} prompts for evaluation.")

    async with aiohttp.ClientSession() as session:
        total_start_time = time.time()
        tasks = [
            _generate_image_eval_async(ctx, session, prompt) for prompt in prompts
        ]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - total_start_time

    logger.info(
        f"Generated {len(prompts)} images concurrently in {total_time:.2f} seconds"
    )

    status_list: list[ImageGenerationTestStatus] = []
    failed_count = 0

    for i, (status, elapsed, base64image) in enumerate(results):
        prompt = prompts[i]
        if not status or base64image is None:
            failed_count += 1
            logger.warning(
                f"❌ Skipping failed image {i + 1}/{num_prompts}: '{prompt}'"
            )
            continue

        inference_steps_per_second = (
            SDXL_SD35_INFERENCE_STEPS / elapsed if elapsed > 0 else 0
        )
        logger.info(f"🚀 Image {i + 1}/{num_prompts}: {prompt} - {elapsed:.2f}s")
        status_list.append(
            ImageGenerationTestStatus(
                status=status,
                elapsed=elapsed,
                num_inference_steps=SDXL_SD35_INFERENCE_STEPS,
                inference_steps_per_second=inference_steps_per_second,
                base64image=base64image,
                prompt=prompt,
            )
        )

    logger.info(f"Total image generations attempted: {num_prompts}")
    logger.info(f"Total failed image generations: {failed_count}")
    logger.info(f"Total successful image generations: {num_prompts - failed_count}")

    if failed_count:
        logger.warning(f"⚠️  {failed_count} image generations failed during eval.")
        raise RuntimeError(
            f"❌ {failed_count} image generations failed - cannot calculate accuracy metrics"
        )

    return status_list, total_time


async def _generate_image_img2img_eval_async(
    ctx: MediaContext,
    session: aiohttp.ClientSession,
    prompt: str,
    image_data: dict,
) -> tuple[bool, float, Optional[str]]:
    logger.info(f"🌆 Generating img2img image for prompt: {prompt}")
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer your-secret-key",
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": prompt,
        "image": image_data["file"],
        "seed": SEED_IMG2IMG,
        "guidance_scale": GUIDANCE_SCALE_IMG2IMG,
        "number_of_images": 1,
        "strength": STRENGTH_IMG2IMG,
        "num_inference_steps": SDXL_IMG2IMG_INFERENCE_STEPS,
    }
    start_time = time.time()
    try:
        async with session.post(
            f"{ctx.base_url}/v1/images/image-to-image",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=25000),
        ) as response:
            elapsed = time.time() - start_time
            if response.status != 200:
                logger.error(
                    f"❌ Img2img generation for eval failed with status: {response.status}"
                )
                return False, elapsed, None
            response_data = await response.json()
            images = response_data.get("images", [])
            base64image = images[0] if images else None
            logger.info(f"✅ Img2img generation for eval succeeded in {elapsed:.2f}s")
            return True, elapsed, base64image
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"❌ Img2img generation for eval failed: {e}")
        return False, elapsed, None


async def _run_img2img_generation_eval(
    ctx: MediaContext, runner: Optional[str] = None
) -> tuple[list[ImageGenerationTestStatus], float]:
    logger.info("Running image2image generation eval.")
    prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"
    logger.info(f"Using 1 prompt for evaluation: {prompt}")

    with open(f"{ctx.test_payloads_path}/image_client_img2img_payload", "r") as f:
        image_data = json.load(f)

    async with aiohttp.ClientSession() as session:
        total_start_time = time.time()
        results = await asyncio.gather(
            _generate_image_img2img_eval_async(ctx, session, prompt, image_data)
        )
        total_time = time.time() - total_start_time

    logger.info(f"Generated 1 img2img image in {total_time:.2f} seconds")

    status_list: list[ImageGenerationTestStatus] = []
    failed_count = 0
    for status, elapsed, base64image in results:
        if not status or base64image is None:
            failed_count += 1
            logger.warning(f"❌ Failed img2img image generation: '{prompt}'")
            continue
        inference_steps_per_second = (
            SDXL_IMG2IMG_INFERENCE_STEPS / elapsed if elapsed > 0 else 0
        )
        logger.info(f"🚀 Img2img image: {prompt} - {elapsed:.2f}s")
        status_list.append(
            ImageGenerationTestStatus(
                status=status,
                elapsed=elapsed,
                num_inference_steps=SDXL_IMG2IMG_INFERENCE_STEPS,
                inference_steps_per_second=inference_steps_per_second,
                base64image=base64image,
                prompt=prompt,
            )
        )

    logger.info("Total img2img generations attempted: 1")
    logger.info(f"Total failed img2img generations: {failed_count}")
    logger.info(f"Total successful img2img generations: {1 - failed_count}")

    if failed_count:
        logger.warning("⚠️  Img2img generation failed during eval.")
        raise RuntimeError(
            "❌ Img2img generation failed - cannot calculate accuracy metrics"
        )

    return status_list, total_time


async def _generate_image_inpainting_eval_async(
    ctx: MediaContext,
    session: aiohttp.ClientSession,
    prompt: str,
    inpaint_image: dict,
    inpaint_mask: dict,
) -> tuple[bool, float, Optional[str]]:
    logger.info(f"🏞️ Generating inpainting image for prompt: {prompt}")
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer your-secret-key",
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": prompt,
        "image": inpaint_image,
        "mask": inpaint_mask,
        "seed": SEED_INPAINTING,
        "guidance_scale": GUIDANCE_SCALE_INPAINTING,
        "number_of_images": 1,
        "strength": STRENGTH_INPAINTING,
        "num_inference_steps": SDXL_INPAINTING_INFERENCE_STEPS,
    }
    start_time = time.time()
    try:
        async with session.post(
            f"{ctx.base_url}/v1/images/edits",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=25000),
        ) as response:
            elapsed = time.time() - start_time
            if response.status != 200:
                logger.error(
                    f"❌ Inpainting generation for eval failed with status: {response.status}"
                )
                return False, elapsed, None
            response_data = await response.json()
            images = response_data.get("images", [])
            base64image = images[0] if images else None
            logger.info(
                f"✅ Inpainting generation for eval succeeded in {elapsed:.2f}s"
            )
            return True, elapsed, base64image
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"❌ Inpainting generation for eval failed: {e}")
        return False, elapsed, None


async def _run_inpainting_generation_eval(
    ctx: MediaContext, runner: Optional[str] = None
) -> tuple[list[ImageGenerationTestStatus], float]:
    logger.info("Running inpainting generation eval.")
    prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
    logger.info(f"Using 1 prompt for evaluation: {prompt}")

    with open(
        f"{ctx.test_payloads_path}/image_client_inpainting_payload", "r"
    ) as f:
        payload_data = json.load(f)
        inpaint_image = payload_data["inpaint_image"]
        inpaint_mask = payload_data["inpaint_mask"]

    async with aiohttp.ClientSession() as session:
        total_start_time = time.time()
        results = await asyncio.gather(
            _generate_image_inpainting_eval_async(
                ctx, session, prompt, inpaint_image, inpaint_mask
            )
        )
        total_time = time.time() - total_start_time

    logger.info(f"Generated 1 inpainting image in {total_time:.2f} seconds")

    status_list: list[ImageGenerationTestStatus] = []
    failed_count = 0
    for status, elapsed, base64image in results:
        if not status or base64image is None:
            failed_count += 1
            logger.warning(f"❌ Failed inpainting image generation: '{prompt}'")
            continue
        inference_steps_per_second = (
            SDXL_INPAINTING_INFERENCE_STEPS / elapsed if elapsed > 0 else 0
        )
        logger.info(f"🚀 Inpainting image: {prompt} - {elapsed:.2f}s")
        status_list.append(
            ImageGenerationTestStatus(
                status=status,
                elapsed=elapsed,
                num_inference_steps=SDXL_INPAINTING_INFERENCE_STEPS,
                inference_steps_per_second=inference_steps_per_second,
                base64image=base64image,
                prompt=prompt,
            )
        )

    logger.info("Total inpainting generations attempted: 1")
    logger.info(f"Total failed inpainting generations: {failed_count}")
    logger.info(f"Total successful inpainting generations: {1 - failed_count}")

    if failed_count:
        logger.warning("⚠️  Inpainting generation failed during eval.")
        raise RuntimeError(
            "❌ Inpainting generation failed - cannot calculate accuracy metrics"
        )

    return status_list, total_time


async def _run_image_generation_eval_test(
    ctx: MediaContext, runner: Optional[str] = None
) -> dict:
    """Flux/Motif path: delegate to ImageGenerationEvalsTest."""
    num_prompts = is_sdxl_num_prompts_enabled(ctx)
    inference_steps = (
        FLUX_1_SCHNELL_INFERENCE_STEPS
        if runner == "tt-flux.1-schnell"
        else FLUX_MOTIF_INFERENCE_STEPS
    )
    logger.info(
        f"Running ImageGenerationEvalsTest for {ctx.model_spec.model_name} "
        f"with {num_prompts} prompts, {inference_steps} inference steps"
    )

    test_config = ServerTestConfig.create_default(timeout=25000)
    request_dict = {
        "model_name": ctx.model_spec.model_name,
        "num_prompts": num_prompts,
        "num_inference_steps": inference_steps,
        "server_url": ctx.base_url,
    }
    eval_test = ImageGenerationEvalsTest(test_config, {"request": request_dict})
    eval_test.service_port = ctx.service_port

    result = await eval_test._run_specific_test_async()

    if not result.get("success"):
        logger.error(
            f"ImageGenerationEvalsTest ACCURACY_CHECK failed: error={result.get('error')}"
        )
        raise RuntimeError(result.get("error", "ImageGenerationEvalsTest failed"))

    logger.info(f"ImageGenerationEvalsTest completed: {result.get('eval_results')}")
    return result


ImageEvalFn = Callable[
    [MediaContext, Optional[str]],
    Awaitable[Any],
]

IMAGE_EVAL_DISPATCH: dict[str, ImageEvalFn] = {
    "tt-sdxl-trace": _run_image_generation_eval,
    "tt-sdxl-image-to-image": _run_img2img_generation_eval,
    "tt-sdxl-edit": _run_inpainting_generation_eval,
    "tt-sd3.5": _run_image_generation_eval,
    "tt-flux.1-dev": _run_image_generation_eval_test,
    "tt-flux.1-schnell": _run_image_generation_eval_test,
    "tt-motif-image-6b-preview": _run_image_generation_eval_test,
}


def run_image_eval(ctx: MediaContext) -> dict:
    """Run evaluations for an image model (SDXL, SD3.5, Flux, Motif)."""
    logger.info(
        f"Running evals for model: {ctx.model_spec.model_name} on device: {ctx.device.name}"
    )

    runner_in_use = _require_health(ctx)

    eval_method = IMAGE_EVAL_DISPATCH.get(runner_in_use, _run_image_generation_eval)
    try:
        eval_result = asyncio.run(eval_method(ctx, runner_in_use))
    except Exception as e:
        logger.error(f"Eval execution encountered an error: {e}")
        raise

    logger.info("Generating eval report...")
    benchmark_data = _common_eval_metadata(ctx, "image")
    benchmark_data["published_score"] = ctx.all_params.tasks[0].score.published_score
    benchmark_data["published_score_ref"] = ctx.all_params.tasks[
        0
    ].score.published_score_ref

    if isinstance(eval_result, dict):
        eval_results = eval_result.get("eval_results", {})
        benchmark_data["fid_score"] = eval_results.get("fid_score")
        benchmark_data["average_clip"] = eval_results.get("average_clip")
        benchmark_data["deviation_clip_score"] = eval_results.get(
            "deviation_clip_score"
        )
        benchmark_data["accuracy_check"] = eval_results.get("accuracy_check")
        benchmark_data["score"] = None
    else:
        status_list, total_time = eval_result

        ttft_value = _image_ttft(status_list)
        logger.info(f"Extracted TTFT value: {ttft_value}")
        benchmark_data["score"] = ttft_value

        logger.info("Running and calculating accuracy and metrics")
        fid_score, average_clip_score, deviation_clip_score = calculate_metrics(
            status_list
        )
        accuracy_check = calculate_accuracy_check(
            fid_score,
            average_clip_score,
            len(status_list),
            ctx.model_spec.model_name,
        )

        benchmark_data["fid_score"] = fid_score
        benchmark_data["average_clip"] = average_clip_score
        benchmark_data["deviation_clip_score"] = deviation_clip_score
        benchmark_data["accuracy_check"] = accuracy_check

        device_spec = ctx.model_spec.device_model_spec
        if device_spec and hasattr(device_spec, "max_concurrency"):
            tput_user = len(status_list) / (total_time * device_spec.max_concurrency)
            benchmark_data["tput_user"] = tput_user
            logger.info(
                f"Calculated tput_user: {tput_user} (prompts: {len(status_list)}, "
                f"time: {total_time}s, max_concurrency: {device_spec.max_concurrency})"
            )
        else:
            logger.warning(f"No device spec found for device: {ctx.device}")

    return benchmark_data


# =============================================================================
# Audio
# =============================================================================


def _audio_ttft(status_list: list[AudioTestStatus]) -> float:
    valid = [s.ttft for s in status_list if s.ttft is not None]
    return sum(valid) / len(valid) if valid else 0


def _audio_rtr(status_list: list[AudioTestStatus]) -> float:
    valid = [s.rtr for s in status_list if s.rtr is not None]
    return sum(valid) / len(valid) if valid else 0


def _audio_tsu(status_list: list[AudioTestStatus]) -> float:
    valid = [s.tsu for s in status_list if s.tsu is not None]
    return sum(valid) / len(valid) if valid else 0


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
    """Streaming transcription. Measures TTFT excluding speaker markers."""
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
            final_tokens / content_streaming_time
            if content_streaming_time > 0
            else 0
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
            AudioTestStatus(
                status=status, elapsed=elapsed, ttft=ttft, tsu=tsu, rtr=rtr
            )
        )
    return status_list


def run_audio_eval(ctx: MediaContext) -> dict:
    """Run evaluations for an audio model (Whisper, etc.)."""
    logger.info(
        f"Running evals for model: {ctx.model_spec.model_name} on device: {ctx.device.name}"
    )
    _require_health(ctx)

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

    benchmark_data = _common_eval_metadata(ctx, "audio")
    benchmark_data["device"] = ctx.device.name
    benchmark_data["published_score"] = ctx.all_params.tasks[0].score.published_score
    benchmark_data["score"] = ttft_value
    benchmark_data["published_score_ref"] = ctx.all_params.tasks[
        0
    ].score.published_score_ref
    # TODO: replace hardcoded PASS with a real accuracy evaluation.
    benchmark_data["accuracy_check"] = ReportCheckTypes.PASS
    benchmark_data["t/s/u"] = tsu_value
    benchmark_data["rtr"] = rtr_value

    return benchmark_data


# =============================================================================
# CNN
# =============================================================================

CNN_MOBILENETV2_RUNNER = "tt-xla-mobilenetv2"


def _cnn_ttft(status_list: list[CnnGenerationTestStatus]) -> float:
    return (
        sum(s.elapsed for s in status_list) / len(status_list) if status_list else 0
    )


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
        status_list.append(
            CnnGenerationTestStatus(status=status, elapsed=elapsed)
        )
    return status_list


def _run_mobilenetv2_eval(ctx: MediaContext) -> dict:
    """Delegate MobileNetV2 accuracy eval to VisionEvalsTest."""
    from server_tests.test_cases.vision_evals_test import (
        VisionEvalsTest,
        VisionEvalsTestRequest,
    )
    from server_tests.test_classes import TestConfig

    logger.info("Running mobilenetv2 eval.")
    request = VisionEvalsTestRequest(
        action="measure_accuracy",
        mode="device",
        models=[CNN_MOBILENETV2_RUNNER],
        server_url=f"{ctx.base_url}/v1/cnn/search-image",
    )
    logger.info(f"Running VisionEvalsTest with request: {request}")

    config = TestConfig.create_default()
    test = VisionEvalsTest(config, {"request": request})

    logger.info("Starting VisionEvalsTest")
    result = test.run_tests()

    eval_results = result.get("result", {}).get("eval_results", {})
    model_results = eval_results.get(CNN_MOBILENETV2_RUNNER, {})
    logger.info(f"VisionEvalsTest model results: {model_results}")

    device_result = model_results.get("device", {})
    device_result["accuracy_status"] = model_results.get(
        "accuracy_status", ReportCheckTypes.NA
    )
    logger.info(f"VisionEvalsTest device eval_results: {device_result}")
    return device_result


def run_cnn_eval(ctx: MediaContext) -> dict:
    """Run evaluations for a CNN model (MobileNetV2, ResNet, etc.)."""
    logger.info(
        f"Running evals for model: {ctx.model_spec.model_name} on device: {ctx.device.name}"
    )

    runner_in_use = _require_health(ctx)

    try:
        eval_result = None
        status_list: list[CnnGenerationTestStatus] = []
        if runner_in_use == CNN_MOBILENETV2_RUNNER:
            eval_result = _run_mobilenetv2_eval(ctx)
        else:
            num_calls = get_num_calls(ctx)
            status_list = _run_image_analysis_benchmark(ctx, num_calls)
    except Exception as e:
        logger.error(f"Eval execution encountered an error: {e}")
        raise

    logger.info("Generating eval report...")
    benchmark_data = _common_eval_metadata(ctx, "cnn")

    if runner_in_use == CNN_MOBILENETV2_RUNNER and eval_result:
        logger.info("Adding eval results from eval spec test to benchmark data")
        benchmark_data["accuracy_check"] = eval_result.get(
            "accuracy_status", ReportCheckTypes.NA
        )
        benchmark_data["correct"] = eval_result["correct"]
        benchmark_data["total"] = eval_result["total"]
        benchmark_data["mismatches_count"] = eval_result["mismatches_count"]
    else:
        logger.info("No eval results from eval spec test to add to benchmark data")
        ttft_value = _cnn_ttft(status_list)
        logger.info(f"Extracted TTFT value: {ttft_value}")
        benchmark_data["published_score"] = ctx.all_params.tasks[
            0
        ].score.published_score
        benchmark_data["score"] = ttft_value
        benchmark_data["published_score_ref"] = ctx.all_params.tasks[
            0
        ].score.published_score_ref

    return benchmark_data


# =============================================================================
# Embedding
# =============================================================================

OPENAI_API_KEY = "your-secret-key"
MTEB_TASKS = ["STS12"]
EMBEDDING_DIMENSIONS = 1000


def _embedding_model_config(ctx: MediaContext) -> tuple[str, int, int]:
    """Return (hf_model_repo, isl, dimensions) derived from model_spec env vars."""
    env = ctx.model_spec.device_model_spec.env_vars
    return (
        ctx.model_spec.hf_model_repo,
        int(env.get("VLLM__MAX_MODEL_LENGTH", 1024)),
        EMBEDDING_DIMENSIONS,
    )


def _parse_embedding_evals_output(results: Any) -> dict:
    """Parse MTEB evaluation results, extracting key metrics from scores['test']."""
    try:
        scores = results.task_results[0].scores["test"]
        if isinstance(scores, list) and len(scores) > 0:
            scores = scores[0]
    except Exception as e:
        logger.error(f"Could not extract scores['test']: {e}")
        raise

    keys = [
        "pearson",
        "spearman",
        "cosine_pearson",
        "cosine_spearman",
        "manhattan_pearson",
        "manhattan_spearman",
        "euclidean_pearson",
        "euclidean_spearman",
        "main_score",
        "languages",
    ]
    report_data = {k: scores.get(k) for k in keys if k in scores}
    logger.info(f"Parsed evaluation results: {report_data}")
    return report_data


def _run_embedding_transcription_eval(ctx: MediaContext) -> dict:
    """Run MTEB eval against the embedding endpoint."""
    import mteb
    import numpy as np
    from mteb.models.model_implementations.openai_models import OpenAIModel
    from mteb.models.model_meta import ModelMeta
    from openai import OpenAI

    model_name, isl, dimensions = _embedding_model_config(ctx)

    def single_string_encode(self, inputs, **kwargs):
        sentences = [text for batch in inputs for text in batch["text"]]
        all_embeddings = []
        for sentence in sentences:
            response = self._client.embeddings.create(
                input=sentence,
                model=model_name,
                encoding_format="float",
                dimensions=self._embed_dim if self._embed_dim else None,
            )
            all_embeddings.extend(self._to_numpy(response))
        return np.array(all_embeddings)

    client = OpenAI(base_url=f"{ctx.base_url}/v1", api_key=OPENAI_API_KEY)

    model = OpenAIModel(
        model_name=model_name,
        max_tokens=isl,
        embed_dim=dimensions,
        client=client,
    )
    model.encode = single_string_encode.__get__(model, type(model))

    model_meta = ModelMeta(
        name=model_name,
        revision=None,
        embed_dim=dimensions,
        max_tokens=isl,
        open_weights=False,
        loader=None,
        loader_kwargs={},
        framework=[],
        similarity_fn_name=None,
        use_instructions=None,
        release_date=None,
        languages=[],
        n_parameters=None,
        memory_usage_mb=None,
        license=None,
        public_training_code=None,
        public_training_data=None,
        training_datasets=None,
    )
    model.mteb_model_meta = model_meta

    tasks = mteb.get_tasks(tasks=MTEB_TASKS)

    logger.info("Running embedding transcription evaluation with STS12...")
    results = mteb.evaluate(
        model,
        tasks=tasks,
        encode_kwargs={"batch_size": 1},
        cache=None,
        overwrite_strategy="always",
    )
    logger.info(f"Evaluation results: {results}")
    return _parse_embedding_evals_output(results)


def run_embedding_eval(ctx: MediaContext) -> dict:
    """Run evaluations for an embedding model."""
    logger.info(
        f"Running evals for model: {ctx.model_spec.model_name} on device: {ctx.device.name}"
    )
    _require_health(ctx)

    try:
        logger.info("Running embedding eval...")
        metrics = _run_embedding_transcription_eval(ctx)
    except Exception as e:
        logger.error(f"Eval execution encountered an error: {e}")
        raise

    logger.info("Generating evals report...")
    report_data = {
        "model": ctx.model_spec.model_name,
        "device": ctx.device.name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "task_type": "embedding",
        "task_name": ctx.all_params.tasks[0].task_name,
    }
    report_data.update(metrics)

    return report_data


# =============================================================================
# TTS
# =============================================================================

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

                logger.info(
                    f"Received audio data (base64 length: {len(audio_base64)})"
                )

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
    _require_health(ctx)

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

    benchmark_data = _common_eval_metadata(ctx, "text_to_speech")
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


# =============================================================================
# Video
# =============================================================================


def _run_video_generation_eval(ctx: MediaContext) -> dict:
    """Delegate to VideoGenerationEvalsTest."""
    from server_tests.test_cases.video_generation_eval_test import (
        VideoGenerationEvalsTest,
        VideoGenerationEvalsTestRequest,
    )
    from server_tests.test_classes import TestConfig

    logger.info("Running video generation eval.")

    num_prompts = 5
    num_inference_steps = 40
    start_from = 0
    frame_sample_rate = 8

    request = VideoGenerationEvalsTestRequest(
        model_name=ctx.model_spec.model_name,
        num_prompts=num_prompts,
        start_from=start_from,
        num_inference_steps=num_inference_steps,
        server_url=ctx.base_url,
        frame_sample_rate=frame_sample_rate,
    )
    logger.info(f"Running VideoGenerationEvalsTest with request: {request}")

    config = TestConfig.create_default()
    test = VideoGenerationEvalsTest(config, {"request": request})

    logger.info("Starting VideoGenerationEvalsTest")
    result = test.run_tests()

    eval_results = result.get("result", {}).get("eval_results", {})
    logger.info(f"VideoGenerationEvalsTest eval_results: {eval_results}")
    return eval_results


def _run_video_fvd_and_fvmd_eval() -> dict:
    """Run FVD + FVMD eval against reference and generated video directories."""
    from server_tests.test_cases.video_fvd_eval_test import (
        DATASET_DIR as FVD_DATASET_DIR,
    )
    from server_tests.test_cases.video_fvd_eval_test import (
        VideoFVDTest,
        VideoFVDTestRequest,
    )
    from server_tests.test_cases.video_fvmd_eval_test import (
        VideoFVMDTest,
        VideoFVMDTestRequest,
    )
    from server_tests.test_classes import TestConfig

    logger.info("Running video FVD and FVMD eval.")

    reference_videos_path = str(Path(FVD_DATASET_DIR) / "videos")
    generated_videos_path = str(Path("server_tests/datasets/videos").resolve())
    if not Path(generated_videos_path).exists():
        generated_videos_path = "/tmp/videos"
    logger.info(
        f"Reference path: {reference_videos_path}, generated path: {generated_videos_path}"
    )

    config = TestConfig.create_default()
    combined_results: dict = {
        "reference_videos_path": reference_videos_path,
        "generated_videos_path": generated_videos_path,
    }

    download_request = VideoFVDTestRequest(
        action="download", download_count=2, category="Sports"
    )
    download_test = VideoFVDTest(config, {"request": download_request})
    download_result = download_test.run_tests()
    if not download_result.get("success") or not download_result.get(
        "result", {}
    ).get("success"):
        logger.warning(
            "Reference video download failed: %s. "
            "FVD/FVMD may fail if reference dir is empty.",
            download_result,
        )
    else:
        logger.info("Reference videos downloaded successfully.")

    fvd_request = VideoFVDTestRequest(
        action="compute_fvd",
        reference_videos_path=reference_videos_path,
        generated_videos_path=generated_videos_path,
    )
    fvd_test = VideoFVDTest(config, {"request": fvd_request})
    fvd_result = fvd_test.run_tests()
    fvd_ok = fvd_result.get("success") and fvd_result.get("result", {}).get("success")
    if fvd_ok:
        combined_results["fvd"] = fvd_result["result"].get("fvd_score")
        logger.info("FVD score: %s", combined_results["fvd"])
    else:
        combined_results["fvd"] = None
        logger.warning("FVD test failed or did not return score: %s", fvd_result)

    fvmd_request = VideoFVMDTestRequest(
        action="compute_fvmd",
        reference_videos_path=reference_videos_path,
        generated_videos_path=generated_videos_path,
    )
    fvmd_test = VideoFVMDTest(config, {"request": fvmd_request})
    fvmd_result = fvmd_test.run_tests()
    fvmd_ok = fvmd_result.get("success") and fvmd_result.get("result", {}).get(
        "success"
    )
    if fvmd_ok:
        combined_results["fvmd"] = fvmd_result["result"].get("fvmd_score")
        logger.info("FVMD score: %s", combined_results["fvmd"])
    else:
        combined_results["fvmd"] = None
        logger.warning("FVMD test failed or did not return score: %s", fvmd_result)

    return combined_results


def run_video_eval(ctx: MediaContext) -> dict:
    """Run evaluations for a video model (Mochi, WAN, etc.)."""
    logger.info(
        f"Running evals for model: {ctx.model_spec.model_name} on device: {ctx.device.name}"
    )
    _require_health(ctx)

    try:
        eval_result = _run_video_generation_eval(ctx)
    except Exception as e:
        logger.error(f"Eval execution encountered an error: {e}")
        raise

    logger.info("Generating eval report...")
    benchmark_data = _common_eval_metadata(ctx, "video")

    if eval_result:
        logger.info("Adding eval results from video generation test to benchmark data")
        benchmark_data["num_prompts"] = eval_result.get("num_prompts", 0)
        benchmark_data["num_inference_steps"] = eval_result.get(
            "num_inference_steps", 0
        )

        clip_results = eval_result.get("clip_results", {})
        benchmark_data["average_clip"] = clip_results.get("average_clip", 0.0)
        benchmark_data["min_clip"] = clip_results.get("min_clip", 0.0)
        benchmark_data["max_clip"] = clip_results.get("max_clip", 0.0)
        benchmark_data["clip_standard_deviation"] = clip_results.get(
            "clip_standard_deviation", 0.0
        )
        benchmark_data["accuracy_check"] = eval_result.get(
            "accuracy_check", ReportCheckTypes.NA
        )
    else:
        logger.warning("No eval results from video generation test")

    benchmark_data["fvd"] = None
    benchmark_data["fvmd"] = None
    try:
        fvd_and_fvmd_result = _run_video_fvd_and_fvmd_eval()
        if fvd_and_fvmd_result:
            benchmark_data["fvd"] = fvd_and_fvmd_result.get("fvd", 0)
            benchmark_data["fvmd"] = fvd_and_fvmd_result.get("fvmd", 0)
    except Exception as e:
        logger.error(f"Error running video FVD and FVMD eval: {e}")

    benchmark_data["published_score"] = ctx.all_params.tasks[0].score.published_score
    benchmark_data["published_score_ref"] = ctx.all_params.tasks[
        0
    ].score.published_score_ref

    return benchmark_data


__all__ = [
    "run_image_eval",
    "run_audio_eval",
    "run_cnn_eval",
    "run_embedding_eval",
    "run_tts_eval",
    "run_video_eval",
    "IMAGE_EVAL_DISPATCH",
]
