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
from typing import Any, Awaitable, Callable, Optional

import aiohttp

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from .image_generation_eval_test import ImageGenerationEvalsTest
from server_tests.test_classes import TestConfig as ServerTestConfig
from utils.sdxl_accuracy_utils.sdxl_accuracy_utils import (
    calculate_accuracy_check,
    calculate_metrics,
    sdxl_get_prompts,
)
from workflows.utils import is_sdxl_num_prompts_enabled

from ..context import MediaContext, common_eval_metadata, require_health
from ..test_status import ImageGenerationTestStatus

logger = logging.getLogger(__name__)


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
    return sum(s.elapsed for s in status_list) / len(status_list) if status_list else 0


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
        tasks = [_generate_image_eval_async(ctx, session, prompt) for prompt in prompts]
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

    with open(f"{ctx.test_payloads_path}/image_client_inpainting_payload", "r") as f:
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

    runner_in_use = require_health(ctx)

    eval_method = IMAGE_EVAL_DISPATCH.get(runner_in_use, _run_image_generation_eval)
    try:
        eval_result = asyncio.run(eval_method(ctx, runner_in_use))
    except Exception as e:
        logger.error(f"Eval execution encountered an error: {e}")
        raise

    logger.info("Generating eval report...")
    benchmark_data = common_eval_metadata(ctx, "image")
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


__all__ = ["IMAGE_EVAL_DISPATCH", "run_image_eval"]
