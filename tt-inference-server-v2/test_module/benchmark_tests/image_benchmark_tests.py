# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC


from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Callable

import requests

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from workflows.utils import get_num_calls

from ..context import MediaContext, common_report_metadata, require_health
from ..test_status import ImageGenerationTestStatus

logger = logging.getLogger(__name__)


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


__all__ = ["IMAGE_BENCHMARK_DISPATCH", "run_image_benchmark"]
