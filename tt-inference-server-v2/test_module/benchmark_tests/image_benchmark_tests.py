# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC


from __future__ import annotations

import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import requests

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from report_module.schema import Block

from workflows.utils import get_num_calls

from .._test_common import (
    MetricSpec,
    ReportCheckTypes,
    block_id,
    run_tiered_check,
)
from ..context import MediaContext, require_health
from ..test_status import ImageGenerationTestStatus

logger = logging.getLogger(__name__)


SDXL_BENCHMARK_NUM_BATCHES = 3

SDXL_SD35_BENCHMARK_NUM_PROMPTS = 20
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

# Z-Image-Turbo is a Decoupled-DMD distilled model: 8 NFEs (≈9 scheduler steps),
# guidance_scale must be 0.0 — non-zero CFG degrades quality on Turbo variants.
# The TT runner hard-codes steps=9 internally; we mirror it for honest reporting.
Z_IMAGE_TURBO_INFERENCE_STEPS = 9
Z_IMAGE_TURBO_GUIDANCE_SCALE = 0.0
Z_IMAGE_TURBO_BENCHMARK_NUM_PROMPTS = 5
Z_IMAGE_TURBO_PROMPTS_PAYLOAD = "tt-z-image-turbo_payload.json"
Z_IMAGE_TURBO_BASE_SEED = 42

_DATASETS_AND_PAYLOADS_DIR = (
    Path(__file__).resolve().parent.parent / "datasets_and_payloads"
)


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
    concurrency: int = 1,
) -> list[ImageGenerationTestStatus]:
    concurrency = max(1, concurrency)

    def _one(call_index: int) -> ImageGenerationTestStatus:
        logger.info(f"Generating image {call_index + 1}/{num_calls}...")
        if generator_steps_kwarg:
            status, elapsed = generator(ctx, inference_steps)
        else:
            status, elapsed = generator(ctx)
        inference_steps_per_second = inference_steps / elapsed if elapsed > 0 else 0
        logger.info(
            f"Generated image with {inference_steps} steps in {elapsed:.2f} seconds."
        )
        return ImageGenerationTestStatus(
            status=status,
            elapsed=elapsed,
            num_inference_steps=inference_steps,
            inference_steps_per_second=inference_steps_per_second,
        )

    status_list: list[ImageGenerationTestStatus] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for batch_start in range(0, num_calls, concurrency):
            batch = range(batch_start, min(batch_start + concurrency, num_calls))
            futures = [pool.submit(_one, i) for i in batch]
            status_list.extend(f.result() for f in futures)
    return status_list


def _run_sdxl_image_generation_benchmark(
    ctx: MediaContext, num_calls: int, concurrency: int = 1
) -> list[ImageGenerationTestStatus]:
    logger.info("Running image generation benchmark.")
    return _build_image_status_list(
        ctx,
        num_calls,
        SDXL_SD35_INFERENCE_STEPS,
        _generate_image,
        concurrency=concurrency,
    )


def _run_img2img_generation_benchmark(
    ctx: MediaContext, num_calls: int, concurrency: int = 1
) -> list[ImageGenerationTestStatus]:
    logger.info("Running image-to-image generation benchmark.")
    return _build_image_status_list(
        ctx,
        num_calls,
        SDXL_IMG2IMG_INFERENCE_STEPS,
        _generate_image_img2img,
        concurrency=concurrency,
    )


def _run_inpainting_generation_benchmark(
    ctx: MediaContext, num_calls: int, concurrency: int = 1
) -> list[ImageGenerationTestStatus]:
    logger.info("Running inpainting generation benchmark.")
    return _build_image_status_list(
        ctx,
        num_calls,
        SDXL_INPAINTING_INFERENCE_STEPS,
        _generate_image_inpainting,
        concurrency=concurrency,
    )


def _run_flux_1_dev_benchmark(
    ctx: MediaContext, num_calls: int, concurrency: int = 1
) -> list[ImageGenerationTestStatus]:
    logger.info("Running Flux 1 Dev or Schnell benchmark.")
    return _build_image_status_list(
        ctx,
        num_calls,
        FLUX_MOTIF_INFERENCE_STEPS,
        _generate_image,
        generator_steps_kwarg=True,
        concurrency=concurrency,
    )


def _run_flux_1_schnell_benchmark(
    ctx: MediaContext, num_calls: int, concurrency: int = 1
) -> list[ImageGenerationTestStatus]:
    logger.info("Running Flux 1 Schnell benchmark.")
    return _build_image_status_list(
        ctx,
        num_calls,
        FLUX_1_SCHNELL_INFERENCE_STEPS,
        _generate_image,
        generator_steps_kwarg=True,
        concurrency=concurrency,
    )


def _run_motif_image_6b_preview_benchmark(
    ctx: MediaContext, num_calls: int, concurrency: int = 1
) -> list[ImageGenerationTestStatus]:
    logger.info("Running Motif Image 6B Preview benchmark.")
    return _build_image_status_list(
        ctx,
        num_calls,
        FLUX_MOTIF_INFERENCE_STEPS,
        _generate_image,
        generator_steps_kwarg=True,
        concurrency=concurrency,
    )


def _load_z_image_turbo_prompts() -> list[str]:
    path = _DATASETS_AND_PAYLOADS_DIR / Z_IMAGE_TURBO_PROMPTS_PAYLOAD
    with open(path, "r") as f:
        data = json.load(f)
    prompts = data.get("prompts") or []
    if not prompts:
        raise RuntimeError(f"No prompts found in {path}")
    return prompts


def _generate_image_z_image_turbo(
    ctx: MediaContext,
    prompt: str,
    seed: int,
    num_inference_steps: int = Z_IMAGE_TURBO_INFERENCE_STEPS,
) -> tuple[bool, float]:
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer your-secret-key",
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": prompt,
        "seed": seed,
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
        logger.error(
            f"❌ Z-Image-Turbo generation failed (status {response.status_code})"
        )
        try:
            logger.error(f"Error details: {response.json()}")
        except Exception as e:
            logger.error(f"Could not parse error response: {e}")
            logger.error(f"Raw response: {response.text[:500]}")
        raise RuntimeError(
            f"Z-Image-Turbo generation failed with status {response.status_code}"
        )

    logger.info(f"✅ Z-Image-Turbo generation successful in {elapsed:.2f}s")
    return True, elapsed


def _run_z_image_turbo_benchmark(
    ctx: MediaContext, num_calls: int, concurrency: int = 1
) -> list[ImageGenerationTestStatus]:
    # concurrency is accepted for dispatch signature compatibility but Z-Image-Turbo
    # is run sequentially to keep deterministic seed/prompt pairing.
    del concurrency
    logger.info("Running Z-Image-Turbo benchmark.")
    prompts = _load_z_image_turbo_prompts()

    status_list: list[ImageGenerationTestStatus] = []
    steps = Z_IMAGE_TURBO_INFERENCE_STEPS
    for i in range(num_calls):
        prompt = prompts[i % len(prompts)]
        seed = Z_IMAGE_TURBO_BASE_SEED + i
        logger.info(
            f"Generating image {i + 1}/{num_calls} (seed={seed}): {prompt[:60]}..."
        )
        status, elapsed = _generate_image_z_image_turbo(ctx, prompt, seed, steps)
        inference_steps_per_second = steps / elapsed if elapsed > 0 else 0
        logger.info(f"Generated image with {steps} steps in {elapsed:.2f} seconds.")
        status_list.append(
            ImageGenerationTestStatus(
                status=status,
                elapsed=elapsed,
                num_inference_steps=steps,
                inference_steps_per_second=inference_steps_per_second,
            )
        )
    return status_list


IMAGE_BENCHMARK_DISPATCH: dict[
    str, Callable[[MediaContext, int, int], list[ImageGenerationTestStatus]]
] = {
    "tt-sdxl-trace": _run_sdxl_image_generation_benchmark,
    "tt-sdxl-image-to-image": _run_img2img_generation_benchmark,
    "tt-sdxl-edit": _run_inpainting_generation_benchmark,
    "tt-sd3.5": _run_sdxl_image_generation_benchmark,
    "tt-flux.1-dev": _run_flux_1_dev_benchmark,
    "tt-flux.1-schnell": _run_flux_1_schnell_benchmark,
    "tt-motif-image-6b-preview": _run_motif_image_6b_preview_benchmark,
    "tt-z-image-turbo": _run_z_image_turbo_benchmark,
}


_SDXL_BENCHMARK_RUNNERS = {
    "tt-sdxl-trace",
    "tt-sdxl-image-to-image",
    "tt-sdxl-edit",
}


def _image_ttft(status_list: list[ImageGenerationTestStatus]) -> float:
    logger.info("Calculating TTFT value")
    return sum(s.elapsed for s in status_list) / len(status_list) if status_list else 0


def _image_target_checks(
    ctx: MediaContext, ttft_seconds: float, tput_user: float
) -> tuple[dict, ReportCheckTypes]:
    # ttft is captured in seconds (time.time() deltas); targets.ttft_ms is ms.
    ttft_ms = ttft_seconds * 1000 if ttft_seconds else None
    logger.info("Computing 3-tier target checks for TTFT, tput_user")
    return run_tiered_check(
        ctx,
        [
            MetricSpec(
                "TTFT", ttft_ms, "ttft_ms", lower_is_better=True, field_name="ttft_ms"
            ),
            MetricSpec(
                "tput_user",
                tput_user,
                "tput_user",
                lower_is_better=False,
                field_name="tput_user",
            ),
        ],
    )


def run_image_benchmark(ctx: MediaContext) -> Block:
    """Run benchmarks for an image model (SDXL, SD3.5, Flux, Motif, Z-Image-Turbo)."""
    logger.info(
        f"Running benchmarks for model: {ctx.model_spec.model_name} on device: {ctx.device.name}"
    )

    runner_in_use = require_health(ctx)

    try:
        num_calls = get_num_calls(ctx)
        max_concurrency = None
        if runner_in_use in _SDXL_BENCHMARK_RUNNERS:
            max_concurrency = ctx.model_spec.device_model_spec.max_concurrency
            if max_concurrency and max_concurrency > 0:
                num_calls = SDXL_BENCHMARK_NUM_BATCHES * max_concurrency
                logger.info(
                    f"Overriding num_calls for {runner_in_use} to {num_calls} prompts "
                    f"({SDXL_BENCHMARK_NUM_BATCHES} batches x {max_concurrency} concurrent requests)"
                )
            else:
                max_concurrency = None
        elif runner_in_use == "tt-z-image-turbo":
            logger.info(
                f"Overriding num_calls for Z-Image-Turbo to "
                f"{Z_IMAGE_TURBO_BENCHMARK_NUM_PROMPTS} prompts"
            )
            num_calls = Z_IMAGE_TURBO_BENCHMARK_NUM_PROMPTS

        benchmark_fn = IMAGE_BENCHMARK_DISPATCH.get(
            runner_in_use, _run_sdxl_image_generation_benchmark
        )
        status_list = benchmark_fn(ctx, num_calls, max_concurrency or 1)
    except Exception as e:
        logger.error(f"Benchmark execution encountered an error: {e}")
        raise

    logger.info("Generating benchmark report...")
    ttft_value = _image_ttft(status_list)
    inference_steps_per_second = (
        sum(s.inference_steps_per_second for s in status_list) / len(status_list)
        if status_list
        else 0
    )
    total_elapsed = sum(s.elapsed for s in status_list)
    # tput_user is per-user image throughput (images/sec = 1 / avg latency).
    tput_user = len(status_list) / total_elapsed if total_elapsed > 0 else 0
    target_checks, accuracy_check = _image_target_checks(ctx, ttft_value, tput_user)
    num_inference_steps_used = status_list[0].num_inference_steps if status_list else 0
    benchmarks_data = {
        "num_requests": len(status_list),
    }
    if max_concurrency:
        benchmarks_data["num_concurrent_requests"] = max_concurrency
    benchmarks_data.update(
        {
            "num_inference_steps": num_inference_steps_used,
            "ttft_ms": ttft_value * 1000,
            "inference_steps_per_second": inference_steps_per_second,
            "tput_user": tput_user,
            "accuracy_check": accuracy_check,
            "target_checks": target_checks,
        }
    )
    return Block(
        kind="benchmarks",
        task_type="image",
        title="Image Benchmark",
        id=block_id(ctx) or None,
        targets={
            "num_prompts": len(status_list),
            "num_inference_steps": num_inference_steps_used,
        },
        data={
            "Benchmarks": benchmarks_data,
        },
    )


__all__ = ["IMAGE_BENCHMARK_DISPATCH", "run_image_benchmark"]
