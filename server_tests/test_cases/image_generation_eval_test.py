# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Eval test for image generation models (Flux, Motif, SD3.5, etc.)."""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Union

import aiohttp
import requests

from server_tests.base_test import BaseTest
from server_tests.test_cases.server_helper import DEFAULT_AUTHORIZATION
from server_tests.test_classes import TestConfig
from utils.media_clients.test_status import ImageGenerationTestStatus
from utils.sdxl_accuracy_utils.sdxl_accuracy_utils import (
    calculate_accuracy_check,
    calculate_metrics,
    sdxl_get_prompts,
)

logger = logging.getLogger(__name__)


class AccuracyResult(IntEnum):
    """Accuracy check result codes."""

    UNDEFINED = 0
    BASELINE = 1
    PASS = 2
    FAIL = 3


@dataclass(frozen=True)
class ImageGenConfig:
    """Image generation configuration."""

    ENDPOINT: str = "v1/images/generations"
    DEFAULT_INFERENCE_STEPS: int = 20
    DEFAULT_NUM_PROMPTS: int = 100
    NEGATIVE_PROMPT: str = (
        "normal quality, low quality, worst quality, low res, blurry, nsfw, nude"
    )
    GUIDANCE_SCALE: float = 8.0
    SEED: int = 0
    IMAGE_RETURN_FORMAT: str = "PNG"
    IMAGE_QUALITY: int = 100
    REQUEST_TIMEOUT: int = 5000


@dataclass(frozen=True)
class HealthCheckConfig:
    """Health check configuration."""

    MAX_ATTEMPTS: int = 230
    RETRY_DELAY: int = 10
    TIMEOUT: int = 10


CONFIG = ImageGenConfig()
HEALTH_CONFIG = HealthCheckConfig()


SDXL_RESOLUTION_MODEL_NAME_SUFFIX = {
    (512, 512): "-512x512",
}


def resolve_model_name(base_model_name: str, image_resolution: Optional[tuple]) -> str:
    """Append resolution suffix to model name for accuracy reference lookup.

    Only appends a suffix for non-default resolutions (e.g. 512x512).
    The default 1024x1024 uses the base model name unchanged.
    """
    if image_resolution is None:
        return base_model_name
    suffix = SDXL_RESOLUTION_MODEL_NAME_SUFFIX.get(tuple(image_resolution), "")
    return f"{base_model_name}{suffix}"


@dataclass
class ImageGenerationEvalsTestRequest:
    """Request parameters for image generation eval."""

    model_name: str
    num_prompts: int = CONFIG.DEFAULT_NUM_PROMPTS
    start_from: int = 0
    num_inference_steps: int = CONFIG.DEFAULT_INFERENCE_STEPS
    server_url: Optional[str] = None
    request_timeout: Optional[int] = None
    image_resolution: Optional[tuple] = None
    lora_path: Optional[str] = None
    lora_scale: Optional[float] = None


@dataclass
class ImageGenContext:
    """Context for image generation requests."""

    base_url: str
    headers: dict
    num_inference_steps: int
    request_timeout_sec: int
    lora_path: Optional[str] = None
    lora_scale: Optional[float] = None


class ImageGenerationEvalsTest(BaseTest):
    """Eval test for image generation models."""

    def __init__(self, config: TestConfig, targets: dict):
        super().__init__(config, targets)
        self.eval_results: dict = {}

    async def _run_specific_test_async(self) -> dict:
        """Run the image generation evaluation test."""
        request = self._parse_request()
        if isinstance(request, dict):
            return request

        timeout_sec = self._request_timeout_from_config()
        logger.info(
            "Running eval - request parameters: model_name=%s, num_prompts=%s, "
            "start_from=%s, num_inference_steps=%s, server_url=%s, timeout=%s (from config), "
            "lora_path=%s, lora_scale=%s",
            request.model_name,
            request.num_prompts,
            request.start_from,
            request.num_inference_steps,
            request.server_url,
            timeout_sec,
            request.lora_path,
            request.lora_scale,
        )

        prompts = self._load_prompts(request)
        if isinstance(prompts, dict):
            return prompts

        status_list = await self._generate_all_images_async(
            request, prompts, timeout_sec
        )
        if len(status_list) < request.num_prompts:
            return self._error(
                f"ImageGenerationEvalTest only {len(status_list)}/{request.num_prompts} images generated"
            )

        return self._compute_and_check_metrics(request, status_list)

    def _compute_and_check_metrics(
        self,
        request: ImageGenerationEvalsTestRequest,
        status_list: list[ImageGenerationTestStatus],
    ) -> dict:
        """Compute metrics and check accuracy."""
        resolution = request.image_resolution or (1024, 1024)
        logger.info("Step 3: Computing FID and CLIP scores (resolution=%s)", resolution)
        fid_score, avg_clip, std_clip = calculate_metrics(
            status_list, image_resolution=resolution
        )
        logger.info("FID: %.2f, CLIP: %.4f ± %.4f", fid_score, avg_clip, std_clip)

        reference_model_name = resolve_model_name(
            request.model_name, request.image_resolution
        )
        logger.info(
            "Step 4: Checking accuracy against reference (key=%s)",
            reference_model_name,
        )
        accuracy_check = calculate_accuracy_check(
            fid_score, avg_clip, request.num_prompts, reference_model_name
        )

        self.eval_results = {
            "model": request.model_name,
            "num_prompts": request.num_prompts,
            "num_inference_steps": request.num_inference_steps,
            "fid_score": fid_score,
            "average_clip": avg_clip,
            "deviation_clip_score": std_clip,
            "accuracy_check": accuracy_check,
        }

        success = accuracy_check == AccuracyResult.PASS
        result_name = (
            AccuracyResult(accuracy_check).name
            if accuracy_check in AccuracyResult._value2member_map_
            else f"UNKNOWN({accuracy_check})"
        )
        status_icon = "✅" if success else "❌"
        clip_display = f"{avg_clip:.4f} ± {std_clip:.4f}"
        header = (
            f"{'Model':25} {'Prompts':>8} {'Steps':>6} "
            f"{'FID':>10} {'CLIP (avg ± std)':>20} {'Result':>8} {'Status':>8}"
        )
        row = (
            f"{request.model_name:25} {request.num_prompts:>8} {request.num_inference_steps:>6} "
            f"{fid_score:>10.4f} {clip_display:>20} {result_name:>8} {status_icon:>8}"
        )
        logger.info("Image Generation Eval Summary:")
        logger.info(header)
        logger.info("-" * len(header))
        logger.info(row)

        return {
            "success": success,
            "eval_results": self.eval_results,
        }

    def _request_timeout_from_config(self) -> int:
        """Single source for timeout: config or default."""
        return (
            self.timeout
            or self.config.get("test_timeout")
            or self.config.get("timeout")
            or CONFIG.REQUEST_TIMEOUT
        )

    def _parse_request(self) -> Union[ImageGenerationEvalsTestRequest, dict]:
        """Parse and validate request from targets. Timeout comes from config only."""
        raw = self.targets.get("request")

        if raw is None:
            return self._error(
                "ImageGenerationEvalTest request not provided in targets"
            )

        if isinstance(raw, ImageGenerationEvalsTestRequest):
            return raw

        if isinstance(raw, dict):
            try:
                return ImageGenerationEvalsTestRequest(**raw)
            except TypeError as e:
                return self._error(
                    f"ImageGenerationEvalTest invalid request parameters: {e}"
                )

        return self._error(
            "ImageGenerationEvalTest request must be dict or ImageGenerationEvalsTestRequest"
        )

    def _load_prompts(
        self, request: ImageGenerationEvalsTestRequest
    ) -> Union[list[str], dict]:
        logger.info("Step 1: Loading %s COCO prompts", request.num_prompts)

        if (
            len(prompts := sdxl_get_prompts(request.start_from, request.num_prompts))
            < request.num_prompts
        ):
            return self._error(
                f"ImageGenerationEvalTest only got {len(prompts)}/{request.num_prompts} prompts"
            )

        return prompts

    async def _generate_all_images_async(
        self,
        request: ImageGenerationEvalsTestRequest,
        prompts: list[str],
        timeout_sec: int,
    ) -> list[ImageGenerationTestStatus]:
        """Generate images for all prompts concurrently."""
        logger.info("Step 2: Generating %s images concurrently", len(prompts))

        if not self._wait_for_server_ready():
            raise RuntimeError("Server health check failed")

        ctx = ImageGenContext(
            base_url=request.server_url or f"http://localhost:{self.service_port}",
            headers=self._build_headers(),
            num_inference_steps=request.num_inference_steps,
            request_timeout_sec=timeout_sec,
            lora_path=request.lora_path,
            lora_scale=request.lora_scale,
        )

        completion_counter = {"count": 0, "last_time": 0.0}
        total_start = time.perf_counter()
        completion_counter["last_time"] = total_start
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._generate_single_image_async(
                    session,
                    ctx,
                    prompt,
                    idx,
                    len(prompts),
                    total_start,
                    completion_counter,
                )
                for idx, prompt in enumerate(prompts, start=1)
            ]
            results = await asyncio.gather(*tasks)
        total_elapsed = time.perf_counter() - total_start

        status_list = [status for status in results if status is not None]
        logger.info("Generated %s/%s images", len(status_list), len(prompts))
        logger.info("✅ Image generation for eval succeeded in %.2fs", total_elapsed)
        return status_list

    async def _generate_single_image_async(
        self,
        session: aiohttp.ClientSession,
        ctx: ImageGenContext,
        prompt: str,
        idx: int,
        total: int,
        total_start: float,
        completion_counter: dict,
    ) -> Optional[ImageGenerationTestStatus]:
        """Generate a single image asynchronously."""
        logger.info("🌅 Generating image %s/%s: %.50s...", idx, total, prompt)

        try:
            start = time.perf_counter()
            payload = self._build_payload(
                prompt, ctx.num_inference_steps, ctx.lora_path, ctx.lora_scale
            )
            async with session.post(
                f"{ctx.base_url}/{CONFIG.ENDPOINT}",
                json=payload,
                headers=ctx.headers,
                timeout=aiohttp.ClientTimeout(total=ctx.request_timeout_sec),
            ) as response:
                now = time.perf_counter()
                elapsed = now - start
                result = await self._parse_image_response_async(
                    response, elapsed, prompt, idx
                )
                if result is not None:
                    completion_counter["count"] += 1
                    order = completion_counter["count"]
                    since_last = now - completion_counter["last_time"]
                    elapsed_total = now - total_start
                    completion_counter["last_time"] = now
                    logger.info(
                        "✅ %s/%s image generated in %.2fs | elapsed %.2fs",
                        order,
                        total,
                        since_last,
                        elapsed_total,
                    )
                return result

        except asyncio.TimeoutError:
            logger.error("Timeout generating image %s", idx)
        except aiohttp.ClientError as e:
            logger.error("Request error for prompt %s: %s", idx, e)

        return None

    async def _parse_image_response_async(
        self,
        response: aiohttp.ClientResponse,
        elapsed: float,
        prompt: str,
        idx: int,
    ) -> Optional[ImageGenerationTestStatus]:
        """Parse image response."""
        if response.status != 200:
            logger.error("Failed: status %s", response.status)
            return None

        data = await response.json()
        images = data.get("images", [])
        if not images:
            logger.warning("No image in response for prompt %s", idx)
            return None

        return ImageGenerationTestStatus(
            status=True,
            elapsed=elapsed,
            base64image=images[0],
            prompt=prompt,
        )

    def _wait_for_server_ready(self) -> bool:
        """Wait for server to be ready."""
        health_url = f"http://localhost:{self.service_port}/tt-liveness"
        logger.info("Waiting for server: %s", health_url)

        for attempt in range(1, HEALTH_CONFIG.MAX_ATTEMPTS + 1):
            if self._check_health(health_url):
                logger.info("Server ready after %s attempt(s)", attempt)
                return True

            logger.debug(
                "Not ready (attempt %s/%s)", attempt, HEALTH_CONFIG.MAX_ATTEMPTS
            )
            time.sleep(HEALTH_CONFIG.RETRY_DELAY)

        logger.error("Server not ready after %s attempts", HEALTH_CONFIG.MAX_ATTEMPTS)
        return False

    def _check_health(self, url: str) -> bool:
        """Single health check attempt."""
        try:
            response = requests.get(url, timeout=HEALTH_CONFIG.TIMEOUT)
        except requests.RequestException:
            return False

        if response.status_code != 200:
            return False

        data = response.json()
        return data.get("status") == "alive" and data.get("model_ready", False)

    @staticmethod
    def _build_headers() -> dict:
        """Build request headers."""
        return {
            "accept": "application/json",
            "Authorization": f"Bearer {DEFAULT_AUTHORIZATION}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _build_payload(
        prompt: str,
        num_inference_steps: int,
        lora_path: Optional[str] = None,
        lora_scale: Optional[float] = None,
    ) -> dict:
        """Build image generation payload."""
        payload = {
            "prompt": prompt,
            "negative_prompt": CONFIG.NEGATIVE_PROMPT,
            "num_inference_steps": num_inference_steps,
            "seed": CONFIG.SEED,
            "guidance_scale": CONFIG.GUIDANCE_SCALE,
            "image_return_format": CONFIG.IMAGE_RETURN_FORMAT,
            "image_quality": CONFIG.IMAGE_QUALITY,
            "number_of_images": 1,
        }
        if lora_path is not None:
            payload["lora_path"] = lora_path
        if lora_scale is not None:
            payload["lora_scale"] = lora_scale
        return payload

    @staticmethod
    def _error(message: str) -> dict:
        """Create error response."""
        return {"success": False, "error": message}
