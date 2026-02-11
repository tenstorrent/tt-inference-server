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

from tests.server_tests.base_test import BaseTest
from tests.server_tests.test_cases.server_helper import DEFAULT_AUTHORIZATION
from tests.server_tests.test_classes import TestConfig
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

    ENDPOINT: str = "image/generations"
    DEFAULT_INFERENCE_STEPS: int = 20
    DEFAULT_NUM_PROMPTS: int = 100
    NEGATIVE_PROMPT: str = (
        "normal quality, low quality, worst quality, low res, blurry, nsfw, nude"
    )
    GUIDANCE_SCALE: float = 8.0
    SEED: int = 0
    REQUEST_TIMEOUT: int = 300


@dataclass(frozen=True)
class HealthCheckConfig:
    """Health check configuration."""

    MAX_ATTEMPTS: int = 230
    RETRY_DELAY: int = 10
    TIMEOUT: int = 10


CONFIG = ImageGenConfig()
HEALTH_CONFIG = HealthCheckConfig()


@dataclass
class ImageGenerationEvalsTestRequest:
    """Request parameters for image generation eval."""

    model_name: str
    num_prompts: int = CONFIG.DEFAULT_NUM_PROMPTS
    start_from: int = 0
    num_inference_steps: int = CONFIG.DEFAULT_INFERENCE_STEPS
    server_url: Optional[str] = None
    request_timeout: Optional[int] = None


@dataclass
class ImageGenContext:
    """Context for image generation requests."""

    base_url: str
    headers: dict
    num_inference_steps: int
    request_timeout_sec: int


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

        logger.info(
            "Running eval - request parameters: model_name=%s, num_prompts=%s, "
            "start_from=%s, num_inference_steps=%s, server_url=%s, request_timeout=%s",
            request.model_name,
            request.num_prompts,
            request.start_from,
            request.num_inference_steps,
            request.server_url,
            request.request_timeout,
        )

        prompts = self._load_prompts(request)
        if isinstance(prompts, dict):
            return prompts

        status_list = await self._generate_all_images_async(request, prompts)
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
        logger.info("Step 3: Computing FID and CLIP scores")
        fid_score, avg_clip, std_clip = calculate_metrics(status_list)
        logger.info("FID: %.2f, CLIP: %.4f ± %.4f", fid_score, avg_clip, std_clip)

        logger.info("Step 4: Checking accuracy against reference")
        accuracy_check = calculate_accuracy_check(
            fid_score, avg_clip, request.num_prompts, request.model_name
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

        return {
            "success": accuracy_check == AccuracyResult.PASS,
            "eval_results": self.eval_results,
        }

    def _parse_request(self) -> Union[ImageGenerationEvalsTestRequest, dict]:
        """Parse and validate request from targets."""
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
    ) -> list[ImageGenerationTestStatus]:
        """Generate images for all prompts concurrently."""
        logger.info("Step 2: Generating %s images concurrently", len(prompts))

        if not self._wait_for_server_ready():
            raise RuntimeError("Server health check failed")

        config_timeout = self.timeout or self.config.get("test_timeout")
        timeout_sec = (
            request.request_timeout
            if request.request_timeout is not None
            else (
                config_timeout if config_timeout is not None else CONFIG.REQUEST_TIMEOUT
            )
        )
        ctx = ImageGenContext(
            base_url=request.server_url or f"http://localhost:{self.service_port}",
            headers=self._build_headers(),
            num_inference_steps=request.num_inference_steps,
            request_timeout_sec=timeout_sec,
        )

        total_start = time.perf_counter()
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._generate_single_image_async(
                    session, ctx, prompt, idx, len(prompts)
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
    ) -> Optional[ImageGenerationTestStatus]:
        """Generate a single image asynchronously."""
        logger.info("Generating image %s/%s: %.50s...", idx, total, prompt)

        try:
            start = time.perf_counter()
            payload = self._build_payload(prompt, ctx.num_inference_steps)
            logger.debug(
                "Request for image %s/%s: url=%s, payload_keys=%s",
                idx,
                total,
                f"{ctx.base_url}/{CONFIG.ENDPOINT}",
                list(payload.keys()),
            )
            async with session.post(
                f"{ctx.base_url}/{CONFIG.ENDPOINT}",
                json=payload,
                headers=ctx.headers,
                timeout=aiohttp.ClientTimeout(total=ctx.request_timeout_sec),
            ) as response:
                elapsed = time.perf_counter() - start
                return await self._parse_image_response_async(
                    response, elapsed, prompt, idx
                )

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
    def _build_payload(prompt: str, num_inference_steps: int) -> dict:
        """Build image generation payload."""
        return {
            "prompt": prompt,
            "negative_prompt": CONFIG.NEGATIVE_PROMPT,
            "num_inference_steps": num_inference_steps,
            "seed": CONFIG.SEED,
            "guidance_scale": CONFIG.GUIDANCE_SCALE,
            "number_of_images": 1,
        }

    @staticmethod
    def _error(message: str) -> dict:
        """Create error response."""
        return {"success": False, "error": message}
