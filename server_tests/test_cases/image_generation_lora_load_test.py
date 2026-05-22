# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Load test for mixed LoRA and baseline image generation requests.

Fires a batch of concurrent requests — some baseline (no LoRA), some with
each configured LoRA adapter — and validates that:
  1. All requests succeed (HTTP 200).
  2. LoRA responses differ from baseline (LoRA is actually applied).
  3. Batch completes within a target wall-clock time.
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import Optional

import aiohttp

from server_tests.base_test import BaseTest
from server_tests.test_cases.server_helper import DEFAULT_AUTHORIZATION
from server_tests.test_classes import TestConfig

logger = logging.getLogger(__name__)

ENDPOINT = "v1/images/generations"
FIXED_PROMPT = "A beautiful sunset over a mountain landscape with vibrant colors"
FIXED_SEED = 42
DEFAULT_BATCH_SIZE = 10
DEFAULT_TARGET_TIME = 120
DEFAULT_INFERENCE_STEPS = 20
REQUEST_TIMEOUT_SEC = 5000


@dataclass(frozen=True)
class LoraSpec:
    lora_path: str
    lora_scale: float


@dataclass
class RequestSpec:
    """Single request in the batch, carrying its LoRA config (or None for baseline)."""

    index: int
    lora: Optional[LoraSpec] = None


@dataclass
class RequestResult:
    index: int
    lora: Optional[LoraSpec]
    status_code: int
    duration: float
    image_data: Optional[str] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.status_code == 200 and self.image_data is not None


class ImageGenerationLoraLoadTest(BaseTest):
    """Concurrent load test mixing baseline and LoRA image-generation requests."""

    def __init__(self, config: TestConfig, targets: dict, description: str = ""):
        super().__init__(config, targets, description)

    async def _run_specific_test_async(self) -> dict:
        lora_configs = self._parse_lora_configs()
        batch_size = self.targets.get("batch_size", DEFAULT_BATCH_SIZE)
        target_time = self.targets.get("target_time", DEFAULT_TARGET_TIME)
        num_inference_steps = self.targets.get(
            "num_inference_steps", DEFAULT_INFERENCE_STEPS
        )

        specs = self._build_request_batch(lora_configs, batch_size)
        logger.info(
            "Firing %d concurrent requests (%d baseline, %s LoRA) with %d inference steps",
            len(specs),
            sum(1 for s in specs if s.lora is None),
            ", ".join(
                f"{sum(1 for s in specs if s.lora == lc)}x {lc.lora_path}"
                for lc in lora_configs
            ),
            num_inference_steps,
        )

        results = await self._fire_batch(specs, num_inference_steps)
        return self._evaluate(results, lora_configs, target_time)

    def _parse_lora_configs(self) -> list[LoraSpec]:
        raw = self.targets.get("lora_configs", [])
        if not raw:
            raise ValueError("lora_configs must be a non-empty list in targets")
        return [
            LoraSpec(lora_path=c["lora_path"], lora_scale=c["lora_scale"]) for c in raw
        ]

    @staticmethod
    def _build_request_batch(
        lora_configs: list[LoraSpec], batch_size: int
    ) -> list[RequestSpec]:
        num_variants = len(lora_configs) + 1  # +1 for baseline
        per_variant = max(1, batch_size // num_variants)

        specs: list[RequestSpec] = []
        for _ in range(per_variant):
            specs.append(RequestSpec(index=len(specs)))
        for lora in lora_configs:
            for _ in range(per_variant):
                specs.append(RequestSpec(index=len(specs), lora=lora))

        while len(specs) < batch_size:
            specs.append(RequestSpec(index=len(specs)))

        random.shuffle(specs)
        for i, spec in enumerate(specs):
            spec.index = i
        return specs

    async def _fire_batch(
        self, specs: list[RequestSpec], num_inference_steps: int
    ) -> list[RequestResult]:
        base_url = f"http://localhost:{self.service_port}"
        url = f"{base_url}/{ENDPOINT}"
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {DEFAULT_AUTHORIZATION}",
            "Content-Type": "application/json",
        }
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SEC)

        async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
            tasks = [
                self._send_request(session, url, spec, num_inference_steps)
                for spec in specs
            ]
            return await asyncio.gather(*tasks)

    async def _send_request(
        self,
        session: aiohttp.ClientSession,
        url: str,
        spec: RequestSpec,
        num_inference_steps: int,
    ) -> RequestResult:
        payload = {
            "prompt": FIXED_PROMPT,
            "negative_prompt": "blurry, low quality, distorted",
            "num_inference_steps": num_inference_steps,
            "seed": FIXED_SEED,
            "guidance_scale": 7.5,
            "number_of_images": 1,
        }
        if spec.lora is not None:
            payload["lora_path"] = spec.lora.lora_path
            payload["lora_scale"] = spec.lora.lora_scale

        lora_label = spec.lora.lora_path if spec.lora else "baseline"
        logger.info("[%d] Sending request (%s)", spec.index, lora_label)

        try:
            start = time.perf_counter()
            async with session.post(url, json=payload) as response:
                duration = time.perf_counter() - start
                if response.status == 200:
                    data = await response.json()
                    images = data.get("images", [])
                    image_data = images[0] if images else None
                    logger.info(
                        "[%d] %s completed in %.2fs", spec.index, lora_label, duration
                    )
                    return RequestResult(
                        index=spec.index,
                        lora=spec.lora,
                        status_code=200,
                        duration=duration,
                        image_data=image_data,
                    )
                else:
                    body = await response.text()
                    logger.error(
                        "[%d] HTTP %d: %s", spec.index, response.status, body[:500]
                    )
                    return RequestResult(
                        index=spec.index,
                        lora=spec.lora,
                        status_code=response.status,
                        duration=time.perf_counter() - start,
                        error=body[:500],
                    )
        except Exception as exc:
            duration = time.perf_counter() - start
            logger.error("[%d] Exception after %.2fs: %s", spec.index, duration, exc)
            return RequestResult(
                index=spec.index,
                lora=spec.lora,
                status_code=0,
                duration=duration,
                error=str(exc),
            )

    def _evaluate(
        self,
        results: list[RequestResult],
        lora_configs: list[LoraSpec],
        target_time: float,
    ) -> dict:
        failed = [r for r in results if not r.success]
        if failed:
            labels = [
                f"[{r.index}] status={r.status_code} error={r.error}" for r in failed
            ]
            logger.error("%d/%d requests failed: %s", len(failed), len(results), labels)
            return {
                "success": False,
                "error": f"{len(failed)} request(s) failed",
                "failures": labels,
            }

        max_duration = max(r.duration for r in results)
        avg_duration = sum(r.duration for r in results) / len(results)
        logger.info(
            "All %d requests succeeded — max=%.2fs, avg=%.2fs, target=%.2fs",
            len(results),
            max_duration,
            avg_duration,
            target_time,
        )

        baseline_images = {
            r.image_data for r in results if r.lora is None and r.image_data
        }
        if not baseline_images:
            return {
                "success": False,
                "error": "No successful baseline responses to compare against",
            }

        differentiation_results = {}
        all_differentiated = True
        for lora in lora_configs:
            lora_images = {
                r.image_data for r in results if r.lora == lora and r.image_data
            }
            differs = not lora_images.issubset(baseline_images)
            differentiation_results[lora.lora_path] = differs
            if not differs:
                logger.error(
                    "LoRA %s produced identical images to baseline — adapter not applied",
                    lora.lora_path,
                )
                all_differentiated = False
            else:
                logger.info(
                    "LoRA %s correctly produced different images", lora.lora_path
                )

        within_target = max_duration <= target_time
        if not within_target:
            logger.warning(
                "Batch exceeded target time: %.2fs > %.2fs", max_duration, target_time
            )

        success = all_differentiated and within_target and not failed
        return {
            "success": success,
            "total_requests": len(results),
            "max_duration": round(max_duration, 2),
            "avg_duration": round(avg_duration, 2),
            "target_time": target_time,
            "within_target": within_target,
            "lora_differentiation": differentiation_results,
        }
