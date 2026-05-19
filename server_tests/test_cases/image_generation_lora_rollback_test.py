# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Rollback test for LoRA adapter loading.

Issues a sequence:
  1. baseline (no LoRA) → record image B1
  2. with LoRA fused   → record image L
  3. baseline (no LoRA) → record image B2

and asserts:
  * all three requests return 200,
  * L != B1 (LoRA was actually applied),
  * B2 == B1 (LoRA was cleanly unloaded — the runner returned to the pre-LoRA state).

This catches regressions where _ensure_lora_state's unload path leaves the
runner in a partially-mutated state.
"""

import logging
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
DEFAULT_INFERENCE_STEPS = 20
REQUEST_TIMEOUT_SEC = 5000


@dataclass
class LoraSpec:
    lora_path: str
    lora_scale: float


@dataclass
class StepResult:
    label: str
    status_code: int
    duration: float
    image_data: Optional[str] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.status_code == 200 and self.image_data is not None


class ImageGenerationLoraRollbackTest(BaseTest):
    """Sequential baseline -> LoRA -> baseline test."""

    def __init__(self, config: TestConfig, targets: dict, description: str = ""):
        super().__init__(config, targets, description)

    async def _run_specific_test_async(self) -> dict:
        lora = self._parse_lora()
        num_inference_steps = self.targets.get(
            "num_inference_steps", DEFAULT_INFERENCE_STEPS
        )

        base_url = f"http://localhost:{self.service_port}"
        url = f"{base_url}/{ENDPOINT}"
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {DEFAULT_AUTHORIZATION}",
            "Content-Type": "application/json",
        }
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SEC)

        async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
            b1 = await self._send(session, url, num_inference_steps, "baseline-pre")
            l = await self._send(
                session, url, num_inference_steps, "with-lora", lora=lora
            )
            b2 = await self._send(session, url, num_inference_steps, "baseline-post")

        return self._evaluate(b1, l, b2)

    def _parse_lora(self) -> LoraSpec:
        raw = self.targets.get("lora")
        if not raw:
            raise ValueError("targets.lora must be set (lora_path, lora_scale)")
        return LoraSpec(lora_path=raw["lora_path"], lora_scale=raw["lora_scale"])

    async def _send(
        self,
        session: aiohttp.ClientSession,
        url: str,
        num_inference_steps: int,
        label: str,
        lora: Optional[LoraSpec] = None,
    ) -> StepResult:
        payload = {
            "prompt": FIXED_PROMPT,
            "negative_prompt": "blurry, low quality, distorted",
            "num_inference_steps": num_inference_steps,
            "seed": FIXED_SEED,
            "guidance_scale": 7.5,
            "number_of_images": 1,
        }
        if lora is not None:
            payload["lora_path"] = lora.lora_path
            payload["lora_scale"] = lora.lora_scale

        logger.info("Sending %s request", label)
        start = time.perf_counter()
        try:
            async with session.post(url, json=payload) as response:
                duration = time.perf_counter() - start
                if response.status == 200:
                    data = await response.json()
                    images = data.get("images", [])
                    return StepResult(
                        label=label,
                        status_code=200,
                        duration=duration,
                        image_data=images[0] if images else None,
                    )
                body = await response.text()
                return StepResult(
                    label=label,
                    status_code=response.status,
                    duration=duration,
                    error=body[:500],
                )
        except Exception as exc:
            return StepResult(
                label=label,
                status_code=0,
                duration=time.perf_counter() - start,
                error=str(exc),
            )

    def _evaluate(
        self, b1: StepResult, l: StepResult, b2: StepResult
    ) -> dict:
        failed = [r for r in (b1, l, b2) if not r.success]
        if failed:
            return {
                "success": False,
                "error": "one or more requests failed",
                "failures": [
                    f"{r.label}: status={r.status_code} err={r.error}" for r in failed
                ],
            }

        lora_differs = l.image_data != b1.image_data
        rollback_clean = b1.image_data == b2.image_data

        if not lora_differs:
            logger.error("LoRA produced the same image as baseline-pre — adapter not applied")
        if not rollback_clean:
            logger.error(
                "baseline-post differs from baseline-pre — LoRA was not cleanly unloaded"
            )

        success = lora_differs and rollback_clean
        return {
            "success": success,
            "lora_differs_from_baseline": lora_differs,
            "rollback_clean": rollback_clean,
            "durations": {
                "baseline_pre": round(b1.duration, 2),
                "with_lora": round(l.duration, 2),
                "baseline_post": round(b2.duration, 2),
            },
        }
