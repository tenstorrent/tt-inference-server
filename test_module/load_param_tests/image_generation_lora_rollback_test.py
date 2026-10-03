# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Rollback test for LoRA adapter loading.

Issues a strictly sequential sequence against a fixed prompt and seed:

  1. baseline (no LoRA) -> image B1
  2. with LoRA fused    -> image L
  3. baseline (no LoRA) -> image B2

and asserts:
  * all three requests return 200,
  * L != B1 -- the adapter was actually applied,
  * B2 == B1 -- the adapter was cleanly removed and the runner is back in its
    pre-LoRA state.

The second assertion is what the concurrent ImageGenerationLoraLoadTest cannot
check: it fires a mixed batch and only verifies that LoRA output differs from
baseline, so a runner that applies an adapter and never unloads it still
passes. Only a sequential baseline -> LoRA -> baseline walk catches a leaked
adapter or a half-reverted state.
"""

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import aiohttp
from report_module.schema import Block

from .._test_common import BaseTest, HardwareRequirement, TestConfig

if TYPE_CHECKING:
    from ..context import MediaContext
from .server_helper import DEFAULT_AUTHORIZATION

logger = logging.getLogger(__name__)

ENDPOINT = "v1/images/generations"
FIXED_PROMPT = "A beautiful sunset over a mountain landscape with vibrant colors"
FIXED_SEED = 42
DEFAULT_INFERENCE_STEPS = 20
REQUEST_TIMEOUT_SEC = 5000


@dataclass(frozen=True)
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
    KIND = "image_generation_lora_rollback"
    TASK_TYPE = "image"
    HARDWARE_REQUIREMENT = HardwareRequirement.FULL_BOARD

    """Sequential baseline -> LoRA -> baseline rollback test."""

    def __init__(
        self,
        config: TestConfig,
        targets: dict,
        description: str = "",
        ctx: Optional["MediaContext"] = None,
    ):
        super().__init__(config, targets, description, ctx=ctx)

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

        # Sequential on purpose: the point is the runner's state transitions,
        # so these must not overlap or be reordered by the batcher.
        async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
            baseline_pre = await self._send(
                session, url, num_inference_steps, "baseline-pre"
            )
            with_lora = await self._send(
                session, url, num_inference_steps, "with-lora", lora=lora
            )
            baseline_post = await self._send(
                session, url, num_inference_steps, "baseline-post"
            )

        return self._evaluate(baseline_pre, with_lora, baseline_post)

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
        self,
        baseline_pre: StepResult,
        with_lora: StepResult,
        baseline_post: StepResult,
    ) -> dict:
        steps = (baseline_pre, with_lora, baseline_post)
        failed = [r for r in steps if not r.success]
        if failed:
            labels = [
                f"{r.label}: status={r.status_code} err={r.error}" for r in failed
            ]
            logger.error("%d/3 requests failed: %s", len(failed), labels)
            return {
                "success": False,
                "error": f"{len(failed)} request(s) failed",
                "failures": labels,
            }

        lora_differs = with_lora.image_data != baseline_pre.image_data
        rollback_clean = baseline_pre.image_data == baseline_post.image_data

        if not lora_differs:
            logger.error(
                "LoRA produced the same image as baseline-pre — adapter not applied"
            )
        else:
            logger.info("LoRA output differs from baseline — adapter applied")

        if not rollback_clean:
            logger.error(
                "baseline-post differs from baseline-pre — LoRA was not cleanly unloaded"
            )
        else:
            logger.info("baseline-post matches baseline-pre — rollback is clean")

        return {
            "success": lora_differs and rollback_clean,
            "lora_differs_from_baseline": lora_differs,
            "rollback_clean": rollback_clean,
            "durations": {
                "baseline_pre": round(baseline_pre.duration, 2),
                "with_lora": round(with_lora.duration, 2),
                "baseline_post": round(baseline_post.duration, 2),
            },
        }


def run_image_generation_lora_rollback(
    ctx: "MediaContext", targets: dict | None = None
) -> Block:
    """Run :class:`ImageGenerationLoraRollbackTest` under ``ctx`` and return its Block."""
    test_config = TestConfig(
        {
            "timeout": 1800,
            "retry_attempts": 1,
            "retry_delay": 10,
            "break_on_failure": False,
        }
    )
    return ImageGenerationLoraRollbackTest(
        test_config, targets or {}, ctx=ctx
    ).run_tests()
