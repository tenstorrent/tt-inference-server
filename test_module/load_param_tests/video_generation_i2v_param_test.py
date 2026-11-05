# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Parameter-sweep test for the Wan2.2 image-to-video endpoint.

Sends several payload variations (different seeds, prompts, a last frame) at the generations API's ``POST /v1/generations/wan-ai/wan2.2-i2v-a14b``
and verifies:

* identical payloads are accepted consistently,
* differing parameters yield distinct job IDs (each request is unique),
* every submission returns 202 Accepted.

Mirrors :mod:`test_module.load_param_tests.video_generation_param_test` but
targets the I2V endpoint and includes a fixed conditioning frame in each
request.

Requires the server to be booted with ``MODEL_RUNNER=tt-wan2.2-i2v``.
"""

import asyncio
import copy
import logging
import time
from typing import TYPE_CHECKING

import aiohttp
from report_module.schema import Block

from .._test_common import (
    BaseTest,
    HardwareRequirement,
    TestConfig,
    fixture_image_data_uri,
    wan_size_for_device,
)

if TYPE_CHECKING:
    from ..context import MediaContext

logger = logging.getLogger(__name__)

# Constants
SUBMIT_PATH = "/v1/generations/wan-ai/wan2.2-i2v-a14b"
HTTP_ACCEPTED = 202
WARMUP_AND_MEASURE_ITERATIONS = 2

HEADERS = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "Bearer your-secret-key",
}


def _build_payload(
    *,
    prompt: str,
    seed: int,
    image: str,
    size: str,
    last_frame: bool = False,
    negative_prompt: str | None = "blurry, low quality, distorted, shaky",
) -> dict:
    """Build an I2V request body with the first frame (and last) attached."""
    payload = {
        "prompt": prompt,
        "seed": seed,
        "first_frame": image,
        "size": size,
        "aspect_ratio": "16:9",
    }
    if last_frame:
        payload["last_frame"] = image
    if negative_prompt is not None:
        payload["negative_prompt"] = negative_prompt
    return payload


class VideoGenerationI2VParamTest(BaseTest):
    """Parameter-sweep happy-path coverage for the I2V endpoint."""

    KIND = "video_generation_i2v_param"
    TASK_TYPE = "video"
    HARDWARE_REQUIREMENT = HardwareRequirement.FULL_BOARD

    async def _run_specific_test_async(self):
        self.url = f"{self.base_url}{SUBMIT_PATH}"
        logger.info(f"Testing I2V parameters at {self.url}")

        frames = {
            "image": fixture_image_data_uri(),
            "size": wan_size_for_device(self.ctx.device if self.ctx else None),
        }
        base_prompt = (
            "A tranquil sunrise over rolling hills, soft golden light, "
            "cinematic camera pan"
        )
        alt_prompt = (
            "A serene ocean wave crashing on a tropical beach at dawn, "
            "crystal clear water, 4K quality"
        )

        # Variations probe the same parameter axes covered by the T2V param
        # test, plus an I2V-only last-frame variation.
        default_payload = _build_payload(prompt=base_prompt, seed=42, **frames)
        payloads = [
            {"name": "default_payload", "payload": default_payload},
            {
                "name": "duplicate_default",
                "payload": copy.deepcopy(default_payload),
            },
            {
                "name": "seed_123_payload",
                "payload": _build_payload(prompt=base_prompt, seed=123, **frames),
            },
            {
                "name": "no_negative_prompt_payload",
                "payload": _build_payload(
                    prompt=base_prompt, seed=42, negative_prompt=None, **frames
                ),
            },
            {
                "name": "different_prompt_payload",
                "payload": _build_payload(prompt=alt_prompt, seed=42, **frames),
            },
            {
                "name": "last_frame_payload",
                "payload": _build_payload(
                    prompt=base_prompt,
                    seed=42,
                    last_frame=True,
                    **frames,
                ),
            },
        ]

        response_data_list = await self._test_concurrent_i2v_generation(payloads)

        logger.info(f"\n📊 Received {len(response_data_list)} responses")
        results = self._summarize_results(payloads, response_data_list)

        all_succeeded = all(r["status"] == HTTP_ACCEPTED for r in response_data_list)
        param_tests_diff = [
            results["tests"][p["name"]]["differs_from_base"] for p in payloads[2:]
        ]
        success = all_succeeded and all(param_tests_diff)
        results["success"] = success

        logger.info(
            f"\n{'✅' if success else '❌'} Test {'PASSED' if success else 'FAILED'}"
        )
        return results

    def _summarize_results(
        self, payloads: list[dict], response_data_list: list[dict]
    ) -> dict:
        """Build the result dict from the measurement-iteration responses."""
        results = {"num_responses": len(response_data_list), "tests": {}}

        # Identical payloads must both succeed; we don't require identical job
        # IDs because the server mints unique IDs per submission.
        base_match = (
            response_data_list[0]["status"] == HTTP_ACCEPTED
            and response_data_list[1]["status"] == HTTP_ACCEPTED
        )
        results["same_requests_match"] = base_match
        logger.info(f"✅ Same requests both accepted: {base_match}")

        for i in range(2, len(response_data_list)):
            test_name = payloads[i]["name"]
            differs_from_base = (
                response_data_list[0]["job_id"] != response_data_list[i]["job_id"]
            )
            results["tests"][test_name] = {
                "differs_from_base": differs_from_base,
                "status": response_data_list[i]["status"],
                "duration": response_data_list[i]["duration"],
            }
            logger.info(
                f"  {test_name}: differs={differs_from_base}, "
                f"status={response_data_list[i]['status']}, "
                f"duration={response_data_list[i]['duration']:.2f}s"
            )

        return results

    async def _test_concurrent_i2v_generation(self, payloads: list[dict]) -> list[dict]:
        """Submit all payloads concurrently in a warm-up + measurement pair."""

        async def timed_request(session, index, test_config):
            test_name = test_config["name"]
            request_payload = test_config["payload"]
            logger.info(f"Starting request {index}: {test_name}")
            start = time.perf_counter()
            try:
                async with session.post(
                    self.url, json=request_payload, headers=HEADERS
                ) as response:
                    duration = time.perf_counter() - start
                    if response.status == HTTP_ACCEPTED:
                        data = await response.json()
                        job_id = data.get("id")
                    else:
                        logger.warning(
                            f"[{index}] {test_name} - Status {response.status}"
                        )
                        data = {
                            "error": f"Status {response.status}",
                            "status": response.status,
                        }
                        job_id = None

                    logger.info(
                        f"[{index}] {test_name} - Status: {response.status}, "
                        f"Job ID: {job_id}, Time: {duration:.2f}s"
                    )
                    return {
                        "index": index,
                        "name": test_name,
                        "duration": duration,
                        "data": data,
                        "job_id": job_id,
                        "status": response.status,
                    }

            except Exception as e:
                duration = time.perf_counter() - start
                logger.error(
                    f"[{index}] {test_name} - Error after {duration:.2f}s: {e}"
                )
                return {
                    "index": index,
                    "name": test_name,
                    "duration": duration,
                    "data": None,
                    "job_id": None,
                    "error": str(e),
                    "status": 0,
                }

        batch_size = len(payloads)
        response_data_list: list[dict] = []

        for iteration in range(WARMUP_AND_MEASURE_ITERATIONS):
            session_timeout = aiohttp.ClientTimeout(total=2000)
            async with aiohttp.ClientSession(
                headers=HEADERS, timeout=session_timeout
            ) as session:
                tasks = [
                    timed_request(session, i + 1, payloads[i])
                    for i in range(batch_size)
                ]
                results = await asyncio.gather(*tasks)

                if iteration == 0:
                    logger.info("🔥 Warm up run done.")
                    continue

                response_data_list = results
                durations = [r["duration"] for r in results]
                logger.info(f"\n🚀 Per-request durations: {durations}")
                logger.info(
                    f"\n🚀 Max time for {batch_size} concurrent requests: "
                    f"{max(durations):.2f}s"
                )
                logger.info(
                    f"\n🚀 Avg time for {batch_size} concurrent requests: "
                    f"{sum(durations) / batch_size:.2f}s"
                )

        return sorted(response_data_list, key=lambda x: x["index"])


def run_video_generation_i2v_param(
    ctx: "MediaContext", targets: dict | None = None
) -> Block:
    """Run :class:`VideoGenerationI2VParamTest` under ``ctx`` and return its Block."""
    test_config = TestConfig(
        {
            "timeout": 1800,
            "retry_attempts": 1,
            "retry_delay": 10,
            "break_on_failure": False,
        }
    )
    return VideoGenerationI2VParamTest(test_config, targets or {}, ctx=ctx).run_tests()
