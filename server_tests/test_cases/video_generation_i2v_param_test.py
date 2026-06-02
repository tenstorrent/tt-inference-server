# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Parameter-sweep test for the Wan2.2 image-to-video endpoint.

Sends several payload variations (different seeds, num_inference_steps,
prompts, frame positions) at ``POST /v1/videos/generations/i2v`` and
verifies:

* identical payloads are accepted consistently,
* differing parameters yield distinct job IDs (each request is unique),
* every submission returns 202 Accepted.

Mirrors :mod:`server_tests.test_cases.video_generation_param_test` but
targets the I2V endpoint and includes a fixed conditioning frame in
each request.

Requires the server to be booted with ``MODEL_RUNNER=tt-wan2.2-i2v``.
"""

import asyncio
import base64
import copy
import json
import logging
import time
from pathlib import Path

import aiohttp

from server_tests.base_test import BaseTest

logger = logging.getLogger(__name__)

ACCURACY_REFERENCE_PATH = "evals/eval_targets/model_accuracy_reference.json"

# Conditioning frame reused from the I2V happy-path test fixture so we
# don't ship a duplicate asset for parameter-sweep coverage.
FIXTURE_IMAGE_PATH = (
    Path(__file__).parent.parent
    / "datasets"
    / "imagenet_subset"
    / "imagenet_002_volcano.jpg"
)

HTTP_ACCEPTED = 202
DEFAULT_NUM_INFERENCE_STEPS = 40
WARMUP_AND_MEASURE_ITERATIONS = 2

HEADERS = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "Bearer your-secret-key",
}


def _load_fixture_image_base64() -> str:
    """Read the repo-checked-in fixture image and return it base64-encoded."""
    if not FIXTURE_IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"I2V fixture image missing at {FIXTURE_IMAGE_PATH}. "
            "Expected a tracked sample from server_tests/datasets/imagenet_subset/."
        )
    return base64.b64encode(FIXTURE_IMAGE_PATH.read_bytes()).decode("ascii")


def _build_payload(
    *,
    prompt: str,
    num_inference_steps: int,
    seed: int,
    image_b64: str,
    frame_pos: int = 0,
    negative_prompt: str | None = "blurry, low quality, distorted, shaky",
) -> dict:
    """Build an I2V request body with the conditioning frame attached."""
    payload = {
        "prompt": prompt,
        "num_inference_steps": num_inference_steps,
        "seed": seed,
        "image_prompts": [{"image": image_b64, "frame_pos": frame_pos}],
    }
    if negative_prompt is not None:
        payload["negative_prompt"] = negative_prompt
    return payload


class VideoGenerationI2VParamTest(BaseTest):
    """Parameter-sweep happy-path coverage for the I2V endpoint."""

    async def _run_specific_test_async(self):
        self.url = f"{self.base_url}/v1/videos/generations/i2v"
        logger.info(f"Testing I2V parameters at {self.url}")

        model_name = self.config.get("model", "test-model")
        default_steps = self._get_num_inference_steps_from_reference(
            model_name, DEFAULT_NUM_INFERENCE_STEPS
        )
        logger.info(f"Using num_inference_steps={default_steps} for model={model_name}")

        image_b64 = _load_fixture_image_base64()
        base_prompt = (
            "A tranquil sunrise over rolling hills, soft golden light, "
            "cinematic camera pan"
        )
        alt_prompt = (
            "A serene ocean wave crashing on a tropical beach at dawn, "
            "crystal clear water, 4K quality"
        )

        # Variations probe the same parameter axes covered by the T2V param
        # test, plus an I2V-only frame_pos variation.
        default_payload = _build_payload(
            prompt=base_prompt,
            num_inference_steps=default_steps,
            seed=42,
            image_b64=image_b64,
        )
        payloads = [
            {"name": "default_payload", "payload": default_payload},
            {
                "name": "duplicate_default",
                "payload": copy.deepcopy(default_payload),
            },
            {
                "name": "inference_steps_15_payload",
                "payload": _build_payload(
                    prompt=base_prompt,
                    num_inference_steps=15,
                    seed=42,
                    image_b64=image_b64,
                ),
            },
            {
                "name": "seed_123_payload",
                "payload": _build_payload(
                    prompt=base_prompt,
                    num_inference_steps=default_steps,
                    seed=123,
                    image_b64=image_b64,
                ),
            },
            {
                "name": "no_negative_prompt_payload",
                "payload": _build_payload(
                    prompt=base_prompt,
                    num_inference_steps=default_steps,
                    seed=42,
                    image_b64=image_b64,
                    negative_prompt=None,
                ),
            },
            {
                "name": "different_prompt_payload",
                "payload": _build_payload(
                    prompt=alt_prompt,
                    num_inference_steps=default_steps,
                    seed=42,
                    image_b64=image_b64,
                ),
            },
            {
                "name": "frame_pos_4_payload",
                "payload": _build_payload(
                    prompt=base_prompt,
                    num_inference_steps=default_steps,
                    seed=42,
                    image_b64=image_b64,
                    frame_pos=4,
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

    def _load_accuracy_reference(self) -> dict:
        """Load accuracy reference data from JSON file."""
        logger.info(f"Loading accuracy reference from: {ACCURACY_REFERENCE_PATH}")
        try:
            with open(ACCURACY_REFERENCE_PATH, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(
                f"Accuracy reference file not found: {ACCURACY_REFERENCE_PATH}, "
                "using defaults"
            )
            return {}
        except json.JSONDecodeError as e:
            logger.warning(
                f"Invalid JSON in accuracy reference file: {e}, using defaults"
            )
            return {}

    def _get_num_inference_steps_from_reference(
        self, model_name: str, default: int
    ) -> int:
        """Get num_inference_steps from reference data for a given model."""
        reference_data = self._load_accuracy_reference()
        if model_name in reference_data:
            return reference_data[model_name].get("num_inference_steps", default)
        return default
