# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import json
import logging
import time
from pathlib import Path

import aiohttp
from server_tests.base_test import BaseTest

logger = logging.getLogger(__name__)

# Path to test payloads
TEST_PAYLOADS_PATH = (
    Path(__file__).parent.parent.parent.parent / "utils" / "test_payloads"
)

# Inpainting specific constants
PROMPT = "a black cat with glowing eyes, sitting on a pillow"
GUIDANCE_SCALE = 7.5
GUIDANCE_SCALE_CHANGED = 8.5
NUM_INFERENCE_STEPS = 30
SEED = 0
STRENGTH = 0.6

headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "Bearer your-secret-key",
}


def load_test_images():
    """Load test image and mask from payload file."""
    payload_path = TEST_PAYLOADS_PATH / "image_client_inpainting_payload"
    try:
        with open(payload_path, "r") as f:
            data = json.load(f)
        return data.get("inpaint_image"), data.get("inpaint_mask")
    except FileNotFoundError:
        logger.error(f"Test payload file not found: {payload_path}")
        return None, None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse test payload JSON: {e}")
        return None, None


def create_payload(
    image: str, mask: str, guidance_scale: float = GUIDANCE_SCALE
) -> dict:
    """Create a payload for inpainting request with given parameters."""
    return {
        "prompt": PROMPT,
        "image": image,
        "mask": mask,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "seed": SEED,
        "guidance_scale": guidance_scale,
        "number_of_images": 1,
        "strength": STRENGTH,
    }


class InpaintingGenerationParamTest(BaseTest):
    """Test inpainting generation with parameter variations.

    This test verifies:
    1. Determinism: Same parameters produce same results
    2. Parameter impact: Different guidance_scale produces different results
    """

    async def _run_specific_test_async(self):
        self.url = f"http://localhost:{self.service_port}/image/edits"
        logger.info(f"Testing endpoint: {self.url}")
        logger.info(f"Test targets: {self.targets}")

        # Load test image and mask
        test_image, test_mask = load_test_images()
        if not test_image or not test_mask:
            return {
                "success": False,
                "error": "Failed to load test image or mask from payload file",
            }

        # Create payloads: default, same (for consistency), different guidance_scale
        payloads = [
            create_payload(test_image, test_mask, GUIDANCE_SCALE),
            create_payload(
                test_image, test_mask, GUIDANCE_SCALE
            ),  # Same for consistency check
            create_payload(test_image, test_mask, GUIDANCE_SCALE_CHANGED),
        ]

        # Run concurrent requests
        response_data_list = await self._run_concurrent_requests(payloads)

        # Validate we got all responses
        if len(response_data_list) != len(payloads):
            return {
                "success": False,
                "error": f"Expected {len(payloads)} responses, got {len(response_data_list)}",
            }

        # Check for errors in responses
        for i, data in enumerate(response_data_list):
            if data is None or "error" in data:
                return {
                    "success": False,
                    "error": f"Request {i + 1} failed: {data}",
                }

        # Verify test assertions
        same_requests_match = response_data_list[0] == response_data_list[1]
        guidance_scale_differs = response_data_list[0] != response_data_list[2]

        logger.info(f"Same requests match: {same_requests_match}")
        logger.info(
            f"Guidance scale change produces different output: {guidance_scale_differs}"
        )

        return {
            "num_responses": len(response_data_list),
            "same_requests_match": same_requests_match,
            "guidance_scale_differs": guidance_scale_differs,
            "success": same_requests_match and guidance_scale_differs,
        }

    async def _run_concurrent_requests(self, payloads: list[dict]) -> list[dict]:
        """
        Run concurrent inpainting requests with warmup.

        Args:
            payloads: List of request payloads.

        Returns:
            List of response data in same order as input payloads.
        """
        batch_size = len(payloads)

        async def send_request(session, index, payload):
            logger.debug(f"Starting request {index}")
            start = time.perf_counter()
            try:
                async with session.post(
                    self.url, json=payload, headers=headers
                ) as response:
                    duration = time.perf_counter() - start
                    if response.status == 200:
                        data = await response.json()
                    else:
                        error_text = await response.text()
                        logger.error(
                            f"[{index}] HTTP {response.status}: {error_text[:200]}"
                        )
                        data = {
                            "error": f"HTTP {response.status}",
                            "status": response.status,
                        }

                    logger.info(
                        f"[{index}] Status: {response.status}, Time: {duration:.2f}s"
                    )
                    return {"index": index, "duration": duration, "data": data}

            except Exception as e:
                duration = time.perf_counter() - start
                logger.error(f"[{index}] Exception after {duration:.2f}s: {e}")
                return {"index": index, "duration": duration, "data": {"error": str(e)}}

        session_timeout = aiohttp.ClientTimeout(total=2000)

        # Warmup run
        logger.info("Starting warmup run...")
        async with aiohttp.ClientSession(timeout=session_timeout) as session:
            tasks = [
                send_request(session, i + 1, payloads[i]) for i in range(batch_size)
            ]
            await asyncio.gather(*tasks)
        logger.info("Warmup complete.")

        # Actual test run
        logger.info("Starting test run...")
        async with aiohttp.ClientSession(timeout=session_timeout) as session:
            tasks = [
                send_request(session, i + 1, payloads[i]) for i in range(batch_size)
            ]
            results = await asyncio.gather(*tasks)

        # Log timing summary
        durations = [r["duration"] for r in results]
        logger.info(f"Request durations: {[f'{d:.2f}s' for d in durations]}")
        logger.info(
            f"Max: {max(durations):.2f}s, Avg: {sum(durations) / len(durations):.2f}s"
        )

        # Return data sorted by index
        sorted_results = sorted(results, key=lambda x: x["index"])
        return [r["data"] for r in sorted_results]
