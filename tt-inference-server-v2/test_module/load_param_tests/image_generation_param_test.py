# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import asyncio
import base64
import io
import logging
import time

import aiohttp
import numpy as np
from PIL import Image

from .._test_common import BaseTest

# Set up logging
logger = logging.getLogger(__name__)

# Model-specific overrides for the "different params" payload.
# Only the fields that differ from default_payload need to be specified.
_MODEL_DIFF_PARAM_OVERRIDES = {
    "FLUX.1-dev": {"seed": 0},
    "FLUX.1-schnell": {"seed": 0},
}
default_payload = {
    "prompt": "A beautiful sunset over a mountain landscape with vibrant colors",
    "negative_prompt": "blurry, low quality, distorted",
    "num_inference_steps": 20,
    "seed": 42,
    "guidance_scale": 7.5,
    "number_of_images": 1,
}

guidance_scale_change_payload = {
    "prompt": "A beautiful sunset over a mountain landscape with vibrant colors",
    "negative_prompt": "blurry, low quality, distorted",
    "num_inference_steps": 20,
    "seed": 42,
    "guidance_scale": 8.5,
    "number_of_images": 1,
}

headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "Bearer your-secret-key",
}

DEFAULT_SAME_SEED_MIN_SSIM = 0.95
DEFAULT_DIFF_PARAMS_MAX_SSIM = 0.98


def _build_diff_payload(model: str) -> dict:
    overrides = _MODEL_DIFF_PARAM_OVERRIDES.get(model)
    if overrides is None:
        return guidance_scale_change_payload

    payload = dict(default_payload)
    payload.update(overrides)
    return payload


def _decode_base64_image(image_base64):
    image_bytes = base64.b64decode(image_base64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _extract_first_image(response_data):
    """Extract the first base64 image string from an API response dict."""
    images = response_data.get("images", [])
    if not images:
        return None
    return images[0]


def compute_image_ssim(response_a, response_b):
    """Compute structural similarity between two API response images.

    Uses a simplified SSIM approximation based on local statistics
    over 8x8 blocks, avoiding the need for scipy/skimage.
    """
    img_a_b64 = _extract_first_image(response_a)
    img_b_b64 = _extract_first_image(response_b)
    if img_a_b64 is None or img_b_b64 is None:
        return 0.0

    arr_a = np.array(_decode_base64_image(img_a_b64), dtype=np.float64)
    arr_b = np.array(_decode_base64_image(img_b_b64), dtype=np.float64)

    if arr_a.shape != arr_b.shape:
        return 0.0

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu_a = arr_a.mean()
    mu_b = arr_b.mean()
    sigma_a_sq = arr_a.var()
    sigma_b_sq = arr_b.var()
    sigma_ab = ((arr_a - mu_a) * (arr_b - mu_b)).mean()

    numerator = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a**2 + mu_b**2 + c1) * (sigma_a_sq + sigma_b_sq + c2)
    return float(numerator / denominator)


class ImageGenerationParamTest(BaseTest):
    async def _run_specific_test_async(self):
        self.url = f"http://localhost:{self.service_port}/v1/images/generations"
        logger.info(f"Targets: {self.targets}")

        model = self.targets.get("model", "")
        diff_payload = _build_diff_payload(model)

        payloads = [default_payload, default_payload, diff_payload]

        # Get response data from all requests
        response_data_list = await self.test_concurrent_image_generation(payloads)

        logger.info("Received %s responses", len(response_data_list))

        same_seed_min_ssim = self.targets.get(
            "same_seed_min_ssim", DEFAULT_SAME_SEED_MIN_SSIM
        )
        diff_params_max_ssim = self.targets.get(
            "diff_params_max_ssim", DEFAULT_DIFF_PARAMS_MAX_SSIM
        )

        ssim_same = compute_image_ssim(response_data_list[0], response_data_list[1])
        ssim_diff = compute_image_ssim(response_data_list[0], response_data_list[2])

        same_requests = ssim_same >= same_seed_min_ssim
        diff_params_differs = ssim_diff < diff_params_max_ssim

        logger.info(
            "SSIM(response[0], response[1])=%.4f (threshold >= %.2f): %s",
            ssim_same,
            same_seed_min_ssim,
            same_requests,
        )
        logger.info(
            "SSIM(response[0], response[2])=%.4f (threshold < %.2f): %s",
            ssim_diff,
            diff_params_max_ssim,
            diff_params_differs,
        )

        return {
            "num_responses": len(response_data_list),
            "same_seed_ssim": ssim_same,
            "diff_params_ssim": ssim_diff,
            "same_requests_match": same_requests,
            "diff_params_differs": diff_params_differs,
            "success": same_requests and diff_params_differs,
        }

    async def test_concurrent_image_generation(self, payloads):
        """
        Test concurrent image generation with a list of payloads.

        Args:
            payloads: List of payload dictionaries to send. Each payload will be sent as a separate request.

        Returns:
            List of response data dictionaries from each request, in the same order as the input payloads.
        """

        async def timed_request(session, index, request_payload):
            logger.info("Starting request %s", index)
            try:
                start = time.perf_counter()
                async with session.post(
                    self.url, json=request_payload, headers=headers
                ) as response:
                    duration = time.perf_counter() - start
                    data = None
                    if response.status == 200:
                        data = await response.json()
                    else:
                        body = await response.text()
                        logger.error(
                            "[%s] HTTP %s: %s", index, response.status, body[:500]
                        )
                        data = {
                            "error": f"Status {response.status}",
                            "status": response.status,
                        }

                    logger.info(
                        "[%s] Status: %s, Time: %.2fs", index, response.status, duration
                    )
                    return {
                        "index": index,
                        "duration": duration,
                        "data": data,
                        "status": response.status,
                    }

            except Exception as e:
                duration = time.perf_counter() - start
                logger.error("[%s] Exception after %.2fs: %s", index, duration, e)
                return {
                    "index": index,
                    "duration": duration,
                    "data": None,
                    "error": str(e),
                }

        batch_size = len(payloads)
        response_data_list = []

        for iteration in range(2):
            session_timeout = aiohttp.ClientTimeout(total=2000)
            async with aiohttp.ClientSession(
                headers=headers, timeout=session_timeout
            ) as session:
                tasks = [
                    timed_request(session, i + 1, payloads[i])
                    for i in range(batch_size)
                ]
                results = await asyncio.gather(*tasks)

                if iteration == 0:
                    logger.info("Warmup run done.")
                else:
                    # Second iteration - collect the actual data
                    response_data_list = results
                    durations = [r["duration"] for r in results]
                    requests_duration = max(durations)
                    avg_duration = sum(durations) / batch_size

                    logger.info(
                        "Durations: %s, max=%.2fs, avg=%.2fs",
                        [f"{d:.2f}s" for d in durations],
                        requests_duration,
                        avg_duration,
                    )

        # Return list of response data in the same order as input payloads
        return [
            result["data"]
            for result in sorted(response_data_list, key=lambda x: x["index"])
        ]
