# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import asyncio
import logging
import time

import aiohttp
from server_tests.base_test import BaseTest

# Set up logging
logger = logging.getLogger(__name__)

payload = {
    "prompt": "A beautiful sunset over a mountain landscape with vibrant colors",
    "negative_prompt": "blurry, low quality, distorted",
    "num_inference_steps": 20,
    "seed": 42,
    "guidance_scale": 7.5,
    "number_of_images": 1,
}

headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "Bearer your-secret-key",
}


class ImageGenerationLoadTest(BaseTest):
    async def _run_specific_test_async(self):
        self.url = f"http://localhost:{self.service_port}/v1/images/generations"
        print(self.targets)
        devices = self.targets.get("num_of_devices", 1)
        image_generation_target_time = self.targets.get(
            "image_generation_time", 9
        )  # in seconds
        num_inference_steps = self.targets.get("num_inference_steps", 20)
        image_resolution = self.targets.get("image_resolution")

        payload["num_inference_steps"] = num_inference_steps

        (
            requests_duration,
            average_duration,
        ) = await self.test_concurrent_image_generation(batch_size=devices)

        success = requests_duration <= image_generation_target_time
        logger.info(
            "Load test result: duration=%.2fs, target=%.2fs, success=%s",
            requests_duration,
            image_generation_target_time,
            success,
        )

        return {
            "requests_duration": requests_duration,
            "average_duration": average_duration,
            "target_time": image_generation_target_time,
            "devices": devices,
            "image_resolution": image_resolution,
            "success": success,
        }

    async def test_concurrent_image_generation(self, batch_size):
        async def timed_request(session, index):
            logger.info("Starting request %s", index)
            try:
                start = time.perf_counter()
                async with session.post(
                    self.url, json=payload, headers=headers
                ) as response:
                    duration = time.perf_counter() - start
                    if response.status == 200:
                        await response.json()
                    else:
                        body = await response.text()
                        logger.error(
                            "[%s] HTTP %s: %s",
                            index,
                            response.status,
                            body[:500],
                        )
                    logger.info(
                        "[%s] Status: %s, Time: %.2fs", index, response.status, duration
                    )
                    return duration

            except Exception as e:
                duration = time.perf_counter() - start
                logger.error("[%s] Exception after %.2fs: %s", index, duration, e)
                raise

        for iteration in range(2):
            session_timeout = aiohttp.ClientTimeout(total=2000)
            async with aiohttp.ClientSession(
                headers=headers, timeout=session_timeout
            ) as session:
                tasks = [timed_request(session, i + 1) for i in range(batch_size)]
                results = await asyncio.gather(*tasks)

                if iteration == 0:
                    logger.info("Warmup run done.")
                else:
                    requests_duration = max(results)
                    total_duration = sum(results)
                    avg_duration = total_duration / batch_size
                    return requests_duration, avg_duration
