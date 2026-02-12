# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

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
        self.url = f"http://localhost:{self.service_port}/image/generations"
        print(self.targets)
        devices = self.targets.get("num_of_devices", 1)
        image_generation_target_time = self.targets.get(
            "image_generation_time", 9
        )  # in seconds
        num_inference_steps = self.targets.get("num_inference_steps", 20)

        payload["num_inference_steps"] = num_inference_steps

        (
            requests_duration,
            average_duration,
        ) = await self.test_concurrent_image_generation(batch_size=devices)

        return {
            "requests_duration": requests_duration,
            "average_duration": average_duration,
            "target_time": image_generation_target_time,
            "devices": devices,
            "success": requests_duration <= image_generation_target_time,
        }

    async def test_concurrent_image_generation(self, batch_size):
        async def timed_request(session, index):
            print(f"Starting request {index}")
            try:
                start = time.perf_counter()
                async with session.post(
                    self.url, json=payload, headers=headers
                ) as response:
                    duration = time.perf_counter() - start
                    if response.status == 200:
                        await response.json()
                    print(
                        f"[{index}] Status: {response.status}, Time: {duration:.2f}s",
                    )
                    return duration

            except Exception as e:
                duration = time.perf_counter() - start
                print(f"[{index}] Error after {duration:.2f}s: {e}")
                raise

        # First iteration is warmup, second is measured (original behavior)
        for iteration in range(2):
            session_timeout = aiohttp.ClientTimeout(total=2000)
            async with aiohttp.ClientSession(
                headers=headers, timeout=session_timeout
            ) as session:
                tasks = [timed_request(session, i + 1) for i in range(batch_size)]
                results = await asyncio.gather(*tasks)
                requests_duration = max(results)
                total_duration = sum(results)
                avg_duration = total_duration / batch_size
                return requests_duration, avg_duration
            if iteration == 0:
                print("🔥 Warm up run done.")

        print(f"\n🚀 Time taken for individual concurrent requests : {results}")
        print(
            f"\n🚀 Total time for {batch_size} concurrent requests: {requests_duration:.2f}s"
        )
        print(
            f"\n🚀 Avg time for {batch_size} concurrent requests: {avg_duration:.2f}s"
        )
        print(f"🚀 Avg time for {batch_size} concurrent requests: {avg_duration:.2f}s")
