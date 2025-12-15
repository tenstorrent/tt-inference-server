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


class ImageGenerationParamTest(BaseTest):
    async def _run_specific_test_async(self):
        self.url = f"http://localhost:{self.service_port}/image/generations"
        print(self.targets)

        # Create list of payloads (one per device)
        payloads = []
        # use two payloads with same params to compare
        payloads.append(default_payload)
        payloads.append(default_payload)
        payloads.append(guidance_scale_change_payload)

        # Get response data from all requests
        response_data_list = await self.test_concurrent_image_generation(payloads)

        # You can now compare the response data
        print(f"\n📊 Received {len(response_data_list)} responses")

        same_requests = (
            True if response_data_list[0] == response_data_list[1] else False
        )
        guidance_scale_differs = response_data_list[0] != response_data_list[2]

        return {
            "num_responses": len(response_data_list),
            "same_requests_match": same_requests,
            "guidance_scale_differs": guidance_scale_differs,
            "success": same_requests and guidance_scale_differs,
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
            print(f"Starting request {index}")
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
                        print(f"[{index}] Error: Status {response.status}")
                        data = {
                            "error": f"Status {response.status}",
                            "status": response.status,
                        }

                    print(
                        f"[{index}] Status: {response.status}, Time: {duration:.2f}s",
                    )
                    return {
                        "index": index,
                        "duration": duration,
                        "data": data,
                        "status": response.status,
                    }

            except Exception as e:
                duration = time.perf_counter() - start
                print(f"[{index}] Error after {duration:.2f}s: {e}")
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
                    print("🔥 Warm up run done.")
                else:
                    # Second iteration - collect the actual data
                    response_data_list = results
                    durations = [r["duration"] for r in results]
                    requests_duration = max(durations)
                    avg_duration = sum(durations) / batch_size

                    print(
                        f"\n🚀 Time taken for individual concurrent requests: {durations}"
                    )
                    print(
                        f"\n🚀 Max time for {batch_size} concurrent requests: {requests_duration:.2f}s"
                    )
                    print(
                        f"\n🚀 Avg time for {batch_size} concurrent requests: {avg_duration:.2f}s"
                    )

        # Return list of response data in the same order as input payloads
        return [
            result["data"]
            for result in sorted(response_data_list, key=lambda x: x["index"])
        ]
