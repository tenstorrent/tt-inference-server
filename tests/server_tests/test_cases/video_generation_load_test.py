# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import asyncio
import logging
import time

import aiohttp
from server_tests.base_test import BaseTest

# Set up logging
logger = logging.getLogger(__name__)

payload = {
    "prompt": "A beautiful sunset over a mountain landscape with vibrant colors, cinematic quality, smooth camera movement",
    "negative_prompt": "blurry, low quality, distorted, shaky",
    "num_inference_steps": 20,
    "seed": 42,
}

headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "Bearer your-secret-key",
}


class VideoGenerationLoadTest(BaseTest):
    async def _run_specific_test_async(self):
        self.url = f"http://localhost:{self.service_port}/video/generations"
        print(self.targets)
        devices = self.targets.get("num_of_devices", 1)
        video_generation_target_time = self.targets.get(
            "video_generation_time", 480
        )  # 8 minutes default in seconds
        num_inference_steps = self.targets.get("num_inference_steps", 20)

        payload["num_inference_steps"] = num_inference_steps

        (
            requests_duration,
            average_duration,
        ) = await self.test_concurrent_video_generation(batch_size=devices)

        return {
            "requests_duration": requests_duration,
            "average_duration": average_duration,
            "target_time": video_generation_target_time,
            "devices": devices,
            "success": requests_duration <= video_generation_target_time,
        }

    async def poll_video_status(self, session, job_id, timeout=700):
        """
        Poll the video generation job status until it's completed or failed.

        Args:
            session: aiohttp client session
            job_id: The video job ID to poll
            timeout: Maximum time to wait in seconds (default 9 minutes)

        Returns:
            dict: Final job status response

        Raises:
            Exception: If job fails or times out
        """
        status_url = f"{self.url}/{job_id}"
        start_time = time.perf_counter()
        poll_interval = 5  # Poll every 5 seconds

        while True:
            elapsed = time.perf_counter() - start_time
            if elapsed > timeout:
                raise Exception(
                    f"Video generation timed out after {timeout}s for job {job_id}"
                )

            async with session.get(status_url, headers=headers) as response:
                if response.status != 200:
                    raise Exception(
                        f"Failed to get job status: {response.status} {response.reason}"
                    )

                data = await response.json()
                status = data.get("status")

                print(f"Job {job_id}: status={status}, elapsed={elapsed:.1f}s")

                if status == "completed":
                    return data
                elif status == "failed":
                    raise Exception(f"Video generation failed for job {job_id}: {data}")
                elif status in ("in_progress", "queued"):
                    await asyncio.sleep(poll_interval)
                else:
                    raise Exception(f"Unknown status '{status}' for job {job_id}")

    async def test_concurrent_video_generation(self, batch_size):
        async def timed_request(session, index):
            print(f"Starting request {index}")
            try:
                start = time.perf_counter()

                # Step 1: Submit video generation job
                async with session.post(
                    self.url, json=payload, headers=headers
                ) as response:
                    if response.status != 202:
                        raise Exception(
                            f"Failed to submit job: {response.status} {response.reason}"
                        )

                    job_data = await response.json()
                    job_id = job_data.get("id")

                    if not job_id:
                        raise Exception(f"No job ID returned: {job_data}")

                    print(
                        f"[{index}] Job submitted: {job_id}, Status: {job_data.get('status')}"
                    )

                # Step 2: Poll until completion
                await self.poll_video_status(session, job_id, timeout=700)

                duration = time.perf_counter() - start
                print(f"[{index}] Completed in {duration:.2f}s")
                return duration

            except Exception as e:
                duration = time.perf_counter() - start
                print(f"[{index}] Error after {duration:.2f}s: {e}")
                raise

        # Single run (no warmup)
        session_timeout = aiohttp.ClientTimeout(
            total=800
        )  # ~13 minute timeout for session
        async with aiohttp.ClientSession(
            headers=headers, timeout=session_timeout
        ) as session:
            tasks = [timed_request(session, i + 1) for i in range(batch_size)]
            results = await asyncio.gather(*tasks)
            requests_duration = max(results)
            total_duration = sum(results)
            avg_duration = total_duration / batch_size

        print(f"\n🚀 Time taken for individual concurrent requests : {results}")
        print(
            f"\n🚀 Total time for {batch_size} concurrent requests: {requests_duration:.2f}s"
        )
        print(f"🚀 Avg time for {batch_size} concurrent requests: {avg_duration:.2f}s")

        return requests_duration, avg_duration
