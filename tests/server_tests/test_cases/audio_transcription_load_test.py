# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import logging
import time

import aiohttp
from server_tests.base_test import BaseTest

# Import the dataset from the Python file
from utils.test_payloads.audio_payload_30s import dataset as dataset30s
from utils.test_payloads.audio_payload_60s import dataset as dataset60s

# Set up logging
logger = logging.getLogger(__name__)

payload = {
    "file": dataset30s["file"],
    "stream": False,
    "is_preprocessing_enabled": True,
    "prompt": "",
}

headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "Bearer your-secret-key",
}


class AudioTranscriptionLoadTest(BaseTest):
    async def _run_specific_test_async(self):
        self.url = f"http://localhost:{self.service_port}/audio/transcriptions"
        print(self.targets)
        devices = self.targets.get("num_of_devices", 1)
        audio_transcription_time = self.targets.get(
            "audio_transcription_time", 9
        )  # in seconds
        dataset_name = self.targets.get("dataset", "30s")  # in seconds

        if dataset_name == "60s":
            payload["file"] = dataset60s["file"]

        (
            requests_duration,
            average_duration,
        ) = await self.test_concurrent_audio_transcription(batch_size=devices)

        self.test_payloads_path = "utils/test_payloads"

        return {
            "requests_duration": requests_duration,
            "average_duration": average_duration,
            "target_time": audio_transcription_time,
            "devices": devices,
            "success": average_duration <= audio_transcription_time,
        }

    async def test_concurrent_audio_transcription(self, batch_size):
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
                    else:
                        raise Exception(f"Status {response.status} {response.reason}")
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
