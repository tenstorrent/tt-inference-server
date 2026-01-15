# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import logging
import time

import aiohttp
from server_tests.base_test import BaseTest

logger = logging.getLogger(__name__)

payload = {
    "input": "The quick brown fox jumps over the lazy dog",
    "model": "test-model",
}

headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "Bearer your-secret-key",
}


class EmbeddingLoadTest(BaseTest):
    async def _run_specific_test_async(self):
        self.url = f"http://localhost:{self.service_port}/v1/embeddings"
        logger.info(self.targets)
        devices = self.targets.get("num_of_devices", 1)
        embedding_target_time = self.targets.get("embedding_time", 5)  # in seconds
        dimensions = self.targets.get("dimensions", None)
        model = self.config.get("model", "test-model")

        payload["model"] = model

        if dimensions is not None:
            payload["dimensions"] = dimensions

        (
            requests_duration,
            average_duration,
        ) = await self.test_concurrent_embedding(batch_size=devices)

        return {
            "requests_duration": requests_duration,
            "average_duration": average_duration,
            "target_time": embedding_target_time,
            "devices": devices,
            "success": requests_duration <= embedding_target_time,
        }

    async def test_concurrent_embedding(self, batch_size):
        async def timed_request(session, index):
            logger.info(f"Starting request {index}")
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
                    logger.info(
                        f"[{index}] Status: {response.status}, Time: {duration:.2f}s",
                    )
                    return duration

            except Exception as e:
                duration = time.perf_counter() - start
                logger.info(f"[{index}] Error after {duration:.2f}s: {e}")
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
                logger.info("🔥 Warm up run done.")

        logger.info(f"\n🚀 Time taken for individual concurrent requests : {results}")
        logger.info(
            f"\n🚀 Total time for {batch_size} concurrent requests: {requests_duration:.2f}s"
        )
        logger.info(
            f"\n🚀 Avg time for {batch_size} concurrent requests: {avg_duration:.2f}s"
        )
        logger.info(
            f"🚀 Avg time for {batch_size} concurrent requests: {avg_duration:.2f}s"
        )
