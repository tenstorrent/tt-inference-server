# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import asyncio
import logging
import time
from pathlib import Path

import aiohttp

from report_module.schema import Block
from .._test_common import BaseTest, TestConfig, TestConfig
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..context import MediaContext

logger = logging.getLogger(__name__)

_PAYLOAD_PATH = Path(__file__).parent / "test_payloads" / "image_client_image_payload"
with open(_PAYLOAD_PATH, "r") as f:
    image_payload_base64 = f.read()

payload = {
    "prompt": image_payload_base64,
    "response_format": "json",
    "top_k": 3,
    "min_confidence": 70.0,
}

headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "Bearer your-secret-key",
}


class CnnLoadTest(BaseTest):
    KIND = "cnn_load"
    TASK_TYPE = "cnn"

    async def _run_specific_test_async(self):
        self.url = f"http://localhost:{self.service_port}/v1/cnn/search-image"
        logger.info(self.targets)
        num_concurrent_requests = self._get_num_concurrent_requests(default=1)
        cnn_target_time = self.targets.get("cnn_time", 5)  # in seconds
        response_format = self.targets.get("response_format", "json")
        top_k = self.targets.get("top_k", 3)
        min_confidence = self.targets.get("min_confidence", 70.0)

        payload["response_format"] = response_format
        payload["top_k"] = top_k
        payload["min_confidence"] = min_confidence

        (
            requests_duration,
            average_duration,
        ) = await self.test_concurrent_cnn(batch_size=num_concurrent_requests)

        return {
            "requests_duration": requests_duration,
            "average_duration": average_duration,
            "target_time": cnn_target_time,
            "num_concurrent_requests": num_concurrent_requests,
            "success": requests_duration <= cnn_target_time,
        }

    async def test_concurrent_cnn(self, batch_size):
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

        # First iteration is warmup, second is measured
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
                if iteration == 0:
                    logger.info("🔥 Warm up run done.")

        logger.info(f"\n🚀 Time taken for individual concurrent requests : {results}")
        logger.info(
            f"\n🚀 Total time for {batch_size} concurrent requests: {requests_duration:.2f}s"
        )
        logger.info(
            f"\n🚀 Avg time for {batch_size} concurrent requests: {avg_duration:.2f}s"
        )

        return requests_duration, avg_duration



def run_cnn_load(ctx: "MediaContext", targets: dict | None = None) -> Block:
    """Run :class:`CnnLoadTest` under ``ctx`` and return its Block."""
    test_config = TestConfig(
        {
            "timeout": 1800,
            "retry_attempts": 1,
            "retry_delay": 10,
            "break_on_failure": False,
        }
    )
    return CnnLoadTest(test_config, targets or {}, ctx=ctx).run_tests()
