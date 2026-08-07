# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import aiohttp
from report_module.schema import Block

from .._test_common import BaseTest, HardwareRequirement, TestConfig

if TYPE_CHECKING:
    from ..context import MediaContext

logger = logging.getLogger(__name__)

DEFAULT_INPUT = "The quick brown fox jumps over the lazy dog"

headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "Bearer your-secret-key",
}


class EmbeddingLoadTest(BaseTest):
    KIND = "embedding_load"
    TASK_TYPE = "embedding"
    HARDWARE_REQUIREMENT = HardwareRequirement.FULL_BOARD

    def _model_name(self) -> str:
        """Model id to send in the request body.

        Forge embedding runners reject a request whose ``model`` is not the one
        they loaded, so this has to be the real repo id — an invented default
        turns the whole test into a 500.
        """
        configured = self.config.get("model")
        if configured:
            return configured
        if self.ctx is not None:
            return self.ctx.model_spec.hf_model_repo
        raise ValueError(
            "EmbeddingLoadTest needs a model: pass test_config['model'] or a ctx."
        )

    async def _run_specific_test_async(self):
        self.url = f"{self.base_url}/v1/embeddings"
        logger.info(self.targets)
        num_concurrent_requests = self._get_num_concurrent_requests(default=1)
        embedding_target_time = self.targets.get("embedding_time", 5)  # in seconds
        dimensions = self.targets.get("dimensions", None)

        self.payload = {"input": DEFAULT_INPUT, "model": self._model_name()}
        if dimensions is not None:
            self.payload["dimensions"] = dimensions
        logger.info(f"Embedding load test payload model={self.payload['model']!r}")

        (
            requests_duration,
            average_duration,
        ) = await self.test_concurrent_embedding(batch_size=num_concurrent_requests)

        return {
            "requests_duration": requests_duration,
            "average_duration": average_duration,
            "target_time": embedding_target_time,
            "num_concurrent_requests": num_concurrent_requests,
            "success": requests_duration <= embedding_target_time,
        }

    async def test_concurrent_embedding(self, batch_size):
        async def timed_request(session, index):
            logger.info(f"Starting request {index}")
            try:
                start = time.perf_counter()
                async with session.post(
                    self.url, json=self.payload, headers=headers
                ) as response:
                    duration = time.perf_counter() - start
                    if response.status == 200:
                        await response.json()
                    else:
                        body = (await response.text())[:500]
                        raise Exception(
                            f"Status {response.status} {response.reason}: {body}"
                        )
                    logger.info(
                        f"[{index}] Status: {response.status}, Time: {duration:.2f}s",
                    )
                    return duration

            except Exception as e:
                duration = time.perf_counter() - start
                logger.info(f"[{index}] Error after {duration:.2f}s: {e}")
                raise

        # First iteration is warmup, second is measured. The warmup matters:
        # the first request against a forge runner can pay trace-capture /
        # compile cost, which would otherwise be charged to the reported time.
        requests_duration = avg_duration = 0.0
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
        logger.info(
            f"🚀 Avg time for {batch_size} concurrent requests: {avg_duration:.2f}s"
        )
        return requests_duration, avg_duration


def run_embedding_load(ctx: "MediaContext", targets: dict | None = None) -> Block:
    """Run :class:`EmbeddingLoadTest` under ``ctx`` and return its Block."""
    test_config = TestConfig(
        {
            "timeout": 1800,
            "retry_attempts": 1,
            "retry_delay": 10,
            "break_on_failure": False,
        }
    )
    return EmbeddingLoadTest(test_config, targets or {}, ctx=ctx).run_tests()
