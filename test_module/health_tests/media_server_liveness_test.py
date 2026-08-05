# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import aiohttp

from report_module.schema import Block

from .._test_common import BaseTest, TestConfig

if TYPE_CHECKING:
    from ..context import MediaContext

# Set up logging
logger = logging.getLogger(__name__)


class MediaServerLivenessTest(BaseTest):
    KIND = "media_server_liveness"
    TASK_TYPE = "health"

    async def _run_specific_test_async(self):
        url = f"{self.base_url}/tt-liveness"

        try:
            timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    assert response.status == 200, (
                        f"Expected status 200, got {response.status}"
                    )
                    data = await response.json()
                    logger.info(f"Liveness check response: {data}")
                    worker_info = data.get("worker_info") or {}
                    return {
                        "status": data.get("status"),
                        "model_ready": data.get("model_ready"),
                        "runner_in_use": data.get("runner_in_use"),
                        "total_workers": len(worker_info),
                        "ready_workers": sum(
                            1 for w in worker_info.values() if w.get("is_ready")
                        ),
                        "alive_workers": sum(
                            1 for w in worker_info.values() if w.get("is_alive")
                        ),
                    }

        except (
            aiohttp.ClientConnectorError,
            aiohttp.ClientConnectionError,
            ConnectionRefusedError,
            OSError,
        ) as e:
            error_msg = f"❌ Media server is not running on port {self.service_port}. Please start the server first.\n🔍 Connection error: {e}"
            raise SystemExit(error_msg)

        except asyncio.TimeoutError as e:
            error_msg = f"❌ Media server on port {self.service_port} is not responding (timeout after 10s). Server may be starting up or overloaded.\n🔍 Error: {e}"
            raise SystemExit(error_msg)

        except Exception as e:
            # Log unexpected errors but don't exit - let retry logic handle it
            logger.error(f"⚠️  Unexpected error during liveness check: {e}")
            raise


def run_media_server_liveness(ctx: MediaContext) -> Block:
    """Run MediaServerLivenessTest under ``ctx`` and return its Block."""
    test_config = TestConfig(
        {
            "timeout": 60,
            "retry_attempts": 3,
            "retry_delay": 5,
            "break_on_failure": False,
        }
    )
    return MediaServerLivenessTest(test_config, targets={}, ctx=ctx).run_tests()


__all__ = ["MediaServerLivenessTest", "run_media_server_liveness"]
