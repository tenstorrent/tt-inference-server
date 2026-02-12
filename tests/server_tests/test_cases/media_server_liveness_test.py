# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import logging

import aiohttp
from server_tests.base_test import BaseTest

# Set up logging
logger = logging.getLogger(__name__)


class MediaServerLivenessTest(BaseTest):
    async def _run_specific_test_async(self):
        url = f"http://localhost:{self.service_port}/tt-liveness"

        try:
            timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    assert response.status == 200, (
                        f"Expected status 200, got {response.status}"
                    )
                    data = await response.json()
                    logger.info(f"Liveness check response: {data}")
                    return data

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
