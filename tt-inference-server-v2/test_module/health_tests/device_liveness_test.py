# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

import aiohttp
from report_module.schema import Block

from .._test_common import BaseTest, TestConfig

if TYPE_CHECKING:
    from ..context import MediaContext

# Set up logging
logger = logging.getLogger(__name__)


class DeviceLivenessTest(BaseTest):
    KIND = "device_liveness"
    TASK_TYPE = "health"

    async def _run_specific_test_async(self):
        url = f"{self.base_url}/tt-liveness"

        if (
            self.targets["num_of_devices"] is None
            or self.targets["num_of_devices"] <= 0
        ):
            raise SystemExit(
                "❌ Number of devices not specified in targets for DeviceLivenessTest."
            )

        expected_devices = self.targets["num_of_devices"]

        try:
            timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    assert response.status == 200, (
                        f"Expected status 200, got {response.status}"
                    )
                    data = await response.json()

                    # Check 1: Verify status is "alive"
                    status = data.get("status")
                    if status != "alive":
                        raise SystemExit(
                            f"❌ Service status is '{status}', expected 'alive'"
                        )
                    logger.info(f"✅ Service status check passed: {status}")

                    # Check 2: Verify worker_info has correct number of ready devices
                    worker_info = data.get("worker_info", {})
                    if not worker_info:
                        raise SystemExit("❌ No worker_info found in response")

                    # Count workers that are ready (is_ready: true)
                    ready_workers = []
                    alive_workers = []

                    for worker_id, worker_data in worker_info.items():
                        if worker_data.get("is_ready", False):
                            ready_workers.append(worker_id)
                        if worker_data.get("is_alive", False):
                            alive_workers.append(worker_id)

                    ready_count = len(ready_workers)
                    alive_count = len(alive_workers)

                    model_ready = data.get("model_ready")
                    runner_in_use = data.get("runner_in_use")
                    logger.info(
                        "📊 Liveness status=%s model_ready=%s runner=%s "
                        "workers ready=%d alive=%d expected=%d",
                        status,
                        model_ready,
                        runner_in_use,
                        ready_count,
                        alive_count,
                        expected_devices,
                    )
                    if ready_count >= expected_devices:
                        logger.info("Liveness check response: %s", data)
                    else:
                        logger.debug("Liveness check response: %s", data)

                    # Check if number of ready workers is equal or bigger than expected devices
                    # we don't use equal to have a possibility to use i.e. 1 device on Galaxy to start some tests
                    if ready_count < expected_devices:
                        # this is just an exception, not a system exit, to allow retries
                        raise Exception(
                            f"❌ Device count mismatch: Expected {expected_devices} ready devices, "
                            f"but found {ready_count} ready workers. "
                            f"Ready workers: {ready_workers}\n"
                        )

                    # Additional check: ensure ready workers are also alive
                    not_alive_but_ready = [
                        w for w in ready_workers if w not in alive_workers
                    ]
                    if not_alive_but_ready:
                        logger.warning(
                            f"⚠️  Warning: Workers {not_alive_but_ready} are ready but not alive"
                        )

                    logger.info(
                        f"✅ Device count check passed: {ready_count}/{expected_devices} devices are ready"
                    )
                    logger.info(f"✅ Ready workers: {ready_workers}")

                    return {
                        "status": status,
                        "expected_devices": expected_devices,
                        "success": True if ready_count >= expected_devices else False,
                        "ready_workers": ready_workers,
                        "alive_workers": alive_workers,
                        "ready_count": ready_count,
                        "alive_count": alive_count,
                        "model_ready": model_ready,
                        "runner_in_use": runner_in_use,
                    }

        except (
            aiohttp.ClientConnectorError,
            aiohttp.ClientConnectionError,
            ConnectionRefusedError,
            OSError,
        ) as e:
            error_msg = f"❌ Media server is not running on port {self.service_port}. Please start the server first.\n🔍 Connection error: {e}"
            raise Exception(error_msg)

        except asyncio.TimeoutError as e:
            error_msg = f"❌ Media server on port {self.service_port} is not responding (timeout after 30s). Server may be starting up or overloaded.\n🔍 Error: {e}"
            raise Exception(error_msg)

        except Exception as e:
            # Log unexpected errors but don't exit - let retry logic handle it
            logger.warning(f"⚠️  Unexpected error during device liveness check: {e}")
            raise


def run_device_liveness(
    ctx: MediaContext,
    min_ready_devices: Optional[int] = None,
) -> Block:
    """Run DeviceLivenessTest under ``ctx`` and return its Block.

    Args:
        ctx: Media context (used for device + service-port resolution).
        min_ready_devices: Minimum number of READY chips required for the
            caller's task. ``None`` (default) means "full board" — i.e. the
            device's ``max_concurrency`` — preserving historical behavior for
            callers that haven't been updated yet. Pass ``1`` for eval-style
            tasks that only need at least one working chip.
    """
    test_config = TestConfig(
        {
            "timeout": 1200,
            "retry_attempts": 229,
            "retry_delay": 10,
            "break_on_failure": False,
        }
    )
    full_board = ctx.model_spec.device_model_spec.max_concurrency
    target = min_ready_devices if min_ready_devices is not None else full_board
    targets = {"num_of_devices": target if target and target > 0 else None}
    return DeviceLivenessTest(test_config, targets, ctx=ctx).run_tests()


__all__ = ["DeviceLivenessTest", "run_device_liveness"]
