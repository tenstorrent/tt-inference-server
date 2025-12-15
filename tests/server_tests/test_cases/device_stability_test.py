# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import logging

import aiohttp

from tests.server_tests.base_test import BaseTest

# Set up logging
logger = logging.getLogger(__name__)


class DeviceStabilityTest(BaseTest):
    async def _run_specific_test_async(self):
        url = f"http://localhost:{self.service_port}/tt-liveness"

        if (
            self.targets.get("num_of_devices") is None
            or self.targets["num_of_devices"] <= 0
        ):
            raise SystemExit(
                "❌ Number of devices not specified in targets for DeviceStabilityTest."
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
                    logger.info("Stability check response received")

                    # Check 1: Verify status is "alive"
                    status = data.get("status")
                    if status != "alive":
                        raise Exception(
                            f"❌ Service status is '{status}', expected 'alive'"
                        )
                    logger.info(f"✅ Service status check passed: {status}")

                    # Check 2: Get worker_info
                    worker_info = data.get("worker_info", {})
                    if not worker_info:
                        raise Exception("❌ No worker_info found in response")

                    # Check 3: Verify no devices have restarted or have errors
                    workers_with_restarts = []
                    workers_with_errors = []
                    unstable_workers = []

                    for worker_id, worker_data in worker_info.items():
                        restart_count = worker_data.get("restart_count", 0)
                        error_count = worker_data.get("error_count", 0)

                        if restart_count > 0:
                            workers_with_restarts.append(
                                {
                                    "worker_id": worker_id,
                                    "restart_count": restart_count,
                                    "pid": worker_data.get("pid"),
                                    "is_ready": worker_data.get("is_ready", False),
                                }
                            )
                            unstable_workers.append(worker_id)

                        if error_count > 0:
                            workers_with_errors.append(
                                {
                                    "worker_id": worker_id,
                                    "error_count": error_count,
                                    "pid": worker_data.get("pid"),
                                    "is_ready": worker_data.get("is_ready", False),
                                }
                            )
                            if worker_id not in unstable_workers:
                                unstable_workers.append(worker_id)

                    total_workers = len(worker_info)
                    stable_workers = total_workers - len(unstable_workers)

                    logger.info(
                        f"📊 Stability check - Total workers: {total_workers}, "
                        f"Stable: {stable_workers}, Unstable: {len(unstable_workers)}"
                    )

                    # Log details about unstable workers
                    if workers_with_restarts:
                        logger.warning(
                            f"⚠️  Workers with restarts ({len(workers_with_restarts)}):"
                        )
                        for worker in workers_with_restarts:
                            logger.warning(
                                f"  - Worker {worker['worker_id']}: "
                                f"restart_count={worker['restart_count']}, "
                                f"pid={worker['pid']}, "
                                f"is_ready={worker['is_ready']}"
                            )

                    if workers_with_errors:
                        logger.warning(
                            f"⚠️  Workers with errors ({len(workers_with_errors)}):"
                        )
                        for worker in workers_with_errors:
                            logger.warning(
                                f"  - Worker {worker['worker_id']}: "
                                f"error_count={worker['error_count']}, "
                                f"pid={worker['pid']}, "
                                f"is_ready={worker['is_ready']}"
                            )

                    # Determine success
                    success = len(unstable_workers) == 0

                    if success:
                        logger.info(
                            f"✅ Device stability check passed: All {total_workers} workers are stable "
                            f"(no restarts, no errors)"
                        )
                    else:
                        error_msg = (
                            f"❌ Device stability check failed: {len(unstable_workers)} of {total_workers} "
                            f"workers are unstable. "
                            f"Unstable workers: {unstable_workers}"
                        )
                        if workers_with_restarts:
                            error_msg += f"\n  Workers with restarts: {[w['worker_id'] for w in workers_with_restarts]}"
                        if workers_with_errors:
                            error_msg += f"\n  Workers with errors: {[w['worker_id'] for w in workers_with_errors]}"

                        raise Exception(error_msg)

                    return {
                        "status": status,
                        "expected_devices": expected_devices,
                        "total_workers": total_workers,
                        "stable_workers": stable_workers,
                        "unstable_workers": unstable_workers,
                        "workers_with_restarts": workers_with_restarts,
                        "workers_with_errors": workers_with_errors,
                        "success": success,
                        "full_response": data,
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
            logger.warning(f"⚠️  Unexpected error during device stability check: {e}")
            raise
