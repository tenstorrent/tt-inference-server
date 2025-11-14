# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio

import aiohttp
from tests.server_tests.base_test import BaseTest


class DeviceLivenessTest(BaseTest):
    async def _run_specific_test_async(self):
        url = f"http://localhost:{self.service_port}/tt-liveness"

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
                    print(f"Liveness check response: {data}")

                    # Check 1: Verify status is "alive"
                    status = data.get("status")
                    if status != "alive":
                        raise SystemExit(
                            f"❌ Service status is '{status}', expected 'alive'"
                        )
                    print(f"✅ Service status check passed: {status}")

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

                    print(
                        f"📊 Worker status - Ready: {ready_count}, Alive: {alive_count}, Expected: {expected_devices}"
                    )

                    # Check if number of ready workers matches expected devices
                    if ready_count != expected_devices:
                        # this is just an exception, not a system exit, to allow retries
                        raise Exception(
                            f"❌ Device count mismatch: Expected {expected_devices} ready devices, "
                            f"but found {ready_count} ready workers. "
                            f"Ready workers: {ready_workers}"
                        )

                    # Additional check: ensure ready workers are also alive
                    not_alive_but_ready = [
                        w for w in ready_workers if w not in alive_workers
                    ]
                    if not_alive_but_ready:
                        print(
                            f"⚠️  Warning: Workers {not_alive_but_ready} are ready but not alive"
                        )

                    print(
                        f"✅ Device count check passed: {ready_count}/{expected_devices} devices are ready"
                    )
                    print(f"✅ Ready workers: {ready_workers}")

                    return {
                        "status": status,
                        "expected_devices": expected_devices,
                        "ready_workers": ready_workers,
                        "alive_workers": alive_workers,
                        "ready_count": ready_count,
                        "alive_count": alive_count,
                        "full_response": data,
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
            error_msg = f"❌ Media server on port {self.service_port} is not responding (timeout after 30s). Server may be starting up or overloaded.\n🔍 Error: {e}"
            raise SystemExit(error_msg)

        except Exception as e:
            # Log unexpected errors but don't exit - let retry logic handle it
            print(f"⚠️  Unexpected error during device liveness check: {e}")
            raise
