# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import asyncio
import logging
import os
import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict

from server_tests.test_classes import TestConfig

logger = logging.getLogger(__name__)

# Targets key used by load tests to control concurrent HTTP request fan-out.
# Introduced to replace the overloaded `num_of_devices` key (which also meant
# "physical chip count" in hardware_defaults / liveness tests).
NUM_CONCURRENT_REQUESTS_KEY = "num_concurrent_requests"
NUM_OF_DEVICES_KEY = "num_of_devices"


class BaseTest(ABC):
    def __init__(
        self, config: TestConfig, targets: Dict[str, Any], description: str = ""
    ):
        self.config = config
        self.targets = targets
        self.description = description
        self.service_port = os.getenv("SERVICE_PORT", "8000")
        self.timeout = config.get("timeout")
        self.retry_attempts = config.get("retry_attempts")
        self.break_on_failure = config.get("break_on_failure")
        self.logs = []
        self.retry_delay = config.get("retry_delay")

    def _get_num_concurrent_requests(self, default: int = 1) -> int:
        """Resolve the number of concurrent HTTP requests a load test should fire.

        The historical key ``num_of_devices`` conflates three different
        concepts (physical chip count, per-model data-parallel factor, and
        client-side concurrency). Load tests only care about concurrency, so
        they read the new key first and fall back for back-compat.
        """
        if NUM_CONCURRENT_REQUESTS_KEY in self.targets:
            return self.targets[NUM_CONCURRENT_REQUESTS_KEY]

        if NUM_OF_DEVICES_KEY in self.targets:
            if not getattr(self, "_warned_num_of_devices", False):
                logger.warning(
                    "targets.%s is deprecated for load tests; please use %s instead.",
                    NUM_OF_DEVICES_KEY,
                    NUM_CONCURRENT_REQUESTS_KEY,
                )
                self._warned_num_of_devices = True
            return self.targets[NUM_OF_DEVICES_KEY]

        return default

    def run_tests(self):
        last_exception = None

        for attempt in range(self.retry_attempts + 1):
            try:
                logger.info(
                    f"Running tests (attempt {attempt + 1}/{self.retry_attempts + 1})"
                )

                result = asyncio.run(
                    asyncio.wait_for(
                        self._run_specific_test_async(), timeout=self.timeout
                    )
                )

                success = result.get("success", False)
                if success:
                    logger.info("Tests completed successfully")
                else:
                    logger.error("Tests failed")

                # Return both result and logs
                return {
                    "success": True if result.get("success", False) else False,
                    "result": result,
                    "logs": self.logs,
                    "attempts": attempt + 1,
                }

            except asyncio.TimeoutError as e:
                last_exception = e
                error_msg = f"Tests timed out after {self.timeout} seconds (attempt {attempt + 1}/{self.retry_attempts + 1})"
                logger.error(error_msg)
                self.logs.append(
                    {
                        "timestamp": time.time(),
                        "level": "ERROR",
                        "attempt": attempt + 1,
                        "type": "TimeoutError",
                        "message": error_msg,
                        "exception": str(e),
                    }
                )

            except SystemExit as e:
                last_exception = e
                error_msg = f"SystemExit encountered (attempt {attempt + 1}/{self.retry_attempts + 1}): {str(e)}"
                logger.error(error_msg)
                self.logs.append(
                    {
                        "timestamp": time.time(),
                        "level": "ERROR",
                        "attempt": attempt + 1,
                        "type": "SystemExit",
                        "message": error_msg,
                        "exception": str(e),
                    }
                )
                # Include logs in SystemExit for immediate access
                raise SystemExit(f"{str(e)}\nLogs: {self.logs}")

            except Exception as e:
                last_exception = e
                error_msg = f"Test failed with exception (attempt {attempt + 1}/{self.retry_attempts + 1}): {str(e)}"
                logger.error(error_msg)
                logger.error(f"Exception type: {type(e).__name__}")
                traceback_str = traceback.format_exc()
                logger.error(f"Traceback: {traceback_str}")
                self.logs.append(
                    {
                        "timestamp": time.time(),
                        "level": "ERROR",
                        "attempt": attempt + 1,
                        "type": type(e).__name__,
                        "message": error_msg,
                        "exception": str(e),
                        "traceback": traceback_str,
                    }
                )

            if attempt < self.retry_attempts:
                logger.info(f"Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)

        # All retries exhausted - return failure with logs
        logger.error(f"All {self.retry_attempts + 1} attempts failed. Last exception:")

        if self.config.get("break_on_failure"):
            raise SystemExit(f"Test failed after all retries. Logs: {self.logs}")

        if last_exception:
            # Attach logs to the exception
            last_exception.test_logs = self.logs
            raise last_exception
        else:
            error = RuntimeError("Tests failed after all retry attempts")
            error.test_logs = self.logs
            raise error

    def get_logs(self):
        """Get all logs collected during test execution"""
        return self.logs

    @abstractmethod
    async def _run_specific_test_async(self):
        pass
