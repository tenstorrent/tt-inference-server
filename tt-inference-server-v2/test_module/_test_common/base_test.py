# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from __future__ import annotations

import asyncio
import logging
import os
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

import requests

from report_module.schema import Block

from .blockify import block_id, block_targets
from .test_classes import TestConfig

if TYPE_CHECKING:
    from ..context import MediaContext

logger = logging.getLogger(__name__)

# Targets key used by load tests to control concurrent HTTP request fan-out.
# Introduced to replace the overloaded `num_of_devices` key (which also meant
# "physical chip count" in hardware_defaults / liveness tests).
NUM_CONCURRENT_REQUESTS_KEY = "num_concurrent_requests"
NUM_OF_DEVICES_KEY = "num_of_devices"


@dataclass(frozen=True)
class HealthCheckConfig:
    """Health check configuration."""

    MAX_ATTEMPTS: int = 230
    RETRY_DELAY: int = 10
    TIMEOUT: int = 10


HEALTH_CHECK_CONFIG = HealthCheckConfig()


class BaseTest(ABC):
    # Subclasses set these so run_tests() can build a properly-tagged Block.
    KIND: str = "base"
    TASK_TYPE: str = "infra"

    def __init__(
        self,
        config: TestConfig,
        targets: Dict[str, Any],
        description: str = "",
        ctx: Optional["MediaContext"] = None,
    ):
        self.config = config
        self.targets = targets
        self.description = description
        self.ctx = ctx
        # Prefer ctx.service_port over the env var so sweep-orchestrated runs
        # don't depend on SERVICE_PORT being exported in the shell.
        self.service_port = (
            str(ctx.service_port)
            if ctx is not None
            else os.getenv("SERVICE_PORT", "8000")
        )
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
            return self._coerce_concurrency(
                self.targets[NUM_CONCURRENT_REQUESTS_KEY],
                NUM_CONCURRENT_REQUESTS_KEY,
            )

        if NUM_OF_DEVICES_KEY in self.targets:
            if not getattr(self, "_warned_num_of_devices", False):
                logger.warning(
                    "targets.%s is deprecated for load tests; please use %s instead.",
                    NUM_OF_DEVICES_KEY,
                    NUM_CONCURRENT_REQUESTS_KEY,
                )
                self._warned_num_of_devices = True
            return self._coerce_concurrency(
                self.targets[NUM_OF_DEVICES_KEY],
                NUM_OF_DEVICES_KEY,
            )

        return default

    @staticmethod
    def _coerce_concurrency(raw: Any, key: str) -> int:
        try:
            value = int(raw)
        except (TypeError, ValueError) as e:
            raise ValueError(f"targets.{key} must be an int >= 1, got {raw!r}") from e
        if value < 1:
            raise ValueError(f"targets.{key} must be >= 1, got {value}")
        return value

    def _block(self, data: Dict[str, Any]) -> Block:
        """Wrap ``data`` in a Block tagged with this test's KIND/TASK_TYPE."""
        if self.ctx is not None:
            bid = block_id(self.ctx) or None
            targets = block_targets(self.ctx, task_type=self.TASK_TYPE)
        else:
            bid = None
            targets = {"task_type": self.TASK_TYPE}
        return Block(kind=self.KIND, id=bid, targets=targets, data=data)

    def run_tests(self) -> Block:
        """Run the test with retry/log accounting and return a Block.

        On success, ``Block.data`` carries the envelope keys
        (``success``, ``attempts``, ``logs``, ``elapsed_seconds``,
        ``test_name``, ``description``) merged with the test's own
        result dict at the top level — so a result like
        ``{"runner_in_use": "vllm", "ttft": 0.18}`` becomes
        ``{"success": True, "attempts": 1, "logs": [...],
        "runner_in_use": "vllm", "ttft": 0.18}``.

        Spreading at the top level (rather than nesting under ``"result"``)
        lets the report renderer recurse one more level into nested test
        data — otherwise dict-of-dicts result fields get JSON-blobbed into
        a single cell.

        Non-dict results fall back to the legacy ``{"result": <raw>}`` shape.
        Envelope keys take precedence on collision so meta-info stays
        reliable.

        On failure (all retries exhausted)::
            {"success": False, "attempts": int, "logs": list,
             "elapsed_seconds": float, "test_name": str,
             "description": str, "error": {"type": str, "message": str}}

        ``break_on_failure=True`` re-raises ``SystemExit`` after building the
        failure Block so callers that opt into hard-fail still get one;
        otherwise the Block is returned and the caller decides.
        """
        last_exception: Optional[BaseException] = None
        attempts_used = 0
        run_started = time.monotonic()

        for attempt in range(self.retry_attempts + 1):
            attempts_used = attempt + 1
            try:
                logger.info(
                    f"Running tests (attempt {attempts_used}/{self.retry_attempts + 1})"
                )

                result = asyncio.run(
                    asyncio.wait_for(
                        self._run_specific_test_async(), timeout=self.timeout
                    )
                )

                # Tests that don't explicitly mark success are treated as
                # passing once the coroutine returns without raising. The
                # legacy "default to False" was a latent bug for tests like
                # MediaServerLivenessTest that just return the response body.
                if isinstance(result, dict) and "success" in result:
                    success = bool(result["success"])
                else:
                    success = True

                if success:
                    logger.info("Tests completed successfully")
                else:
                    logger.error("Tests failed")

                if isinstance(result, dict):
                    data: Dict[str, Any] = {**result}
                else:
                    data = {"result": result}
                data["success"] = success
                data["attempts"] = attempts_used
                data["logs"] = list(self.logs)
                data["elapsed_seconds"] = time.monotonic() - run_started
                data["test_name"] = type(self).__name__
                data["description"] = self.description
                return self._block(data)

            except asyncio.TimeoutError as e:
                last_exception = e
                error_msg = (
                    f"Tests timed out after {self.timeout} seconds "
                    f"(attempt {attempts_used}/{self.retry_attempts + 1})"
                )
                logger.error(error_msg)
                self.logs.append(
                    {
                        "timestamp": time.time(),
                        "level": "ERROR",
                        "attempt": attempts_used,
                        "type": "TimeoutError",
                        "message": error_msg,
                        "exception": str(e),
                    }
                )

            except SystemExit as e:
                last_exception = e
                error_msg = (
                    f"SystemExit encountered "
                    f"(attempt {attempts_used}/{self.retry_attempts + 1}): {e}"
                )
                logger.error(error_msg)
                self.logs.append(
                    {
                        "timestamp": time.time(),
                        "level": "ERROR",
                        "attempt": attempts_used,
                        "type": "SystemExit",
                        "message": error_msg,
                        "exception": str(e),
                    }
                )
                # SystemExit short-circuits the retry loop — no further attempts.
                break

            except Exception as e:
                last_exception = e
                error_msg = (
                    f"Test failed with exception "
                    f"(attempt {attempts_used}/{self.retry_attempts + 1}): {e}"
                )
                logger.error(error_msg)
                logger.error(f"Exception type: {type(e).__name__}")
                traceback_str = traceback.format_exc()
                logger.error(f"Traceback: {traceback_str}")
                self.logs.append(
                    {
                        "timestamp": time.time(),
                        "level": "ERROR",
                        "attempt": attempts_used,
                        "type": type(e).__name__,
                        "message": error_msg,
                        "exception": str(e),
                        "traceback": traceback_str,
                    }
                )

            if attempt < self.retry_attempts:
                logger.info(f"Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)

        logger.error(f"All {attempts_used} attempt(s) failed.")

        error_payload = (
            {
                "type": type(last_exception).__name__,
                "message": str(last_exception),
            }
            if last_exception is not None
            else {
                "type": "RuntimeError",
                "message": "Tests failed after all retry attempts",
            }
        )
        failure_block = self._block(
            {
                "success": False,
                "attempts": attempts_used,
                "logs": list(self.logs),
                "elapsed_seconds": time.monotonic() - run_started,
                "test_name": type(self).__name__,
                "description": self.description,
                "error": error_payload,
            }
        )

        if self.break_on_failure:
            raise SystemExit(f"Test failed after all retries. Logs: {self.logs}")

        return failure_block

    def get_logs(self):
        """Get all logs collected during test execution"""
        return self.logs

    def wait_for_server_ready(
        self,
        service_port: Optional[int] = None,
        max_attempts: Optional[int] = None,
        retry_delay: Optional[int] = None,
    ) -> bool:
        """Wait for server to be ready using simple HTTP health check.

        Args:
            service_port: Port where the server is running.
            max_attempts: Maximum number of retry attempts.
            retry_delay: Seconds to wait between retries.

        Returns:
            bool: True if server is ready, False otherwise.
        """
        logger.info("Waiting for server to be ready...")
        service_port = int(
            service_port if service_port is not None else self.service_port
        )
        max_attempts = (
            max_attempts
            if max_attempts is not None
            else HEALTH_CHECK_CONFIG.MAX_ATTEMPTS
        )
        retry_delay = (
            retry_delay if retry_delay is not None else HEALTH_CHECK_CONFIG.RETRY_DELAY
        )
        health_url = f"http://localhost:{service_port}/tt-liveness"
        logger.info("Waiting for server at %s ...", health_url)

        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.get(health_url, timeout=HEALTH_CHECK_CONFIG.TIMEOUT)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "alive" and data.get("model_ready"):
                        logger.info(
                            "Server is ready after %s attempt(s)",
                            attempt,
                        )
                        return True
                logger.info(
                    "Server not ready (attempt %s/%s), retrying in %ss...",
                    attempt,
                    max_attempts,
                    retry_delay,
                )
            except requests.exceptions.RequestException as e:
                logger.info(
                    "Health check failed (attempt %s/%s): %s, retrying in %ss...",
                    attempt,
                    max_attempts,
                    e,
                    retry_delay,
                )
            time.sleep(retry_delay)

        logger.error("Server health check failed after %s attempts", max_attempts)
        return False

    @abstractmethod
    async def _run_specific_test_async(self):
        pass
