# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Server control protocol used by the runner.

Mirrors the slice of v1's ``utils.prompt_client.PromptClient`` that
``run_benchmarks.py`` actually relies on (health-check, warm-up, trace
capture). Keeps llm_module decoupled from v1's workflow types: callers
pass anything that satisfies this protocol.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Iterable, Optional, Protocol, Tuple, runtime_checkable

import requests

logger = logging.getLogger(__name__)

DEFAULT_WAIT_HEALTHY_TIMEOUT_S = 3600.0
DEFAULT_HEALTH_PATH = "/health"
DEFAULT_POLL_INTERVAL_S = 10


@runtime_checkable
class ServerController(Protocol):
    def wait_for_healthy(
        self, timeout: Optional[float] = None, interval: int = DEFAULT_POLL_INTERVAL_S
    ) -> bool: ...

    def get_health(self, timeout: Optional[float] = DEFAULT_POLL_INTERVAL_S): ...

    def capture_traces(
        self,
        context_lens: Iterable[Tuple[int, int]],
        timeout: Optional[float] = None,
    ) -> None: ...


@dataclass(frozen=True)
class HttpServerController:
    base_url: str
    service_port: int
    auth_token: str = ""
    health_path: str = DEFAULT_HEALTH_PATH

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ValueError("HttpServerController requires a non-empty base_url")
        if int(self.service_port) <= 0:
            raise ValueError(
                f"HttpServerController requires a positive service_port, "
                f"got {self.service_port!r}"
            )

    @property
    def health_url(self) -> str:
        from utils.url_helpers import build_base_url

        base = build_base_url(self.base_url, self.service_port)
        return f"{base}{self.health_path}"

    @property
    def _headers(self) -> dict:
        if self.auth_token:
            return {"Authorization": f"Bearer {self.auth_token}"}
        return {}

    def get_health(
        self, timeout: Optional[float] = DEFAULT_POLL_INTERVAL_S
    ) -> requests.Response:
        return requests.get(self.health_url, headers=self._headers, timeout=timeout)

    def wait_for_healthy(
        self, timeout: Optional[float] = None, interval: int = DEFAULT_POLL_INTERVAL_S
    ) -> bool:
        effective_timeout = (
            DEFAULT_WAIT_HEALTHY_TIMEOUT_S if timeout is None else float(timeout)
        )
        deadline = time.time() + effective_timeout
        logger.info(
            "Waiting for inference server at %s (timeout %.0fs)",
            self.health_url,
            effective_timeout,
        )
        while time.time() < deadline:
            try:
                response = self.get_health(timeout=interval)
                if response.status_code == 200:
                    logger.info("✅ Inference server is healthy at %s", self.health_url)
                    return True
                logger.debug("Health check returned %s", response.status_code)
            except requests.exceptions.RequestException as exc:
                logger.debug("Health check not ready: %s", exc)
            time.sleep(interval)
        logger.error(
            "Inference server did not become healthy within %.0fs at %s",
            effective_timeout,
            self.health_url,
        )
        return False

    def capture_traces(
        self,
        context_lens: Iterable[Tuple[int, int]],
        timeout: Optional[float] = None,
    ) -> None:
        return None
