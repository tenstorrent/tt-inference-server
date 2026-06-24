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
from types import SimpleNamespace
from typing import Iterable, Optional, Protocol, Tuple, runtime_checkable

import requests

logger = logging.getLogger(__name__)

DEFAULT_WAIT_HEALTHY_TIMEOUT_S = 1200.0
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


@dataclass(frozen=True)
class RemoteOpenAIController:
    """Readiness controller for remote OpenAI-compatible endpoints.

    Remote console endpoints do not expose vLLM's ``/health`` route, so use
    ``/v1/models`` as the readiness probe and skip trace warmup.
    """

    base_url: str
    auth_token: str = ""

    @property
    def api_base_url(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/v1"):
            return base
        return f"{base}/v1"

    @property
    def models_url(self) -> str:
        return f"{self.api_base_url}/models"

    @property
    def _headers(self) -> dict:
        if self.auth_token:
            return {"Authorization": f"Bearer {self.auth_token}"}
        return {}

    def get_health(
        self, timeout: Optional[float] = DEFAULT_POLL_INTERVAL_S
    ) -> requests.Response:
        return requests.get(self.models_url, headers=self._headers, timeout=timeout)

    def wait_for_healthy(
        self, timeout: Optional[float] = None, interval: int = DEFAULT_POLL_INTERVAL_S
    ) -> bool:
        from utils.remote_readiness import _wait_for_remote_openai_ready

        effective_timeout = (
            DEFAULT_WAIT_HEALTHY_TIMEOUT_S if timeout is None else float(timeout)
        )
        prompt_client_adapter = SimpleNamespace(
            headers=self._headers,
            _get_api_base_url=lambda: self.api_base_url,
        )
        return _wait_for_remote_openai_ready(
            prompt_client_adapter,
            timeout=effective_timeout,
            interval=interval,
        )

    def capture_traces(
        self,
        context_lens: Iterable[Tuple[int, int]],
        timeout: Optional[float] = None,
    ) -> None:
        return None
