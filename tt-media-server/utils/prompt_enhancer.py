# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Prompt enhancers: an outside service that expands a plain prompt into the
structured caption a model was trained on.

GenerationService calls one from ``pre_process``, while the job is still
`queued` and before it reaches a device, so the device never waits on it.
"""

import asyncio
import time
from typing import Any, Awaitable, Callable, Optional, Protocol

import httpx
from config.settings import Settings, get_settings
from telemetry.telemetry_client import prompt_enhancement_call_duration
from utils.job_manager import JobFailure
from utils.logger import TTLogger

logger = TTLogger()

PROMPT_ENHANCEMENT_ERROR = "prompt_enhancement_error"

# The job messages clients see. They don't name the service behind the enhancer.
REJECTED = "The prompt enhancer rejected the prompt."
UNAVAILABLE = "The prompt enhancer is unavailable. Resubmit later."
UNUSABLE = "The prompt enhancer returned an unusable caption. Resubmit."

# Longest wait before a retry, whatever Retry-After asks for.
_MAX_RETRY_DELAY_SECONDS = 30.0


class PromptEnhancementError(JobFailure):
    """Fails the job with `prompt_enhancement_error`."""

    def __init__(self, message: str) -> None:
        super().__init__(PROMPT_ENHANCEMENT_ERROR, message)


class PromptEnhancer(Protocol):
    async def enhance(self, prompt: str, aspect_ratio: str) -> Any:
        """The service's caption for ``prompt`` at ``aspect_ratio`` ("W:H"),
        unvalidated; the request model checks it."""
        ...


class IdeogramMagicPrompt:
    """Ideogram's hosted magic-prompt API for Ideogram 4.

    ``POST {url}`` with ``{"text_prompt", "aspect_ratio": "WxH"}`` returns
    ``{"json_prompt": {...}, "aspect_ratio": ...}``. Timeouts, 429 and 5xx are
    retried with a backoff; any other error fails at once.
    """

    def __init__(
        self,
        *,
        api_key: str,
        url: str,
        timeout_seconds: float,
        max_in_flight: int,
        max_retries: int,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if not api_key:
            raise ValueError(
                "The Ideogram 4 prompt enhancer needs IDEOGRAM_API_KEY; set it, or "
                "set IDEOGRAM_PROMPT_ENHANCER=none to serve structured_prompt only"
            )
        self._api_key = api_key
        self._url = url
        self._timeout = timeout_seconds
        self._max_in_flight = max_in_flight
        self._max_retries = max_retries
        self._transport = transport
        self._sleep = sleep
        # Per event loop, as utils.media_downloader does: tests run several.
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._client: Optional[httpx.AsyncClient] = None
        self._semaphore: Optional[asyncio.Semaphore] = None

    def _loop_state(self) -> tuple[httpx.AsyncClient, asyncio.Semaphore]:
        loop = asyncio.get_running_loop()
        if self._loop is not loop or self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                transport=self._transport,
                headers={"User-Agent": "tt-media-server"},
            )
            self._semaphore = asyncio.Semaphore(self._max_in_flight)
            self._loop = loop
        return self._client, self._semaphore

    async def enhance(self, prompt: str, aspect_ratio: str) -> Any:
        client, semaphore = self._loop_state()
        body = {"text_prompt": prompt, "aspect_ratio": aspect_ratio.replace(":", "x")}
        for attempt in range(self._max_retries + 1):
            async with semaphore:
                start = time.monotonic()
                outcome, delay, result = await self._call(client, body)
                prompt_enhancement_call_duration.labels(outcome=outcome).observe(
                    time.monotonic() - start
                )
            if outcome == "ok":
                return result
            if delay is None or attempt == self._max_retries:
                logger.warning(
                    f"Prompt enhancer failed after {attempt + 1} call(s): {outcome}"
                )
                raise PromptEnhancementError(
                    REJECTED if outcome == "rejected" else UNAVAILABLE
                )
            logger.info(
                f"Prompt enhancer call {attempt + 1} failed ({outcome}); "
                f"retrying in {delay:.1f} s"
            )
            await self._sleep(delay)
        raise AssertionError("unreachable")

    async def _call(
        self, client: httpx.AsyncClient, body: dict
    ) -> tuple[str, Optional[float], Any]:
        """(outcome, delay before a retry or None to give up, json_prompt)."""
        backoff = None
        try:
            response = await client.post(
                self._url,
                json=body,
                headers={"Api-Key": self._api_key},
                timeout=self._timeout,
            )
        except httpx.TimeoutException:
            return "timeout", self._backoff(), None
        except httpx.TransportError:
            return "transport_error", self._backoff(), None

        status = response.status_code
        if status == 200:
            try:
                return "ok", None, response.json().get("json_prompt")
            except (ValueError, AttributeError):
                # Not JSON, or not an object: unusable like a bad caption.
                return "ok", None, None
        if status in (400, 422):
            return "rejected", backoff, None
        if status in (401, 403):
            # The deployment's key, not the request: needs an operator.
            logger.error(
                f"Prompt enhancer: the API key was refused (HTTP {status}); "
                "check IDEOGRAM_API_KEY"
            )
            return "auth_error", backoff, None
        if status == 429:
            return "rate_limited", self._retry_after(response), None
        if status >= 500:
            return "server_error", self._retry_after(response), None
        return "other_error", backoff, None

    def _backoff(self) -> float:
        return 1.0

    def _retry_after(self, response: httpx.Response) -> float:
        try:
            seconds = float(response.headers.get("Retry-After", ""))
        except ValueError:
            return self._backoff()
        return min(max(seconds, 0.0), _MAX_RETRY_DELAY_SECONDS)


_enhancer: Optional[IdeogramMagicPrompt] = None


def get_ideogram_prompt_enhancer(
    settings: Optional[Settings] = None,
) -> Optional[PromptEnhancer]:
    """The deployment's Ideogram 4 enhancer, or None when it is turned off.

    Raises ValueError when it is on but has no API key, so a misconfigured
    deployment fails at startup rather than on its first request.
    """
    global _enhancer
    settings = settings or get_settings()
    if settings.ideogram_prompt_enhancer == "none":
        return None
    if _enhancer is None:
        _enhancer = IdeogramMagicPrompt(
            api_key=settings.ideogram_api_key.get_secret_value(),
            url=settings.ideogram_magic_prompt_url,
            timeout_seconds=settings.prompt_enhancer_timeout_seconds,
            max_in_flight=settings.prompt_enhancer_max_in_flight,
            max_retries=settings.prompt_enhancer_max_retries,
        )
    return _enhancer


_ENHANCERS: dict[str, Callable[..., Optional[PromptEnhancer]]] = {
    "ideogram": get_ideogram_prompt_enhancer,
}


def get_prompt_enhancer(name: str) -> Optional[PromptEnhancer]:
    """The enhancer a request's ``PROMPT_ENHANCER`` names, or None when the
    deployment turned it off."""
    return _ENHANCERS[name]()
