# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Client of the console's prompt upsampler.

POST {prompt_upsampler_url}/v1/flux3/upsample {"prompt", "images"} returns
{"caption", "finish_reason", "revision", "request_id"}. The upsampler applies
the model's recipe (system prompts, decoding); this side sends the prompt and
the reference as the image model sees it, as a JPEG.
"""

from __future__ import annotations

import asyncio
import base64
import io
import time
from dataclasses import dataclass

import httpx
from config.settings import settings
from PIL import Image

PATH = "/v1/flux3/upsample"
# The upsampler's reference format: JPEG at quality 92.
JPEG_QUALITY = 92
RETRIES = 2
BACKOFF_SECONDS = 0.5
_RETRYABLE = frozenset({429, 500, 502, 503, 504})

# Tests point this at an in-process app.
_transport: httpx.AsyncBaseTransport | None = None


class UpsamplerUnavailable(Exception):
    """No caption: the upsampler was unreachable, overloaded, misconfigured or
    answered without one. Nothing reached the device, so resubmitting is safe."""

    error_code = "upsampler_unavailable"


class UpsamplerRejected(Exception):
    """The upsampler refused the request itself (too long, a bad image)."""


@dataclass(frozen=True)
class Upsampled:
    caption: str
    finish_reason: str
    revision: str | None
    request_id: str | None


def jpeg_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


async def upsample(prompt: str, images: list[Image.Image] = ()) -> Upsampled:
    """The upsampler's caption of ``prompt`` and its reference ``images``.

    Connection errors, timeouts, 429 and 5xx are retried within
    ``prompt_upsampler_timeout_seconds``, honouring Retry-After; then
    UpsamplerUnavailable. Another 4xx is UpsamplerRejected, but a refused
    credential (401, 403) is the server's fault: UpsamplerUnavailable.
    """
    url = settings.prompt_upsampler_url.rstrip("/")
    if not url:
        raise UpsamplerUnavailable(
            "The prompt upsampler is not configured on this server; no image was "
            "generated."
        )
    body = {"prompt": prompt, "images": [jpeg_data_uri(image) for image in images]}
    headers = {}
    if settings.prompt_upsampler_token:
        headers["Authorization"] = f"Bearer {settings.prompt_upsampler_token}"
    timeout = settings.prompt_upsampler_timeout_seconds
    started = time.monotonic()
    deadline = started + timeout
    last = "no attempt"
    attempts = 0
    async with httpx.AsyncClient(
        base_url=url, headers=headers, transport=_transport
    ) as client:
        for attempt in range(RETRIES + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            attempts += 1
            try:
                # wait_for bounds the whole exchange, whatever the transport does
                response = await asyncio.wait_for(
                    client.post(PATH, json=body, timeout=remaining), remaining
                )
            except (asyncio.TimeoutError, httpx.TimeoutException):
                last = "timed out"
                break
            except httpx.TransportError as e:
                last = f"unreachable ({type(e).__name__})"
                wait = BACKOFF_SECONDS * 2**attempt
            else:
                if response.status_code == 200:
                    return _result(response)
                if response.status_code in (401, 403):
                    raise UpsamplerUnavailable(
                        f"The prompt upsampler refused this server's credentials "
                        f"(HTTP {response.status_code}); no image was generated."
                    )
                if response.status_code not in _RETRYABLE:
                    raise UpsamplerRejected(
                        f"The prompt upsampler rejected the request (HTTP "
                        f"{response.status_code}): {_message(response)}"
                    )
                last = f"HTTP {response.status_code}"
                wait = _retry_after(response) or BACKOFF_SECONDS * 2**attempt
            if attempt == RETRIES or wait >= deadline - time.monotonic():
                break
            await asyncio.sleep(wait)
    raise UpsamplerUnavailable(
        f"The prompt upsampler was unavailable ({last}; {attempts} attempt"
        f"{'s' if attempts != 1 else ''} in {time.monotonic() - started:.1f} s). No "
        "image was generated; resubmitting is safe."
    )


def _result(response: httpx.Response) -> Upsampled:
    try:
        data = response.json()
        caption = (data.get("caption") or "").strip()
    except (ValueError, AttributeError):
        raise UpsamplerUnavailable(
            "The prompt upsampler answered with an unreadable response; no image "
            "was generated."
        )
    if not caption:
        raise UpsamplerUnavailable(
            "The prompt upsampler returned an empty caption; no image was generated."
        )
    return Upsampled(
        caption=caption,
        finish_reason=data.get("finish_reason") or "stop",
        revision=data.get("revision"),
        request_id=data.get("request_id"),
    )


def _message(response: httpx.Response) -> str:
    try:
        return str(response.json()["error"]["message"])[:500]
    except (ValueError, KeyError, TypeError):
        return response.text[:200]


def _retry_after(response: httpx.Response) -> float | None:
    try:
        return max(0.0, float(response.headers["Retry-After"]))
    except (KeyError, ValueError):
        return None
