# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Request-body cap for the video generation routes.

The MiniMax input media card puts the whole request body at <= 64 MB and says
large files go by URL, not inline base64. FastAPI reads a JSON body completely
before pydantic sees it, so the cap has to sit in front of the app: this pure
ASGI middleware answers 413 from ``Content-Length`` before a byte of the body
is read, and, for a chunked upload without one, from the running byte count.

Scoped by path prefix to the video routes so the audio and chat services keep
their own limits (``max_audio_size_bytes`` etc.). Only body-carrying methods
are checked; ``max_bytes <= 0`` disables the middleware.

The chunked case raises from ``receive`` while FastAPI is reading the body.
FastAPI wraps any plain exception raised there into a 400 "error parsing the
body", but re-raises a ``fastapi.HTTPException``, so the error IS one (413)
and ``ExceptionMiddleware`` renders it with the same detail text.

Ordering contract: this middleware must sit INSIDE any ``BaseHTTPMiddleware``
(added to the app before it, since the last ``add_middleware`` is outermost).
A ``BaseHTTPMiddleware`` between here and the route reads the body inside an
anyio task group, which would wrap the 413 in an ``ExceptionGroup`` that
FastAPI turns into its 400. ``__call__`` still unwraps such a group as a last
resort for the case where nothing inside rendered a response.
"""

import json
from typing import Iterable

from fastapi import HTTPException

_BODY_METHODS = frozenset({"POST", "PUT", "PATCH"})
_DEFAULT_PREFIXES = ("/v1/videos", "/video")


def _detail_text(max_bytes: int, size_text: str) -> str:
    return (
        f"Request body is {size_text}, over the {max_bytes}-byte limit "
        f"({max_bytes / (1024 * 1024):g} MB) for video generation requests. "
        "Pass large media as URL sources instead of inline base64."
    )


class _BodyTooLarge(HTTPException):
    def __init__(self, received: int, max_bytes: int):
        super().__init__(
            status_code=413, detail=_detail_text(max_bytes, f"over {received} bytes")
        )
        self.received = received


def _find_body_too_large(exc: BaseException):
    """The ``_BodyTooLarge`` inside ``exc``, if any.

    A ``BaseHTTPMiddleware`` between this middleware and the route (the app has
    one) reads the body inside an anyio task group, so the error raised from
    ``receive`` surfaces wrapped in an ``ExceptionGroup`` (the ``exceptiongroup``
    backport on Python 3.10). Walk ``.exceptions`` instead of matching the type.
    """
    if isinstance(exc, _BodyTooLarge):
        return exc
    for inner in getattr(exc, "exceptions", ()) or ():
        found = _find_body_too_large(inner)
        if found is not None:
            return found
    return None


class RequestBodyLimitMiddleware:
    def __init__(
        self,
        app,
        *,
        max_bytes: int,
        path_prefixes: Iterable[str] = _DEFAULT_PREFIXES,
    ):
        self.app = app
        self.max_bytes = int(max_bytes)
        self.path_prefixes = tuple(path_prefixes)

    def _applies(self, scope) -> bool:
        if scope["type"] != "http" or self.max_bytes <= 0:
            return False
        if scope.get("method", "").upper() not in _BODY_METHODS:
            return False
        path = scope.get("path", "")
        return any(
            path == prefix or path.startswith(prefix + "/")
            for prefix in self.path_prefixes
        )

    async def _reject(self, send, size_text: str) -> None:
        body = json.dumps({"detail": _detail_text(self.max_bytes, size_text)}).encode(
            "utf-8"
        )
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                    (b"connection", b"close"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if not self._applies(scope):
            await self.app(scope, receive, send)
            return

        for name, value in scope.get("headers", ()):
            if name.lower() == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    break
                if declared > self.max_bytes:
                    await self._reject(send, f"{declared} bytes")
                    return
                break

        received = 0
        response_started = False

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLarge(received, self.max_bytes)
            return message

        async def tracking_send(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except BaseException as exc:  # noqa: BLE001 - unwrap, re-raise anything else
            too_large = _find_body_too_large(exc)
            if too_large is None or response_started:
                raise
            await self._reject(send, f"over {too_large.received} bytes")
