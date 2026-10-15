# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Session-affine routing across vLLM data-parallel engines.

vLLM runs ``data_parallel_size`` EngineCore processes behind one API server.
Each engine owns its own KV cache, so the automatic prefix cache is per
engine, and the API server's load balancer (``DPLBAsyncMPClient``) picks an
engine per request from load counts only. A multi-turn conversation therefore
lands on a different engine every turn and warms every cache separately.

vLLM already accepts a per-request engine pin: the chat and completions
endpoints read the ``X-data-parallel-rank`` header and the load balancer uses
that rank before any scoring. This ASGI middleware sets that header from a
stable key so repeated requests from one session reach the engine that holds
their prefix.

Configuration comes from the environment because vLLM instantiates
``--middleware`` classes with the app only (``app.add_middleware(cls)``):

``TT_DP_AFFINITY``
    ``auto`` (default): pin only when the request carries an explicit session
    signal: the ``X-TT-Affinity-Key`` header, else the ``user`` field of the
    JSON body. Requests without one keep vLLM's load-aware balancing, so
    benchmark traffic (no ``user``, per-request ``cache_salt``) is unchanged.
    ``prefix``: as ``auto``, and otherwise hash the first
    ``TT_DP_AFFINITY_PREFIX_BYTES`` bytes of the prompt text (messages for
    chat, ``prompt`` for completions). Pins every request; use when callers
    send no session signal and the workload is multi-turn.
    ``off``: pass requests through untouched.
``TT_DP_AFFINITY_DP_SIZE``
    Number of engines (vLLM ``data_parallel_size``). Required when the mode
    is not ``off``; the launcher sets it.
``TT_DP_AFFINITY_PREFIX_BYTES``
    Prefix length for ``prefix`` mode (default 4096).

A client-supplied ``X-data-parallel-rank`` is honoured but reduced modulo the
engine count in every mode except ``off``: vLLM indexes its engine list with
the raw value and an out-of-range rank fails the request.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Awaitable, Callable, MutableMapping

logger = logging.getLogger(__name__)

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

RANK_HEADER = b"x-data-parallel-rank"
AFFINITY_KEY_HEADER = b"x-tt-affinity-key"
ROUTED_PATHS = ("/v1/chat/completions", "/v1/completions")
MODES = ("auto", "prefix", "off")
DEFAULT_PREFIX_BYTES = 4096


def _env_mode() -> str:
    mode = os.environ.get("TT_DP_AFFINITY", "auto").strip().lower() or "auto"
    if mode not in MODES:
        logger.warning(
            "TT_DP_AFFINITY=%r is not one of %s; falling back to 'off'", mode, MODES
        )
        return "off"
    return mode


def _env_int(name: str, default: int | None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using %r", name, raw, default)
        return default


def rank_for_key(key: str, dp_size: int) -> int:
    """Stable engine index for a session key (blake2b, so it survives restarts)."""
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dp_size


def prompt_prefix(body: Any, prefix_bytes: int) -> str | None:
    """The leading prompt text of a chat or completions body, for hashing.

    Chat: the message contents in order (string content, or the ``text`` parts
    of a content list), so every turn of one conversation shares the prefix of
    the first turn. Completions: the ``prompt`` string (or first prompt of a
    list). Returns ``None`` when there is nothing to hash.
    """
    if not isinstance(body, dict):
        return None
    pieces: list[str] = []
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str):
                pieces.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        pieces.append(part["text"])
            if sum(len(p) for p in pieces) >= prefix_bytes:
                break
    else:
        prompt = body.get("prompt")
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt and isinstance(prompt[0], str) else None
        if isinstance(prompt, str):
            pieces.append(prompt)
    text = "\n".join(pieces)
    if not text:
        return None
    return text.encode("utf-8")[:prefix_bytes].decode("utf-8", errors="ignore")


def affinity_key(headers: dict[bytes, bytes], body: Any, mode: str, prefix_bytes: int):
    """The routing key for a request, or ``None`` to leave balancing to vLLM."""
    explicit = headers.get(AFFINITY_KEY_HEADER)
    if explicit:
        return "key:" + explicit.decode("utf-8", errors="ignore")
    if isinstance(body, dict):
        user = body.get("user")
        if isinstance(user, str) and user:
            return "user:" + user
    if mode == "prefix":
        prefix = prompt_prefix(body, prefix_bytes)
        if prefix is not None:
            return "prefix:" + prefix
    return None


def _parse_json(raw: bytes) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return None


class DPAffinityMiddleware:
    """Pure ASGI middleware; registered with ``--middleware <module>.DPAffinityMiddleware``."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.mode = _env_mode()
        self.dp_size = _env_int("TT_DP_AFFINITY_DP_SIZE", None)
        self.prefix_bytes = _env_int(
            "TT_DP_AFFINITY_PREFIX_BYTES", DEFAULT_PREFIX_BYTES
        )
        if self.prefix_bytes is None or self.prefix_bytes <= 0:
            self.prefix_bytes = DEFAULT_PREFIX_BYTES
        if self.mode != "off" and (self.dp_size is None or self.dp_size < 1):
            logger.warning(
                "TT_DP_AFFINITY=%s needs TT_DP_AFFINITY_DP_SIZE >= 1; disabling",
                self.mode,
            )
            self.mode = "off"
        logger.info(
            "DP affinity middleware: mode=%s dp_size=%s prefix_bytes=%s",
            self.mode,
            self.dp_size,
            self.prefix_bytes,
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            self.mode == "off"
            or scope.get("type") != "http"
            or scope.get("method") != "POST"
            or scope.get("path") not in ROUTED_PATHS
        ):
            await self.app(scope, receive, send)
            return

        chunks: list[bytes] = []
        more_body = True
        while more_body:
            message = await receive()
            if message.get("type") == "http.disconnect":
                await self.app(scope, _replay([message]), send)
                return
            chunks.append(message.get("body", b""))
            more_body = message.get("more_body", False)
        raw = b"".join(chunks)

        headers: dict[bytes, bytes] = {
            k.lower(): v for k, v in scope.get("headers", [])
        }
        rank = self._choose_rank(headers, raw)
        if rank is not None:
            scope["headers"] = [
                (k, v) for k, v in scope.get("headers", []) if k.lower() != RANK_HEADER
            ] + [(RANK_HEADER, str(rank).encode("ascii"))]

        await self.app(
            scope,
            _replay([{"type": "http.request", "body": raw, "more_body": False}]),
            send,
        )

    def _choose_rank(self, headers: dict[bytes, bytes], raw: bytes) -> int | None:
        assert self.dp_size is not None
        client_rank = headers.get(RANK_HEADER)
        if client_rank is not None:
            try:
                return int(client_rank) % self.dp_size
            except ValueError:
                pass  # vLLM ignores a non-integer header; so do we
        key = affinity_key(headers, _parse_json(raw), self.mode, self.prefix_bytes)
        if key is None:
            return None
        return rank_for_key(key, self.dp_size)


def _replay(messages: list[Message]) -> Receive:
    queue = list(messages)

    async def receive() -> Message:
        if queue:
            return queue.pop(0)
        return {"type": "http.disconnect"}

    return receive
