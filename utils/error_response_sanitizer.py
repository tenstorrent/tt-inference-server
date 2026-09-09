# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Scrub server internals out of HTTP error responses.

Pure-ASGI middleware for the vLLM API server. Every 4xx/5xx body that passes
through it is rewritten so that it no longer discloses:

* absolute filesystem paths (and with them the container username),
* Python traceback frames (``File "...", line N, in func``) and headers,
* the endpoint context FastAPI >= 0.124 appends to ``str(RequestValidationError)``,
* Pydantic-internal validator markers in error ``loc`` tuples
  (``'function-wrap[__log_extra_fields__()]'``),
* object memory addresses.

2xx/3xx responses (including SSE streams) are passed through untouched, chunk by
chunk, so there is no cost on the hot path. Error bodies are buffered up to
``MAX_BUFFERED_ERROR_BODY_BYTES``; anything larger is replaced by a generic,
bounded error envelope so a malformed request can never amplify into a huge
response. The original body is logged server-side before it is rewritten.

The middleware has no third-party imports on purpose: it is installed via vLLM's
``--middleware <import.path.Class>`` flag and has to work unchanged across every
vLLM commit the release images pin (see ``run_vllm_api_server.py``).

Why here and not in vLLM: the tenstorrent/vllm fork's ``validation_exception_handler``
forwards ``str(exc)`` to the client, which since FastAPI 0.124 (Dec 2025) embeds the
endpoint's source file, line number and function name. Upstream vLLM fixed that in
vllm-project/vllm#46415 (v0.26.0+) but the fork never picked it up, and each release
image pins a different fork commit, so a fix in the fork would need every image to
move to a new vLLM. Wrapping the app instead fixes all of them on the next image
build without changing the pinned engine.
"""

import json
import logging
import re
from http import HTTPStatus
from typing import Any, Awaitable, Callable, Dict, List, MutableMapping, Tuple

logger = logging.getLogger(__name__)

# Import path handed to vLLM's ``--middleware`` flag by run_vllm_api_server.py.
ERROR_SANITIZER_MIDDLEWARE = (
    "utils.error_response_sanitizer.ErrorResponseSanitizerMiddleware"
)

# Error envelopes are small (a few hundred bytes to a few KB). Anything bigger is
# either a bug or an amplification attempt; either way the client gets a bounded
# generic body and the details stay in the server log.
MAX_BUFFERED_ERROR_BODY_BYTES = 256 * 1024
# Cap on any single string inside a JSON error body after sanitization.
MAX_ERROR_STRING_CHARS = 4096
# How much of the original body to keep in the server log when we rewrite it.
LOG_ORIGINAL_BODY_CHARS = 2000

_PATH_PLACEHOLDER = "<path>"

# Python traceback frame + the indented source/context line that follows it.
# Also matches FastAPI's endpoint context, which reuses the traceback format
# ('  File "<file>", line <n>, in <func>\n    <METHOD> <path>').
_TRACEBACK_FRAME_RE = re.compile(
    r'\n?[ \t]*File "[^"\n]+", line \d+, in [^\n]+(\n[ \t]+[^\n]*)?'
)
_TRACEBACK_HEADER_RE = re.compile(
    r"[ \t]*Traceback \(most recent call last\):[ \t]*\n?"
)
# Python 3.11+ caret lines under a traceback source line.
_TRACEBACK_CARETS_RE = re.compile(r"\n[ \t]*[\^~]+[ \t]*(?=\n|$)")
# FastAPI endpoint context when it has a path but no file information.
_ENDPOINT_CONTEXT_RE = re.compile(r"\n?[ \t]*Endpoint: \S+")
# Pydantic-internal validator markers inside a repr'd loc tuple, e.g.
# ('body', 'function-wrap[__log_extra_fields__()]', 'prompt') -> ('body', 'prompt')
_PYDANTIC_INTERNAL_LOC_RE = re.compile(r"'[\w-]+\[[^\]'\n]*\]',\s*")
# Absolute filesystem paths. First the well-known roots (matches directories too,
# so "/home/container_app_user" is caught even without a file name), then any
# multi-segment absolute path that ends in a file with an extension.
_KNOWN_ROOT_PATH_RE = re.compile(
    r"/(?:home|usr|opt|var|tmp|root|lib|lib64|mnt|srv|app|workspace|proc|etc|data)"
    r"(?:/[\w.\-]+)+"
)
_FILE_PATH_RE = re.compile(r"(?:/[\w\-]+)+/[\w\-]+\.\w+")
_MEMORY_ADDRESS_RE = re.compile(r" at 0x[0-9a-fA-F]+>")
_BLANK_LINES_RE = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)+")

_JSON_SEPARATORS = (",", ":")

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]
RawHeaders = List[Tuple[bytes, bytes]]


def sanitize_error_text(message: str) -> str:
    """Remove paths, traceback frames and internal markers from an error string."""
    if not message:
        return message
    message = _TRACEBACK_HEADER_RE.sub("", message)
    message = _TRACEBACK_FRAME_RE.sub("", message)
    message = _TRACEBACK_CARETS_RE.sub("", message)
    message = _ENDPOINT_CONTEXT_RE.sub("", message)
    message = _PYDANTIC_INTERNAL_LOC_RE.sub("", message)
    message = _KNOWN_ROOT_PATH_RE.sub(_PATH_PLACEHOLDER, message)
    message = _FILE_PATH_RE.sub(_PATH_PLACEHOLDER, message)
    message = _MEMORY_ADDRESS_RE.sub(">", message)
    message = _BLANK_LINES_RE.sub("\n", message)
    message = message.strip()
    if len(message) > MAX_ERROR_STRING_CHARS:
        message = message[:MAX_ERROR_STRING_CHARS] + "...[truncated]"
    return message


def _sanitize_json_value(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_error_text(value)
    if isinstance(value, dict):
        return {key: _sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_json_value(item) for item in value]
    return value


def _status_phrase(status: int) -> str:
    try:
        return HTTPStatus(status).phrase
    except ValueError:
        return "Error"


def generic_error_body(status: int, reason: str) -> bytes:
    """OpenAI-shaped error envelope used when the original body cannot be kept."""
    phrase = _status_phrase(status)
    payload = {
        "error": {
            "message": f"{phrase}. {reason}",
            "type": phrase,
            "param": None,
            "code": status,
        }
    }
    return json.dumps(payload, separators=_JSON_SEPARATORS).encode("utf-8")


def sanitize_error_body(body: bytes, status: int, content_type: str) -> bytes:
    """Return a sanitized copy of an error response body.

    JSON bodies are parsed and every string inside them is scrubbed, so the
    OpenAI ``{"error": {...}}`` envelope, FastAPI's ``{"detail": ...}`` and any
    other shape all keep their structure. Non-JSON text bodies are scrubbed as a
    whole. Bodies that cannot be decoded are returned unchanged.
    """
    if not body:
        return body
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return body

    media_type = content_type.split(";", 1)[0].strip().lower()
    looks_like_json = media_type.endswith("json") or text.lstrip()[:1] in "{["
    if looks_like_json:
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = None
        if parsed is not None and isinstance(parsed, (dict, list)):
            sanitized = _sanitize_json_value(parsed)
            if sanitized == parsed:
                return body
            return json.dumps(
                sanitized, ensure_ascii=False, separators=_JSON_SEPARATORS
            ).encode("utf-8")

    if media_type.startswith("text/") or not media_type:
        sanitized_text = sanitize_error_text(text)
        if sanitized_text == text.strip():
            return body
        return sanitized_text.encode("utf-8")

    return body


def _header_value(headers: RawHeaders, name: bytes) -> str:
    for key, value in headers:
        if key.lower() == name:
            return value.decode("latin-1")
    return ""


def _with_content_length(headers: RawHeaders, length: int) -> RawHeaders:
    kept = [(key, value) for key, value in headers if key.lower() != b"content-length"]
    kept.append((b"content-length", str(length).encode("latin-1")))
    return kept


class ErrorResponseSanitizerMiddleware:
    """Pure ASGI middleware: rewrite 4xx/5xx bodies, pass everything else through."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        state: Dict[str, Any] = {
            "start": None,  # held http.response.start for an error response
            "chunks": [],
            "buffered": 0,
            "done": False,  # response fully forwarded; swallow trailing chunks
            "passthrough": False,  # non-error (or undecodable) response
        }

        async def flush(body: bytes) -> None:
            start = state["start"]
            headers = _with_content_length(list(start.get("headers", [])), len(body))
            await send({**start, "headers": headers})
            await send({"type": "http.response.body", "body": body, "more_body": False})
            state["done"] = True

        async def send_wrapper(message: Message) -> None:
            message_type = message["type"]

            if message_type == "http.response.start":
                status = int(message.get("status", 200))
                headers = list(message.get("headers", []))
                encoding = _header_value(headers, b"content-encoding").lower()
                if status < 400 or (encoding and encoding != "identity"):
                    state["passthrough"] = True
                    await send(message)
                    return
                state["start"] = message
                return

            if message_type == "http.response.body" and state["start"] is not None:
                if state["done"]:
                    return
                body = message.get("body", b"") or b""
                state["chunks"].append(body)
                state["buffered"] += len(body)
                status = int(state["start"].get("status", 500))
                if state["buffered"] > MAX_BUFFERED_ERROR_BODY_BYTES:
                    logger.warning(
                        "Replaced oversized %d error body (%d+ bytes) for %s %s",
                        status,
                        state["buffered"],
                        scope.get("method", "?"),
                        scope.get("path", "?"),
                    )
                    await flush(
                        generic_error_body(
                            status,
                            "Error details were withheld because the response "
                            "exceeded the size limit; see server logs.",
                        )
                    )
                    return
                if message.get("more_body", False):
                    return

                original = b"".join(state["chunks"])
                content_type = _header_value(
                    state["start"].get("headers", []), b"content-type"
                )
                sanitized = sanitize_error_body(original, status, content_type)
                if sanitized != original:
                    logger.warning(
                        "Sanitized %d error body for %s %s; original: %s",
                        status,
                        scope.get("method", "?"),
                        scope.get("path", "?"),
                        original[:LOG_ORIGINAL_BODY_CHARS].decode("utf-8", "replace"),
                    )
                await flush(sanitized)
                return

            await send(message)

        await self.app(scope, receive, send_wrapper)
