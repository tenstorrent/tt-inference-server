# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Tests for utils.error_response_sanitizer (tenstorrent/tt-cloud-console#1152).

The first group drives the pure-ASGI middleware directly with a fake app so it
needs no web framework. The last test rebuilds the tenstorrent/vllm fork's
``validation_exception_handler`` on top of the installed FastAPI and checks that
the exact body Cobalt reported (endpoint file path, container username, Pydantic
wrap-validator marker) is cleaned end to end; it is skipped when FastAPI is not
installed.
"""

import asyncio
import json
import re

import pytest

from utils import error_response_sanitizer as ers
from utils.error_response_sanitizer import (
    ERROR_SANITIZER_MIDDLEWARE,
    ErrorResponseSanitizerMiddleware,
    sanitize_error_body,
    sanitize_error_text,
)

# Verbatim from Cobalt PT40893_8 (2026-08-29), response to `POST /tokenize {}`.
COBALT_MESSAGE = (
    "2 validation errors:\n"
    "  {'type': 'missing', 'loc': ('body', 'function-wrap[__log_extra_fields__()]', "
    "'prompt'), 'msg': 'Field required', 'input': {}}\n"
    "  {'type': 'missing', 'loc': ('body', 'function-wrap[__log_extra_fields__()]', "
    "'messages'), 'msg': 'Field required', 'input': {}}\n\n"
    '  File "/home/container_app_user/vllm/vllm/entrypoints/utils.py", line 38, '
    "in tokenize\n"
    "    POST /tokenize [{'type': 'missing', 'loc': ('body', "
    "'function-wrap[__log_extra_fields__()]', 'prompt'), 'msg': 'Field required', "
    "'input': {}}, {'type': 'missing', 'loc': ('body', "
    "'function-wrap[__log_extra_fields__()]', 'messages'), 'msg': 'Field required', "
    "'input': {}}]"
)

LEAK_MARKERS = (
    "container_app_user",
    "/home/",
    'File "',
    "line 38",
    "function-wrap",
    "__log_extra_fields__",
    "entrypoints/utils.py",
    "POST /tokenize",
)


def assert_clean(text: str) -> None:
    for marker in LEAK_MARKERS:
        assert marker not in text, f"{marker!r} still present in {text!r}"


# --------------------------------------------------------------------------- #
# sanitize_error_text
# --------------------------------------------------------------------------- #


def test_sanitize_error_text_cleans_cobalt_validation_message():
    cleaned = sanitize_error_text(COBALT_MESSAGE)

    assert_clean(cleaned)
    # The user-actionable part survives, with the internal loc marker removed.
    assert cleaned == (
        "2 validation errors:\n"
        "  {'type': 'missing', 'loc': ('body', 'prompt'), 'msg': 'Field required', "
        "'input': {}}\n"
        "  {'type': 'missing', 'loc': ('body', 'messages'), 'msg': 'Field required', "
        "'input': {}}"
    )


def test_sanitize_error_text_strips_traceback_but_keeps_exception_line():
    message = (
        "Traceback (most recent call last):\n"
        '  File "/home/container_app_user/tt-metal/python_env/lib/python3.10/'
        'site-packages/vllm/engine.py", line 42, in step\n'
        "    x = foo()\n"
        "        ^^^^^\n"
        "ValueError: prompt too long <SamplingParams at 0x7f00deadbeef> "
        "for /mnt/mldata/weights/model.safetensors"
    )

    cleaned = sanitize_error_text(message)

    assert cleaned == "ValueError: prompt too long <SamplingParams> for <path>"


def test_sanitize_error_text_handles_endpoint_only_context():
    # FastAPI emits this form when it has the route path but no source info.
    cleaned = sanitize_error_text(
        "1 validation error:\n  {'x': 1}\n  Endpoint: /tokenize"
    )
    assert cleaned == "1 validation error:\n  {'x': 1}"


@pytest.mark.parametrize(
    "message",
    [
        "",
        "The model `meta-llama/Llama-3.1-8B-Instruct` does not exist.",
        "Unsupported route. Try GET /v1/models or POST /v1/chat/completions.",
        "max_tokens must be at least 1, got 0",
        '{"error":"Unauthorized"}',
    ],
)
def test_sanitize_error_text_leaves_legitimate_messages_alone(message):
    assert sanitize_error_text(message) == message


def test_sanitize_error_text_bounds_length():
    cleaned = sanitize_error_text("x" * (ers.MAX_ERROR_STRING_CHARS + 500))
    assert len(cleaned) == ers.MAX_ERROR_STRING_CHARS + len("...[truncated]")
    assert cleaned.endswith("...[truncated]")


# --------------------------------------------------------------------------- #
# sanitize_error_body
# --------------------------------------------------------------------------- #


def _openai_envelope(message: str, status: int = 400) -> bytes:
    return json.dumps(
        {
            "error": {
                "message": message,
                "type": "Bad Request",
                "param": None,
                "code": status,
            }
        }
    ).encode()


def test_sanitize_error_body_rewrites_openai_envelope_and_keeps_shape():
    out = sanitize_error_body(_openai_envelope(COBALT_MESSAGE), 400, "application/json")

    parsed = json.loads(out)
    assert set(parsed["error"]) == {"message", "type", "param", "code"}
    assert parsed["error"]["type"] == "Bad Request"
    assert parsed["error"]["code"] == 400
    assert_clean(parsed["error"]["message"])


def test_sanitize_error_body_returns_identical_bytes_when_nothing_to_scrub():
    body = _openai_envelope("max_tokens must be at least 1, got 0")
    assert sanitize_error_body(body, 400, "application/json") is body


def test_sanitize_error_body_handles_fastapi_detail_and_nested_lists():
    body = json.dumps(
        {
            "detail": [
                {
                    "loc": ["body", "function-wrap[__log_extra_fields__()]", "prompt"],
                    "msg": 'see File "/usr/lib/x.py", line 1, in f\n    boom',
                }
            ]
        }
    ).encode()

    parsed = json.loads(sanitize_error_body(body, 422, "application/json"))

    assert parsed["detail"][0]["msg"] == "see"
    # Structure (a JSON list, not a repr'd tuple) is preserved; only strings change.
    assert parsed["detail"][0]["loc"][0] == "body"


def test_sanitize_error_body_scrubs_plain_text():
    out = sanitize_error_body(
        b'Internal Server Error\n  File "/opt/app/x.py", line 3, in g\n    raise',
        500,
        "text/plain; charset=utf-8",
    )
    assert out == b"Internal Server Error"


def test_sanitize_error_body_leaves_binary_and_unknown_types_alone():
    binary = b"\xff\xfe\x00binary"
    assert sanitize_error_body(binary, 500, "application/octet-stream") is binary
    html = b"<html>/home/container_app_user</html>"
    assert sanitize_error_body(html, 502, "text/html") == b"<html><path></html>"
    assert sanitize_error_body(b"", 500, "application/json") == b""


# --------------------------------------------------------------------------- #
# ErrorResponseSanitizerMiddleware (raw ASGI, no framework)
# --------------------------------------------------------------------------- #


def _run_asgi(app, method="POST", path="/tokenize"):
    """Drive an ASGI app once and return (status, headers dict, body bytes)."""
    scope = {"type": "http", "method": method, "path": path, "headers": []}
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))

    start = next(m for m in sent if m["type"] == "http.response.start")
    body = b"".join(
        m.get("body", b"") for m in sent if m["type"] == "http.response.body"
    )
    headers = {k.decode().lower(): v.decode() for k, v in start["headers"]}
    return start["status"], headers, body, sent


def _json_app(status: int, payload, chunks: int = 1, content_type=b"application/json"):
    body = json.dumps(payload).encode()
    sent_chunks = []

    async def app(scope, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", content_type),
                    (b"content-length", str(len(body)).encode()),
                    (b"x-request-id", b"abc"),
                ],
            }
        )
        step = max(1, -(-len(body) // chunks))  # ceil -> at most `chunks` pieces
        pieces = [body[i : i + step] for i in range(0, len(body), step)]
        for piece in pieces[:-1]:
            sent_chunks.append(piece)
            await send({"type": "http.response.body", "body": piece, "more_body": True})
        sent_chunks.append(pieces[-1])
        await send(
            {"type": "http.response.body", "body": pieces[-1], "more_body": False}
        )

    app.sent_chunks = sent_chunks
    return app


def test_middleware_rewrites_400_body_and_fixes_content_length():
    app = ErrorResponseSanitizerMiddleware(
        _json_app(
            400,
            {
                "error": {
                    "message": COBALT_MESSAGE,
                    "type": "Bad Request",
                    "param": None,
                    "code": 400,
                }
            },
        )
    )

    status, headers, body, _ = _run_asgi(app)

    assert status == 400
    assert headers["content-type"] == "application/json"
    assert headers["x-request-id"] == "abc"  # other headers are preserved
    assert int(headers["content-length"]) == len(body)
    assert_clean(body.decode())
    assert json.loads(body)["error"]["code"] == 400


def test_middleware_reassembles_chunked_error_bodies():
    payload = {
        "error": {
            "message": COBALT_MESSAGE,
            "type": "Bad Request",
            "param": None,
            "code": 400,
        }
    }
    app = ErrorResponseSanitizerMiddleware(_json_app(400, payload, chunks=7))

    status, headers, body, sent = _run_asgi(app)

    assert status == 400
    assert int(headers["content-length"]) == len(body)
    assert_clean(body.decode())
    # One start + one body message reach the client, regardless of upstream chunking.
    assert [m["type"] for m in sent] == ["http.response.start", "http.response.body"]


def test_middleware_passes_success_responses_through_untouched_and_unbuffered():
    leaky_but_ok = {
        "choices": [{"text": 'File "/home/container_app_user/x.py", line 1, in f'}]
    }
    inner = _json_app(200, leaky_but_ok, chunks=3)
    app = ErrorResponseSanitizerMiddleware(inner)

    status, headers, body, sent = _run_asgi(app)

    assert status == 200
    assert json.loads(body) == leaky_but_ok  # model output is never rewritten
    # Streaming shape preserved: start + every upstream chunk forwarded as-is.
    assert len(inner.sent_chunks) > 1
    assert [m.get("body") for m in sent[1:]] == inner.sent_chunks


def test_middleware_passes_compressed_error_bodies_through():
    body = b"\x1f\x8b-not-really-gzip-/home/container_app_user"

    async def app(scope, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": 500,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-encoding", b"gzip"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body, "more_body": False})

    status, _, out, _ = _run_asgi(ErrorResponseSanitizerMiddleware(app))
    assert status == 500
    assert out == body  # cannot safely edit; do not corrupt it


def test_middleware_replaces_oversized_error_body_with_bounded_generic_error(
    monkeypatch,
):
    monkeypatch.setattr(ers, "MAX_BUFFERED_ERROR_BODY_BYTES", 1024)
    huge = {
        "error": {
            "message": "x" * 5000,
            "type": "Bad Request",
            "param": None,
            "code": 400,
        }
    }
    app = ErrorResponseSanitizerMiddleware(_json_app(400, huge, chunks=10))

    status, headers, body, sent = _run_asgi(app)

    assert status == 400
    assert len(body) < 1024
    assert int(headers["content-length"]) == len(body)
    parsed = json.loads(body)
    assert parsed["error"]["code"] == 400
    assert parsed["error"]["type"] == "Bad Request"
    assert "withheld" in parsed["error"]["message"]
    assert [m["type"] for m in sent] == ["http.response.start", "http.response.body"]


def test_middleware_ignores_non_http_scopes():
    seen = []

    async def app(scope, receive, send):
        seen.append(scope["type"])

    asyncio.run(ErrorResponseSanitizerMiddleware(app)({"type": "lifespan"}, None, None))
    assert seen == ["lifespan"]


def test_middleware_logs_original_body_server_side(caplog):
    app = ErrorResponseSanitizerMiddleware(
        _json_app(
            400,
            {
                "error": {
                    "message": COBALT_MESSAGE,
                    "type": "Bad Request",
                    "param": None,
                    "code": 400,
                }
            },
        )
    )
    with caplog.at_level("WARNING", logger=ers.__name__):
        _run_asgi(app)

    assert any("container_app_user" in r.getMessage() for r in caplog.records), (
        "verbose details must still reach the server log"
    )


def test_middleware_import_path_constant_resolves():
    module_path, class_name = ERROR_SANITIZER_MIDDLEWARE.rsplit(".", 1)
    assert module_path == ers.__name__
    assert getattr(ers, class_name) is ErrorResponseSanitizerMiddleware


# --------------------------------------------------------------------------- #
# End to end with FastAPI: the fork's handler + FastAPI >= 0.124 endpoint context
# --------------------------------------------------------------------------- #

fastapi = pytest.importorskip("fastapi")


def _build_fork_like_app():
    """Replica of tenstorrent/vllm's validation_exception_handler (fork @7678b70)."""
    from http import HTTPStatus

    from fastapi import FastAPI, Request
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, model_validator

    def fork_sanitize_message(message: str) -> str:
        return re.sub(r" at 0x[0-9a-f]+>", ">", message)

    class OpenAIBaseModel(BaseModel):
        @model_validator(mode="wrap")
        @classmethod
        def __log_extra_fields__(cls, data, handler):
            return handler(data)

    class TokenizeRequest(OpenAIBaseModel):
        prompt: str
        messages: list

    app = FastAPI()

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_: Request, exc: RequestValidationError):
        errors = exc.errors()
        exc_str = str(exc)
        errors_str = str(errors)
        if errors and errors_str and errors_str != exc_str:
            message = f"{exc_str} {errors_str}"
        else:
            message = exc_str
        return JSONResponse(
            {
                "error": {
                    "message": fork_sanitize_message(message),
                    "type": HTTPStatus.BAD_REQUEST.phrase,
                    "param": None,
                    "code": HTTPStatus.BAD_REQUEST,
                }
            },
            status_code=HTTPStatus.BAD_REQUEST,
        )

    @app.post("/tokenize")
    async def tokenize(request: TokenizeRequest, raw_request: Request):
        return {"count": 1}

    return app


def test_end_to_end_fastapi_tokenize_empty_body_no_longer_leaks():
    from fastapi.testclient import TestClient

    unprotected_app = _build_fork_like_app()
    unprotected = (
        TestClient(unprotected_app)
        .post("/tokenize", json={})
        .json()["error"]["message"]
    )

    protected_app = _build_fork_like_app()
    # Same mechanism vLLM uses for `--middleware <Class>` (before startup).
    protected_app.add_middleware(ErrorResponseSanitizerMiddleware)
    response = TestClient(protected_app).post("/tokenize", json={})
    protected = response.json()["error"]

    assert response.status_code == 400
    assert "Field required" in protected["message"]
    assert_clean(protected["message"])
    if 'File "' in unprotected:
        # Installed FastAPI reproduces the leak (>= 0.124): prove we removed it.
        assert __file__ in unprotected
        assert __file__ not in protected["message"]
