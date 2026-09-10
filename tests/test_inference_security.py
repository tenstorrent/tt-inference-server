# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import asyncio
import json
import os

import pytest

from utils import inference_security as security
from utils import secure_media


@pytest.fixture(autouse=True)
def policy_env(monkeypatch):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    for name in (
        "VLLM_API_KEY",
        security._POLICY_ENV,
        "TT_ALLOWED_MEDIA_DOMAINS",
        "TT_INFERENCE_ALLOWED_ROUTES",
        "VLLM_MEDIA_CONNECTOR",
        "VLLM_PLUGINS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(secure_media, "register_media_connector", lambda: None)


def call_guard(
    monkeypatch,
    path,
    *,
    headers=None,
    method="POST",
    routes=None,
    root_path="",
    no_auth=False,
):
    monkeypatch.setenv("VLLM_API_KEY", "" if no_auth else "test-key")
    if routes is not None:
        monkeypatch.setenv("TT_INFERENCE_ALLOWED_ROUTES", json.dumps(routes))
    security.configure_inference_security(["server"], no_auth=no_auth)
    sent, received, forwarded = [], [], []
    body = b'{"messages":[{"content":[{"type":"image_url","image_url":{"url":"https://images.example.com/a.png"}}]}]}'

    async def receive():
        received.append(True)
        return {"type": "http.request", "body": body}

    async def send(event):
        sent.append(event)

    async def app(scope, receive, send):
        forwarded.append(await receive())
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"first", "more_body": True})
        await send({"type": "http.response.body", "body": b"last"})

    scope = {
        "type": "http",
        "path": path,
        "root_path": root_path,
        "method": method,
        "headers": headers
        if headers is not None
        else [(b"authorization", b"Bearer test-key")],
    }
    asyncio.run(security.InferenceSecurityMiddleware(app)(scope, receive, send))
    return sent, received, forwarded, body


@pytest.mark.parametrize(
    "path", ["/invocations", "/tokenizer", "/tokenize", "/pause", "/v1/unknown"]
)
@pytest.mark.parametrize("headers", [[], [(b"authorization", b"Bearer test-key")]])
def test_unused_routes_are_blocked_before_body_read(monkeypatch, path, headers):
    sent, received, forwarded, _ = call_guard(monkeypatch, path, headers=headers)
    assert sent[0]["status"] == 404
    assert not received and not forwarded


@pytest.mark.parametrize(
    "path",
    ["/invocations", "/tokenizer", "/tokenize", "/v1/chat/completions", "/v1/models"],
)
@pytest.mark.parametrize(
    "headers",
    [
        [],
        [(b"authorization", b"Bearer wrong")],
        [(b"authorization", b"Basic test-key")],
        [(b"authorization", b"Bearer test-key"), (b"authorization", b"Bearer wrong")],
    ],
)
def test_opted_in_routes_require_authentication(monkeypatch, path, headers):
    sent, received, forwarded, _ = call_guard(
        monkeypatch, path, headers=headers, routes=[path]
    )
    assert sent[0]["status"] == 401
    assert not received and not forwarded


@pytest.mark.parametrize(
    "path", ["/v1/chat/completions", "/invocations", "/tokenize", "/tokenizer"]
)
def test_allowed_requests_and_streams_are_untouched(monkeypatch, path):
    sent, received, forwarded, body = call_guard(monkeypatch, path, routes=[path])
    assert sent[0]["status"] == 200
    assert received == [True]
    assert forwarded == [{"type": "http.request", "body": body}]
    assert sent[1:] == [
        {"type": "http.response.body", "body": b"first", "more_body": True},
        {"type": "http.response.body", "body": b"last"},
    ]


@pytest.mark.parametrize("path", ["/health", "/ping", "/metrics"])
def test_health_and_metrics_are_public(monkeypatch, path):
    assert (
        call_guard(monkeypatch, path, method="GET", headers=[])[0][0]["status"] == 200
    )


@pytest.mark.parametrize("path", ["/models/qwen/tokenize", "/tokenize"])
def test_root_path_keeps_route_policy(monkeypatch, path):
    assert (
        call_guard(monkeypatch, path, root_path="/models/qwen")[0][0]["status"] == 404
    )
    assert (
        call_guard(
            monkeypatch,
            path,
            root_path="/models/qwen",
            routes=["/tokenize"],
            headers=[],
        )[0][0]["status"]
        == 401
    )


def test_missing_auth_configuration_fails_closed():
    with pytest.raises(RuntimeError, match="authentication requires"):
        security.configure_inference_security(["server"])
    with pytest.raises(RuntimeError, match="not configured"):
        security.InferenceSecurityMiddleware(None)


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["server"], ["env-key"]),
        (["server", "--api-key", "cli-key"], ["cli-key"]),
        (["server", "--api-key=cli-key"], ["cli-key"]),
        (["server", "--api-key", "one", "two", "--model", "model"], ["one", "two"]),
    ],
)
def test_effective_keys_and_middleware(monkeypatch, argv, expected):
    monkeypatch.setenv("VLLM_API_KEY", "env-key")
    security.configure_inference_security(argv)
    assert security.InferenceSecurityMiddleware(None).keys == [
        key.encode() for key in expected
    ]
    assert argv[-2:] == ["--middleware", security._MIDDLEWARE]
    assert json.loads(os.environ["TT_ALLOWED_MEDIA_DOMAINS"]) == []


def test_explicit_development_no_auth_keeps_shared_media_policy(monkeypatch):
    sent, _, _, _ = call_guard(
        monkeypatch, "/v1/chat/completions", headers=[], no_auth=True
    )
    assert sent[0]["status"] == 200
    assert os.environ["VLLM_MEDIA_CONNECTOR"] == secure_media.CONNECTOR_NAME


def test_native_underscore_style_api_key(monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "env-key")
    argv = ["server", "--api_key=cli_key"]
    security.configure_inference_security(argv)
    assert security.InferenceSecurityMiddleware(None).keys == [b"cli_key"]
