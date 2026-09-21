# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""RequestBodyLimitMiddleware: the 64 MB request-body cap of the MiniMax input media card,
scoped to the video routes, answered from Content-Length or from the streamed byte count."""

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from open_ai_api.body_limit import RequestBodyLimitMiddleware
from open_ai_api.deprecation import DeprecatedPathMiddleware

LIMIT = 1024


def _client(max_bytes: int = LIMIT) -> TestClient:
    app = FastAPI()

    async def echo(request: Request):
        return {"received": len(await request.body())}

    app.post("/v1/videos/generations/ref2va")(echo)
    app.post("/video/generations")(echo)
    app.post("/v1/audio/transcriptions")(echo)
    app.get("/v1/videos/jobs")(lambda: {"jobs": []})
    # The real app has a BaseHTTPMiddleware inside the cap; the size error must cross it.
    app.add_middleware(DeprecatedPathMiddleware, sunset_date="2026-06-30")
    app.add_middleware(RequestBodyLimitMiddleware, max_bytes=max_bytes)
    return TestClient(app)


VIDEO = "/v1/videos/generations/ref2va"


def test_under_and_at_the_cap_pass():
    client = _client()
    assert client.post(VIDEO, content=b"x" * 10).json() == {"received": 10}
    assert client.post(VIDEO, content=b"x" * LIMIT).json() == {"received": LIMIT}


def test_content_length_over_the_cap_is_413_before_the_body_is_read():
    client = _client()
    response = client.post(VIDEO, content=b"x" * (LIMIT + 1))
    assert response.status_code == 413
    detail = response.json()["detail"]
    assert f"{LIMIT + 1} bytes" in detail and f"{LIMIT}-byte limit" in detail
    assert "URL" in detail


def test_chunked_body_over_the_cap_is_413():
    client = _client()

    def chunks():
        yield b"x" * 600
        yield b"x" * 600

    response = client.post(
        VIDEO, content=chunks(), headers={"Transfer-Encoding": "chunked"}
    )
    assert response.status_code == 413
    assert "over" in response.json()["detail"]


def test_chunked_body_under_the_cap_passes():
    client = _client()

    def chunks():
        yield b"x" * 300
        yield b"x" * 300

    response = client.post(
        VIDEO, content=chunks(), headers={"Transfer-Encoding": "chunked"}
    )
    assert response.json() == {"received": 600}


def test_legacy_video_prefix_is_capped_too():
    assert (
        _client().post("/video/generations", content=b"x" * (LIMIT + 1)).status_code
        == 413
    )


def test_other_routes_keep_their_own_limits():
    assert _client().post(
        "/v1/audio/transcriptions", content=b"x" * (LIMIT * 4)
    ).json() == {"received": LIMIT * 4}


def test_get_is_untouched():
    assert _client().get("/v1/videos/jobs").json() == {"jobs": []}


def test_zero_disables_the_cap():
    assert _client(max_bytes=0).post(VIDEO, content=b"x" * (LIMIT * 4)).json() == {
        "received": LIMIT * 4
    }


def test_default_in_main_follows_the_card():
    from config.settings import Settings
    from tt_model_runners.minimax_h3_policy import MINIMAX_H3_MAX_REQUEST_BODY_BYTES

    assert MINIMAX_H3_MAX_REQUEST_BODY_BYTES == 64 * 1024 * 1024
    fields = getattr(Settings, "model_fields", None)
    if not isinstance(fields, dict):
        pytest.skip("config.settings is mocked by another test module in this session")
    assert fields["max_request_body_bytes"].default == MINIMAX_H3_MAX_REQUEST_BODY_BYTES


@pytest.mark.parametrize("path", ["/v1/videos", "/v1/videos/x", "/video/x"])
def test_prefix_matching_is_label_anchored(path):
    middleware = RequestBodyLimitMiddleware(lambda *_: None, max_bytes=1)
    assert middleware._applies({"type": "http", "method": "POST", "path": path})
    assert not middleware._applies(
        {"type": "http", "method": "POST", "path": "/v1/videosx"}
    )
    assert not middleware._applies({"type": "http", "method": "GET", "path": path})
