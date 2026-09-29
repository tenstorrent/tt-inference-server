# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""MiniMax-H3 benchmark media transport: URL references / keyframes pointing the server at the
hosted pack, manifest-only asset checks in URL mode, and the inline fallback of ``auto`` whose
oversized files become client-capability rejects (xfail)."""

from __future__ import annotations

import json

import pytest

from test_module._test_common.minimax_h3_bench import adapters as A
from test_module._test_common.minimax_h3_bench import models as M


def _case(case_id):
    return next(c for c in M.load_cases()["cases"] if c["id"] == case_id)


@pytest.fixture
def no_local_media(tmp_path, monkeypatch):
    """An asset search that finds the prompts (repo) but no media file anywhere."""
    real = M.asset_path
    monkeypatch.setattr(
        M, "asset_path", lambda n: real(n) if n.endswith(".txt") else None
    )


def _adapter(media, monkeypatch, caps=None):
    adapter = A.TenstorrentH3(
        {"all": "http://h3.test"},
        api_key="k",
        media=media,
        media_base_url="https://m.test/r",
    )
    monkeypatch.setattr(
        adapter, "caps", lambda task: caps or {"image": 40_000_000, "media": 66_666_668}
    )
    return adapter


def test_url_mode_sends_urls_without_local_media(no_local_media, monkeypatch):
    adapter = _adapter("url", monkeypatch)
    path, payload, _ = adapter.build(_case("FL2VA-H"))
    assert path == A.ROUTE["fl2va"]
    assert [p["image"] for p in payload["image_prompts"]] == [
        "https://m.test/r/img_max_27mb_astronaut.jpg",
        "https://m.test/r/img_min_256px_scientist.jpg",
    ]
    path, payload, _ = adapter.build(_case("REF2VA-H"))
    refs = payload["references"]
    assert {k: len(v) for k, v in refs.items()} == {
        "images": 6,
        "videos": 3,
        "audios": 3,
    }
    assert all(set(e) == {"url"} for v in refs.values() for e in v)


def test_url_mode_checks_only_the_pins(no_local_media):
    names = M.case_assets(_case("REF2VA-H"))
    assert M.verify_assets(names, local_media=False) == []
    assert any("missing asset" in p for p in M.verify_assets(names))
    assert M.verify_assets(["not_pinned.mp4"], local_media=False) == [
        "not_pinned.mp4: not in the asset manifest"
    ]


def test_inline_oversize_is_a_client_capability_reject(monkeypatch):
    adapter = _adapter(
        "b64", monkeypatch, caps={"image": 10_000_000, "media": 80_000_000}
    )
    path, (code, body), _ = adapter.build(_case("FL2VA-H"))
    assert path is None and code == 400
    assert body["error"]["message"].startswith(M.CLIENT_REJECT)
    assert "img_max_27mb_astronaut.jpg" in body["error"]["message"]


def test_inline_body_over_the_request_cap_is_rejected(monkeypatch):
    adapter = _adapter("b64", monkeypatch)
    monkeypatch.setattr(adapter, "MAX_BODY_BYTES", 1_000_000)
    path, (code, body), _ = adapter.build(_case("REF2VA-L"))
    assert path is None and code == 400
    assert "request-body cap" in body["error"]["message"]


def test_auto_resends_inline_when_the_server_cannot_fetch(monkeypatch):
    adapter = _adapter("auto", monkeypatch)
    sent = []

    def fake_http(method, url, body=None, **_):
        payload = json.loads(body)
        sent.append(payload["references"]["images"][0])
        if "url" in sent[-1]:
            return (
                422,
                b'{"detail": "Could not fetch Media URL https://m.test/r/x: HTTP 404"}',
            )
        return (
            202,
            b'{"id": "4b9a1f1e-0000-4000-8000-000000000000", "status": "queued"}',
        )

    monkeypatch.setattr(A, "http", fake_http)
    (code, body), _ = adapter.post(_case("REF2VA-L"))
    assert code == 202 and body["id"]
    assert [set(e) for e in sent] == [{"url"}, {"b64"}]
    assert adapter.last_transport == "b64-fallback"


def test_url_mode_does_not_fall_back(monkeypatch):
    adapter = _adapter("url", monkeypatch)
    monkeypatch.setattr(
        A, "http", lambda *a, **k: (422, b'{"detail": "Could not fetch Media URL x"}')
    )
    (code, _), _ = adapter.post(_case("REF2VA-L"))
    assert code == 422 and adapter.last_transport == "url"


def test_unreachable_urls_switch_auto_to_inline(monkeypatch):
    adapter = _adapter("auto", monkeypatch)
    adapter.use_inline("urls down")
    assert adapter.media == "b64"
    strict = _adapter("url", monkeypatch)
    strict.use_inline("urls down")
    assert strict.media == "url"


def test_default_media_base_url_is_the_pinned_hf_revision():
    assert M.DEFAULT_MEDIA_BASE_URL.startswith(
        "https://huggingface.co/datasets/zhenghaoniTT/minimax-h3-bench-assets/resolve/"
    )
    assert len(M.DEFAULT_MEDIA_BASE_URL.rsplit("/", 1)[1]) == 40
