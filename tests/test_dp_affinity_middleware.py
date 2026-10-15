# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Tests for vllm-tt-metal/src/dp_affinity_middleware.py."""

import importlib.util
import json

import pytest
from starlette.testclient import TestClient

from workflows.utils import get_repo_root_path

MODULE_PATH = (
    get_repo_root_path() / "vllm-tt-metal" / "src" / "dp_affinity_middleware.py"
)
CHAT = "/v1/chat/completions"


@pytest.fixture
def dpa():
    spec = importlib.util.spec_from_file_location("dp_affinity_middleware", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def _echo_app(scope, receive, send):
    """Echo the resolved rank header and the body length the app actually saw."""
    body = b""
    while True:
        message = await receive()
        body += message.get("body", b"")
        if not message.get("more_body"):
            break
    headers = {k.decode(): v.decode() for k, v in scope["headers"]}
    payload = json.dumps(
        {"rank": headers.get("x-data-parallel-rank"), "body_len": len(body)}
    ).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send({"type": "http.response.body", "body": payload})


def _client(dpa, monkeypatch, mode="auto", dp_size="4", prefix_bytes=None):
    monkeypatch.setenv("TT_DP_AFFINITY", mode)
    if dp_size is None:
        monkeypatch.delenv("TT_DP_AFFINITY_DP_SIZE", raising=False)
    else:
        monkeypatch.setenv("TT_DP_AFFINITY_DP_SIZE", dp_size)
    if prefix_bytes is not None:
        monkeypatch.setenv("TT_DP_AFFINITY_PREFIX_BYTES", prefix_bytes)
    return TestClient(dpa.DPAffinityMiddleware(_echo_app))


def _chat(turns, **extra):
    body = {"model": "m", "messages": [{"role": "user", "content": t} for t in turns]}
    body.update(extra)
    return body


def test_rank_for_key_is_stable_and_in_range(dpa):
    ranks = {dpa.rank_for_key(f"user:{i}", 4) for i in range(64)}
    assert ranks <= {0, 1, 2, 3}
    assert len(ranks) == 4, "64 keys should spread over all four engines"
    assert dpa.rank_for_key("user:alice", 4) == dpa.rank_for_key("user:alice", 4)


def test_prompt_prefix_chat_and_completions(dpa):
    chat = _chat(["hello", "world"])
    assert dpa.prompt_prefix(chat, 4096) == "hello\nworld"
    parts = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "a"}, {"type": "image_url"}],
            }
        ]
    }
    assert dpa.prompt_prefix(parts, 4096) == "a"
    assert dpa.prompt_prefix({"prompt": "plain"}, 4096) == "plain"
    assert dpa.prompt_prefix({"prompt": ["first", "second"]}, 4096) == "first"
    assert dpa.prompt_prefix({"prompt": "0123456789"}, 4) == "0123"
    assert dpa.prompt_prefix({"messages": []}, 4096) is None
    assert dpa.prompt_prefix("not a dict", 4096) is None


def test_affinity_key_precedence(dpa):
    headers = {dpa.AFFINITY_KEY_HEADER: b"sess-1"}
    body = _chat(["hi"], user="alice")
    assert dpa.affinity_key(headers, body, "auto", 4096) == "key:sess-1"
    assert dpa.affinity_key({}, body, "auto", 4096) == "user:alice"
    assert dpa.affinity_key({}, _chat(["hi"]), "auto", 4096) is None
    assert dpa.affinity_key({}, _chat(["hi"]), "prefix", 4096) == "prefix:hi"
    assert dpa.affinity_key({}, None, "prefix", 4096) is None


def test_auto_mode_pins_only_with_a_session_signal(dpa, monkeypatch):
    client = _client(dpa, monkeypatch)
    unpinned = client.post(CHAT, json=_chat(["hi"])).json()
    assert unpinned["rank"] is None, "no session signal: leave vLLM's balancer alone"

    first = client.post(CHAT, json=_chat(["hi"], user="alice")).json()
    second = client.post(CHAT, json=_chat(["hi", "again"], user="alice")).json()
    assert first["rank"] is not None and first["rank"] == second["rank"]
    assert 0 <= int(first["rank"]) < 4

    keyed = client.post(CHAT, json=_chat(["hi"]), headers={"X-TT-Affinity-Key": "s"})
    assert keyed.json()["rank"] == str(dpa.rank_for_key("key:s", 4))


def test_body_is_replayed_intact(dpa, monkeypatch):
    client = _client(dpa, monkeypatch)
    body = json.dumps(_chat(["x" * 10000], user="bob")).encode()
    assert client.post(CHAT, content=body).json()["body_len"] == len(body)


def test_client_rank_is_clamped_to_engine_count(dpa, monkeypatch):
    client = _client(dpa, monkeypatch)
    out_of_range = client.post(
        CHAT, json=_chat(["hi"]), headers={"X-data-parallel-rank": "7"}
    )
    assert out_of_range.json()["rank"] == "3"
    in_range = client.post(
        CHAT, json=_chat(["hi"]), headers={"X-data-parallel-rank": "2"}
    )
    assert in_range.json()["rank"] == "2"
    garbage = client.post(
        CHAT, json=_chat(["hi"]), headers={"X-data-parallel-rank": "x"}
    )
    assert garbage.json()["rank"] == "x", "vLLM ignores a non-integer rank itself"


def test_only_generation_posts_are_routed(dpa, monkeypatch):
    client = _client(dpa, monkeypatch)
    assert client.post("/v1/embeddings", json={"user": "alice"}).json()["rank"] is None
    assert client.get(CHAT).json()["rank"] is None
    assert client.post("/v1/completions", json={"prompt": "p", "user": "alice"}).json()[
        "rank"
    ] == str(dpa.rank_for_key("user:alice", 4))


def test_prefix_mode_keeps_a_conversation_on_one_engine(dpa, monkeypatch):
    client = _client(dpa, monkeypatch, mode="prefix", prefix_bytes="64")
    system = "You are a helpful assistant answering about the Galaxy system. " * 2
    turn1 = client.post(CHAT, json=_chat([system, "first question"])).json()
    turn2 = client.post(
        CHAT, json=_chat([system, "first question", "follow-up"])
    ).json()
    assert turn1["rank"] is not None and turn1["rank"] == turn2["rank"]
    assert turn1["rank"] == str(dpa.rank_for_key("prefix:" + system[:64], 4))


def test_off_mode_and_missing_engine_count_pass_through(dpa, monkeypatch):
    off = _client(dpa, monkeypatch, mode="off")
    assert off.post(CHAT, json=_chat(["hi"], user="alice")).json()["rank"] is None
    assert (
        off.post(
            CHAT, json=_chat(["hi"]), headers={"X-data-parallel-rank": "7"}
        ).json()["rank"]
        == "7"
    )
    no_size = _client(dpa, monkeypatch, mode="auto", dp_size=None)
    assert no_size.post(CHAT, json=_chat(["hi"], user="alice")).json()["rank"] is None
    bad_mode = _client(dpa, monkeypatch, mode="sticky")
    assert bad_mode.post(CHAT, json=_chat(["hi"], user="alice")).json()["rank"] is None
