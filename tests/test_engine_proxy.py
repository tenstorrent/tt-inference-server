#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""vllm-tt-metal/src/engine_proxy.py: prefix keys, the prefix index, routing, and the proxy over fake engines."""

import asyncio
import json
import socket
import sys

import pytest

from workflows.utils import get_repo_root_path

sys.path.insert(0, str(get_repo_root_path() / "vllm-tt-metal" / "src"))
aiohttp = pytest.importorskip("aiohttp")
from aiohttp import ClientSession, web  # noqa: E402
from aiohttp.test_utils import TestServer  # noqa: E402

import engine_proxy as lb  # noqa: E402

DOC = "".join(
    f"Paragraph {i}: the quick brown fox jumps over the lazy dog. " for i in range(200)
)


def chat(doc, question, salt=None):
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": doc + "\n\nQuestion: " + question}],
    }
    if salt is not None:
        body["cache_salt"] = salt
    return json.dumps(body).encode()


def hashes_of(body, path="/v1/chat/completions", chunk=256):
    return lb.chunk_hashes(lb.prefix_text(path, body), chunk)


def test_prefix_text_and_salt():
    assert lb.prefix_text(
        "/v1/completions", json.dumps({"prompt": [1, 2, 3]}).encode()
    ).endswith("1,2,3,")
    assert (
        lb.prefix_text("/v1/completions", json.dumps({"prompt": ["a", "b"]}).encode())
        is None
    )
    assert lb.prefix_text("/v1/chat/completions", b"not json") is None
    assert (
        hashes_of(chat(DOC, "a", salt="s1"))[0]
        != hashes_of(chat(DOC, "a", salt="s2"))[0]
    )


def test_index_lru_keeps_heads():
    idx = lb.PrefixIndex(capacity=10)
    h = [bytes([i]) * 16 for i in range(8)]
    idx.record(h, "A")
    idx.record([bytes([100 + i]) * 16 for i in range(4)], "B")
    m = idx.match(h, ["A", "B"])["A"]
    assert 0 < m < 8 and all(h[i] in idx.entries for i in range(m))


def test_router_affinity_and_spill():
    r = lb.Router(["A", "B", "C", "D"], slack=2)
    first = r.pick(hashes_of(chat(DOC, "q0")))
    r.index.record(hashes_of(chat(DOC, "q0")), first)
    for q in range(3):
        assert r.pick(hashes_of(chat(DOC, f"q{q + 1}"))) == first
        r.inflight[first] += 1
    assert r.pick(hashes_of(chat(DOC, "q9"))) != first
    r.inflight = {b: 0 for b in "ABCD"}
    assert sorted(r.pick(hashes_of(chat("short", f"q{i}"))) for i in range(4)) == list(
        "ABCD"
    )


def fake_engine(name, healthy=True):
    async def completions(request):
        await request.json()
        return web.json_response({"backend": name})

    async def health(_):
        return web.Response(status=200 if healthy else 503)

    app = web.Application(client_max_size=1 << 26)
    app.router.add_post("/v1/chat/completions", completions)
    app.router.add_get("/health", health)
    return app


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def with_proxy(engines, fn, dead=0):
    servers = [TestServer(app, host="127.0.0.1") for app in engines]
    for s in servers:
        await s.start_server()
    backends = [f"http://127.0.0.1:{s.port}" for s in servers] + [
        f"http://127.0.0.1:{free_port()}" for _ in range(dead)
    ]
    router = lb.Router(backends)
    proxy = TestServer(lb.make_app(router), host="127.0.0.1")
    await proxy.start_server()
    try:
        async with ClientSession() as session:
            return await fn(session, f"http://127.0.0.1:{proxy.port}", router)
    finally:
        await proxy.close()
        for s in servers:
            await s.close()


def test_proxy_affinity_retry_and_health():
    docs = [f"Document {d}. " + DOC for d in range(2)]

    async def run(session, url, router):
        where = {}
        for q in range(3):
            for d, doc in enumerate(docs):
                async with session.post(
                    url + "/v1/chat/completions", data=chat(doc, f"q{q}")
                ) as r:
                    assert r.status == 200
                    where.setdefault(d, set()).add((await r.json())["backend"])
        async with session.get(url + "/health") as r:
            health = r.status
        return where, health, router.snapshot()

    where, health, snap = asyncio.run(
        with_proxy([fake_engine("A"), fake_engine("B")], run, dead=1)
    )
    assert all(len(e) == 1 for e in where.values()), where
    assert health == 503  # the dead engine is not healthy
    assert snap["prefix"]["prefix_routed"] == 4


# What every engine (a DP-1 vLLM server) exposes: the same families and the same engine="0" label.
ENGINE_METRICS = """# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{engine="0",model_name="m"} 5.0
# HELP vllm:time_to_first_token_seconds Histogram of time to first token in seconds.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.5",model_name="m"} 3.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="+Inf",model_name="m"} 4.0
vllm:time_to_first_token_seconds_count{engine="0",model_name="m"} 4.0
vllm:time_to_first_token_seconds_sum{engine="0",model_name="m"} 1.5
process_open_fds 12.0
"""


def test_merge_metrics_labels_each_engine_and_writes_each_family_once():
    merged = lb.merge_metrics(
        [ENGINE_METRICS, ENGINE_METRICS, 'odd{path="a}b\\"c"} 1\n']
    )
    lines = merged.splitlines()
    assert lines.count("# TYPE vllm:generation_tokens_total counter") == 1
    assert lines.count("# TYPE vllm:time_to_first_token_seconds histogram") == 1
    samples = [line for line in lines if not line.startswith("#")]
    assert len(samples) == len(set(samples)) == 2 * 6 + 1  # every series once
    assert 'vllm:generation_tokens_total{model_name="m",engine="0"} 5.0' in samples
    assert 'vllm:generation_tokens_total{model_name="m",engine="1"} 5.0' in samples
    assert 'process_open_fds{engine="1"} 12.0' in samples
    # A label value with } and an escaped quote.
    assert 'odd{path="a}b\\"c",engine="2"} 1' in samples
    # A family's samples stay together, right after its TYPE line.
    start = lines.index("# TYPE vllm:time_to_first_token_seconds histogram")
    assert all(
        line.startswith("vllm:time_to_first_token_seconds_")
        for line in lines[start + 1 : start + 9]
    )
