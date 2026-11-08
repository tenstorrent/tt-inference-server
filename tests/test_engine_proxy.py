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


def test_router_slack_from_env(monkeypatch):
    monkeypatch.delenv("TT_PROXY_SLACK", raising=False)
    assert lb.Router(["A", "B"]).slack == 2
    monkeypatch.setenv("TT_PROXY_SLACK", "4")
    r = lb.Router(["A", "B"])
    assert r.slack == 4
    assert lb.Router(["A", "B"], slack=1).slack == 1
    first = r.pick(hashes_of(chat(DOC, "q0")))
    r.index.record(hashes_of(chat(DOC, "q0")), first)
    r.inflight[first] = 4  # 4 above the idle engine: still within slack 4
    assert r.pick(hashes_of(chat(DOC, "q1"))) == first
    r.inflight[first] = 5
    assert r.pick(hashes_of(chat(DOC, "q2"))) != first


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


def test_proxy_metrics_without_a_dead_or_hung_engine(monkeypatch):
    monkeypatch.setattr(lb, "PROBE_TIMEOUT_S", 0.5)

    async def engine_metrics(_):
        return web.Response(text=ENGINE_METRICS)

    async def hung_metrics(_):
        await asyncio.sleep(30)

    engine, hung = web.Application(), web.Application()
    engine.router.add_get("/metrics", engine_metrics)
    hung.router.add_get("/metrics", hung_metrics)

    async def run(session, url, router):
        async with session.get(url + "/metrics") as r:
            return r.status, await r.text()

    status, text = asyncio.run(with_proxy([engine, hung], run, dead=1))
    assert status == 200
    assert text.startswith("# engine 1 (")  # the hung engine
    assert "# engine 2 (" in text  # the one nothing listens for
    assert 'vllm:generation_tokens_total{model_name="m",engine="0"} 5.0' in text


@pytest.mark.parametrize(
    "fmt, want",
    [
        (
            {"type": "json_schema", "json_schema": {"name": "answer"}},
            {"type": "json_object"},
        ),
        (
            {
                "type": "json_schema",
                "json_schema": {"name": "a", "strict": True, "schema": None},
            },
            {"type": "json_object"},
        ),
        ({"type": "json_schema"}, {"type": "json_object"}),
    ],
)
def test_normalize_json_schema_without_schema_becomes_json_mode(fmt, want):
    body = json.dumps(
        {
            "model": "m",
            "messages": [],
            "tools": [{"type": "function"}],
            "response_format": fmt,
        }
    ).encode()
    out = json.loads(lb.normalize_structured_outputs(body))
    assert out["response_format"] == want
    assert out["tools"] == [{"type": "function"}] and out["messages"] == []


@pytest.mark.parametrize(
    "body",
    [
        {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        {"model": "m", "messages": [], "response_format": {"type": "json_object"}},
        {"model": "m", "messages": [], "response_format": {"type": "text"}},
        {
            "model": "m",
            "messages": [],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "a", "schema": {"type": "object"}},
            },
        },
        {"model": "m", "messages": [], "structured_outputs": {"choice": ["a", "b"]}},
    ],
)
def test_normalize_leaves_valid_requests_byte_identical(body):
    raw = json.dumps(body, separators=(",", ":")).encode()
    assert lb.normalize_structured_outputs(raw) is raw


def test_normalize_drops_empty_structured_outputs():
    body = json.dumps({"model": "m", "messages": [], "structured_outputs": {}}).encode()
    assert "structured_outputs" not in json.loads(lb.normalize_structured_outputs(body))
    body = json.dumps(
        {
            "model": "m",
            "messages": [],
            "structured_outputs": {"json": None, "regex": None},
        }
    ).encode()
    assert "structured_outputs" not in json.loads(lb.normalize_structured_outputs(body))


def recording_engine(seen):
    async def completions(request):
        seen.append(await request.json())
        return web.json_response({"ok": True})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", completions)
    return app


def test_proxy_rewrites_an_empty_json_schema_before_the_engine():
    seen = []

    async def run(session, url, router):
        body = {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "answer"},
            },
        }
        async with session.post(url + "/v1/chat/completions", json=body) as r:
            assert r.status == 200
        return router.snapshot()

    snap = asyncio.run(with_proxy([recording_engine(seen)], run))
    assert seen[0]["response_format"] == {"type": "json_object"}
    assert snap["prefix"]["structured_outputs_normalized"] == 1


def closing_engine(close_first):
    """A raw HTTP/1.1 engine that reads the first ``close_first`` requests and closes their connections without a
    reply (an engine closing a pooled keep-alive connection as the proxy writes into it), then answers normally."""
    seen = {"n": 0}

    async def handle(reader, writer):
        while True:
            head = await reader.readuntil(b"\r\n\r\n")
            length = 0
            for line in head.decode().split("\r\n"):
                if line.lower().startswith("content-length:"):
                    length = int(line.split(":", 1)[1])
            await reader.readexactly(length)
            seen["n"] += 1
            if seen["n"] <= close_first:
                writer.close()
                return
            data = json.dumps({"backend": "R"}).encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n"
                % len(data)
                + data
            )
            await writer.drain()

    return handle, seen


@pytest.mark.parametrize("close_first, want_status", [(1, 200), (2, 500)])
def test_proxy_resends_once_after_an_unanswered_disconnect(close_first, want_status):
    async def run():
        handle, seen = closing_engine(close_first)
        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        router = lb.Router([f"http://127.0.0.1:{port}"])
        proxy = TestServer(lb.make_app(router), host="127.0.0.1")
        await proxy.start_server()
        try:
            async with ClientSession() as session:
                async with session.post(
                    f"http://127.0.0.1:{proxy.port}/v1/chat/completions",
                    data=chat("short", "q"),
                    headers={"Content-Type": "application/json"},
                ) as r:
                    status = r.status
            return status, seen["n"], router.snapshot()
        finally:
            await proxy.close()
            server.close()
            await server.wait_closed()

    status, requests, snap = asyncio.run(run())
    assert status == want_status
    assert requests == 2  # one resend, never more
    assert snap["prefix"]["stale_retried"] == 1


# Prefill / decode disaggregation (TT_PROXY_PD_PREFILL).


def test_prefill_body_keeps_the_prompt_and_stops_after_one_token():
    body = json.dumps(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{"type": "function", "function": {"name": "f"}}],
            "tool_choice": "required",
            "chat_template_kwargs": {"enable_thinking": True},
            "cache_salt": "s",
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": 4096,
            "max_completion_tokens": 4096,
            "min_tokens": 100,
            "ignore_eos": True,
            "n": 2,
            "logprobs": True,
            "top_logprobs": 5,
            "response_format": {"type": "json_object"},
        }
    ).encode()
    req = json.loads(lb.prefill_body(body))
    assert req["stream"] is False and req["max_tokens"] == 1
    assert req["max_completion_tokens"] == 1 and req["n"] == 1
    assert req["tool_choice"] == "auto"
    for key in (
        "stream_options",
        "min_tokens",
        "logprobs",
        "top_logprobs",
        "response_format",
    ):
        assert key not in req
    for key in ("model", "messages", "tools", "chat_template_kwargs", "cache_salt"):
        assert req[key] == json.loads(body)[key]
    assert lb.prefill_body(b"[1, 2]") is None and lb.prefill_body(b"{") is None


def test_router_pd_split_from_env(monkeypatch):
    monkeypatch.setenv("TT_PROXY_PD_PREFILL", "1")
    monkeypatch.setenv("TT_PROXY_PD_MIN_CHARS", "4096")
    r = lb.Router(["P", "D1", "D2"])
    assert r.pd and r.prefill_backends == ["P"] and r.decode_backends == ["D1", "D2"]
    assert r.pd_min_chars == 4096
    assert r.snapshot()["pd"]["prefill"] == ["P"]
    # every backend a prefill engine (or none): no disaggregation
    assert not lb.Router(["P", "D"], prefill=2).pd
    assert not lb.Router(["P", "D"], prefill=0).pd
    # decode routing never picks the prefill engine, even when it holds the prefix
    h = hashes_of(chat(DOC, "q0"))
    r.index.record(h, "P")
    assert r.pick(h, pool=r.decode_backends) in ("D1", "D2")
    assert r.uncached_chars(h, len(h) * 256 + 100) == 100
    assert r.uncached_chars([], 5000) == 5000


def pd_engine(name, seen):
    async def completions(request):
        req = await request.json()
        seen.append((name, req))
        return web.json_response({"backend": name})

    async def health(_):
        return web.Response(status=200)

    app = web.Application(client_max_size=1 << 26)
    app.router.add_post("/v1/chat/completions", completions)
    app.router.add_get("/health", health)
    return app


def test_proxy_prefills_a_new_long_prompt_then_decodes_elsewhere(monkeypatch):
    monkeypatch.setenv("TT_PROXY_PD_PREFILL", "1")
    monkeypatch.setenv("TT_PROXY_PD_MIN_CHARS", "4096")
    seen = []

    async def run(session, url, router):
        out = []
        for q in ("q0", "q1"):  # a new long prompt, then the same document again
            async with session.post(
                url + "/v1/chat/completions",
                json={
                    "model": "m",
                    "messages": [
                        {"role": "user", "content": DOC + "\n\nQuestion: " + q}
                    ],
                    "stream": True,
                    "max_tokens": 64,
                },
            ) as r:
                assert r.status == 200
                out.append((await r.json())["backend"])
        async with session.post(
            url + "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "short"}]},
        ) as r:
            out.append((await r.json())["backend"])
        return out, router.snapshot()

    out, snap = asyncio.run(
        with_proxy(
            [pd_engine("P", seen), pd_engine("D1", seen), pd_engine("D2", seen)], run
        )
    )
    # q0: prefill on P (one token, not streamed), then the request itself on a decoding engine
    assert seen[0][0] == "P" and seen[0][1]["max_tokens"] == 1
    assert seen[0][1]["stream"] is False
    assert seen[1][0] in ("D1", "D2") and seen[1][1]["max_tokens"] == 64
    assert seen[1][1]["stream"] is True
    # q1 shares the document (now stored): straight to the engine that decoded q0; the short prompt: no prefill
    assert [name for name, _ in seen[2:]] == [seen[1][0], seen[3][0]]
    assert out[0] == out[1] == seen[1][0] and "P" not in out
    assert snap["prefix"]["pd_prefilled"] == 1 and snap["prefix"]["pd_direct"] == 2
    assert snap["prefix"]["pd_prefill_failed"] == 0


def test_proxy_decodes_anyway_when_the_prefill_engine_is_down(monkeypatch):
    monkeypatch.setenv("TT_PROXY_PD_MIN_CHARS", "4096")
    seen = []

    async def run(session, url, router):
        # the dead backend (appended last by with_proxy) as the only prefill engine
        router.prefill_backends = [router.backends[-1]]
        router.decode_backends = router.backends[:-1]
        router.pd = True
        async with session.post(
            url + "/v1/chat/completions", data=chat(DOC, "q0")
        ) as r:
            assert r.status == 200
            return (await r.json())["backend"], router.snapshot()

    backend, snap = asyncio.run(with_proxy([pd_engine("D1", seen)], run, dead=1))
    assert backend == "D1" and len(seen) == 1
    assert snap["prefix"]["pd_prefill_failed"] == 1


def test_prefill_spills_to_an_idle_decoding_engine_when_prefill_is_backed_up():
    r = lb.Router(["P", "D1", "D2"], prefill=1, pd_flex=2)
    h = hashes_of(chat(DOC, "q0"))
    assert r.prefill_target(h) == "P"
    r.inflight["P"] = 3  # more than pd_flex in flight
    r.inflight["D1"] = 1  # decoding
    assert r.prefill_target(h) == "D2"  # the idle decoding engine
    assert r.stats["pd_flex"] == 1
    r.inflight["D2"] = 2
    assert (
        r.prefill_target(h) == "P"
    )  # no idle decoding engine: queue on the prefill engine
    assert lb.Router(["P", "D"], prefill=1, pd_flex=-1).pd_flex == -1
    off = lb.Router(["P", "D"], prefill=1, pd_flex=-1)
    off.inflight["P"] = 9
    assert off.prefill_target(h) == "P"
