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
    # TestServer itself runs with handler_cancellation=True, as lb.RUNNER_KWARGS makes the served proxy run
    await proxy.start_server(
        **{k: v for k, v in lb.RUNNER_KWARGS.items() if k != "handler_cancellation"}
    )
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


def streaming_engine(events, chunks=200, delay_s=0.02):
    """An engine streaming ``chunks`` SSE events; records "finished", or "aborted" when its client goes away."""

    async def completions(request):
        await request.read()
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        try:
            for i in range(chunks):
                await resp.write(f"data: {i}\n\n".encode())
                await asyncio.sleep(delay_s)
            await resp.write_eof()
        except (ConnectionResetError, asyncio.CancelledError):
            events.append("aborted")
            raise
        events.append("finished")
        return resp

    app = web.Application()
    app.router.add_post("/v1/chat/completions", completions)
    return app


def test_proxy_streams_the_engine_reply_through():
    events = []

    async def run(session, url, router):
        async with session.post(url + "/v1/chat/completions", data=chat(DOC, "q")) as r:
            return r.status, (await r.read()).decode(), dict(router.inflight)

    status, text, inflight = asyncio.run(
        with_proxy([streaming_engine(events, chunks=5, delay_s=0.01)], run)
    )
    assert status == 200 and text == "".join(f"data: {i}\n\n" for i in range(5))
    assert events == ["finished"] and set(inflight.values()) == {0}


def test_a_client_disconnect_aborts_the_engine_request():
    """The proxy cancels the request's handler when its client goes away (RUNNER_KWARGS), which closes the upstream
    connection: the engine stops streaming and the request leaves the in-flight count."""
    events = []

    async def run(session, url, router):
        async with session.post(url + "/v1/chat/completions", data=chat(DOC, "q")) as r:
            assert r.status == 200
            await (
                r.content.readany()
            )  # the first event, then the client leaves mid-stream
            busy = dict(router.inflight)
        for _ in range(100):
            if events and not any(router.inflight.values()):
                break
            await asyncio.sleep(0.05)
        return busy, dict(router.inflight)

    assert (
        lb.RUNNER_KWARGS["handler_cancellation"] is True
    )  # what the served proxy runs with (aiohttp default: off)
    busy, after = asyncio.run(with_proxy([streaming_engine(events)], run))
    assert set(busy.values()) == {1}
    assert events == ["aborted"] and set(after.values()) == {0}


def test_every_engine_refusing_is_a_503_and_leaves_nothing_in_flight():
    async def run(session, url, router):
        async with session.post(url + "/v1/chat/completions", data=chat(DOC, "q")) as r:
            return r.status, await r.json(), dict(router.inflight)

    status, body, inflight = asyncio.run(with_proxy([], run, dead=2))
    assert status == 503 and "refused" in body["error"]
    assert set(inflight.values()) == {0}


def image_chat(data, question):
    return json.dumps(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64," + data},
                        },
                        {"type": "text", "text": DOC + question},
                    ],
                }
            ],
        }
    ).encode()


def test_media_payloads_are_keyed_by_their_digest():
    """A base64 image is keyed by its digest, not hashed as megabytes of prompt text: the same image routes like the
    same prefix, another image does not, and the chunk count follows the text only."""
    big = "A" * (4 << 20)
    one, two = hashes_of(image_chat(big, "a")), hashes_of(image_chat(big, "b"))
    other = hashes_of(image_chat("B" * (4 << 20), "a"))
    assert len(one) < 2 * len(DOC) // 256
    assert (
        one[: len(one) - 1] == two[: len(one) - 1]
    )  # the image and the document share their chunks
    assert one[0] != other[0]
