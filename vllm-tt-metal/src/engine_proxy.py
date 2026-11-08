#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Prefix-cache aware HTTP proxy in front of a model's submesh engines (``DeviceModelSpec.submesh_engines``).

``run_vllm_api_server.py`` starts one vLLM engine per device group and serves this proxy on the service port, so every
client keeps talking to one OpenAI-compatible endpoint. Each engine has its own prefix cache, so a request only hits
on the engine that already holds its prefix. The proxy keys each request by its prompt (chat messages and tools, a
completion prompt or its token ids, ``cache_salt``) in ``chunk_chars`` chunks and remembers which engines served which
chunks (an LRU index of chained chunk hashes):

- a request whose longest known prefix is at least ``min_match_chars`` long goes to the least loaded engine that
  holds it, unless that engine has more than ``slack`` requests in flight above the least loaded engine;
- any other request goes to the least loaded engine (fewest in flight; ties round robin).

``slack`` defaults to env ``TT_PROXY_SLACK``, else 2.

So requests sharing a long prefix (a document, a long system prompt, a multi-turn conversation) gather on the engines
that cached it, and spill to another engine (which then caches it too) instead of queueing on one. Responses stream
back unchanged.

GET /health is 200 only when every engine answers its own /health. GET /metrics merges every engine's metrics into one
exposition, each sample labeled with its engine's index (``merge_metrics``).
GET /lb/stats shows in-flight and served counts and the prefix routing counters. A request whose engine refuses the
connection goes to another engine (the refusing one is skipped for ``down_s`` seconds).

Prefill / decode disaggregation (env ``TT_PROXY_PD_PREFILL=N``, off by default): the first N engines only prefill.
A generation request whose prompt is mostly new to every engine (more than ``TT_PROXY_PD_MIN_CHARS`` characters, 8192
by default, beyond the longest prefix any engine has served) is first sent to a prefill engine as a copy that stops
after one token (``prefill_body``); once that answers, the request itself goes to a decoding engine (the other
engines, routed as above). This needs engines that share KV through a host store (vllm-tt-plugin's
TTKVStoreConnector with ttserve-kvstore): the prefill engine saves the prompt's KV as it computes it, and the decoding
engine loads it instead of prefilling, so long prefills never stall the decoding engines' running requests. A request
whose prompt is mostly cached goes straight to a decoding engine, which loads what another engine stored. If the
prefill engine fails, the decoding engine prefills the request itself. Prefill passes are pulled, not pushed: a
prefill engine runs at most ``TT_PROXY_PD_FLEX`` (2) of them at once, a decoding engine with nothing in flight takes
the next one when every prefill engine is full, and the rest wait in the proxy until one of the two frees up (a burst
of new prompts would otherwise sit in a prefill engine's queue while decoding engines idle). A negative value turns
this off: every prefill pass goes straight to a prefill engine.

The same proxy serves tt-metal's MiMo-V2.6-Flash row engines (models/experimental/mimo_v2_6_flash/serving/lb.py).
"""

import asyncio
import hashlib
import itertools
import json
import os
import re
import time
from collections import OrderedDict

from aiohttp import (
    ClientConnectorError,
    ClientSession,
    ClientTimeout,
    ServerDisconnectedError,
    TCPConnector,
    web,
)

HOP = {
    "connection",
    "keep-alive",
    "transfer-encoding",
    "content-length",
    "content-encoding",
    "upgrade",
    "host",
}
PREFIX_PATHS = ("/v1/chat/completions", "/v1/completions", "/v1/responses")
SAMPLE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?(\s.*)$"
)  # name, labels, value [timestamp]
LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')
PROBE_TIMEOUT_S = (
    10.0  # an engine's /health or /metrics answer; generation requests have no timeout
)
# Idle upstream connections are closed by the proxy before the engine's HTTP server closes them (uvicorn's default
# keep-alive is 5 s): a request written into a connection the engine is closing gets "Server disconnected" and no reply.
UPSTREAM_KEEPALIVE_S = 2.0
STRUCTURED_PATHS = ("/v1/chat/completions", "/v1/completions")
PD_PATHS = ("/v1/chat/completions", "/v1/completions")
PD_POLL_S = 0.05  # how often a waiting prefill pass looks for a free engine
# Request fields a prefill-only copy drops: they shape the output, not the prompt (structured outputs would also make the
# prefill engine compile a grammar for one token).
PREFILL_DROP = (
    "stream_options",
    "logprobs",
    "top_logprobs",
    "prompt_logprobs",
    "response_format",
    "structured_outputs",
    "guided_json",
    "guided_regex",
    "guided_choice",
    "guided_grammar",
    "min_tokens",
    "echo",
)


def normalize_structured_outputs(body):
    """Request bytes vLLM would reject for an empty structured-outputs constraint -> bytes it accepts; else unchanged.

    vLLM (0.30) turns ``response_format: {"type": "json_schema", "json_schema": {...}}`` into a JSON-schema constraint
    with ``json_schema.schema``. That field is optional in the OpenAI API, but without it vLLM builds a constraint with
    every field empty and answers 400 ("You must use one kind of structured outputs constraint but none are
    specified"). A json_schema format without a schema asks for JSON with no schema, which is JSON mode, so it becomes
    ``{"type": "json_object"}``. An empty ``structured_outputs`` object (vLLM's own extension) is dropped. Bodies
    without either key are returned as they are, byte for byte.
    """
    if b'"response_format"' not in body and b'"structured_outputs"' not in body:
        return body
    try:
        req = json.loads(body)
    except ValueError:
        return body
    if not isinstance(req, dict):
        return body
    changed = False
    fmt = req.get("response_format")
    if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
        schema = fmt.get("json_schema")
        if not isinstance(schema, dict) or schema.get("schema") is None:
            req["response_format"] = {"type": "json_object"}
            changed = True
    so = req.get("structured_outputs")
    if isinstance(so, dict) and all(
        v in (None, False, "", [], {}) for v in so.values()
    ):
        del req["structured_outputs"]
        changed = True
    return json.dumps(req).encode() if changed else body


def prefill_body(body):
    """The bytes of a prefill-only copy of a generation request (the same prompt, one output token, not streamed), or
    None when the body is not a JSON object. Everything that renders the prompt (messages or prompt, tools,
    chat_template_kwargs, cache_salt, ...) is kept as it is, so the prefill engine computes exactly the prompt the
    decoding engine will be asked for."""
    try:
        req = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(req, dict):
        return None
    for key in PREFILL_DROP:
        req.pop(key, None)
    req["stream"] = False
    req["max_tokens"] = 1
    if "max_completion_tokens" in req:
        req["max_completion_tokens"] = 1
    req["n"] = 1
    if req.get("tool_choice") not in (None, "auto", "none"):
        req["tool_choice"] = (
            "auto"  # "required" / a named tool constrain the output with a grammar, not the prompt
        )
    return json.dumps(req).encode()


def prefix_text(path, body):
    """The text that determines a request's prompt prefix, in prompt order, or None (no prefix routing)."""
    try:
        req = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(req, dict):
        return None
    head = json.dumps(
        {
            "model": req.get("model"),
            "salt": req.get("cache_salt"),
            "tools": req.get("tools"),
            "template": req.get("chat_template_kwargs"),
        },
        sort_keys=True,
    )
    if path == "/v1/chat/completions":
        messages = req.get("messages")
        if not isinstance(messages, list):
            return None
        return head + "".join(
            json.dumps(m, sort_keys=True, ensure_ascii=False) for m in messages
        )
    if path == "/v1/completions":
        prompt = req.get("prompt")
        if isinstance(prompt, list) and len(prompt) == 1:
            prompt = prompt[0]  # one prompt in a list
        if isinstance(prompt, str):
            return head + prompt
        if isinstance(prompt, list) and all(isinstance(t, int) for t in prompt):
            return head + ",".join(map(str, prompt)) + ","
        return None  # a batch of prompts: no single prefix
    if path == "/v1/responses":
        inp = req.get("input")
        text = (
            inp
            if isinstance(inp, str)
            else json.dumps(inp, sort_keys=True, ensure_ascii=False)
        )
        return head + json.dumps(req.get("instructions")) + text
    return None


def chunk_hashes(text, chunk_chars):
    """Chained hashes of the text's full chunks: hash i covers chunks 0..i (a partial last chunk is not keyed)."""
    out, h = [], b""
    data = text.encode("utf-8", "surrogatepass")
    step = chunk_chars  # bytes; equal to characters for ASCII prompts
    for start in range(0, len(data) - step + 1, step):
        h = hashlib.blake2b(h + data[start : start + step], digest_size=16).digest()
        out.append(h)
    return out


def merge_metrics(texts):
    """The engines' Prometheus texts (in backend order) as one exposition.

    Each engine is its own vLLM server, so all of them label their samples ``engine="0"``: concatenated, every series
    and every family's HELP / TYPE would appear once per engine, which a Prometheus parser rejects or collapses. Here
    every sample gets ``engine="<index of its engine>"``, as vLLM labels the engines of one data-parallel server, and
    each family is written once: HELP / TYPE from the first engine that has them, then every engine's samples."""
    families = (
        OrderedDict()
    )  # family name -> {"HELP": line, "TYPE": line, "samples": [lines]}
    for index, text in enumerate(texts):
        family = None
        for line in text.splitlines():
            if line.startswith("#"):
                parts = line.split(None, 3)
                if len(parts) >= 3 and parts[1] in ("HELP", "TYPE"):
                    family = parts[2]
                    families.setdefault(family, {"samples": []}).setdefault(
                        parts[1], line
                    )
                continue
            match = SAMPLE.match(line)
            if match is None:
                continue
            name, labels, rest = match.groups()
            if family is None or not (name == family or name.startswith(family + "_")):
                family = name  # an untyped sample: its own family
            labels = [
                (k, v) for k, v in LABEL.findall(labels or "") if k != "engine"
            ] + [("engine", str(index))]
            sample = name + "{" + ",".join(f'{k}="{v}"' for k, v in labels) + "}" + rest
            families.setdefault(family, {"samples": []})["samples"].append(sample)
    out = []
    for lines in families.values():
        out += [lines[key] for key in ("HELP", "TYPE") if key in lines]
        out += lines["samples"]
    return "\n".join(out) + "\n"


class PrefixIndex:
    """Chunk hash -> the engines that served a prompt with that prefix, as an LRU of at most ``capacity`` hashes. A
    request's hashes are touched longest first, so eviction drops the tails of old prompts before their heads (as an
    engine's prefix cache does) and every prefix of a kept hash stays kept."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.entries = OrderedDict()

    def match(self, hashes, backends):
        """{backend: matched chunks} for the backends holding at least the first chunk."""
        alive = set(backends)
        lengths = {}
        for i, h in enumerate(hashes):
            holders = self.entries.get(h)
            now = alive & holders if holders else set()
            for b in alive - now:
                if i:
                    lengths[b] = i
            alive = now
            if not alive:
                break
        for b in alive:
            lengths[b] = len(hashes)
        return lengths

    def record(self, hashes, backend):
        for h in reversed(hashes):
            holders = self.entries.pop(h, None) or set()
            holders.add(backend)
            self.entries[h] = holders
        while len(self.entries) > self.capacity:
            self.entries.popitem(last=False)


class Router:
    def __init__(
        self,
        backends,
        chunk_chars=256,
        min_match_chars=1024,
        slack=None,
        capacity=262144,
        down_s=5.0,
        prefill=None,
        pd_min_chars=None,
        pd_flex=None,
    ):
        self.backends = list(backends)
        # Prefill / decode disaggregation: the first ``prefill`` backends only prefill (module docstring).
        if prefill is None:
            prefill = int(os.environ.get("TT_PROXY_PD_PREFILL") or 0)
        if pd_min_chars is None:
            pd_min_chars = int(os.environ.get("TT_PROXY_PD_MIN_CHARS") or 8192)
        self.pd = 0 < prefill < len(self.backends)
        self.prefill_backends = self.backends[:prefill] if self.pd else []
        self.decode_backends = (
            self.backends[prefill:] if self.pd else list(self.backends)
        )
        self.pd_min_chars = pd_min_chars
        if pd_flex is None:
            pd_flex = int(os.environ.get("TT_PROXY_PD_FLEX") or 2)
        self.pd_flex = pd_flex
        self.chunk_chars = chunk_chars
        self.min_match_chunks = max(1, -(-min_match_chars // chunk_chars))
        if slack is None:
            slack = int(os.environ.get("TT_PROXY_SLACK") or 2)
        self.slack = slack
        self.down_s = down_s
        self.index = PrefixIndex(capacity)
        self.inflight = {b: 0 for b in self.backends}
        self.served = {b: 0 for b in self.backends}
        self.down_until = {b: 0.0 for b in self.backends}
        self.rr = itertools.count()
        self.stats = {
            "prefix_routed": 0,
            "prefix_spilled": 0,
            "least_loaded": 0,
            "matched_chars": 0,
            "retried": 0,
            "stale_retried": 0,
            "structured_outputs_normalized": 0,
            "pd_prefilled": 0,
            "pd_prefill_failed": 0,
            "pd_prefill_s": 0.0,
            "pd_direct": 0,
            "pd_flex": 0,
            "pd_wait_s": 0.0,
        }

    def up(self, exclude=(), pool=None):
        now = time.monotonic()
        pool = self.backends if pool is None else pool
        ups = [b for b in pool if b not in exclude and self.down_until[b] <= now]
        return ups or [b for b in pool if b not in exclude]

    def least_loaded(self, cands):
        low = min(self.inflight[b] for b in cands)
        ties = [b for b in cands if self.inflight[b] == low]
        return ties[next(self.rr) % len(ties)]

    def pick(self, hashes, exclude=(), pool=None):
        cands = self.up(exclude, pool)
        if not cands:
            return None
        if hashes:
            lengths = self.index.match(hashes, cands)
            best = max(lengths.values(), default=0)
            if best >= self.min_match_chunks:
                holder = self.least_loaded([b for b, n in lengths.items() if n == best])
                if (
                    self.inflight[holder]
                    <= min(self.inflight[b] for b in cands) + self.slack
                ):
                    self.stats["prefix_routed"] += 1
                    self.stats["matched_chars"] += best * self.chunk_chars
                    return holder
                self.stats["prefix_spilled"] += 1
        self.stats["least_loaded"] += 1
        return self.least_loaded(cands)

    def prefill_target(self, hashes):
        """The engine for a prefill-only pass now, or None to wait: a prefill engine running fewer than ``pd_flex``
        prefill passes (prefix holders first), else a decoding engine with nothing in flight."""
        if self.pd_flex < 0:
            return self.pick(hashes, pool=self.prefill_backends)
        free = [
            b
            for b in self.up(pool=self.prefill_backends)
            if self.inflight[b] < self.pd_flex
        ]
        if free:
            return self.pick(hashes, pool=free)
        idle = [b for b in self.up(pool=self.decode_backends) if self.inflight[b] == 0]
        if idle:
            self.stats["pd_flex"] += 1
            return self.least_loaded(idle)
        return None

    def uncached_chars(self, hashes, total_chars):
        """Characters of a prompt beyond the longest prefix any backend has served (all of them share KV through the
        store in disaggregated mode)."""
        lengths = self.index.match(hashes, self.backends) if hashes else {}
        return total_chars - max(lengths.values(), default=0) * self.chunk_chars

    def mark_down(self, backend):
        self.down_until[backend] = time.monotonic() + self.down_s

    def snapshot(self):
        out = {
            "inflight": self.inflight,
            "served": self.served,
            "prefix": dict(self.stats, indexed_chunks=len(self.index.entries)),
        }
        if self.pd:
            out["pd"] = {
                "prefill": self.prefill_backends,
                "decode": self.decode_backends,
                "min_chars": self.pd_min_chars,
            }
        return out


def make_app(router, session_factory=None):
    state = {}

    async def proxy(request: web.Request):
        body = await request.read()
        if request.method == "POST" and request.path in STRUCTURED_PATHS:
            fixed = normalize_structured_outputs(body)
            if fixed is not body:
                router.stats["structured_outputs_normalized"] += 1
                body = fixed
        hashes = []
        text = None
        if request.method == "POST" and request.path in PREFIX_PATHS:
            text = prefix_text(request.path, body)
            if text is not None:
                hashes = chunk_hashes(text, router.chunk_chars)
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
        if router.pd and text is not None and request.path in PD_PATHS:
            total = len(text.encode("utf-8", "surrogatepass"))
            if router.uncached_chars(hashes, total) >= router.pd_min_chars:
                await prefill(request.path, body, headers, hashes)
            else:
                router.stats["pd_direct"] += 1
        pool = router.decode_backends if router.pd else None
        tried = []
        stale_retry = True
        while True:
            b = router.pick(hashes, exclude=tried, pool=pool)
            if b is None:
                return web.json_response(
                    {"error": "no backend accepted the request"}, status=503
                )
            router.inflight[b] += 1
            if hashes:
                # Now, not when the response starts: a request sent while this one prefills (or a non-streaming one,
                # whose headers come at its end) should find the prefix where it is being computed.
                router.index.record(hashes, b)
            started = False
            try:
                async with state["session"].request(
                    request.method,
                    b + request.path_qs,
                    data=body or None,
                    headers=headers,
                ) as up:
                    router.served[b] += 1
                    resp = web.StreamResponse(
                        status=up.status,
                        headers={
                            k: v for k, v in up.headers.items() if k.lower() not in HOP
                        },
                    )
                    await resp.prepare(request)
                    started = True
                    async for chunk in up.content.iter_any():
                        await resp.write(chunk)
                    await resp.write_eof()
                    return resp
            except ServerDisconnectedError:
                # The engine closed the connection before answering: almost always a pooled keep-alive connection it
                # was closing as we wrote into it (the engine never read the request). Resend once on a new
                # connection; a disconnect after the reply started cannot be retried.
                if started or not stale_retry:
                    raise
                stale_retry = False
                router.stats["stale_retried"] += 1
            except ClientConnectorError:
                # Nothing reached the client yet: try another engine.
                router.mark_down(b)
                tried.append(b)
                router.stats["retried"] += 1
                if len(tried) == len(pool or router.backends):
                    return web.json_response(
                        {"error": "every backend refused the connection"}, status=503
                    )
            finally:
                router.inflight[b] -= 1

    async def prefill(path, body, headers, hashes):
        """Run the prompt on a prefill engine (one output token) so that its KV is in the store before the decoding
        engine gets the request. Any failure is logged in the stats and leaves the prefill to the decoding engine."""
        pbody = prefill_body(body)
        if pbody is None:
            return
        t_wait = time.monotonic()
        while (p := router.prefill_target(hashes)) is None:
            await asyncio.sleep(PD_POLL_S)  # every engine that may prefill is busy
        router.stats["pd_wait_s"] += time.monotonic() - t_wait
        router.inflight[p] += 1
        t0 = time.monotonic()
        ok = False
        try:
            for attempt in range(
                2
            ):  # resend once after an unanswered keep-alive disconnect, as proxy() does
                try:
                    async with state["session"].post(
                        p + path, data=pbody, headers=headers
                    ) as up:
                        await up.read()
                        ok = up.status == 200
                    break
                except ServerDisconnectedError:
                    if attempt:
                        break
                    router.stats["stale_retried"] += 1
                except ClientConnectorError:
                    router.mark_down(p)
                    break
        finally:
            router.inflight[p] -= 1
        if ok:
            router.served[p] += 1
            # Recorded once the prompt is computed (and stored), not when it was sent: a request arriving meanwhile
            # with the same new prefix is prefilled too instead of being sent to a decoding engine as if cached.
            router.index.record(hashes, p)
            router.stats["pd_prefilled"] += 1
            router.stats["pd_prefill_s"] += time.monotonic() - t0
        else:
            router.stats["pd_prefill_failed"] += 1

    async def stats(_):
        return web.json_response(router.snapshot())

    async def all_metrics(_):
        # An engine that fails or hangs leaves out only its own samples (its index keeps its place).
        texts, missing = [], []
        for i, b in enumerate(router.backends):
            try:
                async with state["session"].get(
                    b + "/metrics", timeout=ClientTimeout(total=PROBE_TIMEOUT_S)
                ) as up:
                    texts.append(await up.text())
            except Exception:  # noqa: BLE001 - any failure means no metrics from this engine
                texts.append("")
                missing.append(f"# engine {i} ({b}) did not answer\n")
        return web.Response(
            text="".join(missing) + merge_metrics(texts), content_type="text/plain"
        )

    async def health(_):
        bad = []
        for b in router.backends:
            try:
                async with state["session"].get(
                    b + "/health", timeout=ClientTimeout(total=PROBE_TIMEOUT_S)
                ) as up:
                    if up.status != 200:
                        bad.append(b)
            except Exception:  # noqa: BLE001 - any failure means not healthy
                bad.append(b)
        if bad:
            return web.json_response({"unhealthy": bad}, status=503)
        return web.Response(status=200)

    async def on_start(app):
        state["session"] = (session_factory or ClientSession)(
            timeout=ClientTimeout(total=None),
            connector=TCPConnector(limit=0, keepalive_timeout=UPSTREAM_KEEPALIVE_S),
        )

    async def on_stop(app):
        await state["session"].close()

    app = web.Application(client_max_size=1 << 30)
    app.on_startup.append(on_start)
    app.on_cleanup.append(on_stop)
    app.router.add_get("/lb/stats", stats)
    app.router.add_get("/metrics", all_metrics)
    app.router.add_get("/health", health)
    app.router.add_route("*", "/{tail:.*}", proxy)
    return app
