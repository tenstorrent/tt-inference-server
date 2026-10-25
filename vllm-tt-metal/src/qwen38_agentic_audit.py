# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""Bounded request-level evidence for goal12; no prompts, tokens or headers logged.

Non-streaming vLLM0.30 responses can return prompt IDs. They are used once to
measure prior-prompt LCP, then restored to null unless the caller requested IDs.
Streaming traffic passes through unchanged and explicitly lacks prefix evidence.
"""
from array import array
from collections import OrderedDict
import hashlib
import json
import logging
import os
from pathlib import Path
import time
import uuid


def number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def opaque(value):
    if value is None:
        return None
    return hashlib.sha256(value if isinstance(value, bytes) else str(value).encode()).hexdigest()


class AgenticAuditMiddleware:
    def __init__(self, app, *, path=None, max_sessions=64, max_tokens=8388608, ttl_s=3600):
        self.app = app
        self.path = path or os.getenv("QWEN38_HTTP_AUDIT")
        self.sessions = OrderedDict()
        self.session_inflight = {}
        self.active = 0
        self.stored_tokens = 0
        self.evictions = 0
        self.audit_write_failures = 0
        self.max_sessions, self.max_tokens, self.ttl_s = max_sessions, max_tokens, ttl_s
        if self.path:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)

    def emit(self, data):
        # No credentials, headers, request/response bodies or token arrays enter data.
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(data, separators=(",", ":")) + "\n")
        except OSError:
            self.audit_write_failures += 1
            if self.audit_write_failures == 1:
                logging.getLogger(__name__).error("Goal12 audit writes failed; request evidence unavailable")

    def evict(self, now):
        for key, (_, stamp) in list(self.sessions.items()):
            if now - stamp > self.ttl_s:
                self.stored_tokens -= len(self.sessions.pop(key)[0])
                self.evictions += 1

    def remember(self, key, ids, now):
        old = self.sessions.pop(key, None)
        if old:
            self.stored_tokens -= len(old[0])
        if len(ids) > self.max_tokens:
            return False
        while self.sessions and (len(self.sessions) >= self.max_sessions or self.stored_tokens + len(ids) > self.max_tokens):
            self.stored_tokens -= len(self.sessions.popitem(last=False)[1][0])
            self.evictions += 1
        self.sessions[key] = (array("I", ids), now)
        self.stored_tokens += len(ids)
        return True

    async def __call__(self, scope, receive, send):
        if not self.path or scope.get("type") != "http" or scope.get("path") != "/v1/chat/completions":
            return await self.app(scope, receive, send)
        original_messages = []
        while True:
            message = await receive()
            original_messages.append(message)
            if message["type"] != "http.request" or not message.get("more_body", False):
                break
        body = b"".join(m.get("body", b"") for m in original_messages)
        try:
            request = json.loads(body)
            if not isinstance(request, dict):
                raise ValueError("not object")
        except (ValueError, UnicodeDecodeError):
            request = None
        streaming = bool(request and request.get("stream"))
        inject = request is not None and not streaming and not request.get("return_token_ids") and not request.get("echo")
        replay = list(original_messages)
        child_scope = scope
        if inject:
            request = {**request, "return_token_ids": True}
            changed = json.dumps(request, separators=(",", ":")).encode()
            replay = [{"type": "http.request", "body": changed, "more_body": False}]
            child_scope = {**scope, "headers": [(k, v) for k, v in scope.get("headers", []) if k.lower() != b"content-length"] + [(b"content-length", str(len(changed)).encode())]}
        async def replay_receive():
            return replay.pop(0) if replay else await receive()
        session = next((opaque(v) for k, v in scope.get("headers", []) if k.lower() == b"x-session-id"), None)
        started = time.monotonic()
        self.evict(started)
        self.active += 1
        overlap = bool(session and self.session_inflight.get(session, 0))
        if session:
            self.session_inflight[session] = self.session_inflight.get(session, 0) + 1
        base = {"audit_id": uuid.uuid4().hex, "session_sha256": session, "started_unix": time.time(),
                "seed": number((request or {}).get("seed")), "max_tokens": number((request or {}).get("max_completion_tokens", (request or {}).get("max_tokens"))),
                "stream": streaming, "active_http_at_start": self.active,
                "messages_sha256": opaque(json.dumps((request or {}).get("messages"), sort_keys=True)),
                "request_id_sha256": opaque((request or {}).get("request_id")), "session_overlap": overlap}
        self.emit({"event": "request_start", **base})
        response_start, chunks = None, []
        record = {**base}
        async def audit_send(message):
            nonlocal response_start
            if message["type"] == "http.response.start":
                record["http_status"] = message["status"]
                if not streaming and request is not None:
                    response_start = message
                    return
            if message["type"] == "http.response.body" and response_start is not None:
                chunks.append(message.get("body", b""))
                if message.get("more_body", False):
                    return
                raw = b"".join(chunks)
                try:
                    response = json.loads(raw)
                    if not isinstance(response, dict):
                        raise ValueError("not object")
                    self.observe(response, session, started, overlap, record)
                    if inject:
                        # vLLM0.30 emits these fields as null without return_token_ids.
                        if "prompt_token_ids" in response:
                            response["prompt_token_ids"] = None
                        for choice in response.get("choices", []):
                            if isinstance(choice, dict) and "token_ids" in choice:
                                choice["token_ids"] = None
                        raw = json.dumps(response, ensure_ascii=False, separators=(",", ":")).encode()
                except (ValueError, TypeError, OverflowError, AttributeError):
                    record["missing_reason"] = "non-JSON or invalid response token metadata"
                headers = [(k, v) for k, v in response_start.get("headers", []) if k.lower() != b"content-length"]
                await send({**response_start, "headers": headers + [(b"content-length", str(len(raw)).encode())]})
                await send({**message, "body": raw})
                return
            await send(message)
        try:
            await self.app(child_scope, replay_receive, audit_send)
        except BaseException as error:
            record["exception_type"] = type(error).__name__
            raise
        finally:
            self.active -= 1
            if session:
                self.session_inflight[session] -= 1
                if not self.session_inflight[session]:
                    del self.session_inflight[session]
            record.update(event="request_end", finished_unix=time.time(), elapsed_s=time.monotonic() - started,
                          active_http_after=self.active, state_evictions=self.evictions, stored_prefix_tokens=self.stored_tokens,
                          audit_write_failures=self.audit_write_failures)
            if streaming:
                record["missing_reason"] = "streaming passed through unchanged; no token or usage inspection"
            elif request and request.get("echo") and not request.get("return_token_ids"):
                record["prefix_missing_reason"] = "echo request passed through without token-ID injection"
            elif request is None:
                record["missing_reason"] = "malformed JSON request passed through unchanged"
            self.emit(record)

    def observe(self, response, session, started, overlap, record):
        usage = response.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        record.update(response_id_sha256=opaque(response.get("id")),
                      prompt_tokens=number(usage.get("prompt_tokens")), completion_tokens=number(usage.get("completion_tokens")),
                      cached_tokens=number(details.get("cached_tokens")),
                      finish_reasons=[c.get("finish_reason") for c in response.get("choices", []) if isinstance(c, dict)])
        ids = response.get("prompt_token_ids")
        valid_ids = isinstance(ids, list) and all(isinstance(t, int) and not isinstance(t, bool) and 0 <= t < 2**32 for t in ids)
        if not valid_ids:
            record["prefix_missing_reason"] = "response lacks valid prompt token IDs"
            return
        record["prompt_ids_count"] = len(ids)
        record["prompt_token_count_matches_usage"] = len(ids) == record["prompt_tokens"]
        record["prompt_ids_sha256"] = hashlib.sha256(array("I", ids).tobytes()).hexdigest()
        previous = self.sessions.get(session) if session else None
        if not session:
            record["prefix_missing_reason"] = "no X-Session-ID; independent request"
        elif overlap or self.session_inflight.get(session, 0) > 1:
            record["prefix_missing_reason"] = "overlapping requests in same session"
            self.sessions.pop(session, None)
            self.stored_tokens = sum(len(v[0]) for v in self.sessions.values())
        else:
            if previous:
                prior = previous[0]
                length = 0
                for a, b in zip(prior, ids):
                    if a != b:
                        break
                    length += 1
                record.update(previous_sent_prompt_tokens=len(prior), previous_prompt_lcp_tokens=length,
                              previous_turn_away_s=max(0, started - previous[1]))
            else:
                record["prefix_missing_reason"] = "first observed session turn or audit-state eviction"
            record["prefix_state_saved"] = self.remember(session, ids, time.monotonic())
