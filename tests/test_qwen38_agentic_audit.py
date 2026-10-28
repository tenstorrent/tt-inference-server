# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
import asyncio
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("qwen38_agentic_audit", Path(__file__).resolve().parents[1] / "vllm-tt-metal/src/qwen38_agentic_audit.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Audit = module.AgenticAuditMiddleware


class AuditTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.received = []
        self.prompt_ids = [1, 2, 3]
        self.status = 200

        async def app(scope, receive, send):
            request = await receive()
            self.received.append(request["body"])
            try:
                body = json.loads(request["body"])
            except ValueError:
                body = {}
            if body.get("stream"):
                await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/event-stream")]})
                await send({"type": "http.response.body", "body": b"data: SSE-secret\n\n", "more_body": True})
                await send({"type": "http.response.body", "body": b"data: [DONE]\n\n"})
                return
            response = {"id": "chatcmpl-secret", "choices": [{"message": {"content": "answer-secret", "reasoning_content": "reasoning-secret"}, "finish_reason": "stop", "token_ids": [20]}],
                        "prompt_token_ids": list(self.prompt_ids), "usage": {"prompt_tokens": len(self.prompt_ids), "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 2}}}
            if self.status != 200:
                response = {"error": {"message": "failure-secret"}}
            raw = json.dumps(response).encode()
            await send({"type": "http.response.start", "status": self.status, "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(raw)).encode())]})
            await send({"type": "http.response.body", "body": raw[:4], "more_body": True})
            await send({"type": "http.response.body", "body": raw[4:]})
        self.audit = Audit(app, path=str(self.path))

    async def request(self, body=None, session="session-secret"):
        if body is None:
            body = {"messages": [{"role": "user", "content": "prompt-secret"}], "seed": 1234, "max_tokens": 128}
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        headers = [(b"authorization", b"Bearer credential-secret"), (b"content-length", str(len(raw)).encode())]
        if session is not None:
            headers.append((b"x-session-id", session.encode()))
        incoming = [{"type": "http.request", "body": raw}]
        outgoing = []
        async def receive():
            return incoming.pop(0) if incoming else {"type": "http.disconnect"}
        async def send(value):
            outgoing.append(value)
        await self.audit({"type": "http", "path": "/v1/chat/completions", "headers": headers}, receive, send)
        return outgoing

    def ends(self):
        return [r for r in map(json.loads, self.path.read_text().splitlines()) if r['event']=='request_end']

    async def test_policy_selectors_are_recorded_without_arbitrary_template_values(self):
        await self.request({'messages': [], 'chat_template_kwargs': {
            'reasoning_effort': 'medium', 'enable_thinking': True,
            'preserve_thinking': False, 'private': 'never-record-this'}})
        assert self.ends()[-1]['requested_chat_template_policy'] == {
            'reasoning_effort': 'medium', 'enable_thinking': True, 'preserve_thinking': False}
        assert 'never-record-this' not in self.path.read_text()

    async def test_lcp_seed_and_response_semantics_without_raw_artifacts(self):
        output = await self.request()
        self.assertTrue(json.loads(self.received[0])["return_token_ids"])
        response = json.loads(output[-1]["body"])
        self.assertIsNone(response["prompt_token_ids"])
        self.assertIsNone(response["choices"][0]["token_ids"])
        self.assertEqual(response["choices"][0]["message"]["reasoning_content"], "reasoning-secret")
        self.assertEqual(int(dict(output[0]["headers"])[b"content-length"]), len(output[-1]["body"]))
        self.prompt_ids = [1, 2, 3, 4, 5]
        await self.request()
        record = self.ends()[-1]
        self.assertEqual(record["seed"], 1234)
        self.assertEqual(record["cached_tokens"], 2)
        self.assertEqual(record["previous_prompt_lcp_tokens"], 3)
        self.assertEqual(record["previous_sent_prompt_tokens"], 3)
        self.assertNotIn("secret", self.path.read_text())

    async def test_sessions_are_isolated_and_state_bounded(self):
        self.audit.max_sessions = 1
        await self.request(session="one")
        await self.request(session="two")
        self.assertNotIn("previous_prompt_lcp_tokens", self.ends()[-1])
        await self.request(session="one")
        self.assertNotIn("previous_prompt_lcp_tokens", self.ends()[-1])
        self.assertEqual(len(self.audit.sessions), 1)
        self.assertEqual(self.audit.stored_tokens, 4)
        self.assertEqual(self.audit.evictions, 2)

    async def test_streaming_passes_through_without_injection(self):
        body = {"stream": True, "messages": [], "seed": 1234}
        out = await self.request(body)
        self.assertEqual(json.loads(self.received[-1]), body)
        self.assertEqual(out[1]["body"], b"data: SSE-secret\n\n")
        self.assertEqual(out[2]["body"], b"data: [DONE]\n\n")
        self.assertIn("streaming passed through", self.ends()[-1]["missing_reason"])
        self.assertNotIn("secret", self.path.read_text())

    async def test_malformed_error_and_explicit_token_request(self):
        self.status = 400
        out = await self.request(b'{malformed')
        self.assertEqual(self.received[-1], b'{malformed')
        self.assertEqual(out[0]["status"], 400)
        self.assertIn("malformed", self.ends()[-1]["missing_reason"])
        self.status = 200
        out = await self.request({"messages": [], "return_token_ids": True}, session=None)
        self.assertEqual(json.loads(out[-1]["body"])["prompt_token_ids"], self.prompt_ids)
        self.assertIn("independent request", self.ends()[-1]["prefix_missing_reason"])
        self.assertEqual(self.audit.active, 0)

    async def test_echo_request_not_modified(self):
        body = {"messages": [], "echo": True}
        await self.request(body)
        self.assertEqual(json.loads(self.received[-1]), body)
        self.assertIn("echo request", self.ends()[-1]["prefix_missing_reason"])

    async def test_exception_cleans_request_counters(self):
        async def failing(scope, receive, send):
            raise RuntimeError("private-secret")
        self.audit.app = failing
        with self.assertRaises(RuntimeError):
            await self.request()
        self.assertEqual(self.audit.active, 0)
        self.assertEqual(self.audit.session_inflight, {})
        self.assertEqual(self.ends()[-1]["exception_type"], "RuntimeError")
        self.assertNotIn("secret", self.path.read_text())


if __name__ == "__main__":
    unittest.main()
