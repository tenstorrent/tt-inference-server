# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from llm_module.agentic.request_telemetry import (
    REPETITION_FEEDBACK,
    RequestTelemetryProxy,
    recent_tool_summary,
    repeated_failure_count,
    response_text_stats,
)


def test_response_repetition_stats_do_not_record_text():
    stats = response_text_stats({"content": "secret repeated long output line\n" * 12})
    assert stats["max_identical_line_count"] == 12
    assert stats["repeated_line_char_fraction"] > 0.8
    assert "secret" not in json.dumps(stats)
    assert response_text_stats({})["max_identical_line_count"] == 0


def test_tool_summary_records_categories_without_payloads():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "a",
                    "function": {"arguments": '{"command":"pytest private_file.py"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "a",
            "content": '{"returncode":1,"output":"private failure"}',
        },
    ]
    rows = recent_tool_summary(messages)
    assert rows[0]["categories_heuristic"] == ["test"]
    assert rows[0]["returncode"] == 1
    assert rows[0]["output_chars"] == 15
    assert "private" not in json.dumps(rows)
    assert recent_tool_summary([]) == []


def test_repetition_detector_requires_identical_failed_results():
    action = {
        "role": "assistant",
        "tool_calls": [
            {
                "function": {
                    "name": "bash",
                    "arguments": '{"command":"grep missing file"}',
                }
            }
        ],
    }
    failed = {"role": "tool", "content": '{"returncode":1,"output":""}'}
    assert repeated_failure_count([action, failed] * 3) == 3
    assert repeated_failure_count([action, failed] * 2) == 2
    succeeded = {"role": "tool", "content": '{"returncode":0,"output":""}'}
    assert repeated_failure_count([action, failed] * 3 + [action, succeeded]) == 0
    changed = {"role": "tool", "content": '{"returncode":1,"output":"new evidence"}'}
    assert repeated_failure_count([action, failed] * 3 + [action, changed]) == 1
    assert (
        repeated_failure_count(
            [action, failed] * 3 + [{"role": "user", "content": "new instruction"}]
        )
        == 0
    )


@pytest.mark.parametrize(
    "status,body",
    [
        (
            200,
            b'{"choices":[{"message":{"content":"private output"},"finish_reason":"length"}],"usage":{"completion_tokens":32768}}',
        ),
        (400, b'{"error":{"message":"private error detail"}}'),
        (503, b"upstream unavailable"),
    ],
)
def test_nonstream_response_bytes_and_errors_are_preserved(tmp_path, status, body):
    seen = []

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            seen.append(
                (
                    self.path,
                    self.headers.get("Authorization"),
                    self.rfile.read(int(self.headers["Content-Length"])),
                )
            )
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    path = tmp_path / "requests.jsonl"
    payload = (
        b'{"messages":[{"role":"user","content":"private prompt"}],"max_tokens":32768}'
    )
    try:
        with RequestTelemetryProxy(
            f"http://127.0.0.1:{upstream.server_port}/v1", path, 5
        ) as endpoint:
            request = Request(
                endpoint + "/chat/completions",
                payload,
                {"Authorization": "Bearer secret", "X-Session-ID": "trial"},
            )
            try:
                response = urlopen(request)
            except HTTPError as error:
                response = error
            assert response.status == status
            assert response.read() == body
        assert seen == [("/v1/chat/completions", "Bearer secret", payload)]
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert [r["event"] for r in records] == ["request_start", "response"]
        assert records[0]["session"] == "trial"
        assert records[1]["elapsed_s"] >= 0
        assert "private" not in path.read_text()
        assert "secret" not in path.read_text()
        if status == 200:
            assert records[1]["tool_counts"] == [0]
            assert records[1]["usage"]["completion_tokens"] == 32768
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join()


def test_feedback_is_opt_in_and_preserves_history_and_sampling(tmp_path):
    seen = []

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            seen.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"choices":[]}')

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    action = {
        "role": "assistant",
        "tool_calls": [
            {
                "function": {
                    "name": "bash",
                    "arguments": '{"command":"grep missing file"}',
                }
            }
        ],
    }
    failure = {"role": "tool", "content": '{"returncode":1,"output":""}'}
    payload = {
        "messages": [action, failure] * 3,
        "temperature": 1,
        "top_p": 0.95,
        "max_tokens": 32768,
    }
    try:
        for threshold in (0, 3):
            path = tmp_path / f"requests-{threshold}.jsonl"
            with RequestTelemetryProxy(
                f"http://127.0.0.1:{upstream.server_port}/v1",
                path,
                5,
                repetition_feedback_after=threshold,
            ) as endpoint:
                with urlopen(
                    Request(
                        endpoint + "/chat/completions", json.dumps(payload).encode()
                    )
                ) as response:
                    assert response.status == 200
            records = [json.loads(line) for line in path.read_text().splitlines()]
            assert sum(r["event"] == "repetition_feedback" for r in records) == bool(
                threshold
            )
        assert seen[0] == payload
        assert seen[1] == {
            **payload,
            "messages": payload["messages"]
            + [{"role": "user", "content": REPETITION_FEEDBACK}],
        }
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join()
