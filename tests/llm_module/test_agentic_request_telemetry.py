# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import copy
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from llm_module.agentic.request_telemetry import (
    REPEATED_TOOL_FEEDBACK,
    REPETITION_FEEDBACK,
    SUBMISSION_COMMAND,
    RequestTelemetryProxy,
    limit_reasoning_history,
    normalize_submission_marker,
    parse_server_counters,
    recent_tool_summary,
    repeated_failure_count,
    repeated_tool_count,
    response_text_stats,
)


@pytest.mark.parametrize("keep", [0, 1, 2, 10])
def test_reasoning_history_policy_preserves_all_visible_evidence(keep):
    messages = [
        {"role": "system", "content": "instructions", "reasoning": "untouched"},
        {
            "role": "assistant",
            "content": "visible",
            "reasoning_content": "old analysis",
            "tool_calls": [{"id": "a"}],
        },
        {"role": "tool", "content": "observed result", "tool_call_id": "a"},
        {"role": "assistant", "content": "next visible", "reasoning": "latest plan"},
        {"role": "user", "content": "task", "reasoning_content": "untouched"},
    ]
    saved = copy.deepcopy(messages)
    result, removed = limit_reasoning_history(messages, keep)
    assert messages == saved
    assert [row["message_index"] for row in removed] == (
        [1, 3] if keep == 0 else [1] if keep == 1 else []
    )
    for before, after in zip(messages, result):
        for key in ("role", "content", "tool_calls", "tool_call_id"):
            assert before.get(key) == after.get(key)
    assert result[0] == messages[0]
    assert result[-1] == messages[-1]
    if keep:
        assert result[3]["reasoning"] == "latest plan"


@pytest.mark.parametrize("invalid", [-1, True, 1.5, "1"])
def test_reasoning_history_policy_rejects_invalid_limits(invalid):
    with pytest.raises(ValueError, match="nonnegative integer"):
        limit_reasoning_history([], invalid)


def test_server_counters_drop_labels_and_nonfinite_or_unrelated_metrics():
    assert parse_server_counters(
        'vllm:request_success_total{model_name="private",finished_reason="stop"} 2\n'
        'vllm:request_success_total{finished_reason="length"} 3\n'
        "vllm:time_to_first_token_seconds_sum 1.25e2\n"
        "vllm:generation_tokens_total 1e999\n"
        "vllm:prompt_tokens_total -1\n"
        "private_metric 999\n"
    ) == {"request_success_total": 5.0, "time_to_first_token_seconds_sum": 125.0}


def test_server_metric_snapshots_preserve_timing_and_disable_on_failure(tmp_path):
    hits = []

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            hits.append(self.path)
            self.send_response(200 if len(hits) == 1 else 503)
            self.end_headers()
            self.wfile.write(b'vllm:request_success_total{model_name="private"} 7\n')

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    try:
        path = tmp_path / "metrics.jsonl"
        proxy = RequestTelemetryProxy(
            f"http://127.0.0.1:{upstream.server_port}/v1",
            path,
            5,
            collect_server_metrics=True,
        )
        for phase in ("before_request", "after_response", "before_request"):
            proxy.snapshot_server_metrics({"session": "trial"}, phase, {})
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == 2
        assert rows[0]["counters"] == {"request_success_total": 7}
        assert rows[0]["collection_s"] >= 0
        assert rows[1]["error_type"] == "HTTPError"
        assert hits == ["/metrics", "/metrics"]
        assert "private" not in path.read_text()
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join()


def submission_example():
    return (
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "Done.\n\n" + SUBMISSION_COMMAND,
                    },
                }
            ],
            "usage": {"completion_tokens": 100},
        },
        {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "parameters": {"properties": {"command": {"type": "string"}}},
                    },
                }
            ]
        },
    )


def test_submission_normalization_preserves_output_and_usage():
    response, payload = submission_example()
    original = copy.deepcopy(response)
    normalized = normalize_submission_marker(response, payload)
    assert response == original
    assert normalized["usage"] == response["usage"]
    message = normalized["choices"][0]["message"]
    assert message["content"] == response["choices"][0]["message"]["content"]
    assert normalized["choices"][0]["finish_reason"] == "tool_calls"
    assert message["tool_calls"][0]["function"] == {
        "name": "bash",
        "arguments": json.dumps({"command": SUBMISSION_COMMAND}),
    }


@pytest.mark.parametrize(
    "content",
    [
        "Not done",
        SUBMISSION_COMMAND + " && touch arbitrary",
        "`" + SUBMISSION_COMMAND + "`",
        "> " + SUBMISSION_COMMAND,
        "```bash\n" + SUBMISSION_COMMAND,
        "~~~bash\n" + SUBMISSION_COMMAND,
        SUBMISSION_COMMAND + "\nMore work remains.",
        "I might run " + SUBMISSION_COMMAND,
        None,
    ],
)
def test_submission_normalization_rejects_ambiguous_text(content):
    response, payload = submission_example()
    response["choices"][0]["message"]["content"] = content
    assert normalize_submission_marker(response, payload) is None


@pytest.mark.parametrize(
    "reason", ["length", "repetition", "error", "tool_calls", None]
)
def test_submission_normalization_requires_natural_stop(reason):
    response, payload = submission_example()
    response["choices"][0]["finish_reason"] = reason
    assert normalize_submission_marker(response, payload) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("tool_calls", [{"id": "existing"}]),
        ("function_call", {"name": "other"}),
        ("refusal", "Refused"),
        ("role", "user"),
    ],
)
def test_submission_normalization_preserves_other_protocol_actions(field, value):
    response, payload = submission_example()
    response["choices"][0]["message"][field] = value
    assert normalize_submission_marker(response, payload) is None


def test_submission_normalization_requires_one_choice_and_available_bash():
    response, payload = submission_example()
    assert normalize_submission_marker(response, {}) is None
    assert (
        normalize_submission_marker(response, {**payload, "tool_choice": "none"})
        is None
    )
    assert (
        normalize_submission_marker(response, {**payload, "tool_choice": "required"})
        is None
    )
    assert (
        normalize_submission_marker(
            response,
            {
                **payload,
                "tool_choice": {"type": "function", "function": {"name": "other"}},
            },
        )
        is None
    )
    response["choices"] *= 2
    assert normalize_submission_marker(response, payload) is None


def test_submission_normalization_never_reads_reasoning_for_marker():
    response, payload = submission_example()
    response["choices"][0]["message"] = {
        "role": "assistant",
        "content": None,
        "reasoning": SUBMISSION_COMMAND,
    }
    assert normalize_submission_marker(response, payload) is None


def test_submission_normalization_is_audited_and_default_off(tmp_path):
    response, payload = submission_example()
    body = json.dumps(response).encode()

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    try:
        for enabled in (False, True):
            path = tmp_path / f"submission_{enabled}.jsonl"
            with RequestTelemetryProxy(
                f"http://127.0.0.1:{upstream.server_port}/v1",
                path,
                5,
                normalize_submission=enabled,
            ) as endpoint:
                with urlopen(
                    Request(
                        endpoint + "/chat/completions", json.dumps(payload).encode()
                    )
                ) as received:
                    forwarded = received.read()
            events = [json.loads(line) for line in path.read_text().splitlines()]
            assert sum(
                e["event"] == "submission_marker_normalized" for e in events
            ) == int(enabled)
            assert events[1]["finish_reasons"] == ["stop"]
            assert events[1]["tool_counts"] == [0]
            if enabled:
                assert (
                    json.loads(forwarded)["choices"][0]["finish_reason"] == "tool_calls"
                )
            else:
                assert forwarded == body
            assert SUBMISSION_COMMAND not in path.read_text()
    finally:
        upstream.shutdown()
        thread.join()
        upstream.server_close()


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
    payload = b'{"messages":[{"role":"user","content":"private prompt"}],"max_tokens":32768,"seed":9472}'
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
        assert records[0]["request_seed"] == 9472
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


def test_exact_repeat_detector_includes_nonconsecutive_successes():
    def pair(command="inspect", output="same", returncode=0):
        return [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "x",
                        "function": {
                            "name": "bash",
                            "arguments": json.dumps({"command": command}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "x",
                "content": json.dumps({"returncode": returncode, "output": output}),
            },
        ]

    history = pair() + pair("different") + pair() + pair("other") + pair()
    original = copy.deepcopy(history)
    assert repeated_tool_count(history) == 3
    assert repeated_tool_count(history, window=3) == 2
    assert repeated_tool_count(history + pair(output="changed")) == 1
    assert repeated_tool_count(history + pair(returncode=1)) == 1
    assert (
        repeated_tool_count(history + [{"role": "tool", "content": "malformed"}]) == 0
    )
    assert history == original


@pytest.mark.parametrize("mode", ["failed", "exact", "reasoning"])
def test_feedback_is_opt_in_and_preserves_history_and_sampling(tmp_path, mode):
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
        "reasoning_content": "private diagnostic reasoning",
        "tool_calls": [
            {
                "function": {
                    "name": "bash",
                    "arguments": '{"command":"grep missing file"}',
                }
            }
        ],
    }
    failure = {
        "role": "tool",
        "content": json.dumps(
            {"returncode": 1 if mode == "failed" else 0, "output": ""}
        ),
    }
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
                repetition_feedback_after=threshold if mode == "failed" else 0,
                repeated_tool_feedback=bool(threshold) if mode == "exact" else False,
                reasoning_history_limit=(
                    1 if mode == "reasoning" and threshold else None
                ),
            ) as endpoint:
                with urlopen(
                    Request(
                        endpoint + "/chat/completions", json.dumps(payload).encode()
                    )
                ) as response:
                    assert response.status == 200
            records = [json.loads(line) for line in path.read_text().splitlines()]
            event = (
                "repetition_feedback"
                if mode == "failed"
                else (
                    "repeated_tool_feedback"
                    if mode == "exact"
                    else "reasoning_history_limited"
                )
            )
            assert sum(r["event"] == event for r in records) == bool(threshold)
            assert "private diagnostic reasoning" not in path.read_text()
        assert seen[0] == payload
        if mode == "reasoning":
            expected, _ = limit_reasoning_history(payload["messages"], 1)
            assert seen[1] == {**payload, "messages": expected}
            return
        assert seen[1] == {
            **payload,
            "messages": payload["messages"]
            + [
                {
                    "role": "user",
                    "content": (
                        REPETITION_FEEDBACK
                        if mode == "failed"
                        else REPEATED_TOOL_FEEDBACK
                    ),
                }
            ],
        }
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join()
