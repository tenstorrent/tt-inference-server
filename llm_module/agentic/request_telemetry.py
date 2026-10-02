# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Payload-free timing and opt-in, audited mini-swe protocol interventions."""

import copy
import hashlib
import json
import math
import re
import threading
import time
import uuid
from collections import Counter
from contextlib import AbstractContextManager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from llm_module.agentic.abortable_request import (
    ClientDisconnected,
    post_until_disconnect,
)

REPETITION_FEEDBACK = (
    "Loop check: the same command has been repeated with the same unsuccessful "
    "result. Repeating it again without new evidence or a relevant change will "
    "not add evidence. Choose a different inspection or test, update your "
    "hypothesis using the observations already present, and continue solving "
    "the original issue. Do not submit until you have made and checked the "
    "required fix."
)
SUBMISSION_COMMAND = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
SUBMISSION_REVIEW_COMMAND = (
    "git diff --check; review_status=$?; git diff --stat; git diff --; "
    "printf '%s\\n' 'Submission paused once for a generic validation checkpoint. "
    "Review the actual tracked diff above. If it is empty, no tracked implementation "
    "change has been made. Confirm the original issue is fixed with executable "
    "checks, run relevant existing repository tests, and investigate failures or "
    "unintended changes before submitting again. This checkpoint provides no "
    'task-specific solution and does not claim tests passed.\'; exit "$review_status"'
)
REPEATED_TOOL_FEEDBACK = (
    "Loop check: the most recent shell command and its exact result have appeared "
    "at least twice in the recent tool history. Reuse that evidence rather "
    "than repeating equivalent inspections, unless a relevant change makes a "
    "rerun necessary. Reconsider your current hypothesis, make the smallest "
    "justified source change, and validate the original issue. Before submission, "
    "run relevant existing repository tests for the edited code, not only custom "
    "examples, and investigate any regression. Review the actual diff for "
    "unintended deletions and submit only the checked patch. "
    "This note supplies no task-specific solution."
)
SERVER_COUNTERS = frozenset(
    {
        "prompt_tokens_total",
        "generation_tokens_total",
        "request_success_total",
        "time_to_first_token_seconds_count",
        "time_to_first_token_seconds_sum",
        "e2e_request_latency_seconds_count",
        "e2e_request_latency_seconds_sum",
    }
)


def parse_server_counters(text):
    """Retain numeric aggregate counters only, never labels or metric payloads."""
    counters = {}
    for line in text.splitlines():
        match = re.fullmatch(r"vllm:(\w+)(?:\{[^\n]*\})? ([\d.eE+\-]+)", line)
        if match is None or match[1] not in SERVER_COUNTERS:
            continue
        value = float(match[2])
        if math.isfinite(value) and value >= 0:
            counters[match[1]] = counters.get(match[1], 0) + value
    return counters


def normalize_submission_marker(response, payload):
    """Opt-in mini-swe protocol adapter; never interpret arbitrary shell text."""
    choices = response.get("choices") or []
    if len(choices) != 1 or payload.get("tool_choice") not in (None, "auto"):
        return None
    bash_available = any(
        tool.get("type") == "function"
        and tool.get("function", {}).get("name") == "bash"
        and tool.get("function", {})
        .get("parameters", {})
        .get("properties", {})
        .get("command", {})
        .get("type")
        == "string"
        for tool in payload.get("tools") or []
    )
    choice = choices[0]
    message = choice.get("message") or {}
    content = message.get("content")
    if (
        not bash_available
        or choice.get("finish_reason") != "stop"
        or message.get("role") != "assistant"
        or message.get("refusal")
        or message.get("tool_calls")
        or message.get("function_call")
        or not isinstance(content, str)
    ):
        return None
    lines = content.rstrip().splitlines()
    if (
        not lines
        or lines[-1] != SUBMISSION_COMMAND
        or sum(line.lstrip().startswith("```") for line in lines) % 2
        or sum(line.lstrip().startswith("~~~") for line in lines) % 2
    ):
        return None
    normalized = copy.deepcopy(response)
    normalized["choices"][0]["finish_reason"] = "tool_calls"
    normalized["choices"][0]["message"]["tool_calls"] = [
        {
            "id": "call_submission_" + uuid.uuid4().hex,
            "type": "function",
            "function": {
                "name": "bash",
                "arguments": json.dumps({"command": SUBMISSION_COMMAND}),
            },
        }
    ]
    return normalized


def response_text_stats(message):
    """Measure repeated text without retaining any generated content."""
    pieces = [
        message.get(key) or "" for key in ("content", "reasoning", "reasoning_content")
    ]
    for call in message.get("tool_calls") or []:
        arguments = call.get("function", {}).get("arguments", "")
        try:
            parsed = json.loads(arguments)
        except (ValueError, TypeError):
            parsed = None
        pieces.append(
            parsed.get("command", "") if isinstance(parsed, dict) else arguments
        )
    text = "\n".join(piece for piece in pieces if isinstance(piece, str))
    counts = Counter(
        line.strip() for line in text.splitlines() if len(line.strip()) >= 20
    )
    repeated_chars = sum((count - 1) * len(line) for line, count in counts.items())
    return {
        "chars": len(text),
        "long_line_count": sum(counts.values()),
        "unique_long_lines": len(counts),
        "max_identical_line_count": max(counts.values(), default=0),
        "repeated_line_char_fraction": repeated_chars / len(text) if text else 0.0,
    }


def review_submission(response):
    """Replace one exact submission action with an audited read-only review action."""
    choices = response.get("choices") or []
    if len(choices) != 1 or choices[0].get("finish_reason") != "tool_calls":
        return None
    message = choices[0].get("message") or {}
    calls = message.get("tool_calls") or []
    if message.get("role") != "assistant" or message.get("refusal") or len(calls) != 1:
        return None
    function = calls[0].get("function") or {}
    try:
        args = json.loads(function.get("arguments", ""))
    except (TypeError, ValueError):
        return None
    if (
        function.get("name") != "bash"
        or not isinstance(args, dict)
        or not isinstance(args.get("command"), str)
        or args["command"].strip() != SUBMISSION_COMMAND
    ):
        return None
    reviewed = copy.deepcopy(response)
    reviewed["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = (
        json.dumps({**args, "command": SUBMISSION_REVIEW_COMMAND})
    )
    return reviewed


def recent_tool_summary(messages):
    """Describe the latest tool phase without logging command/output contents."""
    assistant = next(
        (
            i
            for i in range(len(messages) - 1, -1, -1)
            if messages[i].get("role") == "assistant"
        ),
        None,
    )
    if assistant is None:
        return []
    calls = {
        call.get("id"): call.get("function", {})
        for call in messages[assistant].get("tool_calls") or []
    }
    rows = []
    for message in messages[assistant + 1 :]:
        if message.get("role") != "tool":
            continue
        function = calls.get(message.get("tool_call_id"), {})
        try:
            arguments = json.loads(function.get("arguments", "{}"))
            result = json.loads(message.get("content", "{}"))
        except (ValueError, TypeError):
            continue
        if not isinstance(arguments, dict) or not isinstance(result, dict):
            continue
        command = arguments.get("command", "")
        if not isinstance(command, str):
            continue
        categories = [
            name
            for name, pattern in (
                ("test", r"\b(pytest|unittest|tox|runtests)\b"),
                ("install", r"\b(pip|conda|apt|apt-get|uv)\b.*\binstall\b"),
                ("build", r"\b(make|cmake|ninja|setup\.py)\b"),
                ("inspect", r"\b(grep|rg|head|tail|find|ls|sed)\b"),
                ("edit", r"\b(apply_patch|patch)\b|>\s*\S+"),
            )
            if re.search(pattern, command)
        ]
        rows.append(
            {
                "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
                "command_chars": len(command),
                "categories_heuristic": categories or ["other"],
                "returncode": result.get("returncode"),
                "output_chars": len(str(result.get("output", ""))),
            }
        )
    return rows


def repeated_failure_count(messages):
    """Count consecutive identical single-tool failures; never suppress a tool."""
    signatures = []
    pending = None
    for message in messages:
        if message.get("role") == "assistant":
            calls = message.get("tool_calls") or []
            pending = calls[0].get("function") if len(calls) == 1 else None
            if pending is None:
                signatures.append(None)
        elif message.get("role") == "tool" and pending:
            try:
                result = json.loads(message.get("content", ""))
            except (TypeError, ValueError):
                result = {}
            if isinstance(result, dict) and result.get("returncode", 0) != 0:
                signatures.append(json.dumps([pending, result], sort_keys=True))
            else:
                signatures.append(None)
            pending = None
        elif message.get("role") == "user":
            signatures.append(None)
    if not signatures or signatures[-1] is None:
        return 0
    count = 0
    for signature in reversed(signatures):
        if signature != signatures[-1]:
            break
        count += 1
    return count


def repeated_tool_count(messages, window=32):
    """Count exact repeats of the latest bash command/result; never skip execution."""
    signatures, pending = [], {}
    for message in messages:
        if message.get("role") == "assistant":
            pending = {
                call.get("id"): call.get("function", {})
                for call in message.get("tool_calls") or []
            }
        elif message.get("role") == "tool":
            function = pending.pop(message.get("tool_call_id"), {})
            signature = None
            if function.get("name") == "bash":
                try:
                    args = json.loads(function.get("arguments", ""))
                    result = json.loads(message.get("content", ""))
                    if (
                        isinstance(args, dict)
                        and isinstance(args.get("command"), str)
                        and isinstance(result, dict)
                    ):
                        signature = json.dumps(
                            [
                                args["command"],
                                result.get("returncode"),
                                result.get("output"),
                            ],
                            sort_keys=True,
                        )
                except (TypeError, ValueError):
                    pass
            signatures.append(signature)
    recent = signatures[-window:]
    return recent.count(recent[-1]) if recent and recent[-1] is not None else 0


def limit_reasoning_history(messages, keep):
    """Explicit context-policy control; preserve visible answers and tool evidence."""
    if type(keep) is not int or keep < 0:
        raise ValueError("Reasoning history limit must be a nonnegative integer")
    result = [copy.deepcopy(message) for message in messages]
    indices = [
        i for i, message in enumerate(result) if message.get("role") == "assistant"
    ]
    removed = []
    for index in indices[:-keep] if keep else indices:
        fields = {}
        for key in ("reasoning", "reasoning_content"):
            if key in result[index]:
                value = result[index].pop(key)
                fields[key] = len(value) if isinstance(value, str) else 0
        if fields:
            removed.append(
                {"message_index": index, "removed_chars": sum(fields.values())}
            )
    return result, removed


class RequestTelemetryProxy(AbstractContextManager):
    def __init__(
        self,
        upstream: str,
        output: Path,
        timeout: float,
        repetition_feedback_after: int = 0,
        normalize_submission: bool = False,
        collect_server_metrics: bool = False,
        repeated_tool_feedback: bool = False,
        reasoning_history_limit: int | None = None,
        submission_review_once: bool = False,
        abort_on_client_disconnect: bool = False,
    ):
        self.upstream = upstream.rstrip("/")
        self.output = output
        self.timeout = timeout
        self.repetition_feedback_after = repetition_feedback_after
        self.normalize_submission = normalize_submission
        self.collect_server_metrics = collect_server_metrics
        self.repeated_tool_feedback = repeated_tool_feedback
        if reasoning_history_limit is not None and (
            type(reasoning_history_limit) is not int or reasoning_history_limit < 0
        ):
            raise ValueError("Reasoning history limit must be a nonnegative integer")
        self.reasoning_history_limit = reasoning_history_limit
        self.submission_review_once = submission_review_once
        self.abort_on_client_disconnect = abort_on_client_disconnect
        self.reviewed_sessions = set()
        self.lock = threading.Lock()
        self.previous_response_times = {}
        self.server = None

    def record(self, event):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.lock, self.output.open("a") as stream:
            stream.write(json.dumps(event) + "\n")

    def snapshot_server_metrics(self, common, phase, headers):
        """C1-only diagnostic snapshots; counter lag is preserved, not hidden."""
        if not self.collect_server_metrics:
            return
        started = time.monotonic()
        event = {**common, "event": "server_metrics", "phase": phase}
        try:
            endpoint = self.upstream.removesuffix("/v1") + "/metrics"
            auth = {k: v for k, v in headers.items() if k == "Authorization"}
            with urlopen(Request(endpoint, headers=auth), timeout=0.5) as response:
                event["counters"] = parse_server_counters(
                    response.read(1024 * 1024).decode()
                )
        except (OSError, ValueError) as error:
            event["error_type"] = type(error).__name__
            # Missing metrics must not repeatedly delay agent work.
            self.collect_server_metrics = False
        event.update(unix_s=time.time(), collection_s=time.monotonic() - started)
        self.record(event)

    def __enter__(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                if self.path != "/v1/chat/completions":
                    self.send_error(404)
                    return
                body = self.rfile.read(int(self.headers["Content-Length"]))
                payload = json.loads(body)
                if payload.get("stream"):
                    self.send_error(
                        400, "Telemetry proxy requires non-streaming requests"
                    )
                    return
                request_id = uuid.uuid4().hex
                started = time.monotonic()
                common = {
                    "request_id": request_id,
                    "session": self.headers.get("X-Session-ID"),
                }
                with owner.lock:
                    previous = owner.previous_response_times.get(common["session"])
                owner.record(
                    {
                        **common,
                        "event": "request_start",
                        "unix_s": time.time(),
                        "body_bytes": len(body),
                        "message_count": len(payload.get("messages", [])),
                        "body_sha256": hashlib.sha256(body).hexdigest(),
                        "max_tokens": payload.get("max_tokens"),
                        "request_seed": (
                            payload.get("seed")
                            if isinstance(payload.get("seed"), int)
                            else None
                        ),
                        "tools_since_previous_assistant": recent_tool_summary(
                            payload.get("messages", [])
                        ),
                        # Includes tool execution plus agent/harness work; not
                        # an exact subprocess-only timer. Session boundaries
                        # must be considered when no X-Session-ID is supplied.
                        "since_previous_response_s": (
                            None if previous is None else started - previous
                        ),
                    }
                )
                if owner.reasoning_history_limit is not None:
                    payload["messages"], removed = limit_reasoning_history(
                        payload.get("messages", []), owner.reasoning_history_limit
                    )
                    if removed:
                        body = json.dumps(payload).encode()
                        owner.record(
                            {
                                **common,
                                "event": "reasoning_history_limited",
                                "unix_s": time.time(),
                                "keep_assistant_messages": owner.reasoning_history_limit,
                                "removed": removed,
                                "forwarded_body_sha256": hashlib.sha256(
                                    body
                                ).hexdigest(),
                            }
                        )
                repetitions = repeated_failure_count(payload.get("messages", []))
                if (
                    owner.repetition_feedback_after
                    and repetitions >= owner.repetition_feedback_after
                ):
                    payload["messages"].append(
                        {"role": "user", "content": REPETITION_FEEDBACK}
                    )
                    body = json.dumps(payload).encode()
                    owner.record(
                        {
                            **common,
                            "event": "repetition_feedback",
                            "unix_s": time.time(),
                            "consecutive_failures": repetitions,
                            "feedback": REPETITION_FEEDBACK,
                            "forwarded_body_sha256": hashlib.sha256(body).hexdigest(),
                        }
                    )
                exact_repeats = repeated_tool_count(payload.get("messages", []))
                if owner.repeated_tool_feedback and exact_repeats >= 2:
                    payload["messages"].append(
                        {"role": "user", "content": REPEATED_TOOL_FEEDBACK}
                    )
                    body = json.dumps(payload).encode()
                    owner.record(
                        {
                            **common,
                            "event": "repeated_tool_feedback",
                            "unix_s": time.time(),
                            "recent_exact_repeats": exact_repeats,
                            "feedback": REPEATED_TOOL_FEEDBACK,
                            "forwarded_body_sha256": hashlib.sha256(body).hexdigest(),
                        }
                    )
                headers = {"Content-Type": "application/json"}
                for key in ("Authorization", "X-Session-ID"):
                    if self.headers.get(key):
                        headers[key] = self.headers[key]
                owner.snapshot_server_metrics(common, "before_request", headers)
                try:
                    if owner.abort_on_client_disconnect:
                        result, status = post_until_disconnect(
                            owner.upstream + "/chat/completions",
                            body,
                            headers,
                            owner.timeout,
                            self.connection,
                        )
                    else:
                        try:
                            response = urlopen(
                                Request(
                                    owner.upstream + "/chat/completions", body, headers
                                ),
                                timeout=owner.timeout,
                            )
                        except HTTPError as error:
                            response = error
                        with response:
                            result = response.read()
                            status = response.status
                    try:
                        parsed = json.loads(result)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        parsed = {}
                    choices = parsed.get("choices", [])
                    with owner.lock:
                        owner.previous_response_times[common["session"]] = (
                            time.monotonic()
                        )
                    owner.record(
                        {
                            **common,
                            "event": "response",
                            "unix_s": time.time(),
                            "elapsed_s": time.monotonic() - started,
                            "status": status,
                            "usage": parsed.get("usage"),
                            "finish_reasons": [c.get("finish_reason") for c in choices],
                            "tool_counts": [
                                len(c.get("message", {}).get("tool_calls") or [])
                                for c in choices
                            ],
                            "text_stats": [
                                response_text_stats(c.get("message", {}))
                                for c in choices
                            ],
                            "response_sha256": hashlib.sha256(result).hexdigest(),
                        }
                    )
                    owner.snapshot_server_metrics(common, "after_response", headers)
                    normalized = (
                        normalize_submission_marker(parsed, payload)
                        if owner.normalize_submission and status == 200
                        else None
                    )
                    if normalized is not None:
                        result = json.dumps(normalized).encode()
                        owner.record(
                            {
                                **common,
                                "event": "submission_marker_normalized",
                                "unix_s": time.time(),
                                "policy": "explicit_final_plaintext_sentinel_to_fixed_bash_call",
                                "forwarded_response_sha256": hashlib.sha256(
                                    result
                                ).hexdigest(),
                            }
                        )
                    with owner.lock:
                        reviewed = (
                            review_submission(normalized or parsed)
                            if owner.submission_review_once
                            and status == 200
                            and common["session"]
                            and common["session"] not in owner.reviewed_sessions
                            else None
                        )
                        if reviewed is not None:
                            owner.reviewed_sessions.add(common["session"])
                    if reviewed is not None:
                        result = json.dumps(reviewed).encode()
                        owner.record(
                            {
                                **common,
                                "event": "submission_review_requested",
                                "unix_s": time.time(),
                                "policy": "one_exact_submission_to_read_only_diff_review_per_session",
                                "command": SUBMISSION_REVIEW_COMMAND,
                                "forwarded_response_sha256": hashlib.sha256(
                                    result
                                ).hexdigest(),
                            }
                        )
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(result)))
                    self.end_headers()
                    self.wfile.write(result)
                except ClientDisconnected:
                    owner.record(
                        {
                            **common,
                            "event": "upstream_abort_requested",
                            "unix_s": time.time(),
                            "elapsed_s": time.monotonic() - started,
                        }
                    )
                    owner.snapshot_server_metrics(
                        common, "after_client_disconnect", headers
                    )
                except (BrokenPipeError, ConnectionResetError):
                    owner.record(
                        {
                            **common,
                            "event": "client_disconnected",
                            "unix_s": time.time(),
                            "elapsed_s": time.monotonic() - started,
                        }
                    )
                except Exception as error:
                    owner.record(
                        {
                            **common,
                            "event": "upstream_error",
                            "unix_s": time.time(),
                            "elapsed_s": time.monotonic() - started,
                            "error_type": type(error).__name__,
                        }
                    )
                    self.send_error(502, "Upstream request failed")

        self.server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
