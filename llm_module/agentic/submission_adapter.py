# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Opt-in adapter for mini-swe's explicit submission sentinel."""

from __future__ import annotations

import copy
import json
import threading
import uuid
from contextlib import AbstractContextManager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

SUBMISSION_COMMAND = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def normalize_submission_marker(response, payload):
    """Turn only mini-swe's final plaintext sentinel into its bash tool call."""
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


class SubmissionMarkerProxy(AbstractContextManager):
    """Forward non-streaming chat requests and apply the sentinel adapter."""

    def __init__(self, upstream: str, timeout: float):
        self.upstream = upstream.rstrip("/")
        self.timeout = timeout
        self.server = None
        self.thread = None

    def __enter__(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                if self.path != "/v1/chat/completions":
                    self.send_error(404)
                    return
                try:
                    body = self.rfile.read(int(self.headers["Content-Length"]))
                    payload = json.loads(body)
                    if payload.get("stream"):
                        self.send_error(
                            400, "Submission adapter requires non-streaming requests"
                        )
                        return
                    headers = {"Content-Type": "application/json"}
                    for key in ("Authorization", "X-Session-ID"):
                        if self.headers.get(key):
                            headers[key] = self.headers[key]
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
                    if status == 200:
                        parsed = json.loads(result)
                        normalized = normalize_submission_marker(parsed, payload)
                        if normalized is not None:
                            result = json.dumps(normalized).encode()
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(result)))
                    self.end_headers()
                    self.wfile.write(result)
                except (BrokenPipeError, ConnectionResetError):
                    return
                except Exception:
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
