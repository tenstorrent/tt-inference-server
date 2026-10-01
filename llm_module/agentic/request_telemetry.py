# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Transparent, payload-free timing for local non-streaming agent requests."""

import hashlib
import json
import threading
import time
import uuid
from contextlib import AbstractContextManager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


class RequestTelemetryProxy(AbstractContextManager):
    def __init__(self, upstream: str, output: Path, timeout: float):
        self.upstream = upstream.rstrip("/")
        self.output = output
        self.timeout = timeout
        self.lock = threading.Lock()
        self.server = None

    def record(self, event):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.lock, self.output.open("a") as stream:
            stream.write(json.dumps(event) + "\n")

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
                owner.record(
                    {
                        **common,
                        "event": "request_start",
                        "unix_s": time.time(),
                        "body_bytes": len(body),
                        "message_count": len(payload.get("messages", [])),
                        "body_sha256": hashlib.sha256(body).hexdigest(),
                        "max_tokens": payload.get("max_tokens"),
                    }
                )
                headers = {"Content-Type": "application/json"}
                for key in ("Authorization", "X-Session-ID"):
                    if self.headers.get(key):
                        headers[key] = self.headers[key]
                try:
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
                            "response_sha256": hashlib.sha256(result).hexdigest(),
                        }
                    )
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(result)))
                    self.end_headers()
                    self.wfile.write(result)
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
