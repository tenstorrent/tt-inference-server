# SPDX-License-Identifier: Apache-2.0
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import pytest

from llm_module.agentic.request_telemetry import RequestTelemetryProxy


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("status", [200, 503])
def test_live_caller_keeps_exact_response_and_status(tmp_path, enabled, status):
    observed = []
    response = b'{"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 2}}'

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            observed.append(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(status)
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    body = b'{"messages": [], "temperature": 1, "max_tokens": 32768}'
    try:
        with RequestTelemetryProxy(
            "http://127.0.0.1:{}/v1".format(server.server_port),
            tmp_path / "requests.jsonl",
            3,
            abort_on_client_disconnect=enabled,
        ) as endpoint:
            try:
                result = urlopen(
                    Request(
                        endpoint + "/chat/completions",
                        body,
                        {"Content-Type": "application/json"},
                    )
                )
            except HTTPError as error:
                result = error
            with result:
                assert result.status == status
                assert result.read() == response
        assert observed == [body]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_disconnected_caller_closes_its_upstream_before_response(tmp_path):
    started, closed = threading.Event(), threading.Event()

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            started.set()
            self.connection.settimeout(3)
            if self.connection.recv(1) == b"":
                closed.set()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    evidence = tmp_path / "requests.jsonl"
    try:
        with RequestTelemetryProxy(
            "http://127.0.0.1:{}/v1".format(server.server_port),
            evidence,
            3,
            abort_on_client_disconnect=True,
        ) as endpoint:
            parsed = urlsplit(endpoint)
            client = socket.create_connection((parsed.hostname, parsed.port), timeout=3)
            body = b'{"messages": []}'
            client.sendall(
                b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
                + "Content-Length: {}\r\n\r\n".format(len(body)).encode()
                + body
            )
            assert started.wait(2)
            start = time.monotonic()
            client.close()
            assert closed.wait(1)
            assert time.monotonic() - start < 1
        events = [json.loads(line) for line in evidence.read_text().splitlines()]
        assert any(event["event"] == "upstream_abort_requested" for event in events)
        assert not any(event["event"] == "response" for event in events)
        assert not any(event["event"] == "upstream_error" for event in events)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
