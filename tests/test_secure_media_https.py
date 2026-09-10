# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Exercise the pinned downloader over a real local TLS connection."""

import shutil
import socket
import ssl
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from utils import secure_media


@pytest.mark.skipif(
    shutil.which("openssl") is None,
    reason="openssl is needed for an ephemeral test certificate",
)
def test_https_connection_close_and_tls_hostname(tmp_path, monkeypatch):
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "1",
            "-subj",
            "/CN=images.example.com",
            "-addext",
            "subjectAltName=DNS:images.example.com",
            "-keyout",
            str(key),
            "-out",
            str(cert),
        ],
        check=True,
        capture_output=True,
    )
    observed = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            observed.append((self.path, self.headers["Host"]))
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", "5")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(b"image")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        trusted = ssl.create_default_context(cafile=str(cert))
        monkeypatch.setattr(secure_media.ssl, "create_default_context", lambda: trusted)
        monkeypatch.setattr(
            secure_media.socket,
            "getaddrinfo",
            lambda *args, **kwargs: [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", server.server_address)
            ],
        )
        # Only this TLS integration fixture allows localhost. Policy tests
        # separately prove that production rejects this DNS answer before dialing.
        monkeypatch.setattr(secure_media, "_public_ip", lambda _: True)
        client = secure_media.SafeMediaHTTPConnection(["images.example.com"])
        try:
            assert client.get_bytes("https://images.example.com/a.png") == b"image"
        except ValueError as exc:
            raise AssertionError(f"HTTPS fixture failure: {exc.__context__!r}") from exc
        assert observed == [("/a.png", "images.example.com")]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
