# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import asyncio
import io
import json
import os
import socket
from unittest.mock import MagicMock

import pytest

from utils import secure_media as media

HOST = "images.example.com"
URL = "https://images.example.com/a.png?signature=test"
PUBLIC_ADDRESS = (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    monkeypatch.setattr(os, "environ", os.environ.copy())


@pytest.fixture
def transport(monkeypatch):
    response = MagicMock()
    response.status = 200
    headers = {"Content-Type": "image/png"}
    response.getheader.side_effect = lambda key, default=None: headers.get(key, default)
    response.read1.side_effect = io.BytesIO(b"image bytes").read1
    connection = MagicMock()
    connection.tls_socket = None
    connection.getresponse.return_value = response
    factory = MagicMock(return_value=connection)
    lookup = MagicMock(return_value=[PUBLIC_ADDRESS])
    monkeypatch.setattr(media, "_PinnedHTTPSConnection", factory)
    monkeypatch.setattr(media.socket, "getaddrinfo", lookup)
    return (
        media.SafeMediaHTTPConnection([HOST]),
        lookup,
        factory,
        connection,
        response,
        headers,
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8000/health",
        "http://127.0.0.1:8000/v1/models",
        "http://127.0.0.1:8080/",
        "http://169.254.169.254/latest/meta-data/",
        "http://kubernetes.default.svc/",
        "https://10.233.0.1/",
        "http://240.0.0.1/",
        "https://[::1]/",
        "https://[::ffff:127.0.0.1]/",
        "https://2130706433/",
        "https://0177.0.0.1/",
        "file:///etc/passwd",
        "//images.example.com/a",
        "http://images.example.com/a",
        "https://images.example.com:8000/a",
        "https://images.example.com.evil.test/a",
        "https://evil.test@images.example.com/a",
        "https://127.0.0.1\\@images.example.com/a",
        "https://images.example.com\\@127.0.0.1/a",
        "https://images.example.com./a",
        "https://images%2eexample.com/a",
        " https://images.example.com/a",
        "https://images.example.com/a\r\nHost: 127.0.0.1",
        "https://images.example.com/a#fragment",
    ],
)
def test_unapproved_urls_never_resolve_or_connect(transport, url):
    client, lookup, factory, *_ = transport
    with pytest.raises(ValueError, match="not permitted"):
        client.get_bytes(url)
    lookup.assert_not_called()
    factory.assert_not_called()


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",
        "10.0.0.1",
        "172.16.0.1",
        "192.168.0.1",
        "169.254.169.254",
        "100.64.0.1",
        "0.0.0.0",
        "240.0.0.1",
        "224.0.0.1",
        "::1",
        "fe80::1",
        "fd00::1",
        "::ffff:127.0.0.1",
        "64:ff9b::7f00:1",
        "2002:7f00:1::1",
    ],
)
def test_private_dns_answers_never_connect(transport, ip):
    client, lookup, factory, *_ = transport
    private = (
        socket.AF_INET6 if ":" in ip else socket.AF_INET,
        socket.SOCK_STREAM,
        6,
        "",
        (ip, 443),
    )
    lookup.return_value = [PUBLIC_ADDRESS, private]
    with pytest.raises(ValueError, match="Unable to fetch"):
        client.get_bytes(URL)
    factory.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_approved_https_fetch_pins_validated_resolution(transport, asynchronous):
    client, lookup, factory, connection, *_ = transport
    # A second resolution would rebind to loopback. It must never occur.
    lookup.side_effect = [
        [PUBLIC_ADDRESS],
        [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))],
    ]
    result = (
        asyncio.run(client.async_get_bytes(URL))
        if asynchronous
        else client.get_bytes(URL)
    )
    assert result == b"image bytes"
    lookup.assert_called_once_with(HOST, 443, type=socket.SOCK_STREAM)
    assert factory.call_args.args[:2] == (HOST, [PUBLIC_ADDRESS])
    assert connection.request.call_args.args == ("GET", "/a.png?signature=test")
    connection.close.assert_called_once()


def test_dial_uses_numeric_ip_and_keeps_certificate_hostname(monkeypatch):
    sock, tls, context = MagicMock(), MagicMock(), MagicMock()
    context.wrap_socket.return_value = tls
    factory = MagicMock(return_value=sock)
    monkeypatch.setattr(media.socket, "socket", factory)
    monkeypatch.setattr(media.ssl, "create_default_context", lambda: context)
    connection = media._PinnedHTTPSConnection(
        HOST, [PUBLIC_ADDRESS], media.time.monotonic() + 5
    )
    connection.connect()
    sock.connect.assert_called_once_with(("93.184.216.34", 443))
    context.wrap_socket.assert_called_once_with(sock, server_hostname=HOST)
    assert connection.sock is tls
    connection.close()


@pytest.mark.parametrize("status", [301, 302, 307, 308, 401, 404, 500])
def test_redirects_and_upstream_details_are_not_exposed(transport, status):
    client, lookup, _, _, response, headers = transport
    response.status = status
    headers["Location"] = "http://127.0.0.1:8000/v1/models"
    with pytest.raises(ValueError) as exc:
        client.get_bytes(URL, allow_redirects=True)
    assert str(exc.value) == media._FETCH_FAILED
    assert lookup.call_count == 1
    response.read1.assert_not_called()


def test_unannounced_oversize_body_is_bounded(transport, monkeypatch):
    client, _, _, connection, response, _ = transport
    monkeypatch.setattr(media, "MAX_DOWNLOAD_BYTES", 8)
    response.read1.side_effect = io.BytesIO(b"x" * 100).read1
    with pytest.raises(ValueError, match="Unable to fetch"):
        client.get_bytes(URL)
    assert response.read1.call_args.args == (9,)
    connection.close.assert_called_once()


@pytest.mark.parametrize(
    "headers",
    [
        {"Content-Length": "99999999999"},
        {"Content-Type": "text/html"},
        {"Content-Encoding": "gzip"},
    ],
)
def test_response_headers_checked_before_read(transport, headers):
    client, _, _, _, response, actual_headers = transport
    actual_headers.update(headers)
    with pytest.raises(ValueError, match="Unable to fetch"):
        client.get_bytes(URL)
    response.read1.assert_not_called()


def test_expired_deadline_cannot_connect(transport):
    client, _, _, connection, *_ = transport
    # Factory is mocked, so assert the timeout while reading as well.
    with pytest.raises(ValueError, match="Unable to fetch"):
        client.get_bytes(URL, timeout=0)
    connection.close.assert_called_once()


def test_empty_policy_denies_every_remote_url(transport):
    _, lookup, factory, *_ = transport
    with pytest.raises(ValueError, match="not permitted"):
        media.SafeMediaHTTPConnection([]).get_bytes(URL)
    lookup.assert_not_called()
    factory.assert_not_called()


@pytest.mark.parametrize(
    "domains",
    [
        ["*.example.com"],
        ["https://example.com"],
        ["example.com:443"],
        ["127.0.0.1"],
        ["localhost"],
        ["example.com/"],
        ["example.com."],
        [None],
        "example.com",
    ],
)
def test_allowlist_rejects_ambiguous_configuration(domains):
    with pytest.raises(ValueError):
        media.allowed_domains(domains)


def test_native_flags_and_worker_plugin_configuration(monkeypatch):
    monkeypatch.setenv("TT_ALLOWED_MEDIA_DOMAINS", json.dumps([HOST]))
    monkeypatch.setenv("VLLM_PLUGINS", "tt_model_registry")
    register = MagicMock()
    monkeypatch.setattr(media, "register_media_connector", register)
    argv = ["server"]
    media.configure_media_policy(argv)
    assert argv == ["server", "--allowed-media-domains", HOST]
    assert os.environ["VLLM_MEDIA_URL_ALLOW_REDIRECTS"] == "0"
    assert os.environ["VLLM_MEDIA_CONNECTOR"] == "tt_secure"
    assert os.environ["VLLM_PLUGINS"] == "tt_model_registry,tt_media_security"
    register.assert_called_once()


def test_cli_cannot_widen_an_operator_allowlist(monkeypatch):
    monkeypatch.setenv("TT_ALLOWED_MEDIA_DOMAINS", json.dumps([HOST]))
    with pytest.raises(ValueError, match="conflicts"):
        media.configure_media_policy(["server"], ["unapproved.example"])


def test_missing_shared_connector_support_fails_closed(monkeypatch):
    import sys
    import types

    monkeypatch.setattr(media, "_registered", False)
    monkeypatch.setitem(
        sys.modules, "vllm.multimodal.media", types.ModuleType("vllm.multimodal.media")
    )
    with pytest.raises(RuntimeError, match="shared media-connector registry"):
        media.register_media_connector()
