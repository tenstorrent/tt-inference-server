# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Shared media policy for TT vLLM, independent of the requesting API route."""

import asyncio
import http.client
import ipaddress
import json
import os
import re
import socket
import ssl
import time
import warnings
from urllib.parse import urlsplit

CONNECTOR_NAME = "tt_secure"
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
FETCH_TIMEOUT_SECONDS = 5
MAX_IMAGE_PIXELS = 16 * 1024 * 1024
_DENIED = "Media URL is not permitted by the workload's media policy"
_FETCH_FAILED = "Unable to fetch media from the approved host"
_registered = False


def allowed_domains(values):
    if not isinstance(values, list):
        raise ValueError(
            "TT_ALLOWED_MEDIA_DOMAINS must be a JSON array of exact hostnames"
        )
    result = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError("Media domains must be exact hostnames")
        host = value.lower()
        labels = host.split(".")
        if (
            len(host) > 253
            or len(labels) < 2
            or any(
                not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
                for label in labels
            )
        ):
            raise ValueError(
                "Media domains must be exact DNS hostnames without ports or wildcards"
            )
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            raise ValueError("Media domains must not be IP addresses")
        if host not in result:
            result.append(host)
    return result


def configure_media_policy(argv, cli_domains=None):
    configured = os.environ.get("TT_ALLOWED_MEDIA_DOMAINS")
    domains = allowed_domains(
        json.loads(configured) if configured is not None else (cli_domains or [])
    )
    if (
        configured is not None
        and cli_domains is not None
        and allowed_domains(cli_domains) != domains
    ):
        raise ValueError(
            "--allowed-media-domains conflicts with the operator's TT_ALLOWED_MEDIA_DOMAINS policy"
        )
    os.environ["TT_ALLOWED_MEDIA_DOMAINS"] = json.dumps(domains)
    if cli_domains is None and domains:
        argv.extend(["--allowed-media-domains", *domains])
    os.environ["VLLM_MEDIA_URL_ALLOW_REDIRECTS"] = "0"
    os.environ["VLLM_MEDIA_CONNECTOR"] = CONNECTOR_NAME
    if "VLLM_PLUGINS" in os.environ:
        plugins = [name for name in os.environ["VLLM_PLUGINS"].split(",") if name]
        if "tt_media_security" not in plugins:
            plugins.append("tt_media_security")
        os.environ["VLLM_PLUGINS"] = ",".join(plugins)
    # An empty native allowlist means unrestricted. The mandatory connector
    # below instead treats an empty policy as deny-all for remote media.
    register_media_connector()


def _public_ip(address):
    ip = ipaddress.ip_address(address)
    if not ip.is_global or ip.is_multicast:
        return False
    if isinstance(ip, ipaddress.IPv6Address):
        # Do not allow alternate encodings/tunnels to reach IPv4 destinations.
        if ip.ipv4_mapped or ip.sixtofour or ip.teredo:
            return False
        if ip in ipaddress.ip_network("64:ff9b::/96") or ip in ipaddress.ip_network(
            "64:ff9b:1::/48"
        ):
            return False
    return True


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Media fetch deadline exceeded")
    return remaining


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Connect only to previously validated addresses, retaining TLS SNI/verification."""

    def __init__(self, host, addresses, deadline):
        super().__init__(
            host,
            port=443,
            timeout=_remaining(deadline),
            context=ssl.create_default_context(),
        )
        self.addresses = addresses
        self.deadline = deadline
        self.tls_socket = None

    def connect(self):
        # socket.connect receives a numeric sockaddr from getaddrinfo, never a
        # hostname. The HTTP Host and TLS certificate name remain the allowlisted
        # hostname. There is no second DNS lookup between validation and dialing.
        for family, socktype, proto, _, sockaddr in self.addresses:
            sock = socket.socket(family, socktype, proto)
            try:
                sock.settimeout(_remaining(self.deadline))
                sock.connect(sockaddr)
                self.sock = self._context.wrap_socket(sock, server_hostname=self.host)
                self.tls_socket = self.sock
                return
            except OSError:
                sock.close()
        raise OSError(_FETCH_FAILED)


class SafeMediaHTTPConnection:
    """The narrow get_bytes/async_get_bytes interface consumed by MediaConnector."""

    def __init__(self, domains):
        self.domains = frozenset(allowed_domains(domains))

    def validate_url(self, url):
        try:
            if (
                not isinstance(url, str)
                or any(ord(c) <= 32 or ord(c) == 127 for c in url)
                or "\\" in url
            ):
                raise ValueError(_DENIED)
            parsed = urlsplit(url)
            if (
                parsed.scheme != "https"
                or parsed.hostname not in self.domains
                or parsed.username is not None
                or parsed.password is not None
                or parsed.port not in (None, 443)
                or parsed.fragment
            ):
                raise ValueError(_DENIED)
            return parsed
        except (ValueError, UnicodeError):
            raise ValueError(_DENIED) from None

    def get_bytes(self, url, *, timeout=None, allow_redirects=False):
        parsed = self.validate_url(url)
        duration = min(
            timeout if timeout is not None else FETCH_TIMEOUT_SECONDS,
            FETCH_TIMEOUT_SECONDS,
        )
        deadline = time.monotonic() + duration
        connection = None
        response = None
        try:
            addresses = socket.getaddrinfo(
                parsed.hostname, 443, type=socket.SOCK_STREAM
            )
            if not addresses or any(not _public_ip(item[4][0]) for item in addresses):
                raise ValueError(_DENIED)
            connection = _PinnedHTTPSConnection(parsed.hostname, addresses, deadline)
            target = parsed.path or "/"
            if parsed.query:
                target += "?" + parsed.query
            connection.request(
                "GET",
                target,
                headers={
                    "Accept": "image/*, audio/*, video/*",
                    "Accept-Encoding": "identity",
                    "User-Agent": "tt-inference-server-media",
                },
            )
            response = connection.getresponse()
            # Never follow redirects, even if a caller passes allow_redirects=True.
            if response.status != 200:
                raise ValueError(_FETCH_FAILED)
            content_type = (
                response.getheader("Content-Type", "").split(";", 1)[0].lower()
            )
            if not content_type.startswith(("image/", "audio/", "video/")):
                raise ValueError(
                    "Remote media must have an image, audio, or video content type"
                )
            if response.getheader("Content-Encoding", "identity").lower() != "identity":
                raise ValueError("Encoded media HTTP responses are not supported")
            length = response.getheader("Content-Length")
            if length is not None and not 0 <= int(length) <= MAX_DOWNLOAD_BYTES:
                raise ValueError("Media download exceeds the size limit")
            data = bytearray()
            while True:
                remaining = _remaining(deadline)
                if (
                    connection.tls_socket is not None
                    and connection.tls_socket.fileno() >= 0
                ):
                    connection.tls_socket.settimeout(remaining)
                chunk = response.read1(min(65536, MAX_DOWNLOAD_BYTES + 1 - len(data)))
                if not chunk:
                    return bytes(data)
                data.extend(chunk)
                if len(data) > MAX_DOWNLOAD_BYTES:
                    raise ValueError("Media download exceeds the size limit")
        except (OSError, http.client.HTTPException, ValueError):
            # Do not reflect internal DNS/connection/status details or signed URLs.
            raise ValueError(_FETCH_FAILED) from None
        finally:
            if response is not None:
                response.close()
            if connection is not None:
                connection.close()

    async def async_get_bytes(self, url, *, timeout=None, allow_redirects=False):
        return await asyncio.to_thread(
            self.get_bytes, url, timeout=timeout, allow_redirects=allow_redirects
        )


def register_media_connector():
    global _registered
    if _registered:
        return
    try:
        from PIL import Image
        from vllm.multimodal.media import MEDIA_CONNECTOR_REGISTRY, MediaConnector
    except ImportError as exc:
        raise RuntimeError(
            "This workload requires a TT vLLM build with the shared media-connector registry"
        ) from exc

    @MEDIA_CONNECTOR_REGISTRY.register(CONNECTOR_NAME)
    class SecureMediaConnector(MediaConnector):
        def __init__(
            self,
            media_io_kwargs=None,
            connection=None,
            *,
            allowed_local_media_path="",
            allowed_media_domains=None,
        ):
            domains = allowed_domains(
                json.loads(os.environ.get("TT_ALLOWED_MEDIA_DOMAINS", "[]"))
            )
            if allowed_local_media_path:
                raise ValueError("Local media paths are disabled for this workload")
            super().__init__(
                media_io_kwargs,
                SafeMediaHTTPConnection(domains),
                allowed_local_media_path="",
                allowed_media_domains=domains,
            )

        def _assert_url_in_allowed_media_domains(self, url_spec):
            super()._assert_url_in_allowed_media_domains(url_spec)
            # Also run for an empty allowlist and before any native disk-cache hit.
            self.connection.validate_url(url_spec.url)

    Image.MAX_IMAGE_PIXELS = min(
        Image.MAX_IMAGE_PIXELS or MAX_IMAGE_PIXELS, MAX_IMAGE_PIXELS
    )
    warnings.filterwarnings("error", category=Image.DecompressionBombWarning)
    _registered = True
