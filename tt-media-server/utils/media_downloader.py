# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Hardened downloader for client-supplied media URLs (#4974).

Lets request media fields carry a presigned URL (e.g. S3) instead of inline
base64. Every remote fetch in the request path must go through
``download_media_url`` — side-channel fetches are how reference servers
(vLLM, SGLang) reintroduced SSRF after hardening their main path.

Policy, following SGLang's ``download_remote_media`` with vLLM's error
taxonomy on top:

* http(s) only; the optional exact-hostname allowlist is normalized (IDNA,
  case, IP literals) and checked on the initial URL and on every redirect hop.
* URLs are validated as ``httpx.URL`` — the same parser the client uses — so
  a validator/client parser mismatch cannot bypass the allowlist.
* Redirects are followed manually (301/302/303/307/308, capped) so each
  destination is validated before a connection is made.
* One total deadline covers all hops and the body read; the body is streamed
  against the byte cap instead of buffered blindly. Callers can pass a shared
  ``deadline`` so one request's assets share a single budget. Known
  limitation: a origin that trickles bytes just inside the per-read timeout
  can stretch a hop somewhat past the deadline (checked between chunks;
  ``asyncio.timeout`` would close this but needs Python 3.11+).

Error taxonomy (mapped to HTTP statuses at the endpoint):

* ``MediaDownloadPolicyError`` — the URL violates server policy (400).
* ``MediaDownloadTooLargeError`` — the body exceeds the byte cap (413).
* ``MediaDownloadFetchError`` — the origin failed us: HTTP error such as an
  expired presigned URL, network error, or deadline exceeded (422).
* ``MediaDownloadError`` (base, raised directly) — server-side
  misconfiguration; deliberately NOT caught at the endpoint so it surfaces
  as a 500, not a client-fault 4xx.
"""

import asyncio
import ipaddress
import time
from typing import Optional

import httpx
from config.settings import settings
from utils.logger import TTLogger

logger = TTLogger()

_REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307, 308})
_READ_CHUNK_BYTES = 64 * 1024

_shared_client: Optional[httpx.AsyncClient] = None
_shared_client_loop: Optional[asyncio.AbstractEventLoop] = None
_warned_open_allowlist = False


class MediaDownloadError(Exception):
    """Base class; raised directly only for server-side misconfiguration."""


class MediaDownloadPolicyError(MediaDownloadError):
    """URL violates server download policy (scheme, domain, redirects)."""


class MediaDownloadTooLargeError(MediaDownloadError):
    """Remote body exceeds the configured byte cap."""


class MediaDownloadFetchError(MediaDownloadError):
    """Origin returned an error, the network failed, or the deadline passed."""


def is_media_url(value: str) -> bool:
    """True when a media field value is a remote URL rather than base64.

    Case-insensitive: ``HTTPS://`` must route to the URL path, not the
    base64 decoder. No base64 payload can collide — ``:`` and ``/`` in the
    prefix are outside the base64 alphabet.
    """
    return value[:8].lower().startswith(("http://", "https://"))


def _normalize_hostname(hostname: str) -> str:
    """Canonicalize a hostname for allowlist comparison.

    IP literals collapse to their canonical text form; names are IDNA-encoded
    and lowercased so unicode/case variants of an allowlisted domain match.
    """
    hostname = hostname.strip().rstrip(".")
    if hostname.startswith("[") and hostname.endswith("]"):
        hostname = hostname[1:-1]
    try:
        return str(ipaddress.ip_address(hostname))
    except ValueError:
        pass
    try:
        return hostname.encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise MediaDownloadPolicyError(f"Invalid hostname {hostname!r}") from exc


def _allowed_domains() -> frozenset:
    raw = settings.media_url_allowed_domains or ""
    try:
        return frozenset(
            _normalize_hostname(domain) for domain in raw.split(",") if domain.strip()
        )
    except MediaDownloadPolicyError as exc:
        # A bad allowlist entry is operator error, not client error: raise the
        # base class so the endpoint surfaces 500 instead of blaming the caller.
        raise MediaDownloadError(
            f"Misconfigured media_url_allowed_domains: {exc}"
        ) from exc


def check_media_url_policy(url: str) -> httpx.URL:
    """Validate one URL against the download policy; return it parsed.

    Parses with ``httpx.URL`` — the exact representation the HTTP client
    connects with — and is re-run on every redirect destination.
    """
    if not settings.media_url_download_enabled:
        raise MediaDownloadPolicyError(
            "Media URL download is disabled on this server "
            "(media_url_download_enabled=false); send the asset inline."
        )

    try:
        parsed = httpx.URL(url)
    except httpx.InvalidURL as exc:
        raise MediaDownloadPolicyError(f"Invalid media URL: {url!r}") from exc

    if parsed.scheme not in ("http", "https") or not parsed.host:
        raise MediaDownloadPolicyError(
            f"Media URL must be http(s) with a hostname, got: {url!r}"
        )

    allowed = _allowed_domains()
    if allowed:
        hostname = _normalize_hostname(parsed.host)
        if hostname not in allowed:
            # Do not echo the allowlist: it is deployment configuration.
            raise MediaDownloadPolicyError(
                f"Media URL domain {hostname!r} is not in the allowed domains list"
            )
    return parsed


def _get_shared_client() -> httpx.AsyncClient:
    """Pooled client bound to the running loop.

    Rebuilt if the loop changed (repeated TestClient runs) or the client was
    closed; per-request deadlines are passed per call. Cross-request cookie
    persistence is a state channel between unrelated clients' downloads, so
    the jar is cleared on every acquisition.
    """
    global _shared_client, _shared_client_loop
    loop = asyncio.get_running_loop()
    if (
        _shared_client is None
        or _shared_client.is_closed
        or _shared_client_loop is not loop
    ):
        _shared_client = httpx.AsyncClient()
        _shared_client_loop = loop
    _shared_client.cookies.clear()
    return _shared_client


def _warn_if_open_allowlist() -> None:
    global _warned_open_allowlist
    if _warned_open_allowlist:
        return
    if not _allowed_domains():
        logger.warning(
            "Media URL download is enabled with no media_url_allowed_domains; "
            "any authenticated client can make this server fetch arbitrary "
            "http(s) URLs (SSRF surface). Set an allowlist on deployments "
            "reachable by untrusted clients."
        )
    _warned_open_allowlist = True


async def download_media_url(
    url: str,
    *,
    client: Optional[httpx.AsyncClient] = None,
    deadline: Optional[float] = None,
) -> bytes:
    """Download one media asset under the configured URL policy.

    Redirects are followed manually so every destination is validated before
    a connection is made. The body is streamed to enforce both the total
    deadline and the byte cap without first buffering an attacker-controlled
    payload in memory.

    Args:
        url: The http(s) URL to fetch.
        client: Override client (tests); defaults to the shared pooled client.
        deadline: Optional ``time.monotonic()`` deadline shared across several
            downloads (one budget per request). Defaults to now +
            ``media_url_timeout_seconds``.
    """
    max_bytes = settings.media_url_max_bytes
    max_redirects = max(0, settings.media_url_max_redirects)
    timeout_seconds = settings.media_url_timeout_seconds
    if timeout_seconds <= 0:
        # Operator error → base class → 500 at the endpoint.
        raise MediaDownloadError("media_url_timeout_seconds must be positive")

    current = check_media_url_policy(url)
    _warn_if_open_allowlist()
    http_client = client if client is not None else _get_shared_client()
    if deadline is None:
        deadline = time.monotonic() + timeout_seconds

    try:
        for redirect_count in range(max_redirects + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MediaDownloadFetchError(f"Timed out downloading media URL: {url}")

            async with http_client.stream(
                "GET",
                current,
                timeout=remaining,
                follow_redirects=False,
            ) as response:
                if response.status_code in _REDIRECT_STATUS_CODES:
                    location = response.headers.get("location")
                    if not location:
                        raise MediaDownloadFetchError(
                            f"Media URL returned redirect HTTP "
                            f"{response.status_code} without a Location "
                            f"header: {url}"
                        )
                    if redirect_count == max_redirects:
                        raise MediaDownloadPolicyError(
                            f"Media URL exceeded {max_redirects} redirects: {url}"
                        )
                    try:
                        target = str(current.join(location))
                    except httpx.InvalidURL as exc:
                        raise MediaDownloadPolicyError(
                            f"Invalid redirect location {location!r}"
                        ) from exc
                    current = check_media_url_policy(target)
                    continue

                if response.status_code >= 400:
                    raise MediaDownloadFetchError(
                        f"Media URL returned HTTP {response.status_code} "
                        f"(an expired presigned URL typically returns 403): {url}"
                    )

                content_length = response.headers.get("content-length")
                if content_length is not None:
                    try:
                        declared = int(content_length)
                    except ValueError:
                        declared = None
                    if declared is not None and declared > max_bytes:
                        raise MediaDownloadTooLargeError(
                            f"Remote media declares {declared} bytes, over the "
                            f"{max_bytes}-byte download cap"
                        )

                body = bytearray()
                async for chunk in response.aiter_bytes(_READ_CHUNK_BYTES):
                    if time.monotonic() > deadline:
                        raise MediaDownloadFetchError(
                            f"Timed out downloading media URL: {url}"
                        )
                    if len(body) + len(chunk) > max_bytes:
                        raise MediaDownloadTooLargeError(
                            f"Remote media exceeds the {max_bytes}-byte download cap"
                        )
                    body.extend(chunk)
                return bytes(body)
    except httpx.TimeoutException as exc:
        raise MediaDownloadFetchError(
            f"Timed out downloading media URL: {url}"
        ) from exc
    except httpx.HTTPError as exc:
        raise MediaDownloadFetchError(
            f"Failed to download media URL: {url} ({exc})"
        ) from exc

    raise AssertionError("unreachable: redirect loop must return or raise")
