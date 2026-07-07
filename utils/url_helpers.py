# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Shared helpers for resolving the inference-server URL across workflows.

Every workflow (benchmarks, evals, stress_tests, tests, spec_tests) needs the
same answer to "where is the server?" with the same ``urlparse``-port-wins
rule. Previously each runner re-derived this inline; this module is the
single source of truth so a fix applied here propagates everywhere.

Resolution precedence (used by :func:`resolve_deploy_url`):

1. ``runtime_config.server_url`` — set from ``--server-url`` on ``run.py``
2. ``DEPLOY_URL`` env var — for direct script invocations
3. ``http://127.0.0.1`` — default
"""

from __future__ import annotations

import os
from typing import Tuple
from urllib.parse import urlparse

DEFAULT_DEPLOY_URL = "http://127.0.0.1"


def normalize_server_url(value: str) -> str:
    """Normalize a user-supplied ``--server-url`` value.

    Strips surrounding whitespace and a trailing slash, prepends ``http://``
    when no scheme is given, and validates that a hostname is present. Shared
    by v1 and v2 ``run.py`` so both apply the same rule.

    Raises ``ValueError`` (with a CLI-friendly message) when no hostname can be
    derived; callers should surface it via ``parser.error``.
    """
    server_url = value.strip().rstrip("/")
    parsed = urlparse(server_url)
    if not parsed.scheme:
        server_url = f"http://{server_url}"
        parsed = urlparse(server_url)
    if not parsed.hostname:
        raise ValueError(
            "--server-url must include a hostname (e.g. 'http://127.0.0.1')."
        )
    return server_url


def resolve_deploy_url(runtime_config=None) -> str:
    """Resolve the deploy URL using the standard precedence.

    Pass ``runtime_config`` when available; otherwise the function falls back
    to ``DEPLOY_URL`` env var, then the localhost default.
    """
    if runtime_config is not None:
        server_url = getattr(runtime_config, "server_url", None)
        if server_url:
            return server_url
    return os.environ.get("DEPLOY_URL", DEFAULT_DEPLOY_URL)


def is_remote_server(runtime_config=None, args=None) -> bool:
    """Return ``True`` when a remote ``--server-url`` was configured.

    Checks the explicit ``--server-url`` CLI flag first (when ``args`` is
    provided), then falls back to ``runtime_config.server_url`` propagated
    through the v2 bridge. A truthy value means tests/benchmarks should target
    a remote OpenAI-compatible endpoint rather than a locally launched server.
    """
    return bool(
        (args is not None and getattr(args, "server_url", None))
        or getattr(runtime_config, "server_url", None)
    )


def build_base_url(deploy_url: str, service_port) -> str:
    """Return ``scheme://host[:port]`` for building endpoint URLs.

    An explicit port on ``deploy_url`` wins over ``service_port`` so callers
    can't end up with malformed double-port URLs like ``http://host:9000:8000``
    when the user passes ``--server-url http://host:9000``.
    """
    deploy_url = deploy_url.rstrip("/")
    parsed = urlparse(deploy_url)
    if parsed.port is not None:
        return deploy_url
    return f"{deploy_url}:{service_port}"


def resolve_host_port(deploy_url: str, service_port) -> Tuple[str, str]:
    """Split into ``(host, port)`` for ``--host``/``--port`` style CLI args.

    Same port-wins rule as :func:`build_base_url`: an explicit port on
    ``deploy_url`` overrides ``service_port``.
    """
    parsed = urlparse(deploy_url.rstrip("/"))
    host = parsed.hostname or "localhost"
    port = str(parsed.port) if parsed.port is not None else str(service_port)
    return host, port
