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
