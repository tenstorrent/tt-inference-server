# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Unit tests for the centralized server-URL resolution helpers."""

import pytest

from utils.url_helpers import (
    DEFAULT_DEPLOY_URL,
    build_base_url,
    normalize_server_url,
    resolve_deploy_url,
    resolve_host_port,
)


class _Cfg:
    """Minimal stand-in for RuntimeConfig with a server_url attribute."""

    def __init__(self, server_url=None):
        self.server_url = server_url


# --- normalize_server_url ---------------------------------------------------


def test_normalize_prepends_http_scheme():
    assert normalize_server_url("192.168.1.10") == "http://192.168.1.10"


def test_normalize_strips_trailing_slash():
    assert normalize_server_url("http://host:9000/") == "http://host:9000"


def test_normalize_preserves_explicit_scheme_and_port():
    assert normalize_server_url("https://host:8443") == "https://host:8443"


def test_normalize_strips_surrounding_whitespace():
    assert normalize_server_url("  http://127.0.0.1  ") == "http://127.0.0.1"


@pytest.mark.parametrize("bad", ["", "   ", "http://", "://nohost"])
def test_normalize_requires_hostname(bad):
    with pytest.raises(ValueError):
        normalize_server_url(bad)


# --- resolve_deploy_url -----------------------------------------------------


def test_resolve_prefers_runtime_config(monkeypatch):
    monkeypatch.delenv("DEPLOY_URL", raising=False)
    assert resolve_deploy_url(_Cfg("http://from-config")) == "http://from-config"


def test_resolve_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("DEPLOY_URL", "http://from-env")
    assert resolve_deploy_url(_Cfg(None)) == "http://from-env"


def test_resolve_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("DEPLOY_URL", raising=False)
    assert resolve_deploy_url(None) == DEFAULT_DEPLOY_URL


# --- build_base_url / resolve_host_port (port-wins rule) --------------------


def test_build_base_url_appends_service_port():
    assert build_base_url("http://host", 8000) == "http://host:8000"


def test_build_base_url_explicit_port_wins():
    assert build_base_url("http://host:9000", 8000) == "http://host:9000"


def test_resolve_host_port_splits_and_applies_port_wins():
    assert resolve_host_port("http://host:9000", 8000) == ("host", "9000")
    assert resolve_host_port("http://host", 8000) == ("host", "8000")
