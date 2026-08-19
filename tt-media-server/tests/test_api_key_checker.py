# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Tests for API key auth and the tenant id derived from it.

The module reads its config at import time, so each test reloads it under a
patched environment rather than mutating module globals.
"""

import importlib
import os

import pytest
from fastapi import HTTPException

_AUTH_VARS = ("API_KEY", "API_KEYS", "NO_AUTH")


@pytest.fixture(autouse=True)
def _restore_api_key_checker():
    """Reload the module under the original env after each test.

    These tests reload a module whose config is read at import time, which
    mutates global state. Without this, whichever auth config ran last would
    leak into every other test file in the session.
    """
    saved = {var: os.environ.get(var) for var in _AUTH_VARS}
    yield
    for var, value in saved.items():
        if value is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = value
    import security.api_key_checker as mod

    importlib.reload(mod)


def _load(monkeypatch, **env):
    """Reload the checker with exactly the given auth env vars set."""
    for var in ("API_KEY", "API_KEYS", "NO_AUTH"):
        monkeypatch.delenv(var, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    import security.api_key_checker as mod

    return importlib.reload(mod)


class TestApiKeyAuth:
    def test_accepts_configured_key(self, monkeypatch):
        mod = _load(monkeypatch, API_KEY="secret-a")
        assert mod.get_api_key("Bearer secret-a") == "Bearer secret-a"

    def test_rejects_wrong_key(self, monkeypatch):
        mod = _load(monkeypatch, API_KEY="secret-a")
        with pytest.raises(HTTPException) as exc:
            mod.get_api_key("Bearer wrong")
        assert exc.value.status_code == 401

    def test_rejects_missing_key(self, monkeypatch):
        mod = _load(monkeypatch, API_KEY="secret-a")
        with pytest.raises(HTTPException) as exc:
            mod.get_api_key(None)
        assert exc.value.status_code == 401

    def test_api_keys_accepts_any_configured_key(self, monkeypatch):
        mod = _load(monkeypatch, API_KEYS="secret-a, secret-b ,secret-c")
        for key in ("secret-a", "secret-b", "secret-c"):
            assert mod.get_api_key(f"Bearer {key}") is not None

    def test_api_keys_supersedes_api_key(self, monkeypatch):
        mod = _load(monkeypatch, API_KEY="legacy", API_KEYS="secret-a")
        assert mod.get_api_key("Bearer secret-a") is not None
        with pytest.raises(HTTPException):
            mod.get_api_key("Bearer legacy")

    def test_no_auth_disables_checking(self, monkeypatch):
        mod = _load(monkeypatch, API_KEY="secret-a", NO_AUTH="1")
        assert mod.get_api_key(None) is None
        assert mod.get_api_key("Bearer anything") is None


class TestOrgIdDerivation:
    def test_distinct_keys_get_distinct_org_ids(self, monkeypatch):
        """The isolation guarantee: two tenants must not collide."""
        mod = _load(monkeypatch, API_KEYS="secret-a,secret-b")
        assert mod.get_org_id("Bearer secret-a") != mod.get_org_id("Bearer secret-b")

    def test_org_id_is_stable_across_calls(self, monkeypatch):
        """A tenant must still see its own jobs on a later request."""
        mod = _load(monkeypatch, API_KEYS="secret-a,secret-b")
        assert mod.get_org_id("Bearer secret-a") == mod.get_org_id("Bearer secret-a")

    def test_org_id_does_not_leak_the_key(self, monkeypatch):
        """org_id lands in job records, so it must not contain the credential."""
        mod = _load(monkeypatch, API_KEY="super-secret-value")
        org_id = mod.get_org_id("Bearer super-secret-value")
        assert "super-secret-value" not in org_id

    def test_bad_key_raises_rather_than_returning_none(self, monkeypatch):
        """None means 'unscoped' to JobManager, so this must never fail open."""
        mod = _load(monkeypatch, API_KEY="secret-a")
        with pytest.raises(HTTPException) as exc:
            mod.get_org_id("Bearer wrong")
        assert exc.value.status_code == 401

    def test_no_auth_yields_unscoped(self, monkeypatch):
        """With auth off there is no tenant, and job access stays unscoped."""
        mod = _load(monkeypatch, API_KEY="secret-a", NO_AUTH="true")
        assert mod.get_org_id("Bearer anything") is None

    def test_single_key_deployment_shares_one_org(self, monkeypatch):
        """Documents the limitation: one key means one tenant, so no isolation."""
        mod = _load(monkeypatch, API_KEY="shared")
        assert mod.get_org_id("Bearer shared") == mod.get_org_id("Bearer shared")
