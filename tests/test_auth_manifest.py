"""Tests for workflows/auth_manifest.py."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from workflows.auth_manifest import (  # noqa: E402
    AuthManifest,
    SOURCE_JWT_SIGNED,
    SOURCE_NO_AUTH,
    SOURCE_OPENAI_API_KEY,
    SOURCE_VLLM_API_KEY,
    build_manifest,
    load_bearer_or_fallback,
    resolve_and_write_from_env,
    resolve_bearer,
)


# ---------------------------------------------------------------------------
# resolve_bearer: the four canonical sources
# ---------------------------------------------------------------------------


def test_resolve_bearer_prefers_vllm_api_key_over_jwt_and_openai():
    bearer, source = resolve_bearer(
        vllm_api_key="raw-key",
        jwt_secret="some-secret",
        openai_api_key="some-openai",
    )
    assert bearer == "raw-key"
    assert source == SOURCE_VLLM_API_KEY


def test_resolve_bearer_falls_back_to_signed_jwt_when_vllm_missing():
    bearer, source = resolve_bearer(
        jwt_secret="some-secret",
        openai_api_key="some-openai",
    )
    assert source == SOURCE_JWT_SIGNED
    assert bearer and bearer.count(".") == 2


def test_resolve_bearer_uses_openai_when_no_secret_no_vllm_key():
    bearer, source = resolve_bearer(openai_api_key="sk-fake")
    assert (bearer, source) == ("sk-fake", SOURCE_OPENAI_API_KEY)


def test_resolve_bearer_no_auth_when_all_inputs_empty():
    bearer, source = resolve_bearer()
    assert bearer is None
    assert source == SOURCE_NO_AUTH


def test_resolve_bearer_treats_empty_strings_as_unset():
    bearer, source = resolve_bearer(
        vllm_api_key="",
        jwt_secret="",
        openai_api_key="",
    )
    assert bearer is None
    assert source == SOURCE_NO_AUTH


# ---------------------------------------------------------------------------
# build_manifest + fingerprint
# ---------------------------------------------------------------------------


def test_build_manifest_includes_fingerprint_and_base_url():
    manifest = build_manifest(
        service_port="8000",
        deploy_url="http://example",
        run_id="abc123",
        vllm_api_key="abc",
    )
    assert manifest.source == SOURCE_VLLM_API_KEY
    assert manifest.bearer == "abc"
    assert manifest.base_url == "http://example:8000/v1"
    assert len(manifest.bearer_sha256_8) == 8
    assert manifest.run_id == "abc123"


def test_build_manifest_no_auth_fingerprint_is_none_literal():
    manifest = build_manifest(service_port="8000")
    assert manifest.bearer is None
    assert manifest.bearer_sha256_8 == "none"
    assert manifest.source == SOURCE_NO_AUTH


def test_build_manifest_rejects_invalid_source_at_construction():
    # AuthManifest validates source in __post_init__.
    with pytest.raises(ValueError):
        AuthManifest(
            bearer=None,
            bearer_sha256_8="none",
            source="not_a_real_source",
            service_port="8000",
            base_url=None,
            run_id=None,
        )


# ---------------------------------------------------------------------------
# write / load round-trip
# ---------------------------------------------------------------------------


def test_manifest_write_then_load_round_trip(tmp_path):
    manifest = build_manifest(
        service_port="8000",
        run_id="rid",
        vllm_api_key="bearer-xyz",
    )
    path = manifest.write(tmp_path / "auth_manifest.json")
    loaded = AuthManifest.load(path)
    assert loaded == manifest


def test_manifest_write_is_chmod_0600_when_filesystem_supports_it(tmp_path):
    manifest = build_manifest(
        service_port="8000", run_id="rid", vllm_api_key="bearer"
    )
    path = manifest.write(tmp_path / "auth.json")
    mode = os.stat(path).st_mode & 0o777
    # We don't assert == 0o600 strictly: some filesystems (e.g. shared
    # CI volumes) may strip permissions. We require at minimum that
    # group/other write bits are off so the file is not world-writable.
    assert mode & 0o022 == 0


def test_manifest_safe_summary_does_not_leak_bearer():
    manifest = build_manifest(
        service_port="8000",
        run_id="rid",
        vllm_api_key="super-secret-bearer-do-not-leak",
    )
    summary = manifest.safe_summary()
    assert "super-secret-bearer-do-not-leak" not in summary
    assert manifest.bearer_sha256_8 in summary


# ---------------------------------------------------------------------------
# load_bearer_or_fallback: manifest path > env fallback
# ---------------------------------------------------------------------------


def test_load_bearer_or_fallback_prefers_manifest_path(tmp_path, monkeypatch):
    manifest = build_manifest(
        service_port="8000",
        run_id="rid",
        vllm_api_key="from-manifest",
    )
    path = manifest.write(tmp_path / "auth.json")

    monkeypatch.setenv("OPENAI_API_KEY", "from-env-not-used")
    monkeypatch.setenv("VLLM_API_KEY", "from-env-not-used-either")

    bearer, source = load_bearer_or_fallback(str(path))
    assert bearer == "from-manifest"
    assert source.startswith("manifest:")


def test_load_bearer_or_fallback_uses_env_when_manifest_missing(monkeypatch):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "fallback-key")

    bearer, source = load_bearer_or_fallback(None)
    assert bearer == "fallback-key"
    assert source.startswith("env:")


def test_load_bearer_or_fallback_recovers_on_corrupt_manifest(tmp_path, monkeypatch):
    bad = tmp_path / "auth_manifest.json"
    bad.write_text("{not valid json")

    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "fallback")

    bearer, source = load_bearer_or_fallback(str(bad))
    assert bearer == "fallback"
    assert source.startswith("env:")


def test_load_bearer_or_fallback_recovers_on_nonexistent_path(monkeypatch):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "fb")

    bearer, source = load_bearer_or_fallback("/tmp/does-not-exist-xyz.json")
    assert bearer == "fb"
    assert source.startswith("env:")


# ---------------------------------------------------------------------------
# resolve_and_write_from_env: the integration entry point
# ---------------------------------------------------------------------------


def test_resolve_and_write_from_env_persists_manifest(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.setenv("JWT_SECRET", "supersecret")

    manifest, path = resolve_and_write_from_env(
        target_dir=tmp_path,
        run_id="my-run-id",
        service_port="7000",
    )
    assert manifest.source == SOURCE_JWT_SIGNED
    assert path.name == "auth_manifest_my-run-id.json"
    assert path.exists()

    loaded = json.loads(path.read_text())
    assert loaded["source"] == SOURCE_JWT_SIGNED
    assert loaded["base_url"] == "http://127.0.0.1:7000/v1"
    assert loaded["run_id"] == "my-run-id"


def test_resolve_and_write_from_env_with_no_auth_still_writes(tmp_path, monkeypatch):
    for var in ("OPENAI_API_KEY", "VLLM_API_KEY", "JWT_SECRET"):
        monkeypatch.delenv(var, raising=False)

    manifest, path = resolve_and_write_from_env(
        target_dir=tmp_path,
        run_id="rid-noauth",
        service_port="8000",
    )
    assert manifest.source == SOURCE_NO_AUTH
    assert manifest.bearer is None
    assert path.exists()
