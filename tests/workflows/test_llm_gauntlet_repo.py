# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for resolving ``--requirements-json llm-gauntlet;<path>``.

No test here may reach the network: ``fetch_specs`` is stubbed, or driven with
``urlopen`` monkeypatched to serve an in-memory tarball.
"""

from __future__ import annotations

import io
import json
import tarfile
import urllib.error
from pathlib import Path

import pytest

from workflows import llm_gauntlet_repo as gauntlet
from workflows.llm_gauntlet_repo import (
    DEFAULT_LLM_GAUNTLET_REF,
    LLM_GAUNTLET_PREFIX,
    LLM_GAUNTLET_REF_ENV,
    LLM_GAUNTLET_TOKEN_ENV,
    LLMGauntletError,
    fetch_specs,
    is_llm_gauntlet_ref,
    resolve_ref,
    resolve_requirements_location,
)

_DOC = {
    "schemaVersion": "2.7.0",
    "id": "d",
    "model": {"name": "Qwen/Qwen3-32B", "contextLength": 131072},
    "deployment": {"hardware": "WH GLX", "maxConcurrencyPerInstance": 32},
}


@pytest.fixture
def fake_fetch(monkeypatch, tmp_path):
    """Redirect the download at tmp_path and record every fetch request."""
    dest = tmp_path / "llm-gauntlet"
    calls = []

    def fake(d, ref, token=None):
        calls.append({"dest": d, "ref": ref, "token": token})
        (d / "specs" / "tt-internal" / "qwen3-32b").mkdir(parents=True, exist_ok=True)
        (d / "specs" / "tt-internal" / "qwen3-32b" / "x.json").write_text(
            json.dumps(_DOC)
        )
        (d / "specs" / "x.json").write_text(json.dumps(_DOC))
        return 2

    monkeypatch.delenv(LLM_GAUNTLET_TOKEN_ENV, raising=False)
    monkeypatch.delenv(LLM_GAUNTLET_REF_ENV, raising=False)
    monkeypatch.setattr(gauntlet, "download_dir", lambda: dest)
    monkeypatch.setattr(gauntlet, "fetch_specs", fake)
    return calls, dest


def test_plain_paths_never_touch_the_network(fake_fetch):
    calls, _ = fake_fetch

    assert resolve_requirements_location("/tmp/doc.json") == "/tmp/doc.json"
    assert resolve_requirements_location("relative/doc.json") == "relative/doc.json"
    assert calls == []


@pytest.mark.parametrize(
    "value, expected",
    [
        ("llm-gauntlet;specs/a.json", True),
        ("/abs/specs/a.json", False),
        ("", False),
        (None, False),
    ],
)
def test_is_llm_gauntlet_ref(value, expected):
    assert is_llm_gauntlet_ref(value) is expected


def test_resolves_to_a_path_inside_the_download(fake_fetch):
    calls, dest = fake_fetch

    out = resolve_requirements_location(
        f"{LLM_GAUNTLET_PREFIX}specs/tt-internal/qwen3-32b/x.json"
    )

    assert Path(out) == (dest / "specs/tt-internal/qwen3-32b/x.json").resolve()
    assert Path(out).is_file()
    assert len(calls) == 1


def test_resolved_document_loads(fake_fetch):
    """End of the line: the resolved path is a document the loader accepts."""
    from workflow_module.requirements_schema import load_requirements

    out = resolve_requirements_location(
        f"{LLM_GAUNTLET_PREFIX}specs/tt-internal/qwen3-32b/x.json"
    )

    assert load_requirements(out).model.name == "Qwen/Qwen3-32B"


def test_leading_slash_in_the_repo_path_is_tolerated(fake_fetch):
    _, dest = fake_fetch

    out = resolve_requirements_location(
        f"{LLM_GAUNTLET_PREFIX}/specs/tt-internal/qwen3-32b/x.json"
    )

    assert Path(out) == (dest / "specs/tt-internal/qwen3-32b/x.json").resolve()


@pytest.mark.parametrize(
    "value",
    [
        LLM_GAUNTLET_PREFIX,
        f"{LLM_GAUNTLET_PREFIX}   ",
        f"{LLM_GAUNTLET_PREFIX}../../etc/passwd",
        f"{LLM_GAUNTLET_PREFIX}specs/../../outside.json",
    ],
)
def test_rejects_empty_and_escaping_paths(fake_fetch, value):
    with pytest.raises(LLMGauntletError):
        resolve_requirements_location(value)


def test_missing_document_is_reported(fake_fetch):
    with pytest.raises(LLMGauntletError, match="not found"):
        resolve_requirements_location(f"{LLM_GAUNTLET_PREFIX}README.md")


def test_ref_precedence_is_argument_then_env_then_main(monkeypatch):
    monkeypatch.delenv(LLM_GAUNTLET_REF_ENV, raising=False)
    assert resolve_ref() == DEFAULT_LLM_GAUNTLET_REF == "main"
    assert resolve_ref("from-arg") == "from-arg"

    monkeypatch.setenv(LLM_GAUNTLET_REF_ENV, "from-env")
    assert resolve_ref() == "from-env"
    assert resolve_ref("from-arg") == "from-arg"


def test_env_ref_and_token_reach_the_fetch(fake_fetch, monkeypatch):
    calls, _ = fake_fetch
    monkeypatch.setenv(LLM_GAUNTLET_REF_ENV, "my-test-branch")
    monkeypatch.setenv(LLM_GAUNTLET_TOKEN_ENV, "tok")

    resolve_requirements_location(f"{LLM_GAUNTLET_PREFIX}specs/x.json")

    assert calls[0]["ref"] == "my-test-branch"
    assert calls[0]["token"] == "tok"


# --- fetch_specs: the tarball download, without the network -----------------


def _tarball(files: dict, top: str = "tenstorrent-llm-gauntlet-abc123") -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(f"{top}/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        link = tarfile.TarInfo(f"{top}/specs/link.json")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        tar.addfile(link)
    return buf.getvalue()


@pytest.fixture
def served(monkeypatch):
    """Serve a tarball from a fake urlopen; record the requests it received."""
    requests = []
    payload = {"body": b""}

    def fake_urlopen(request, timeout=None):
        requests.append(request)
        if isinstance(payload["body"], Exception):
            raise payload["body"]
        return io.BytesIO(payload["body"])

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return requests, payload


def test_keeps_only_spec_json_with_the_top_dir_stripped(served, tmp_path):
    requests, payload = served
    payload["body"] = _tarball(
        {
            "specs/tt-internal/qwen3-32b/x.json": b"{}",
            "specs/README.md": b"docs",
            "src/other.json": b"{}",
            "package.json": b"{}",
        }
    )
    dest = tmp_path / "llm-gauntlet"

    assert fetch_specs(dest, "main") == 1

    files = sorted(
        p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()
    )
    assert files == ["specs/tt-internal/qwen3-32b/x.json"]
    assert requests[0].full_url == (
        "https://api.github.com/repos/tenstorrent/llm-gauntlet/tarball/main"
    )


def test_branch_with_a_slash_is_kept_in_the_url(served, tmp_path):
    requests, payload = served
    payload["body"] = _tarball({"specs/x.json": b"{}"})

    fetch_specs(tmp_path / "llm-gauntlet", "ipastalTT/my-branch")

    assert requests[0].full_url.endswith("/tarball/ipastalTT/my-branch")


def test_token_is_sent_as_bearer_and_not_forwarded_on_redirect(served, tmp_path):
    requests, payload = served
    payload["body"] = _tarball({"specs/x.json": b"{}"})

    fetch_specs(tmp_path / "llm-gauntlet", "main", token="ghp_SECRET")

    req = requests[0]
    assert req.unredirected_hdrs["Authorization"] == "Bearer ghp_SECRET"
    assert "Authorization" not in req.headers


def test_no_token_sends_no_auth_header(served, tmp_path):
    requests, payload = served
    payload["body"] = _tarball({"specs/x.json": b"{}"})

    fetch_specs(tmp_path / "llm-gauntlet", "main")

    assert not requests[0].has_header("Authorization")


def test_a_previous_download_is_replaced(served, tmp_path):
    _, payload = served
    payload["body"] = _tarball({"specs/new.json": b"{}"})
    dest = tmp_path / "llm-gauntlet"
    (dest / "specs").mkdir(parents=True)
    (dest / "specs" / "stale.json").write_text("{}")

    fetch_specs(dest, "main")

    assert (dest / "specs" / "new.json").is_file()
    assert not (dest / "specs" / "stale.json").exists()


def test_a_failed_download_keeps_the_previous_copy(served, tmp_path):
    _, payload = served
    payload["body"] = urllib.error.HTTPError(
        "https://api.github.com/x", 404, "Not Found", {}, None
    )
    dest = tmp_path / "llm-gauntlet"
    (dest / "specs").mkdir(parents=True)
    (dest / "specs" / "kept.json").write_text("{}")

    with pytest.raises(LLMGauntletError, match=LLM_GAUNTLET_TOKEN_ENV):
        fetch_specs(dest, "main")

    assert (dest / "specs" / "kept.json").is_file()


def test_a_tarball_without_specs_is_an_error(served, tmp_path):
    _, payload = served
    payload["body"] = _tarball({"README.md": b"x"})

    with pytest.raises(LLMGauntletError, match="no specs"):
        fetch_specs(tmp_path / "llm-gauntlet", "main")


def test_the_token_never_reaches_the_log(served, monkeypatch, tmp_path, caplog):
    _, payload = served
    payload["body"] = _tarball({"specs/x.json": b"{}"})
    secret = "ghp_SUPERSECRET"
    monkeypatch.setenv(LLM_GAUNTLET_TOKEN_ENV, secret)
    monkeypatch.setattr(gauntlet, "download_dir", lambda: tmp_path / "llm-gauntlet")

    with caplog.at_level("DEBUG"):
        resolve_requirements_location(f"{LLM_GAUNTLET_PREFIX}specs/x.json")

    assert secret not in caplog.text
