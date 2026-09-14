# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for resolving ``--requirements-json llm-gauntlet;<path>``.

No test here may reach the network. ``checkout_repo_ref`` is either stubbed or
driven with ``run_command`` monkeypatched, and ``clone_dir`` is redirected at
``tmp_path`` so a bug can never materialize a clone in the real repo root.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from workflows import llm_gauntlet_repo as gauntlet
from workflows.llm_gauntlet_repo import (
    DEFAULT_LLM_GAUNTLET_REPO,
    LLM_GAUNTLET_PREFIX,
    LLM_GAUNTLET_REF_ENV,
    LLM_GAUNTLET_REPO_ENV,
    LLMGauntletError,
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
def fake_clone(monkeypatch, tmp_path):
    """Redirect the clone at tmp_path and record every checkout request."""
    dest = tmp_path / "llm-gauntlet"
    calls = []

    def fake_checkout(d, repo, ref):
        calls.append({"dest": d, "repo": repo, "ref": ref})
        (d / "specs" / "tt-internal" / "qwen3-32b").mkdir(parents=True, exist_ok=True)
        (d / "specs" / "tt-internal" / "qwen3-32b" / "x.json").write_text(
            json.dumps(_DOC)
        )
        return True

    monkeypatch.setattr(gauntlet, "clone_dir", lambda: dest)
    monkeypatch.setattr(gauntlet, "checkout_repo_ref", fake_checkout)
    return calls, dest


def test_plain_paths_never_touch_git(fake_clone):
    """The no-prefix fast path is the guarantee that nothing else regresses."""
    calls, _ = fake_clone

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


def test_resolves_to_a_path_inside_the_clone(fake_clone):
    calls, dest = fake_clone

    out = resolve_requirements_location(
        f"{LLM_GAUNTLET_PREFIX}specs/tt-internal/qwen3-32b/x.json"
    )

    assert Path(out) == (dest / "specs/tt-internal/qwen3-32b/x.json").resolve()
    assert Path(out).is_file()
    assert len(calls) == 1


def test_resolved_document_loads(fake_clone):
    """End of the line: the resolved path is a document the loader accepts."""
    from workflow_module.requirements_schema import load_requirements

    out = resolve_requirements_location(
        f"{LLM_GAUNTLET_PREFIX}specs/tt-internal/qwen3-32b/x.json"
    )

    assert load_requirements(out).model.name == "Qwen/Qwen3-32B"


def test_leading_slash_in_the_repo_path_is_tolerated(fake_clone):
    _, dest = fake_clone

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
def test_rejects_empty_and_escaping_paths(fake_clone, value):
    with pytest.raises(LLMGauntletError):
        resolve_requirements_location(value)


def test_reports_a_failed_checkout(monkeypatch, tmp_path):
    monkeypatch.setattr(gauntlet, "clone_dir", lambda: tmp_path / "llm-gauntlet")
    monkeypatch.setattr(gauntlet, "checkout_repo_ref", lambda d, repo, ref: False)

    with pytest.raises(LLMGauntletError, match="credentials"):
        resolve_requirements_location(f"{LLM_GAUNTLET_PREFIX}specs/x.json")


def test_ref_precedence_is_argument_then_env(monkeypatch):
    monkeypatch.delenv(LLM_GAUNTLET_REF_ENV, raising=False)
    assert resolve_ref() is None
    assert resolve_ref("from-arg") == "from-arg"

    monkeypatch.setenv(LLM_GAUNTLET_REF_ENV, "from-env")
    assert resolve_ref() == "from-env"
    assert resolve_ref("from-arg") == "from-arg"


def test_defaults_to_ssh_because_the_repo_is_private(fake_clone):
    """Anonymous HTTPS 404s on a private repo; every runner has an SSH key."""
    calls, _ = fake_clone

    resolve_requirements_location(f"{LLM_GAUNTLET_PREFIX}specs/x.json")

    assert calls[0]["repo"] == DEFAULT_LLM_GAUNTLET_REPO
    assert calls[0]["repo"].startswith("git@")


def test_repo_url_comes_from_the_env_when_set(fake_clone, monkeypatch):
    """Escape hatch for a runner where a token, not a key, is the credential."""
    calls, _ = fake_clone
    https = "https://github.com/tenstorrent/llm-gauntlet.git"
    monkeypatch.setenv(LLM_GAUNTLET_REPO_ENV, https)

    resolve_requirements_location(f"{LLM_GAUNTLET_PREFIX}specs/x.json")

    assert calls[0]["repo"] == https


def test_env_ref_reaches_the_checkout(fake_clone, monkeypatch):
    calls, _ = fake_clone
    monkeypatch.setenv(LLM_GAUNTLET_REF_ENV, "my-test-branch")

    resolve_requirements_location(f"{LLM_GAUNTLET_PREFIX}specs/x.json")

    assert calls[0]["ref"] == "my-test-branch"


# --- checkout_repo_ref: the git command sequence, without git ---------------


@pytest.fixture
def git_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "workflow_module.proc.run_command",
        lambda cmd, logger=None, **kw: (calls.append(cmd), 0)[1],
    )
    return calls


def test_fresh_clone_without_a_ref_takes_the_default_branch(git_calls, tmp_path):
    dest = tmp_path / "llm-gauntlet"

    assert gauntlet.checkout_repo_ref(dest, "REPO", None) is True

    # --depth 1 with no --branch is the remote's default branch, and nothing
    # else is needed: no fetch, no detach.
    assert git_calls == [f"git clone --depth 1 REPO {dest}"]


def test_fresh_clone_with_a_ref_asks_for_that_branch(git_calls, tmp_path):
    dest = tmp_path / "llm-gauntlet"

    gauntlet.checkout_repo_ref(dest, "REPO", "my-branch")

    assert git_calls[0] == f"git clone --depth 1 --branch my-branch REPO {dest}"
    assert any("checkout --detach --force FETCH_HEAD" in c for c in git_calls)


def test_existing_clone_is_always_refetched(git_calls, tmp_path):
    """A default branch moves, so reuse must converge, not short-circuit."""
    dest = tmp_path / "llm-gauntlet"
    (dest / ".git").mkdir(parents=True)

    gauntlet.checkout_repo_ref(dest, "REPO", None)

    # startswith, not "in": tmp_path embeds the test name, which says "clone".
    assert not any(c.startswith("git clone") for c in git_calls)
    assert f"git -C {dest} fetch --depth 1 origin HEAD" in git_calls
    assert f"git -C {dest} checkout --detach --force FETCH_HEAD" in git_calls


def test_existing_clone_fetches_a_named_ref_directly(git_calls, tmp_path):
    dest = tmp_path / "llm-gauntlet"
    (dest / ".git").mkdir(parents=True)

    gauntlet.checkout_repo_ref(dest, "REPO", "abc123")

    assert f"git -C {dest} fetch --depth 1 origin abc123" in git_calls


def test_a_non_git_directory_is_discarded(monkeypatch, tmp_path):
    dest = tmp_path / "llm-gauntlet"
    dest.mkdir()
    (dest / "junk").write_text("x")
    monkeypatch.setattr(
        "workflow_module.proc.run_command", lambda cmd, logger=None, **kw: 0
    )

    gauntlet.checkout_repo_ref(dest, "REPO", None)

    assert not (dest / "junk").exists()


def test_checkout_failure_is_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "workflow_module.proc.run_command", lambda cmd, logger=None, **kw: 1
    )

    assert gauntlet.checkout_repo_ref(tmp_path / "llm-gauntlet", "REPO", None) is False
