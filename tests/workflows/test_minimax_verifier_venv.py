# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""The MINIMAX_VERIFIER venv's pinned, sparse checkout of MiniMax-Provider-Verifier."""

from __future__ import annotations

from types import SimpleNamespace

import workflows.workflow_venvs as wv
from workflow_module.engine_types import WorkflowVenvType


def _record(monkeypatch):
    calls = []
    monkeypatch.setattr(wv, "run_command", lambda cmd, **kw: calls.append(cmd) or 0)
    return calls


def test_sparse_paths_are_set_before_the_checkout(monkeypatch, tmp_path):
    calls = _record(monkeypatch)

    assert wv.checkout_pinned_repo(tmp_path / "repo", "url", "abc", ["a", "b/c"])

    sparse = calls.index(f"git -C {tmp_path / 'repo'} sparse-checkout set --cone a b/c")
    checkout = next(i for i, c in enumerate(calls) if " checkout --detach " in c)
    assert sparse < checkout
    assert calls[0].startswith("git clone --filter=blob:none --no-checkout url ")


def test_without_sparse_paths_the_commands_are_unchanged(monkeypatch, tmp_path):
    calls = _record(monkeypatch)
    dest = tmp_path / "repo"

    assert wv.checkout_pinned_repo(dest, "url", "abc")

    assert calls == [
        f"git clone --filter=blob:none --no-checkout url {dest}",
        f"git -C {dest} remote set-url origin url",
        f"git -C {dest} fetch --depth 1 origin abc",
        f"git -C {dest} checkout --detach --force FETCH_HEAD",
    ]


def test_the_verifier_is_checked_out_at_its_pinned_commit(monkeypatch, tmp_path):
    seen = {}

    def checkout(dest, repo, ref, sparse_paths=None):
        seen.update(dest=dest, repo=repo, ref=ref, sparse_paths=sparse_paths)
        return True

    monkeypatch.setattr(wv, "checkout_pinned_repo", checkout)
    venv = SimpleNamespace(venv_path=tmp_path)

    assert wv.setup_minimax_verifier(venv, model_spec=None)

    assert seen["dest"] == tmp_path / "MiniMax-Provider-Verifier"
    assert seen["repo"] == wv.MINIMAX_VERIFIER_REPO
    # A full commit SHA, not a branch: the checkout must not drift.
    assert len(seen["ref"]) == 40 and int(seen["ref"], 16) >= 0
    # verify.py imports validator/; the pytest suites and fixtures live in
    # m3_format_check/.
    for path in ("validator", "m3_format_check"):
        assert path in seen["sparse_paths"]


def test_every_configured_baseline_is_checked_out():
    """A verify case's verify_baseline must be inside the sparse checkout, or
    the run fails for a missing baseline."""
    from test_module.test_categorization_system.test_filter import (
        TestFilter as SuiteTestFilter,  # aliased so pytest does not collect it
    )

    baselines = {
        case["test_config"]["verify_baseline"]
        for suite in SuiteTestFilter().get_tests()
        for case in suite.get("test_cases", [])
        if case.get("name") == "MiniMaxProviderVerifierTest"
        and case["test_config"].get("verify_baseline")
    }
    assert baselines
    for baseline in baselines:
        assert any(
            baseline == path or baseline.startswith(path + "/")
            for path in wv.MINIMAX_VERIFIER_SPARSE_PATHS
        ), baseline


def test_the_verifier_venv_is_registered():
    config = wv.VENV_CONFIGS[WorkflowVenvType.MINIMAX_VERIFIER]

    assert config.python_version == "3.12"
    assert config.setup_function is wv.setup_minimax_verifier
    assert (wv.REQUIREMENTS_DIR / config.requirements_file).is_file()
