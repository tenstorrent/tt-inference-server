# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""The MINIMAX_VERIFIER venv's export of MiniMax-Provider-Verifier."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import workflows.workflow_venvs as wv
from workflow_module.engine_types import WorkflowVenvType


class _Git:
    """Fakes git: ls-remote returns ``head``; a clone writes the files."""

    def __init__(self, monkeypatch, head="a" * 40):
        self.head = head
        self.commands = []
        monkeypatch.setattr(wv, "run_command", self.run)
        monkeypatch.setattr(wv, "_git_output", self.output)

    def output(self, *args):
        if args[0] == "ls-remote":
            return f"{self.head}\trefs/heads/b" if self.head else None
        return self.head  # rev-parse HEAD

    def run(self, cmd, **kw):
        self.commands.append(cmd)
        if cmd.startswith("git clone"):
            staging = Path(cmd.split()[-1])
            (staging / ".git").mkdir(parents=True)
            (staging / "verify.py").write_text("")
        return 0


def test_the_branch_head_is_exported_without_git_metadata(monkeypatch, tmp_path):
    git = _Git(monkeypatch)
    dest = tmp_path / "verifier"

    assert wv.export_repo_branch(dest, "url", "b", ["validator", "m3_format_check"])

    assert (dest / "verify.py").exists()
    assert not (dest / ".git").exists()
    assert wv.read_source_commit(dest) == git.head
    assert not (tmp_path / "verifier.staging").exists()
    clone, sparse, checkout = git.commands
    assert clone.startswith("git clone --filter=blob:none --no-checkout --depth 1 ")
    assert "--branch b url" in clone
    assert sparse.endswith("sparse-checkout set --cone validator m3_format_check")
    assert checkout.endswith("checkout b")


def test_an_export_at_the_branch_head_is_kept(monkeypatch, tmp_path):
    git = _Git(monkeypatch)
    dest = tmp_path / "verifier"
    wv.export_repo_branch(dest, "url", "b", ["validator"])
    git.commands.clear()

    assert wv.export_repo_branch(dest, "url", "b", ["validator"])

    assert git.commands == []


def test_a_moved_branch_is_exported_again(monkeypatch, tmp_path):
    git = _Git(monkeypatch)
    dest = tmp_path / "verifier"
    wv.export_repo_branch(dest, "url", "b", ["validator"])
    (dest / "stale.txt").write_text("")
    git.head = "b" * 40

    assert wv.export_repo_branch(dest, "url", "b", ["validator"])

    assert wv.read_source_commit(dest) == "b" * 40
    assert not (dest / "stale.txt").exists()


def test_an_unreachable_remote_keeps_an_existing_export(monkeypatch, tmp_path):
    git = _Git(monkeypatch)
    dest = tmp_path / "verifier"
    wv.export_repo_branch(dest, "url", "b", ["validator"])
    git.head = None

    assert wv.export_repo_branch(dest, "url", "b", ["validator"])
    assert not wv.export_repo_branch(tmp_path / "other", "url", "b", ["validator"])


def test_a_failed_clone_leaves_no_staging_directory(monkeypatch, tmp_path):
    _Git(monkeypatch)
    monkeypatch.setattr(wv, "run_command", lambda cmd, **kw: 1)

    assert not wv.export_repo_branch(tmp_path / "verifier", "url", "b", ["v"])
    assert not (tmp_path / "verifier.staging").exists()


def test_the_verifier_branch_is_exported(monkeypatch, tmp_path):
    seen = {}

    def export(dest, repo, branch, paths):
        seen.update(dest=dest, repo=repo, branch=branch, paths=paths)
        return True

    monkeypatch.setattr(wv, "export_repo_branch", export)

    assert wv.setup_minimax_verifier(SimpleNamespace(venv_path=tmp_path), None)

    assert seen["dest"] == tmp_path / "MiniMax-Provider-Verifier"
    assert seen["repo"] == wv.MINIMAX_VERIFIER_REPO
    assert seen["branch"] == wv.MINIMAX_VERIFIER_BRANCH
    # verify.py imports validator/; the pytest suites and fixtures live in
    # m3_format_check/.
    for path in ("validator", "m3_format_check"):
        assert path in seen["paths"]


def test_every_configured_baseline_is_checked_out():
    """A verify case's verify_baseline must be inside the exported paths, or
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
