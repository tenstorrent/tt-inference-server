# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""The MINIMAX_VERIFIER venv's export of MiniMax-Provider-Verifier."""

from __future__ import annotations

import io
import tarfile
import urllib.error
from types import SimpleNamespace

import workflows.workflow_venvs as wv
from workflow_module.engine_types import WorkflowVenvType

_FILES = {
    "verify.py": "",
    "sample.jsonl": "",
    "README.md": "",
    "validator/__init__.py": "",
    "output-dir/MiniMax-M3/loop_01/a.jsonl": "",
    "output-dir/MiniMax-M3/loop_02/a.jsonl": "",
}


def _tarball(files):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(f"repo-abc/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    buf.seek(0)
    return buf


class _Remote:
    """Fakes the remote: ls-remote returns ``head``; the tarball holds ``files``."""

    def __init__(self, monkeypatch, head="a" * 40):
        self.head = head
        self.files = dict(_FILES)
        self.urls = []
        monkeypatch.setattr(wv, "_git_output", self.output)
        monkeypatch.setattr(wv.urllib.request, "urlopen", self.urlopen)

    def output(self, *args):
        return f"{self.head}\trefs/heads/b" if self.head else None

    def urlopen(self, url, timeout=None):
        self.urls.append(url)
        if self.files is None:
            raise urllib.error.URLError("down")
        return _tarball(self.files)


_PATHS = ["verify.py", "validator", "output-dir/MiniMax-M3/loop_01"]


def test_only_the_listed_paths_are_exported_at_the_head(monkeypatch, tmp_path):
    remote = _Remote(monkeypatch)
    dest = tmp_path / "verifier"

    assert wv.export_repo_branch(dest, "https://github.com/o/r.git", "b", _PATHS)

    assert remote.urls == [f"https://codeload.github.com/o/r/tar.gz/{remote.head}"]
    exported = sorted(
        str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file()
    )
    assert exported == [
        wv.SOURCE_COMMIT_FILE,
        "output-dir/MiniMax-M3/loop_01/a.jsonl",
        "validator/__init__.py",
        "verify.py",
    ]
    assert wv.read_source_commit(dest) == remote.head
    assert not (tmp_path / "verifier.staging").exists()


def test_a_missing_path_fails_the_export(monkeypatch, tmp_path):
    remote = _Remote(monkeypatch)
    del remote.files["verify.py"]
    dest = tmp_path / "verifier"

    assert not wv.export_repo_branch(dest, "url", "b", _PATHS)

    assert not dest.exists()
    assert not (tmp_path / "verifier.staging").exists()


def test_an_export_at_the_branch_head_is_kept(monkeypatch, tmp_path):
    remote = _Remote(monkeypatch)
    dest = tmp_path / "verifier"
    wv.export_repo_branch(dest, "url", "b", _PATHS)
    remote.urls.clear()

    assert wv.export_repo_branch(dest, "url", "b", _PATHS)

    assert remote.urls == []


def test_a_moved_branch_is_exported_again(monkeypatch, tmp_path):
    remote = _Remote(monkeypatch)
    dest = tmp_path / "verifier"
    wv.export_repo_branch(dest, "url", "b", _PATHS)
    (dest / "stale.txt").write_text("")
    remote.head = "b" * 40

    assert wv.export_repo_branch(dest, "url", "b", _PATHS)

    assert wv.read_source_commit(dest) == "b" * 40
    assert not (dest / "stale.txt").exists()


def test_an_unreachable_remote_keeps_an_existing_export(monkeypatch, tmp_path):
    remote = _Remote(monkeypatch)
    dest = tmp_path / "verifier"
    wv.export_repo_branch(dest, "url", "b", _PATHS)
    remote.head = None

    assert wv.export_repo_branch(dest, "url", "b", _PATHS)
    assert not wv.export_repo_branch(tmp_path / "other", "url", "b", _PATHS)


def test_a_failed_download_keeps_the_previous_export(monkeypatch, tmp_path):
    remote = _Remote(monkeypatch)
    dest = tmp_path / "verifier"
    wv.export_repo_branch(dest, "url", "b", _PATHS)
    remote.head, remote.files = "b" * 40, None

    assert not wv.export_repo_branch(dest, "url", "b", _PATHS)

    assert wv.read_source_commit(dest) == "a" * 40
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
    for path in ("verify.py", "sample.jsonl", "validator", "m3_format_check"):
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
            for path in wv.MINIMAX_VERIFIER_PATHS
        ), baseline


def test_the_verifier_venv_is_registered():
    config = wv.VENV_CONFIGS[WorkflowVenvType.MINIMAX_VERIFIER]

    assert config.python_version == "3.12"
    assert config.setup_function is wv.setup_minimax_verifier
    assert (wv.REQUIREMENTS_DIR / config.requirements_file).is_file()
