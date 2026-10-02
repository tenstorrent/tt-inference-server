# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import logging
import subprocess
from pathlib import Path

import pytest

from workflows import workflow_venvs as wv

logger = logging.getLogger(__name__)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture
def checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = tmp_path / "harbor"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "agent.py").write_text("MAX_TOKENS = 16384\n")
    _git(repo, "add", "agent.py")
    _git(repo, "commit", "-q", "-m", "base")
    (repo / "agent.py").write_text("MAX_TOKENS = 4096\n")
    patch_text = _git(repo, "diff")
    _git(repo, "checkout", "--", "agent.py")

    patch_dir = tmp_path / "patches"
    patch_dir.mkdir()
    (patch_dir / "0001-compact.patch").write_text(patch_text)
    monkeypatch.setattr(wv, "_HARBOR_PATCH_DIR", patch_dir)
    return repo


def test_harbor_patch_applies_once(checkout: Path) -> None:
    target = checkout / "agent.py"
    assert wv._apply_harbor_patches(checkout, logger)
    assert target.read_text() == "MAX_TOKENS = 4096\n"
    assert wv._apply_harbor_patches(checkout, logger)
    assert target.read_text() == "MAX_TOKENS = 4096\n"


def test_harbor_patch_mismatch_fails_setup(checkout: Path) -> None:
    (checkout / "agent.py").write_text("MAX_TOKENS = 2048\n")
    assert not wv._apply_harbor_patches(checkout, logger)


def test_harbor_patch_requires_git_checkout(tmp_path: Path) -> None:
    target = tmp_path / "not-a-checkout"
    target.mkdir()
    assert not wv._apply_harbor_patches(target, logger)


def test_carried_length_recovery_patch_is_present() -> None:
    assert [patch.name for patch in wv._harbor_patches()] == [
        "0001-recover-compactly-from-length-truncation.patch"
    ]
