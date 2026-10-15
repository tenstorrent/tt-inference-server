# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Patches carried in workflows/patches/agentx_harness are applied to the
agentx-harness checkout, exactly once, and change the ref stamp."""

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
    """A fake agentx-harness checkout and one patch for it."""
    repo = tmp_path / "agentx-harness"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "mod.py").write_text("VALUE = 1\n")
    _git(repo, "add", "mod.py")
    _git(repo, "commit", "-q", "-m", "base")
    (repo / "mod.py").write_text("VALUE = 2\n")
    patch = _git(repo, "diff")
    _git(repo, "checkout", "--", "mod.py")

    patch_dir = tmp_path / "patches"
    patch_dir.mkdir()
    (patch_dir / "0001-bump.patch").write_text(patch)
    monkeypatch.setattr(wv, "_AGENTX_HARNESS_PATCH_DIR", patch_dir)
    return repo


def test_patch_applies_once(checkout: Path) -> None:
    target = checkout / "mod.py"
    assert wv._apply_agentx_harness_patches(checkout, logger)
    assert target.read_text() == "VALUE = 2\n"
    # Second pass sees the patch already applied and leaves the tree alone.
    assert wv._apply_agentx_harness_patches(checkout, logger)
    assert target.read_text() == "VALUE = 2\n"


def test_patch_that_does_not_apply_fails_setup(checkout: Path) -> None:
    (checkout / "mod.py").write_text("VALUE = 3\n")
    assert not wv._apply_agentx_harness_patches(checkout, logger)


def test_missing_checkout_fails_setup(checkout: Path) -> None:
    assert not wv._apply_agentx_harness_patches(checkout / "elsewhere", logger)


def test_stamp_includes_patch_digest(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ref = "deadbeef"
    with_patch = wv._agentx_harness_stamp(ref)
    assert with_patch.startswith(f"{ref}+patches:")
    (wv._AGENTX_HARNESS_PATCH_DIR / "0001-bump.patch").write_text("changed\n")
    assert wv._agentx_harness_stamp(ref) != with_patch
    monkeypatch.setattr(wv, "_AGENTX_HARNESS_PATCH_DIR", checkout / "no-patches")
    assert wv._agentx_harness_stamp(ref) == ref


def test_carried_aiperf_patch_is_present() -> None:
    names = [p.name for p in wv._agentx_harness_patches()]
    assert "0001-wait-for-all-service-replicas.patch" in names
