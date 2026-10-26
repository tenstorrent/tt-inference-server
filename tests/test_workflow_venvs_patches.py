# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Patches carried in workflows/patches/inferencex are applied to the InferenceX
checkout after the submodule update, exactly once, and change the ref stamp."""

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
    """A fake InferenceX checkout with a git repo at utils/aiperf and one patch for it."""
    repo = tmp_path / "InferenceX"
    aiperf = repo / "utils" / "aiperf"
    aiperf.mkdir(parents=True)
    _git(aiperf, "init", "-q")
    _git(aiperf, "config", "user.email", "t@example.com")
    _git(aiperf, "config", "user.name", "t")
    (aiperf / "mod.py").write_text("VALUE = 1\n")
    _git(aiperf, "add", "mod.py")
    _git(aiperf, "commit", "-q", "-m", "base")
    (aiperf / "mod.py").write_text("VALUE = 2\n")
    patch = _git(aiperf, "diff")
    _git(aiperf, "checkout", "--", "mod.py")

    patch_dir = tmp_path / "patches" / "aiperf"
    patch_dir.mkdir(parents=True)
    (patch_dir / "0001-bump.patch").write_text(patch)
    monkeypatch.setattr(wv, "_INFERENCEX_PATCH_DIR", tmp_path / "patches")
    return repo


def test_patch_applies_once(checkout: Path) -> None:
    target = checkout / "utils" / "aiperf" / "mod.py"
    assert wv._apply_inferencex_patches(checkout, logger)
    assert target.read_text() == "VALUE = 2\n"
    # Second pass sees the patch already applied and leaves the tree alone.
    assert wv._apply_inferencex_patches(checkout, logger)
    assert target.read_text() == "VALUE = 2\n"


def test_patch_that_does_not_apply_fails_setup(checkout: Path) -> None:
    (checkout / "utils" / "aiperf" / "mod.py").write_text("VALUE = 3\n")
    assert not wv._apply_inferencex_patches(checkout, logger)


def test_missing_target_fails_setup(checkout: Path) -> None:
    assert not wv._apply_inferencex_patches(checkout / "elsewhere", logger)


def test_unpopulated_submodule_fails_setup(tmp_path: Path, checkout: Path) -> None:
    # A superproject whose submodule directory exists but was never checked out:
    # `git apply` from inside it would ignore the patch and report success.
    superproject = tmp_path / "super"
    (superproject / "utils" / "aiperf").mkdir(parents=True)
    _git(superproject, "init", "-q")
    assert not wv._apply_inferencex_patches(superproject, logger)


def test_stamp_includes_patch_digest(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ref = "deadbeef"
    with_patch = wv._inferencex_stamp(ref)
    assert with_patch.startswith(f"{ref}+patches:")
    patch_dir = wv._INFERENCEX_PATCH_DIR / "aiperf"
    (patch_dir / "0001-bump.patch").write_text("changed\n")
    assert wv._inferencex_stamp(ref) != with_patch
    monkeypatch.setattr(wv, "_INFERENCEX_PATCH_DIR", checkout / "no-patches")
    assert wv._inferencex_stamp(ref) == ref


def test_carried_aiperf_patch_is_present() -> None:
    patches = wv._inferencex_patches()
    names = [p.name for p, _ in patches]
    assert "0001-wait-for-all-service-replicas.patch" in names
    assert all(target == Path("utils") / "aiperf" for _, target in patches)
