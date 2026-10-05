# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""--host-hf-cache must mount enough of the HF cache for snapshot symlinks to resolve."""

import os
from pathlib import Path

from workflows.setup_host import widen_mount_for_symlinks


def _hub_v1_layout(root: Path) -> Path:
    """Classic layout: snapshots/<rev>/f -> ../../blobs/<hash> (inside the repo dir)."""
    repo = root / "hub" / "models--org--name"
    (repo / "blobs").mkdir(parents=True)
    (repo / "blobs" / "aaa").write_text("x")
    snap = repo / "snapshots" / "rev"
    snap.mkdir(parents=True)
    os.symlink("../../blobs/aaa", snap / "model.safetensors")
    return snap


def _hub_v2_layout(root: Path) -> Path:
    """huggingface_hub >= 2: repo blobs/<hash> -> ../../blobs/<xx>/<sha> (shared store)."""
    hub = root / "hub"
    (hub / "blobs" / "86").mkdir(parents=True)
    (hub / "blobs" / "86" / "86abc").write_text("x")
    repo = hub / "models--org--name"
    (repo / "blobs").mkdir(parents=True)
    os.symlink("../../blobs/86/86abc", repo / "blobs" / "0189")
    (repo / "blobs" / "cfg").write_text("{}")
    snap = repo / "snapshots" / "rev"
    snap.mkdir(parents=True)
    os.symlink("../../blobs/0189", snap / "model.safetensors")
    os.symlink("../../blobs/cfg", snap / "config.json")
    return snap


def test_classic_layout_keeps_repo_dir(tmp_path):
    snap = _hub_v1_layout(tmp_path)
    repo = snap.parent.parent
    assert widen_mount_for_symlinks(snap, repo) == repo


def test_shared_blob_store_widens_to_hub(tmp_path):
    snap = _hub_v2_layout(tmp_path)
    repo = snap.parent.parent
    mount = widen_mount_for_symlinks(snap, repo)
    assert mount == (tmp_path / "hub").resolve()
    # every snapshot file resolves inside the widened mount
    for f in snap.iterdir():
        assert f.resolve().is_relative_to(mount)
    # and the container-side relative path still points at the snapshot
    assert snap.relative_to(mount) == Path("models--org--name/snapshots/rev")


def test_missing_snapshot_is_a_noop(tmp_path):
    assert widen_mount_for_symlinks(tmp_path / "nope", tmp_path) == tmp_path
