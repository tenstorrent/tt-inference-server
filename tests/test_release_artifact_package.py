# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""The release package layout is a compatibility surface, so pin it.

Every published asset from v0.13.0 on has the same shape: one top-level folder
named after the archive, one uncompressed ``workflow_logs_release_*.zip`` per
released leaf, each bundle carried exactly as tt-shield produced it. Consumers
depend on that, and it survived a change of packaging tool at v0.16.0 without
anyone testing it. These tests exist so that changing it has to be deliberate.
"""

import sys
import zipfile
from pathlib import Path

import pytest

# build_release_artifacts.py runs as a script and resolves its own imports off
# sys.path; import it the way the release pipeline does.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "release"))

from scripts.release.build_release_artifacts import package  # noqa: E402


BUNDLES = (
    "workflow_logs_release_Qwen__Qwen3-32B_galaxy.zip",
    "workflow_logs_release_google__diffusiongemma-26B-A4B-it_p300x2.zip",
)


@pytest.fixture
def staged(tmp_path: Path) -> dict[str, Path]:
    out = {}
    for name in BUNDLES:
        src = tmp_path / name
        with zipfile.ZipFile(src, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("run_logs/run.log", f"log for {name}\n")
            zf.writestr("runtime_model_specs/spec.json", '{"ok": true}')
        out[name] = src
    return out


def _entries(path: Path) -> dict[str, zipfile.ZipInfo]:
    with zipfile.ZipFile(path) as zf:
        return {i.filename: i for i in zf.infolist()}


def test_layout_is_one_folder_holding_the_bundles(tmp_path, staged):
    out = package("v0.0.0", staged, tmp_path)
    assert out.name == "v0.0.0-release_artifacts.zip"
    assert set(_entries(out)) == {"v0.0.0-release_artifacts/"} | {
        f"v0.0.0-release_artifacts/{name}" for name in BUNDLES
    }


def test_bundles_are_carried_byte_for_byte(tmp_path, staged):
    """The bundle is what tt-shield produced; packaging must not rewrite it."""
    out = package("v0.0.0", staged, tmp_path)
    with zipfile.ZipFile(out) as zf:
        for name, src in staged.items():
            assert zf.read(f"v0.0.0-release_artifacts/{name}") == src.read_bytes()


def test_entries_are_stored_not_recompressed(tmp_path, staged):
    """The bundles are already deflate zips; recompressing them buys under 1%."""
    out = package("v0.0.0", staged, tmp_path)
    for info in _entries(out).values():
        assert info.compress_type == zipfile.ZIP_STORED
