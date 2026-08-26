"""The release artifact zip has to open in a stock macOS Finder.

`package()` writes entries with create_system=3 (Unix), which tells an extractor
to read the external_attr high word as st_mode. A mode without file-type bits
declares "type unknown": unzip and Keka fall back to a default and open the
archive anyway, but macOS Archive Utility honours the declaration and rejects the
whole archive as an unsupported format. Every release up to v0.20.0 shipped an
asset that no macOS user could open by double-clicking it.
"""

import stat
import sys
import zipfile
from pathlib import Path

import pytest

# build_release_artifacts.py is written to run as a script and does
# `from _bootstrap import REPO_ROOT`, which only resolves with its own directory
# on sys.path. Import it the way the release pipeline does rather than changing
# the module just to make it importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "release"))

from scripts.release.build_release_artifacts import package  # noqa: E402


@pytest.fixture
def staged(tmp_path: Path) -> dict[str, Path]:
    bundle = tmp_path / "workflow_logs_release_Qwen__Qwen3-32B_galaxy.zip"
    with zipfile.ZipFile(bundle, "w") as zf:
        zf.writestr("run_logs/run.log", "ok")
    return {bundle.name: bundle}


def _modes(zip_path: Path) -> dict[str, int]:
    with zipfile.ZipFile(zip_path) as zf:
        return {i.filename: i.external_attr >> 16 for i in zf.infolist()}


def test_file_entries_declare_a_regular_file_type(tmp_path, staged):
    out = package("v0.0.0", staged, tmp_path)
    for name, mode in _modes(out).items():
        if name.endswith("/"):
            continue
        assert stat.S_ISREG(mode), (
            f"{name} has mode 0o{mode:06o} with file type 0o{mode & 0o170000:06o}; "
            "Archive Utility rejects the archive when the type bits are missing"
        )


def test_directory_entry_declares_a_directory_type(tmp_path, staged):
    out = package("v0.0.0", staged, tmp_path)
    dirs = [n for n in _modes(out) if n.endswith("/")]
    assert dirs, "the package should carry an explicit top-level directory entry"
    for name in dirs:
        assert stat.S_ISDIR(_modes(out)[name])


def test_permissions_are_preserved_alongside_the_type_bits(tmp_path, staged):
    """The fix must add S_IFREG without disturbing the 0644 it already set."""
    out = package("v0.0.0", staged, tmp_path)
    for name, mode in _modes(out).items():
        if not name.endswith("/"):
            assert stat.S_IMODE(mode) == 0o644


def test_inner_bundles_are_stored_not_recompressed(tmp_path, staged):
    """Bundles are already-compressed zips; storing them keeps packaging cheap."""
    out = package("v0.0.0", staged, tmp_path)
    with zipfile.ZipFile(out) as zf:
        for info in zf.infolist():
            assert info.compress_type == zipfile.ZIP_STORED
