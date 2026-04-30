# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Unit tests for workflows.compose_config.

Covers version parsing, contract lookup, and sidecar writing. The compose-up /
compose-down wrappers are exercised via integration tests (run.py end-to-end
against real Docker), not here.
"""

import pytest


def test_module_imports():
    """Smoke test: the module imports cleanly."""
    import workflows.compose_config  # noqa: F401


from packaging.version import Version

from workflows.compose_config import parse_image_version


class TestParseImageVersion:
    def test_release_tag_with_build_suffix(self):
        image = "ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.11.0-fae3df"
        assert parse_image_version(image) == Version("0.11.0")

    def test_release_tag_no_suffix(self):
        image = "ghcr.io/foo/bar:0.11.0"
        assert parse_image_version(image) == Version("0.11.0")

    def test_two_part_version(self):
        # Tag like "0.9-abc" should still parse — packaging.Version("0.9") is valid
        assert parse_image_version("foo/bar:0.9-abc") == Version("0.9")

    def test_dev_tag_returns_none(self):
        assert parse_image_version("ghcr.io/foo/bar:dev") is None

    def test_latest_tag_returns_none(self):
        assert parse_image_version("ghcr.io/foo/bar:latest") is None

    def test_no_tag_returns_none(self):
        assert parse_image_version("ghcr.io/foo/bar") is None

    def test_empty_string_returns_none(self):
        assert parse_image_version("") is None

    def test_version_in_image_path_not_tag_returns_none(self):
        # No `:` so no tag, returns None even though "0.11.0" appears in the path
        assert parse_image_version("ghcr.io/foo/0.11.0/bar") is None


from pathlib import Path

from workflows.compose_config import (
    NoMatchingContractError,
    lookup_contract,
)


@pytest.fixture
def contracts_yaml(tmp_path: Path) -> Path:
    """Two engines, each with two eras."""
    p = tmp_path / "contracts.yml"
    p.write_text(
        """\
contracts:
  vllm:
    - min_version: "0.0.0"
      file: docker-compose.vllm-pre-0.11.yml
    - min_version: "0.11.0"
      file: docker-compose.vllm-0.11.yml
  media:
    - min_version: "0.0.0"
      file: docker-compose.media-pre-0.11.yml
    - min_version: "0.11.0"
      file: docker-compose.media-0.11.yml
"""
    )
    return p


@pytest.fixture
def contracts_yaml_only_current(tmp_path: Path) -> Path:
    """Phase-1 state: only the current era is populated."""
    p = tmp_path / "contracts.yml"
    p.write_text(
        """\
contracts:
  vllm:
    - min_version: "0.11.0"
      file: docker-compose.vllm-0.11.yml
  media:
    - min_version: "0.11.0"
      file: docker-compose.media-0.11.yml
"""
    )
    return p


class TestLookupContract:
    def test_exact_match_picks_that_era(self, contracts_yaml):
        path = lookup_contract("vllm", Version("0.11.0"), contracts_path=contracts_yaml)
        assert path.name == "docker-compose.vllm-0.11.yml"

    def test_above_match_picks_that_era(self, contracts_yaml):
        path = lookup_contract("vllm", Version("0.13.5"), contracts_path=contracts_yaml)
        assert path.name == "docker-compose.vllm-0.11.yml"

    def test_below_match_picks_legacy_era(self, contracts_yaml):
        path = lookup_contract("vllm", Version("0.10.4"), contracts_path=contracts_yaml)
        assert path.name == "docker-compose.vllm-pre-0.11.yml"

    def test_pre_zero_picks_lowest_era(self, contracts_yaml):
        # 0.0.0 is the lower bound — anything >= it (which is everything)
        # picks pre-0.11 if there's nothing larger that fits.
        path = lookup_contract("vllm", Version("0.0.1"), contracts_path=contracts_yaml)
        assert path.name == "docker-compose.vllm-pre-0.11.yml"

    def test_media_engine_isolated_from_vllm(self, contracts_yaml):
        path = lookup_contract("media", Version("0.11.0"), contracts_path=contracts_yaml)
        assert path.name == "docker-compose.media-0.11.yml"

    def test_unknown_engine_raises(self, contracts_yaml):
        with pytest.raises(NoMatchingContractError, match="forge"):
            lookup_contract("forge", Version("0.11.0"), contracts_path=contracts_yaml)

    def test_none_version_falls_back_to_newest(self, contracts_yaml):
        path = lookup_contract("vllm", None, contracts_path=contracts_yaml)
        assert path.name == "docker-compose.vllm-0.11.yml"

    def test_below_oldest_era_raises(self, contracts_yaml_only_current):
        # contracts_yaml_only_current has no pre-0.11 entry; a 0.10.4 image
        # has no era that fits (smallest min_version is 0.11.0)
        with pytest.raises(NoMatchingContractError, match="0.10.4"):
            lookup_contract(
                "vllm", Version("0.10.4"), contracts_path=contracts_yaml_only_current
            )

    def test_returned_path_is_relative_to_contracts_dir(self, contracts_yaml):
        # The returned path lives next to contracts.yml, not as a bare basename
        path = lookup_contract("vllm", Version("0.11.0"), contracts_path=contracts_yaml)
        assert path.parent == contracts_yaml.parent


from workflows.compose_config import write_compose_files_sidecar


class TestWriteComposeFilesSidecar:
    def test_writes_dash_f_args(self, tmp_path: Path):
        files = [
            tmp_path / "docker-compose.vllm-0.11.yml",
            tmp_path / "overlays" / "dev-mode.yml",
        ]
        sidecar = tmp_path / ".env.compose.files"
        result = write_compose_files_sidecar(files, path=sidecar)
        assert result == sidecar
        contents = sidecar.read_text().strip()
        # Format: "-f /abs/...vllm-0.11.yml -f /abs/.../dev-mode.yml"
        assert contents.startswith("-f ")
        assert str(files[0]) in contents
        assert str(files[1]) in contents

    def test_single_file(self, tmp_path: Path):
        f = tmp_path / "docker-compose.vllm-0.11.yml"
        sidecar = tmp_path / ".env.compose.files"
        write_compose_files_sidecar([f], path=sidecar)
        assert sidecar.read_text().strip() == f"-f {f}"

    def test_empty_list_writes_empty(self, tmp_path: Path):
        sidecar = tmp_path / ".env.compose.files"
        write_compose_files_sidecar([], path=sidecar)
        assert sidecar.read_text() == "\n"
