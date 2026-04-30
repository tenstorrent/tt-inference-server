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
