# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from unittest.mock import patch

import pytest

from utils.lora_utils import _find_safetensors_filename, resolve_lora_path

REPO_ID = "artificialguybr/ColoringBookRedmond-V2"
LORA_FILENAME = "ColoringBookRedmond-ColoringBook-ColoringBookAF.safetensors"


class TestFindSafetensorsFilename:
    def test_single_safetensors_file(self):
        with patch(
            "utils.lora_utils.list_repo_files",
            return_value=["README.md", LORA_FILENAME],
        ):
            assert _find_safetensors_filename(REPO_ID) == LORA_FILENAME

    def test_no_safetensors_file_raises(self):
        with patch(
            "utils.lora_utils.list_repo_files",
            return_value=["README.md", "config.json"],
        ):
            with pytest.raises(FileNotFoundError, match="No .safetensors file found"):
                _find_safetensors_filename(REPO_ID)

    def test_multiple_files_prefers_root(self):
        with patch(
            "utils.lora_utils.list_repo_files",
            return_value=[
                "subfolder/other.safetensors",
                "root_model.safetensors",
                "README.md",
            ],
        ):
            assert _find_safetensors_filename(REPO_ID) == "root_model.safetensors"

    def test_multiple_root_files_returns_first(self):
        with patch(
            "utils.lora_utils.list_repo_files",
            return_value=["b_model.safetensors", "a_model.safetensors"],
        ):
            assert _find_safetensors_filename(REPO_ID) == "b_model.safetensors"

    def test_all_nested_returns_first(self):
        with patch(
            "utils.lora_utils.list_repo_files",
            return_value=[
                "v1/model.safetensors",
                "v2/model.safetensors",
            ],
        ):
            assert _find_safetensors_filename(REPO_ID) == "v1/model.safetensors"


class TestResolveLoraPath:
    def test_local_file_returns_resolved_path(self, tmp_path):
        lora_file = tmp_path / "adapter.safetensors"
        lora_file.write_bytes(b"fake weights")

        result = resolve_lora_path(str(lora_file))
        assert result == str(lora_file.resolve())

    def test_hf_repo_delegates_to_hf_hub_download(self, tmp_path):
        hf_cached = str(tmp_path / "hub" / "snapshots" / "abc123" / LORA_FILENAME)

        with patch(
            "utils.lora_utils.list_repo_files",
            return_value=[LORA_FILENAME],
        ):
            with patch(
                "utils.lora_utils.hf_hub_download",
                return_value=hf_cached,
            ) as mock_download:
                result = resolve_lora_path(REPO_ID)

        assert result == hf_cached
        mock_download.assert_called_once_with(
            repo_id=REPO_ID,
            filename=LORA_FILENAME,
        )

    def test_nonexistent_local_path_treated_as_hf_repo(self, tmp_path):
        hf_cached = str(tmp_path / "hub" / "snapshots" / "abc123" / "model.safetensors")

        with patch(
            "utils.lora_utils.list_repo_files",
            return_value=["model.safetensors"],
        ):
            with patch(
                "utils.lora_utils.hf_hub_download",
                return_value=hf_cached,
            ):
                result = resolve_lora_path("/nonexistent/path.safetensors")

        assert result == hf_cached

    def test_download_failure_propagates(self):
        with patch(
            "utils.lora_utils.list_repo_files",
            return_value=["model.safetensors"],
        ):
            with patch(
                "utils.lora_utils.hf_hub_download",
                side_effect=OSError("Network error"),
            ):
                with pytest.raises(OSError, match="Network error"):
                    resolve_lora_path("bad/repo")

    def test_list_files_failure_propagates(self):
        with patch(
            "utils.lora_utils.list_repo_files",
            side_effect=OSError("Network error"),
        ):
            with pytest.raises(OSError, match="Network error"):
                resolve_lora_path("bad/repo")
