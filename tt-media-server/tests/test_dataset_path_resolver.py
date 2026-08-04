# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import os
from unittest.mock import MagicMock, patch

import pytest
from utils.dataset_path_resolver import dataset_file_type, resolve_dataset_path

RESOLVER_MODULE = "utils.dataset_path_resolver"


@pytest.fixture
def datasets_dir(tmp_path):
    """A dataset directory holding one file, with a secret alongside it."""
    base = tmp_path / "datasets"
    (base / "nested").mkdir(parents=True)
    (base / "train.json").write_text("{}")
    (base / "nested" / "train.jsonl").write_text("{}")
    (tmp_path / "secret.json").write_text("not yours")

    settings = MagicMock()
    settings.training_datasets_dir = str(base)
    with patch(f"{RESOLVER_MODULE}.get_settings", return_value=settings):
        yield tmp_path


class TestResolveDatasetPath:
    def test_resolves_a_relative_path(self, datasets_dir):
        resolved = resolve_dataset_path("train.json")
        assert resolved == str(datasets_dir / "datasets" / "train.json")

    def test_resolves_a_nested_path(self, datasets_dir):
        assert resolve_dataset_path("nested/train.jsonl") == str(
            datasets_dir / "datasets" / "nested" / "train.jsonl"
        )

    def test_accepts_an_absolute_path_inside_the_directory(self, datasets_dir):
        inside = str(datasets_dir / "datasets" / "train.json")
        assert resolve_dataset_path(inside) == inside

    def test_rejects_traversal_out_of_the_directory(self, datasets_dir):
        with pytest.raises(ValueError, match="resolves outside"):
            resolve_dataset_path("../secret.json")

    def test_rejects_an_absolute_path_outside_the_directory(self, datasets_dir):
        with pytest.raises(ValueError, match="resolves outside"):
            resolve_dataset_path(str(datasets_dir / "secret.json"))

    def test_rejects_a_symlink_escaping_the_directory(self, datasets_dir):
        # The link itself lives inside the directory, so this only fails if
        # symlinks are resolved before the containment check.
        link = datasets_dir / "datasets" / "escape.json"
        os.symlink(datasets_dir / "secret.json", link)

        with pytest.raises(ValueError, match="resolves outside"):
            resolve_dataset_path("escape.json")

    def test_rejects_a_missing_file(self, datasets_dir):
        with pytest.raises(FileNotFoundError, match="not found"):
            resolve_dataset_path("absent.json")

    def test_rejects_a_directory(self, datasets_dir):
        with pytest.raises(FileNotFoundError, match="not found"):
            resolve_dataset_path("nested")


class TestDatasetFileType:
    @pytest.mark.parametrize(
        "path, expected",
        [
            ("train.json", "json"),
            ("train.jsonl", "jsonl"),
            ("/data/TRAIN.JSON", "json"),
        ],
    )
    def test_names_the_loader(self, path, expected):
        assert dataset_file_type(path) == expected

    @pytest.mark.parametrize("path", ["train.csv", "train.parquet", "train"])
    def test_rejects_other_files(self, path):
        with pytest.raises(ValueError, match="Unsupported dataset file"):
            dataset_file_type(path)
