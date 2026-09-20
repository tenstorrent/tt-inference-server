# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import json

import pytest
from utils.adapter_resolver import resolve_adapter


class TestResolveAdapter:
    def test_resolve_adapter(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CACHE_ROOT", str(tmp_path))
        ckpt = tmp_path / "adapters" / "job123" / "ckpt-step-20"
        ckpt.mkdir(parents=True)
        config = {"base_model_name_or_path": "google/gemma-1.1-2b-it"}
        (ckpt / "adapter_config.json").write_text(json.dumps(config))

        info = resolve_adapter("job123/ckpt-step-20")

        assert info.base_model_name == "google/gemma-1.1-2b-it"
        assert info.adapter_path == str(ckpt)

    def test_resolve_adapter_uses_cache_root_not_cwd(self, tmp_path, monkeypatch):
        """Adapters are looked up under $CACHE_ROOT/adapters, matching where
        TrainingService writes them, rather than relative to the process CWD."""
        cache_root = tmp_path / "cache_root"
        ckpt = cache_root / "adapters" / "job" / "ckpt"
        ckpt.mkdir(parents=True)
        (ckpt / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "google/gemma-1.1-2b-it"})
        )
        # CWD is elsewhere; only CACHE_ROOT points at the adapters store.
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("CACHE_ROOT", str(cache_root))

        info = resolve_adapter("job/ckpt")

        assert info.adapter_path == str(ckpt)

    def test_resolve_adapter_rejects_traversal(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CACHE_ROOT", str(tmp_path))
        with pytest.raises(ValueError, match="outside adapters root"):
            resolve_adapter("../evil")

    def test_resolve_adapter_errors(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CACHE_ROOT", str(tmp_path))
        with pytest.raises(FileNotFoundError, match="Adapter not found"):
            resolve_adapter("nonexistent/ckpt")

        dir_only = tmp_path / "adapters" / "job" / "ckpt"
        dir_only.mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="adapter_config.json not found"):
            resolve_adapter("job/ckpt")

        (dir_only / "adapter_config.json").write_text(json.dumps({"lora_alpha": 16}))
        with pytest.raises(ValueError, match="base_model_name_or_path missing"):
            resolve_adapter("job/ckpt")

        (dir_only / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": ""})
        )
        with pytest.raises(ValueError, match="base_model_name_or_path missing"):
            resolve_adapter("job/ckpt")

    def test_dataset_loader_loaded_from_metadata(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CACHE_ROOT", str(tmp_path))
        ckpt = tmp_path / "adapters" / "job" / "ckpt"
        ckpt.mkdir(parents=True)
        (ckpt / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "google/gemma-1.1-2b-it"})
        )
        (ckpt / "dataset_metadata.json").write_text(
            json.dumps({"dataset_loader": "alpaca"})
        )

        info = resolve_adapter("job/ckpt")

        assert info.dataset_loader == "alpaca"
        assert info.base_model_name == "google/gemma-1.1-2b-it"
