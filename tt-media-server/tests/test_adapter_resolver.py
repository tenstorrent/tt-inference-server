# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import json
from unittest.mock import patch

import pytest
from utils.adapter_resolver import resolve_adapter


class TestResolveAdapter:
    def test_resolve_adapter(self, tmp_path):
        ckpt = tmp_path / "job123" / "ckpt-step-20"
        ckpt.mkdir(parents=True)
        config = {"base_model_name_or_path": "google/gemma-1.1-2b-it"}
        (ckpt / "adapter_config.json").write_text(json.dumps(config))

        with patch("utils.adapter_resolver.TRAINING_STORE_ADAPTERS_DIR", str(tmp_path)):
            info = resolve_adapter("job123/ckpt-step-20")

        assert info.base_model_name == "google/gemma-1.1-2b-it"
        assert info.adapter_path == str(ckpt)

    def test_resolve_adapter_errors(self, tmp_path):
        with patch("utils.adapter_resolver.TRAINING_STORE_ADAPTERS_DIR", str(tmp_path)):
            with pytest.raises(FileNotFoundError, match="Adapter not found"):
                resolve_adapter("nonexistent/ckpt")

        dir_only = tmp_path / "job" / "ckpt"
        dir_only.mkdir(parents=True)
        with patch("utils.adapter_resolver.TRAINING_STORE_ADAPTERS_DIR", str(tmp_path)):
            with pytest.raises(
                FileNotFoundError, match="adapter_config.json not found"
            ):
                resolve_adapter("job/ckpt")

        (dir_only / "adapter_config.json").write_text(json.dumps({"lora_alpha": 16}))
        with patch("utils.adapter_resolver.TRAINING_STORE_ADAPTERS_DIR", str(tmp_path)):
            with pytest.raises(ValueError, match="base_model_name_or_path missing"):
                resolve_adapter("job/ckpt")

        (dir_only / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": ""})
        )
        with patch("utils.adapter_resolver.TRAINING_STORE_ADAPTERS_DIR", str(tmp_path)):
            with pytest.raises(ValueError, match="base_model_name_or_path missing"):
                resolve_adapter("job/ckpt")

    def test_dataset_loader_loaded_from_metadata(self, tmp_path):
        ckpt = tmp_path / "job" / "ckpt"
        ckpt.mkdir(parents=True)
        (ckpt / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "google/gemma-1.1-2b-it"})
        )
        (ckpt / "dataset_metadata.json").write_text(
            json.dumps({"dataset_loader": "alpaca"})
        )
        with patch("utils.adapter_resolver.TRAINING_STORE_ADAPTERS_DIR", str(tmp_path)):
            info = resolve_adapter("job/ckpt")
        assert info.dataset_loader == "alpaca"
        assert info.base_model_name == "google/gemma-1.1-2b-it"
