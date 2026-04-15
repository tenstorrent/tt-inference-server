# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import json
from unittest.mock import MagicMock, patch

import pytest

from utils.adapter_resolver import resolve_adapter
from utils.adapter_storage import LocalAdapterStorage


class TestResolveAdapter:
    def test_resolve_adapter(self, tmp_path):
        ckpt = tmp_path / "job123" / "ckpt-step-20"
        ckpt.mkdir(parents=True)
        config = {"base_model_name_or_path": "google/gemma-1.1-2b-it"}
        (ckpt / "adapter_config.json").write_text(json.dumps(config))

        storage = LocalAdapterStorage(base_dir=str(tmp_path))
        info = resolve_adapter("job123/ckpt-step-20", storage=storage)

        assert info.base_model_name == "google/gemma-1.1-2b-it"
        assert info.adapter_path == str(ckpt)

    def test_resolve_adapter_errors(self, tmp_path):
        storage = LocalAdapterStorage(base_dir=str(tmp_path))

        with pytest.raises(FileNotFoundError, match="Adapter not found"):
            resolve_adapter("nonexistent/ckpt", storage=storage)

        dir_only = tmp_path / "job" / "ckpt"
        dir_only.mkdir(parents=True)
        with pytest.raises(
            FileNotFoundError, match="adapter_config.json not found"
        ):
            resolve_adapter("job/ckpt", storage=storage)

        (dir_only / "adapter_config.json").write_text(json.dumps({"lora_alpha": 16}))
        with pytest.raises(ValueError, match="base_model_name_or_path missing"):
            resolve_adapter("job/ckpt", storage=storage)

        (dir_only / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": ""})
        )
        with pytest.raises(ValueError, match="base_model_name_or_path missing"):
            resolve_adapter("job/ckpt", storage=storage)

    def test_resolve_adapter_uses_default_storage(self, tmp_path):
        ckpt = tmp_path / "job123" / "ckpt-step-20"
        ckpt.mkdir(parents=True)
        config = {"base_model_name_or_path": "google/gemma-1.1-2b-it"}
        (ckpt / "adapter_config.json").write_text(json.dumps(config))

        mock_storage = LocalAdapterStorage(base_dir=str(tmp_path))
        with patch(
            "utils.adapter_resolver.get_adapter_storage",
            return_value=mock_storage,
        ):
            info = resolve_adapter("job123/ckpt-step-20")

        assert info.base_model_name == "google/gemma-1.1-2b-it"
