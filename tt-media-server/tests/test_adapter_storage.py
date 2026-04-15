# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from utils.adapter_storage import (
    AdapterInfo,
    AdapterStorage,
    HfHubAdapterStorage,
    LocalAdapterStorage,
    get_adapter_storage,
)


class TestLocalAdapterStorage:
    @pytest.fixture
    def storage(self, tmp_path):
        return LocalAdapterStorage(base_dir=str(tmp_path))

    def test_save_checkpoint(self, storage, tmp_path):
        peft_model = MagicMock()
        peft_model.state_dict.return_value = {}

        ref = storage.save_checkpoint(peft_model, "job-1", "ckpt-step-10")

        expected_path = str(tmp_path / "job-1" / "ckpt-step-10")
        assert ref == expected_path
        assert os.path.isdir(expected_path)
        peft_model.save_pretrained.assert_called_once_with(expected_path, state_dict={})

    def test_resolve_adapter_success(self, storage, tmp_path):
        ckpt = tmp_path / "job-1" / "ckpt-step-10"
        ckpt.mkdir(parents=True)
        config = {"base_model_name_or_path": "meta-llama/Llama-3.1-8B"}
        (ckpt / "adapter_config.json").write_text(json.dumps(config))

        info = storage.resolve_adapter("job-1/ckpt-step-10")

        assert info == AdapterInfo(
            base_model_name="meta-llama/Llama-3.1-8B",
            adapter_path=str(ckpt),
        )

    def test_resolve_adapter_missing_dir(self, storage):
        with pytest.raises(FileNotFoundError, match="Adapter not found"):
            storage.resolve_adapter("nonexistent/ckpt")

    def test_resolve_adapter_missing_config(self, storage, tmp_path):
        (tmp_path / "job" / "ckpt").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="adapter_config.json not found"):
            storage.resolve_adapter("job/ckpt")

    def test_resolve_adapter_missing_base_model(self, storage, tmp_path):
        ckpt = tmp_path / "job" / "ckpt"
        ckpt.mkdir(parents=True)
        (ckpt / "adapter_config.json").write_text(json.dumps({"lora_alpha": 16}))
        with pytest.raises(ValueError, match="base_model_name_or_path missing"):
            storage.resolve_adapter("job/ckpt")

    def test_resolve_adapter_empty_base_model(self, storage, tmp_path):
        ckpt = tmp_path / "job" / "ckpt"
        ckpt.mkdir(parents=True)
        (ckpt / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": ""})
        )
        with pytest.raises(ValueError, match="base_model_name_or_path missing"):
            storage.resolve_adapter("job/ckpt")

    def test_get_checkpoint_path_exists(self, storage, tmp_path):
        ckpt = tmp_path / "job-1" / "ckpt-step-10"
        ckpt.mkdir(parents=True)
        assert storage.get_checkpoint_path("job-1", "ckpt-step-10") == str(ckpt)

    def test_get_checkpoint_path_missing(self, storage):
        assert storage.get_checkpoint_path("job-1", "ckpt-step-99") is None

    def test_ensure_job_dir(self, storage, tmp_path):
        path = storage.ensure_job_dir("new-job")
        expected = str(tmp_path / "new-job")
        assert path == expected
        assert os.path.isdir(expected)

    def test_ensure_job_dir_idempotent(self, storage, tmp_path):
        storage.ensure_job_dir("job-1")
        storage.ensure_job_dir("job-1")
        assert os.path.isdir(tmp_path / "job-1")


class TestHfHubAdapterStorage:
    def test_init_requires_hf_org(self):
        with pytest.raises(ValueError, match="hf_adapter_org must be set"):
            HfHubAdapterStorage(hf_org="", token="tok")

    def test_save_checkpoint(self):
        mock_api = MagicMock()
        with patch("utils.adapter_storage.HfApi", return_value=mock_api):
            storage = HfHubAdapterStorage(hf_org="my-org", token="tok")

        peft_model = MagicMock()
        peft_model.state_dict.return_value = {}

        ref = storage.save_checkpoint(peft_model, "job-1", "ckpt-step-10")

        assert ref == "my-org/job-1/ckpt-step-10"
        mock_api.create_repo.assert_called_once_with(
            "my-org/job-1", exist_ok=True, private=True
        )
        mock_api.upload_folder.assert_called_once()
        upload_kwargs = mock_api.upload_folder.call_args[1]
        assert upload_kwargs["repo_id"] == "my-org/job-1"
        assert upload_kwargs["path_in_repo"] == "ckpt-step-10"

    @patch("utils.adapter_storage.HfApi", return_value=MagicMock())
    def test_resolve_adapter(self, _mock_api_cls, tmp_path):
        storage = HfHubAdapterStorage(hf_org="my-org", token="tok")

        ckpt_dir = tmp_path / "ckpt-step-10"
        ckpt_dir.mkdir()
        config = {"base_model_name_or_path": "meta-llama/Llama-3.1-8B"}
        (ckpt_dir / "adapter_config.json").write_text(json.dumps(config))

        with patch(
            "huggingface_hub.snapshot_download", return_value=str(tmp_path)
        ):
            info = storage.resolve_adapter("job-1/ckpt-step-10")

        assert info.base_model_name == "meta-llama/Llama-3.1-8B"
        assert info.adapter_path == str(ckpt_dir)

    @patch("utils.adapter_storage.HfApi", return_value=MagicMock())
    def test_ensure_job_dir(self, _mock_api_cls):
        storage = HfHubAdapterStorage(hf_org="my-org", token="tok")
        repo_id = storage.ensure_job_dir("job-1")
        assert repo_id == "my-org/job-1"
        storage.api.create_repo.assert_called_once_with(
            "my-org/job-1", exist_ok=True, private=True
        )

    @patch("utils.adapter_storage.HfApi", return_value=MagicMock())
    def test_get_checkpoint_path_success(self, _mock_api_cls, tmp_path):
        storage = HfHubAdapterStorage(hf_org="my-org", token="tok")
        ckpt_dir = tmp_path / "ckpt-step-10"
        ckpt_dir.mkdir()

        with patch(
            "huggingface_hub.snapshot_download", return_value=str(tmp_path)
        ):
            result = storage.get_checkpoint_path("job-1", "ckpt-step-10")

        assert result == str(ckpt_dir)

    @patch("utils.adapter_storage.HfApi", return_value=MagicMock())
    def test_get_checkpoint_path_download_fails(self, _mock_api_cls):
        storage = HfHubAdapterStorage(hf_org="my-org", token="tok")
        with patch(
            "huggingface_hub.snapshot_download",
            side_effect=Exception("not found"),
        ):
            assert storage.get_checkpoint_path("job-1", "ckpt-step-10") is None


class TestGetAdapterStorage:
    def test_returns_local_by_default(self):
        mock_settings = MagicMock()
        mock_settings.adapter_storage_backend = "local"
        with patch("utils.adapter_storage.get_settings", return_value=mock_settings):
            storage = get_adapter_storage()
        assert isinstance(storage, LocalAdapterStorage)

    def test_returns_hf_hub(self):
        mock_settings = MagicMock()
        mock_settings.adapter_storage_backend = "hf_hub"
        mock_settings.hf_adapter_org = "test-org"
        with patch("utils.adapter_storage.get_settings", return_value=mock_settings):
            with patch("utils.adapter_storage.HfApi", return_value=MagicMock()):
                storage = get_adapter_storage()
        assert isinstance(storage, HfHubAdapterStorage)
