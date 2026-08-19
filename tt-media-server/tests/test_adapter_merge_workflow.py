# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""
Integration test for the LoRA adapter -> base model merge workflow.
"""

import json
import os
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

from config.constants import ModelNames, SupportedModels

BASE_REPO = SupportedModels.LLAMA_3_1_8B.value


def fake_run_merge_subprocess(
    base_model_name,
    adapter_path,
    output_dir,
    *,
    python_executable=None,
    cwd=None,
    dtype_str="torch.bfloat16",
):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump({"base_model": base_model_name}, f)


class FakeJobManager:
    """Records the source job's single checkpoint; create_job just returns."""

    def __init__(self, checkpoints, result_path):
        self._checkpoints = checkpoints
        self._result_path = result_path

    async def create_job(self, **kwargs):
        return {"job_id": kwargs["job_id"]}

    def get_job_checkpoints(self, job_id, org_id=None):
        return self._checkpoints

    def get_job_result_path(self, job_id, org_id=None):
        return self._result_path


@pytest.fixture
def merge_service(tmp_path, monkeypatch):
    """A TrainingService wired to a FakeJobManager, with heavy init patched out."""
    monkeypatch.setenv("CACHE_ROOT", str(tmp_path))
    settings = MagicMock()
    settings.training_model = ModelNames.LLAMA_3_1_8B.value
    settings.download_weights_from_service = False
    fake_jm = FakeJobManager(checkpoints=[], result_path=None)
    with ExitStack() as stack:
        stack.enter_context(
            patch("model_services.training_service.get_settings", return_value=settings)
        )
        stack.enter_context(patch("model_services.base_service.get_scheduler"))
        stack.enter_context(patch("model_services.base_service.TTLogger"))
        stack.enter_context(patch("model_services.base_service.HuggingFaceUtils"))
        stack.enter_context(patch("model_services.base_service.settings", settings))
        stack.enter_context(patch("model_services.base_job_service.settings", settings))
        stack.enter_context(
            patch(
                "model_services.base_job_service.get_job_manager", return_value=fake_jm
            )
        )
        from model_services.training_service import TrainingService

        yield TrainingService(), fake_jm


def test_is_within_dir(tmp_path):
    from model_services.training_service import TrainingService

    base = str(tmp_path)
    assert TrainingService._is_within_dir(base, str(tmp_path / "a" / "b")) is True
    assert TrainingService._is_within_dir(base, str(tmp_path / ".." / "evil")) is False
    assert TrainingService._is_within_dir(base, "/etc") is False
    assert TrainingService._is_within_dir(base, base) is False


def test_get_checkpoint_download_path_rejects_traversal(merge_service, tmp_path):
    service, fake_jm = merge_service
    # The malicious id is even allowlisted, yet the containment guard rejects it.
    fake_jm._checkpoints = [{"id": "../../evil"}]
    fake_jm._result_path = str(tmp_path / "adapters" / "train-1")
    assert service.get_checkpoint_download_path("train-1", "../../evil") is None


@pytest.mark.asyncio
async def test_adapter_merge_workflow_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_ROOT", str(tmp_path))

    # A completed training job with one adapter checkpoint on disk.
    checkpoint_dir = tmp_path / "adapters" / "train-1" / "ckpt-100"
    checkpoint_dir.mkdir(parents=True)
    fake_jm = FakeJobManager(
        checkpoints=[{"id": "ckpt-100"}],
        result_path=str(tmp_path / "adapters" / "train-1"),
    )

    settings = MagicMock()
    settings.training_model = ModelNames.LLAMA_3_1_8B.value
    settings.download_weights_from_service = False

    with ExitStack() as stack:
        stack.enter_context(
            patch("model_services.training_service.get_settings", return_value=settings)
        )
        stack.enter_context(patch("model_services.base_service.get_scheduler"))
        stack.enter_context(patch("model_services.base_service.TTLogger"))
        stack.enter_context(patch("model_services.base_service.HuggingFaceUtils"))
        stack.enter_context(patch("model_services.base_service.settings", settings))
        stack.enter_context(patch("model_services.base_job_service.settings", settings))
        stack.enter_context(
            patch(
                "model_services.base_job_service.get_job_manager", return_value=fake_jm
            )
        )
        # Route the heavy merge (normally a 4.x-venv subprocess) to a stub.
        stack.enter_context(
            patch(
                "model_services.training_service.run_merge_subprocess",
                fake_run_merge_subprocess,
            )
        )

        from domain.adapter_merge_request import AdapterMergeRequest
        from model_services.training_service import (
            MERGE_INFO_FILE_NAME,
            TrainingService,
        )

        service = TrainingService()
        request = AdapterMergeRequest(source_job_id="train-1", checkpoint_id="ckpt-100")

        await service.create_adapter_merge_job(request)
        output_dir = await service.run_adapter_merge(request)

    # Adapter checkpoint resolved and merge ran in the subprocess for this model.
    assert request._adapter_path == str(checkpoint_dir)
    # Merged checkpoint is named after the model it came from, then the merge id.
    assert os.path.basename(output_dir) == (
        f"{ModelNames.LLAMA_3_1_8B.value}-{request._task_id}"
    )
    assert os.path.dirname(output_dir) == str(tmp_path / "merged_models")
    assert json.load(open(os.path.join(output_dir, "config.json")))["base_model"] == (
        BASE_REPO
    )

    # Provenance file records the merge correctly.
    info = json.load(open(os.path.join(output_dir, MERGE_INFO_FILE_NAME)))
    assert info["merge_id"] == request._task_id
    assert info["model"] == BASE_REPO
    assert info["source_job_id"] == "train-1"
    assert info["checkpoint_id"] == "ckpt-100"
