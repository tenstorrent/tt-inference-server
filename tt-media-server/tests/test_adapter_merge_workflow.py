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
from utils.adapter_merge_utils import MergeResult

BASE_REPO = SupportedModels.LLAMA_3_1_8B.value
BASE_REVISION = "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"


def fake_merge_adapter(base_model_name, adapter_path, output_dir, dtype_str=None):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump({"base_model": base_model_name}, f)
    return MergeResult(output_dir=output_dir, base_revision=BASE_REVISION)


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
        # Route the heavy merge to the importable stub (survives the spawn).
        stack.enter_context(
            patch("model_services.training_service.merge_adapter", fake_merge_adapter)
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
    assert json.load(open(os.path.join(output_dir, "config.json")))["base_model"] == (
        BASE_REPO
    )

    # Provenance file records the merge correctly.
    info = json.load(open(os.path.join(output_dir, MERGE_INFO_FILE_NAME)))
    assert info["merge_id"] == request._task_id
    assert info["model"] == BASE_REPO
    assert info["base_revision"] == BASE_REVISION
    assert info["source_job_id"] == "train-1"
    assert info["checkpoint_id"] == "ckpt-100"
