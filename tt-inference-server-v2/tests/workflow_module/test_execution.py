# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for ``workflow_module.execution`` result + options dataclasses."""

from __future__ import annotations

import json

from unittest.mock import MagicMock

from report_module import ReportSchema
from workflow_module.execution import (
    OrchestratorMetadata,
    TaskOutcome,
    WorkflowResult,
)
from workflow_module.workflows import ReleaseWorkflow


class TestTaskOutcome:
    def test_succeeded_when_exit_zero(self):
        o = TaskOutcome(
            task_type="benchmark",
            exit_code=0,
            elapsed_seconds=1.0,
            block_kind="benchmarks",
        )
        assert o.succeeded is True

    def test_not_succeeded_on_nonzero_exit(self):
        o = TaskOutcome(
            task_type="benchmark", exit_code=2, elapsed_seconds=1.0, block_kind=None
        )
        assert o.succeeded is False


class TestWorkflowResult:
    def test_succeeded_reflects_return_code(self):
        assert WorkflowResult("w", return_code=0).succeeded is True
        assert WorkflowResult("w", return_code=1, error="boom").succeeded is False


def _write_spec(tmp_path, spec: dict) -> str:
    path = tmp_path / "runtime_model_spec.json"
    path.write_text(json.dumps({"runtime_model_spec": spec}), encoding="utf-8")
    return str(path)


def _make_workflow(orchestrator_metadata):
    ctx = MagicMock()
    return ReleaseWorkflow(ctx, orchestrator_metadata=orchestrator_metadata)


class TestInjectMetadata:
    """Central metadata injection — the single source of truth for the six
    identity/provenance fields on both media and LLM reports."""

    def test_copies_spec_identity_and_provenance_fields(self, tmp_path):
        spec_path = _write_spec(
            tmp_path,
            {
                "model_id": "id_tt-transformers_Llama-3.1-8B-Instruct_galaxy",
                "hf_model_repo": "meta-llama/Llama-3.1-8B-Instruct",
                "inference_engine": "vLLM",
                "tt_metal_commit": "6593e60",
                "vllm_commit": "9a72cb9",
                "impl": {"impl_id": "tt_transformers", "impl_name": "tt-transformers"},
                "status": "READY",
            },
        )
        meta = OrchestratorMetadata(
            server_mode="docker", runtime_model_spec_json=spec_path
        )
        schema = ReportSchema(
            metadata={"model_name": "m", "device": "GALAXY"}, sections=[]
        )

        _make_workflow(meta).inject_metadata(schema)

        assert schema.metadata["workflow"] == "release"
        assert schema.metadata["server_mode"] == "docker"
        assert schema.metadata["model_id"] == (
            "id_tt-transformers_Llama-3.1-8B-Instruct_galaxy"
        )
        assert schema.metadata["model_repo"] == "meta-llama/Llama-3.1-8B-Instruct"
        assert schema.metadata["inference_engine"] == "vLLM"
        assert schema.metadata["tt_metal_commit"] == "6593e60"
        assert schema.metadata["vllm_commit"] == "9a72cb9"
        assert schema.metadata["model_impl"] == "tt-transformers"

    def test_media_spec_missing_commits_are_written_as_none(self, tmp_path):
        # A media image has no tt-metal / vLLM commits; the keys must still be
        # present (as None) so media and LLM reports share one schema.
        spec_path = _write_spec(
            tmp_path,
            {
                "model_id": "id_tt-transformers_FLUX.1-schnell_p300x2",
                "hf_model_repo": "black-forest-labs/FLUX.1-schnell",
                "inference_engine": "media",
                "impl": {"impl_name": "tt-transformers"},
            },
        )
        meta = OrchestratorMetadata(runtime_model_spec_json=spec_path)
        schema = ReportSchema(
            metadata={"model_name": "FLUX.1-schnell", "device": "P300X2"}, sections=[]
        )

        _make_workflow(meta).inject_metadata(schema)

        assert schema.metadata["model_repo"] == "black-forest-labs/FLUX.1-schnell"
        assert schema.metadata["inference_engine"] == "media"
        assert schema.metadata["model_impl"] == "tt-transformers"
        assert schema.metadata["tt_metal_commit"] is None
        assert schema.metadata["vllm_commit"] is None

    def test_no_spec_leaves_fields_absent(self):
        meta = OrchestratorMetadata(server_mode="API")
        schema = ReportSchema(
            metadata={"model_name": "m", "device": "N150"}, sections=[]
        )

        _make_workflow(meta).inject_metadata(schema)

        assert schema.metadata["workflow"] == "release"
        for absent in (
            "model_id",
            "model_repo",
            "inference_engine",
            "tt_metal_commit",
            "vllm_commit",
            "model_impl",
        ):
            assert absent not in schema.metadata
