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
                "model_name": "Llama-3.1-8B-Instruct",
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
        assert schema.metadata["model_name"] == "Llama-3.1-8B-Instruct"
        # Display property prefers the full HF repo.
        assert schema.model_name == "meta-llama/Llama-3.1-8B-Instruct"
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
                "model_name": "FLUX.1-schnell",
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
        assert schema.metadata["model_name"] == "FLUX.1-schnell"
        assert schema.model_name == "black-forest-labs/FLUX.1-schnell"
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


class _StagePack:
    """Tags evals with one stage and benchmarks with another."""

    def stage_of(self, block):
        position = {"evals": 1, "benchmarks": 2}.get(block.kind)
        if position is None:
            return None
        return {"key": block.kind, "name": block.kind.title(), "position": position}

    def extra_spec_metadata_fields(self):
        return ()


class TestDeliveryStages:
    def _schema(self):
        from report_module.schema import Block

        return ReportSchema(
            metadata={"model_name": "m", "device": "SUPER_CLUSTER"},
            sections=[
                Block(kind="evals", data={"task_name": "t", "accuracy_check": 2}),
                Block(kind="benchmarks", data={"concurrency": 1}),
                Block(kind="spec_tests", data={}),
            ],
        )

    def test_acceptance_grades_each_tagged_stage(self, monkeypatch):
        from workflow_module import target_pack

        monkeypatch.setattr(target_pack, "_target_pack", _StagePack())
        workflow = _make_workflow(OrchestratorMetadata(server_mode="API"))
        schema = self._schema()

        workflow._tag_stages(schema)
        workflow.apply_acceptance_criteria(schema, task_outcomes=[])

        assert [b.targets.get("stage", {}).get("key") for b in schema.sections] == [
            "evals",
            "benchmarks",
            None,
        ]
        stages = schema.metadata["acceptance_stages"]
        assert [s["key"] for s in stages] == ["evals", "benchmarks"]
        assert "#### Delivery stages" in schema.metadata["acceptance_summary_markdown"]

    def test_a_checkpoint_report_is_tagged_too(self, monkeypatch):
        from workflow_module import target_pack

        monkeypatch.setattr(target_pack, "_target_pack", _StagePack())
        schema = self._schema()

        _make_workflow(OrchestratorMetadata(server_mode="API")).inject_metadata(schema)

        assert schema.sections[0].targets["stage"]["key"] == "evals"
