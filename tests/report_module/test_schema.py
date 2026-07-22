# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for ``report_module.schema`` Block / ReportSchema construction."""

from __future__ import annotations

import pytest

from report_module.schema import Block, ReportSchema


class TestBlock:
    def test_slug_without_id_is_kind(self):
        assert Block(kind="benchmarks", data={}).slug == "benchmarks"

    def test_slug_with_id_joins_kind_and_id(self):
        assert (
            Block(kind="benchmarks", data={}, id="m_n300").slug == "benchmarks__m_n300"
        )

    def test_from_dict_requires_kind(self):
        with pytest.raises(ValueError, match="kind"):
            Block.from_dict({"data": {}})

    def test_from_dict_round_trips_through_to_dict(self):
        payload = {
            "kind": "evals",
            "data": {"score": 1},
            "title": "Eval",
            "task_type": "image",
            "id": "x",
            "targets": {"a": 1},
        }
        assert Block.from_dict(payload).to_dict() == payload

    def test_to_dict_omits_unset_optionals(self):
        assert Block(kind="evals", data={"x": 1}).to_dict() == {
            "kind": "evals",
            "data": {"x": 1},
        }

    def test_to_dict_omits_empty_targets(self):
        out = Block(kind="evals", data={}, targets={}).to_dict()
        assert "targets" not in out


class TestReportSchemaFromDict:
    def test_parses_metadata_and_sections(self):
        schema = ReportSchema.from_dict(
            {
                "metadata": {"report_id": "r1"},
                "sections": [{"kind": "evals", "data": {}}],
            }
        )
        assert schema.report_id == "r1"
        assert [b.kind for b in schema.sections] == ["evals"]

    def test_missing_sections_defaults_to_empty(self):
        assert ReportSchema.from_dict({"metadata": {}}).sections == []

    def test_non_list_sections_raises(self):
        with pytest.raises(TypeError):
            ReportSchema.from_dict({"sections": {"kind": "x"}})

    def test_report_id_required_when_absent(self):
        with pytest.raises(ValueError, match="report_id"):
            _ = ReportSchema.from_dict({"metadata": {}}).report_id


class TestReportSchemaFromRecords:
    def test_groups_by_kind_model_device(self):
        records = [
            {"kind": "benchmarks", "model": "m", "device": "n300", "ttft": 1},
            {"kind": "benchmarks", "model": "m", "device": "n300", "ttft": 2},
            {"kind": "evals", "model": "m", "device": "n300", "score": 0.9},
        ]
        schema = ReportSchema.from_records(records)
        assert [b.kind for b in schema.sections] == ["benchmarks", "evals"]
        bench = schema.sections[0]
        assert len(bench.data["records"]) == 2
        assert bench.data["model"] == "m" and bench.data["device"] == "n300"

    def test_block_id_slugifies_model_and_device(self):
        schema = ReportSchema.from_records(
            [{"kind": "benchmarks", "model": "meta/Llama 3", "device": "n300"}]
        )
        assert schema.sections[0].id == "meta__Llama_3_n300"

    def test_metadata_synthesised_from_first_record(self):
        schema = ReportSchema.from_records(
            [
                {
                    "kind": "benchmarks",
                    "model": "m",
                    "device": "n300",
                    "timestamp": "2026-04-11 01:50:50",
                }
            ]
        )
        assert schema.model_name == "m"
        assert schema.device == "n300"
        assert schema.report_id == "m_20260411_015050"
        assert schema.metadata["generated_at"] == "2026-04-11 01:50:50"

    def test_explicit_metadata_not_overwritten(self):
        schema = ReportSchema.from_records(
            [{"kind": "benchmarks", "model": "m", "device": "n300"}],
            metadata={"report_id": "fixed", "model_name": "override"},
        )
        assert schema.report_id == "fixed"
        assert schema.model_name == "override"

    def test_empty_records_still_valid(self):
        schema = ReportSchema.from_records([])
        assert schema.sections == []
        assert schema.report_id  # synthesised "report_..."

    def test_record_missing_kind_raises(self):
        with pytest.raises(ValueError, match="kind"):
            ReportSchema.from_records([{"model": "m"}])

    def test_non_dict_record_raises(self):
        with pytest.raises(TypeError):
            ReportSchema.from_records([["not", "a", "dict"]])

    def test_string_is_not_a_records_sequence(self):
        with pytest.raises(TypeError):
            ReportSchema.from_records("notrecords")
