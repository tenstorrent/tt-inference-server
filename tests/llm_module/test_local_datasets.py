# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from pathlib import Path
from unittest.mock import patch

from llm_module.agentic import local_datasets
from llm_module.agentic.local_datasets import local_task_pattern, prepare_local_dataset

_DATASET = "sierra-research/tau3-bench"


class TestLocalTaskPattern:
    def test_strips_the_registry_prefix(self):
        assert (
            local_task_pattern(
                _DATASET, "sierra-research/tau3-bench__tau3-banking_knowledge-*"
            )
            == "tau3-banking_knowledge-*"
        )

    def test_leaves_other_patterns_alone(self):
        assert local_task_pattern(_DATASET, "tau3-airline-*") == "tau3-airline-*"


class TestPrepareLocalDataset:
    def test_datasets_without_a_generator_use_the_registry(self, tmp_path):
        assert prepare_local_dataset("swebench-verified", tmp_path / "python") is None

    def test_a_registered_generator_builds_the_tasks(self, tmp_path):
        calls = []

        def generate(python: Path) -> Path:
            calls.append(python)
            return tmp_path / "tasks"

        with patch.dict(local_datasets._GENERATORS, {_DATASET: generate}):
            assert prepare_local_dataset(_DATASET, tmp_path / "python") == (
                tmp_path / "tasks"
            )
        assert calls == [tmp_path / "python"]
