# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from pathlib import Path
from unittest.mock import patch

import pytest

from llm_module.agentic import local_datasets
from llm_module.agentic.local_datasets import (
    TAU3_BENCH_DATASET,
    local_task_pattern,
    prepare_local_dataset,
    tau2_bench_ref,
    tree_digest,
)

_REF = "b7ea9074c1cba482b30687fecdb5c8425fd6f619"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _fake_harbor(root: Path, dockerfile: str) -> Path:
    _write(root / local_datasets._TAU3_TEMPLATE_DOCKERFILE, dockerfile)
    _write(root / local_datasets._TAU3_ADAPTER_SRC / "tau3_bench" / "main.py", "")
    return root


class TestLocalTaskPattern:
    def test_strips_the_registry_prefix(self):
        assert (
            local_task_pattern(
                TAU3_BENCH_DATASET,
                "sierra-research/tau3-bench__tau3-banking_knowledge-*",
            )
            == "tau3-banking_knowledge-*"
        )

    def test_leaves_other_patterns_alone(self):
        assert local_task_pattern(TAU3_BENCH_DATASET, "tau3-airline-*") == (
            "tau3-airline-*"
        )


class TestTau2BenchRef:
    def test_reads_the_pinned_commit(self, tmp_path):
        dockerfile = _write(
            tmp_path / "Dockerfile",
            f'FROM python:3.12-slim\nARG TAU2_BENCH_REF="{_REF}"\n',
        )
        assert tau2_bench_ref(dockerfile) == _REF

    def test_unpinned_template_is_rejected(self, tmp_path):
        dockerfile = _write(
            tmp_path / "Dockerfile",
            'ARG TAU2_BENCH_REPO="https://github.com/sierra-research/tau2-bench.git"\n',
        )
        with pytest.raises(RuntimeError, match="does not pin TAU2_BENCH_REF"):
            tau2_bench_ref(dockerfile)


class TestTreeDigest:
    def test_changes_with_file_contents(self, tmp_path):
        _write(tmp_path / "a" / "x.txt", "one")
        before = tree_digest(tmp_path / "a")
        _write(tmp_path / "a" / "x.txt", "two")
        assert tree_digest(tmp_path / "a") != before

    def test_ignores_bytecode_caches(self, tmp_path):
        _write(tmp_path / "a" / "x.py", "pass")
        before = tree_digest(tmp_path / "a")
        _write(tmp_path / "a" / "__pycache__" / "x.cpython-312.pyc", "junk")
        assert tree_digest(tmp_path / "a") == before


class TestPrepareLocalDataset:
    def test_registry_datasets_are_not_generated(self, tmp_path):
        with patch.object(local_datasets.subprocess, "run") as run:
            assert (
                prepare_local_dataset("swebench-verified", tmp_path / "python") is None
            )
        run.assert_not_called()

    def test_generates_once_then_reuses_the_cache(self, tmp_path):
        harbor_root = _fake_harbor(
            tmp_path / "venv" / "harbor", f'ARG TAU2_BENCH_REF="{_REF}"\n'
        )
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if cmd[:2] == ["git", "clone"]:
                (Path(cmd[-1]) / ".git").mkdir(parents=True)
            elif "tau3_bench.main" in cmd:
                assert kwargs["env"]["TAU2_BENCH_ROOT"].endswith(
                    f"tau2-bench-{_REF[:12]}"
                )
                out = Path(cmd[cmd.index("--output-dir") + 1])
                (out / "tau3-banking_knowledge-task-001").mkdir(parents=True)

        with patch.object(
            local_datasets, "_harbor_root", return_value=harbor_root
        ), patch.object(local_datasets.subprocess, "run", side_effect=fake_run):
            first = prepare_local_dataset(TAU3_BENCH_DATASET, tmp_path / "python")
            n_calls = len(calls)
            second = prepare_local_dataset(TAU3_BENCH_DATASET, tmp_path / "python")

        assert first == second
        assert first.name == "tau3-bench"
        assert first.parent.parent == tmp_path / "venv" / "harbor-datasets"
        assert (first / "tau3-banking_knowledge-task-001").is_dir()
        assert len(calls) == n_calls
        assert not list(first.parent.parent.glob("*.tmp-*"))
