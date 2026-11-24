# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Tests for ``python -m report_module.list_agentic_tasks`` (mocked Harbor)."""

from __future__ import annotations

import importlib.machinery
import json
import sys
import types
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import pytest

from report_module import list_agentic_tasks as lat
from workflows.workflow_types import EvalLimitMode

# A fake Harbor registry: dataset name -> task ids in registry order.
REGISTRY: Dict[str, List[str]] = {
    "terminal-bench/terminal-bench-2-1": [f"terminal-bench/t{i:02d}" for i in range(6)],
}
SEEN: List[dict] = []


class _TaskId:
    def __init__(self, name):
        self._name = name

    def get_name(self):
        return self._name


class _TaskConfig:
    def __init__(self, name):
        self._name = name

    def get_task_id(self):
        return _TaskId(self._name)


class FakeDatasetConfig:
    """Harbor's DatasetConfig selection rule over the fake registry or a dir."""

    def __init__(self, data):
        self.data = data

    @classmethod
    def model_validate(cls, data):
        SEEN.append(dict(data))
        return cls(data)

    async def get_task_configs(self, disable_verification: bool = False):
        if "path" in self.data:
            ids = sorted(p.name for p in Path(self.data["path"]).iterdir())
        else:
            ids = REGISTRY[self.data["name"]]
        if self.data.get("task_names"):
            ids = [
                i for i in ids if any(fnmatch(i, p) for p in self.data["task_names"])
            ]
        if self.data.get("exclude_task_names"):
            ids = [
                i
                for i in ids
                if not any(fnmatch(i, p) for p in self.data["exclude_task_names"])
            ]
        if self.data.get("n_tasks") is not None:
            ids = ids[: self.data["n_tasks"]]
        return [_TaskConfig(i) for i in ids]


@pytest.fixture
def fake_harbor(monkeypatch):
    SEEN.clear()
    names = ["harbor", "harbor.models", "harbor.models.job", "harbor.models.job.config"]
    for name in names:
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, None)
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["harbor.models.job.config"].DatasetConfig = FakeDatasetConfig
    monkeypatch.setattr(
        "llm_module.agentic.local_datasets.prepare_local_dataset",
        lambda dataset, python: None,
    )
    return SEEN


@dataclass
class FakeAgenticConfig:
    dataset: str = "terminal-bench/terminal-bench-2-1"
    dataset_ref: Optional[str] = None
    task_names: List[str] = field(default_factory=list)
    exclude_task_names: List[str] = field(default_factory=list)
    n_tasks: Optional[int] = None
    task_names_map: dict = field(default_factory=dict)


def _use_task(monkeypatch, **cfg):
    task = SimpleNamespace(
        task_name="terminal_bench_2_1",
        agentic_eval_config=FakeAgenticConfig(**cfg),
        limit_samples_map={EvalLimitMode.SMOKE_TEST: 1},
    )
    monkeypatch.setattr(lat, "eval_task", lambda model, name: task)
    return task


def _main(tmp_path, *extra):
    out = tmp_path / "tasks.json"
    rc = lat.main(["--model", "m", "--task", "t", "--out", str(out), *extra])
    return rc, json.loads(out.read_text()) if out.exists() else None


def test_lists_the_registry_set_with_config_filters(tmp_path, monkeypatch, fake_harbor):
    _use_task(
        monkeypatch,
        dataset_ref="sha256:abc",
        exclude_task_names=["terminal-bench/t01"],
        n_tasks=4,
        # Limit-mode lists never shape the full run's set.
        task_names_map={EvalLimitMode.CI_NIGHTLY: ["terminal-bench/t05"]},
    )
    rc, names = _main(tmp_path)
    assert rc == 0
    assert names == [
        "terminal-bench/t00",
        "terminal-bench/t02",
        "terminal-bench/t03",
        "terminal-bench/t04",
    ]
    assert fake_harbor == [
        {
            "name": "terminal-bench/terminal-bench-2-1",
            "ref": "sha256:abc",
            "n_tasks": 4,
            "exclude_task_names": ["terminal-bench/t01"],
        }
    ]


def test_include_globs_select_tasks(tmp_path, monkeypatch, fake_harbor):
    _use_task(monkeypatch, task_names=["terminal-bench/t0[23]"])
    rc, names = _main(tmp_path)
    assert names == ["terminal-bench/t02", "terminal-bench/t03"]


def test_local_dataset_lists_directory_names(tmp_path, monkeypatch, fake_harbor):
    local = tmp_path / "tau3-bench"
    for name in ("tau3-banking_knowledge-task-002", "tau3-banking_knowledge-task-001"):
        (local / name).mkdir(parents=True)
    (local / "tau3-airline-task-001").mkdir()
    _use_task(
        monkeypatch,
        dataset="sierra-research/tau3-bench",
        dataset_ref="sha256:ignored",
        task_names=["sierra-research/tau3-bench__tau3-banking_knowledge-*"],
        n_tasks=97,
    )
    monkeypatch.setattr(
        "llm_module.agentic.local_datasets.prepare_local_dataset",
        lambda dataset, python: local,
    )
    rc, names = _main(tmp_path)
    assert rc == 0
    # Harbor filters a local copy by directory name: these names are what
    # --agentic-task-names and merge_agentic_runs --expected-tasks take.
    assert names == [
        "tau3-banking_knowledge-task-001",
        "tau3-banking_knowledge-task-002",
    ]
    assert fake_harbor[0]["path"] == str(local)
    assert fake_harbor[0]["task_names"] == ["tau3-banking_knowledge-*"]
    assert "ref" not in fake_harbor[0]


def test_writes_stdout_without_out(monkeypatch, fake_harbor, capsys):
    _use_task(monkeypatch, n_tasks=2)
    assert lat.main(["--model", "m", "--task", "t"]) == 0
    assert json.loads(capsys.readouterr().out) == [
        "terminal-bench/t00",
        "terminal-bench/t01",
    ]


def test_empty_selection_is_refused(tmp_path, monkeypatch, fake_harbor):
    _use_task(monkeypatch, exclude_task_names=["*"])
    rc, names = _main(tmp_path)
    assert rc == 2 and names is None


def test_unknown_task_is_refused_before_any_venv_setup(monkeypatch, capsys):
    monkeypatch.setattr(
        lat, "_reexec_in_agentic_venv", lambda: pytest.fail("must not set up a venv")
    )
    rc = lat.main(["--model", "zai-org/GLM-5.2", "--task", "nope"])
    assert rc == 2
    assert "terminal_bench_2_1" in capsys.readouterr().err


def test_without_harbor_reexecs_in_the_agentic_venv(monkeypatch):
    calls = []
    _use_task(monkeypatch)
    monkeypatch.setattr(lat.importlib.util, "find_spec", lambda name: None)

    def fake_reexec():
        calls.append(True)
        raise lat.ListError("stop")

    monkeypatch.setattr(lat, "_reexec_in_agentic_venv", fake_reexec)
    assert lat.main(["--model", "m", "--task", "t"]) == 2
    assert calls == [True]


def test_real_config_resolves():
    task = lat.eval_task("zai-org/GLM-5.2", "tau3_bench_banking")
    assert task.agentic_eval_config.n_tasks == 97


def test_listed_file_feeds_parse_task_names(tmp_path, monkeypatch, fake_harbor):
    from workflows.agentic_task_names import parse_task_names

    _use_task(monkeypatch, n_tasks=3)
    rc, names = _main(tmp_path)
    assert parse_task_names(f"@{tmp_path / 'tasks.json'}") == names
    bad = tmp_path / "bad.json"
    bad.write_text('["a", ""]')
    with pytest.raises(ValueError):
        parse_task_names(f"@{bad}")
