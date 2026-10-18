# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Prevent unscored diagnostic routing from replacing scored evaluations."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from llm_module.drivers.agentic import HarborAgenticDriver
from reference_config.evals.eval_config import ALL_EVAL_CONFIGS


def driver(marker="v1", **score_fields):
    obj = object.__new__(HarborAgenticDriver)
    score = {
        "published_score": None,
        "published_score_ref": None,
        "gpu_reference_score": None,
        "gpu_reference_score_ref": None,
        "mode_reference_scores": {},
    }
    score.update(score_fields)
    kwargs = {} if marker is None else {"gemma_formation_probe": marker}
    obj.task = SimpleNamespace(
        task_name="terminal_bench_2",
        score=SimpleNamespace(**score),
        agentic_eval_config=SimpleNamespace(agent_kwargs=kwargs),
    )
    obj.runtime_config = None
    obj.venv_python = None
    return obj


def server(model="google/gemma-4-31B-it"):
    return SimpleNamespace(model=model)


def test_valid_probe_reports_na():
    obj = driver()
    raw = {"unscored_policy_probe": True, "status": "completed"}
    with patch("llm_module.drivers.agentic.resolve_n_tasks", return_value=1), patch(
        "llm_module.agentic.ci_formation_probe.run_probe", return_value=(raw, None)
    ) as probe, patch("llm_module.drivers.agentic.run_harbor") as harbor:
        result = obj.run(None, server(), None)
    assert result.raw == raw
    probe.assert_called_once()
    harbor.assert_not_called()
    assert obj.parse(raw).data["status"] == "na"


@pytest.mark.parametrize(
    "field,value",
    [
        ("published_score", 0),
        ("published_score_ref", "ref"),
        ("gpu_reference_score", 44.94),
        ("gpu_reference_score_ref", "ref"),
        ("mode_reference_scores", {"test": 1}),
    ],
)
def test_marked_scored_task_rejected(field, value):
    obj = driver(**{field: value})
    with patch("llm_module.drivers.agentic.resolve_n_tasks", return_value=1), patch(
        "llm_module.agentic.ci_formation_probe.run_probe"
    ) as probe, patch("llm_module.drivers.agentic.run_harbor") as harbor, pytest.raises(
        ValueError, match="absent score references"
    ):
        obj.run(None, server(), None)
    probe.assert_not_called()
    harbor.assert_not_called()


@pytest.mark.parametrize("mismatch", ["marker", "model", "task"])
def test_marked_identity_mismatch_rejected(mismatch):
    obj = driver(marker="other" if mismatch == "marker" else "v1")
    if mismatch == "task":
        obj.task.task_name = "swe_bench_verified"
    with patch(
        "llm_module.drivers.agentic.resolve_n_tasks", return_value=1
    ), pytest.raises(ValueError, match="exact model/task"):
        obj.run(None, server("other/model") if mismatch == "model" else server(), None)


def test_zero_task_skip_precedes_probe():
    obj = driver()
    with patch("llm_module.drivers.agentic.resolve_n_tasks", return_value=0), patch(
        "llm_module.agentic.ci_formation_probe.run_probe"
    ) as probe:
        assert obj.run(None, server(), None).raw is None
    probe.assert_not_called()
    assert not obj._formation_probe_validated


def test_unmarked_scored_task_retains_harbor_path():
    obj = driver(marker=None, gpu_reference_score=44.94)
    obj.result_path = lambda *_: "existing-result"
    obj._load_result = lambda rc, path: (rc, path)
    with patch("llm_module.drivers.agentic.resolve_n_tasks", return_value=1), patch(
        "llm_module.drivers.agentic.build_harbor_config", return_value="existing-config"
    ), patch("llm_module.drivers.agentic.run_harbor", return_value=0) as harbor, patch(
        "llm_module.agentic.ci_formation_probe.run_probe"
    ) as probe:
        assert obj.run(None, server(), None) == (0, "existing-result")
    harbor.assert_called_once_with("existing-config")
    probe.assert_not_called()


def test_unvalidated_na_output_rejected():
    with pytest.raises(ValueError, match="without validated execution"):
        driver().parse({"unscored_policy_probe": True, "status": "completed"})


def test_catalog_marker_is_unique_and_unscored():
    marked = []
    for key, config in ALL_EVAL_CONFIGS.items():
        for task in config.tasks:
            cfg = getattr(task, "agentic_eval_config", None)
            if cfg and cfg.agent_kwargs.get("gemma_formation_probe"):
                marked.append((key, task))
    assert len(marked) == 1
    key, task = marked[0]
    assert key == "google/gemma-4-31B-it" and task.task_name == "terminal_bench_2"
    assert task.score.published_score is None and task.score.gpu_reference_score is None
    assert not task.score.mode_reference_scores
