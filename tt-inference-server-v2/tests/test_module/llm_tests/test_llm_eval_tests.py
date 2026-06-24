# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for standard LLM evals: scoring, result-loading, orchestration, wiring.
These tests cover the copied
scoring loop, the result-JSON loader, ``run_llm_eval`` orchestration, and the
``EvalsWorkflow`` LLM override.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from test_module._test_common import ReportCheckTypes
from test_module.llm_tests import llm_eval_tests as mod

_MOD = "test_module.llm_tests.llm_eval_tests"


# --- fixtures ----------------------------------------------------------------


def _single_key(results, task_name, kwargs):
    score = results[task_name][kwargs["result_keys"][0]]
    if kwargs.get("unit") == "percent":
        score *= 100.0
    return score


def _score(
    published=None,
    reference=None,
    tolerance=0.05,
    result_keys=("acc,none",),
    unit="percent",
    score_func=_single_key,
):
    return SimpleNamespace(
        published_score=published,
        published_score_ref="ref://published",
        gpu_reference_score=reference,
        tolerance=tolerance,
        score_func=score_func,
        score_func_kwargs={"result_keys": list(result_keys), "unit": unit},
    )


def _task(task_name="gpqa", score=None, min_context_required=None):
    return SimpleNamespace(
        task_name=task_name,
        score=score,
        min_context_required=min_context_required,
    )


def _ctx(max_context=131072):
    ctx = MagicMock()
    ctx.model_spec.model_name = "test-llm"
    ctx.model_spec.hf_model_repo = "org/test-llm"
    ctx.model_spec.device_model_spec.max_context = max_context
    ctx.device.name = "gpu"
    ctx.server_host = "http://127.0.0.1"
    ctx.server_port = 8000
    ctx.output_path = "/tmp/out"
    ctx.runtime_config = None
    return ctx


# --- scoring -> Block (the copied logic) -------------------------------------


class TestBlocksForTask:
    def _blocks(self, task, results):
        with patch(f"{_MOD}.block_id", return_value=""):
            return mod.blocks_for_task(_ctx(), task, results)

    def test_reference_pass(self):
        (b,) = self._blocks(
            _task(score=_score(published=90.5, reference=90.91)),
            {"gpqa": {"acc,none": 0.9}},
        )
        assert b.kind == "evals"
        assert b.data["score"] == 90.0
        assert b.data["accuracy_check"] == ReportCheckTypes.PASS

    def test_reference_fail(self):
        (b,) = self._blocks(
            _task(score=_score(published=90.5, reference=90.91)),
            {"gpqa": {"acc,none": 0.5}},
        )
        assert b.data["accuracy_check"] == ReportCheckTypes.FAIL

    def test_published_only_drives_accuracy(self):
        (b,) = self._blocks(
            _task(score=_score(published=90.0, reference=None)),
            {"gpqa": {"acc,none": 0.88}},  # 88/90 = 0.977 >= 0.95
        )
        assert b.data["accuracy_check"] == ReportCheckTypes.PASS
        assert b.data["ratio_to_reference"] == "N/A"

    def test_no_targets_is_na(self):
        (b,) = self._blocks(
            _task(score=_score(published=None, reference=None)),
            {"gpqa": {"acc,none": 0.88}},
        )
        assert b.data["accuracy_check"] == ReportCheckTypes.NA
        assert b.data["ratio_to_published"] == "N/A"

    def test_subtask_prefix_expansion(self):
        blocks = self._blocks(
            _task("longbench", _score(published=50.0, reference=50.0)),
            {
                "longbench_2wikimqa": {"acc,none": 0.6},
                "longbench_hotpotqa": {"acc,none": 0.4},
            },
        )
        names = sorted(b.data["task_name"] for b in blocks)
        assert names == ["longbench_2wikimqa", "longbench_hotpotqa"]

    def test_wer_is_inverted(self):
        def wer_func(results, task_name, kwargs):
            return results[task_name]["wer,none"]

        (b,) = self._blocks(
            _task(
                "librispeech",
                _score(published=92.0, reference=92.0, unit="WER", score_func=wer_func),
            ),
            {"librispeech": {"wer,none": 8.0}},
        )
        assert b.data["score"] == 92.0
        assert b.data["accuracy_check"] == ReportCheckTypes.PASS

    def test_auto_detect_replacement_metric(self):
        (b,) = self._blocks(
            _task(score=_score(published=90.0, result_keys=("missing,none",))),
            {"gpqa": {"exact_match,none": 0.9, "acc_stderr,none": 0.01}},
        )
        assert b.data["score"] == 90.0

    def test_score_func_raises_non_wer_scores_zero(self):
        def boom(results, task_name, kwargs):
            raise KeyError("nope")

        (b,) = self._blocks(
            _task(score=_score(published=90.0, score_func=boom)),
            {"gpqa": {"acc,none": 0.9}},
        )
        assert b.data["score"] == 0.0
        assert b.data["accuracy_check"] == ReportCheckTypes.FAIL

    def test_task_without_score_is_skipped(self):
        assert self._blocks(_task(score=None), {"gpqa": {"acc,none": 0.9}}) == []


# --- reading lm-eval result JSON ---------------------------------------------


class TestResultLoading:
    def _write(self, path, task_name, metric):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "results": {task_name: {"acc,none": metric, "alias": task_name}},
                    "configs": {task_name: {"task": task_name, "dataset_path": "d"}},
                }
            )
        )

    def test_discover_text_and_lmms_patterns(self, tmp_path):
        ms = SimpleNamespace(
            hf_model_repo="meta/Llama-3.2-1B", model_id="llama-1b", model_name="Llama"
        )
        base = tmp_path / "eval_llama-1b" / "meta__Llama-3.2-1B"
        self._write(base / "results_2026.json", "gpqa", 0.9)
        self._write(base / "librispeech_results.json", "librispeech", 0.1)
        assert len(mod.discover_eval_results(str(tmp_path), ms)) == 2

    def test_merge_strips_alias_and_dedupes(self, tmp_path):
        self._write(tmp_path / "results_1.json", "gpqa", 0.9)
        self._write(tmp_path / "results_2.json", "mmlu", 0.7)
        results = mod.merge_eval_results(
            [str(tmp_path / "results_1.json"), str(tmp_path / "results_2.json")]
        )
        assert set(results) == {"gpqa", "mmlu"}
        assert results["gpqa"]["acc,none"] == 0.9
        assert "alias" not in results["gpqa"]


# --- orchestration -----------------------------------------------------------


class TestRunLLMEval:
    def _run(self, tasks, *, healthy=True, blocks=None, run_rc=0, results=None):
        server = MagicMock()
        server.wait_for_healthy.return_value = healthy
        server.get_health.return_value = SimpleNamespace(status_code=200)
        with patch(f"{_MOD}.get_llm_eval_tasks", return_value=tasks), patch(
            f"{_MOD}.HttpServerController", return_value=server
        ), patch(f"{_MOD}._run_eval_task", return_value=run_rc) as run_task, patch(
            f"{_MOD}.discover_eval_results", return_value=["f.json"]
        ), patch(f"{_MOD}.merge_eval_results", return_value=results or {}), patch(
            f"{_MOD}.blocks_for_task", return_value=blocks if blocks is not None else []
        ), patch(f"{_MOD}.accept_blocks") as accept, patch(
            f"{_MOD}.block_id", return_value=""
        ):
            out = mod.run_llm_eval(_ctx())
        return out, run_task, accept

    def test_happy_path(self):
        blk = MagicMock()
        blk.kind = "evals"
        out, run_task, accept = self._run([_task()], blocks=[blk])
        assert out == [blk]
        run_task.assert_called_once()
        accept.assert_called_once()

    def test_no_tasks(self):
        out, run_task, accept = self._run([])
        assert out == []
        run_task.assert_not_called()
        accept.assert_not_called()

    def test_unhealthy_emits_fail_blocks(self):
        out, run_task, _accept = self._run([_task("a"), _task("b")], healthy=False)
        assert len(out) == 2
        assert all(b.data["accuracy_check"] == ReportCheckTypes.FAIL for b in out)
        run_task.assert_not_called()

    def test_ran_but_no_block_gets_fail_block(self):
        out, run_task, _accept = self._run([_task("gpqa")], blocks=[], run_rc=1)
        assert len(out) == 1
        assert out[0].data["accuracy_check"] == ReportCheckTypes.FAIL
        assert "no eval results parsed" in out[0].data["error"]
        run_task.assert_called_once()

    def test_min_context_skip(self):
        out, run_task, _accept = self._run(
            [_task("longctx", min_context_required=200000)]
        )
        run_task.assert_not_called()
        assert out == []


# --- EvalsWorkflow override --------------------------------------------------


class TestEvalsWorkflowLLMOverride:
    def _wf(self, model_type):
        from workflow_module.workflows import EvalsWorkflow
        from workflows.workflow_types import ModelType

        ctx = _ctx()
        ctx.model_spec.model_type = (
            ModelType.LLM if model_type == "llm" else ModelType.IMAGE
        )
        return EvalsWorkflow(ctx, accumulator=MagicMock())

    def test_llm_routes_to_run_llm_eval(self):
        wf = self._wf("llm")
        block = MagicMock()
        block.kind = "evals"
        with patch(f"{_MOD}.run_llm_eval", return_value=[block]) as run:
            outcomes = wf.run_tasks()
        run.assert_called_once()
        assert outcomes[0].exit_code == 0
        assert outcomes[0].block_kind == "evals"

    def test_llm_no_tasks_is_clean_noop(self):
        wf = self._wf("llm")
        with patch(f"{_MOD}.run_llm_eval", return_value=[]):
            outcomes = wf.run_tasks()
        assert outcomes[0].exit_code == 0
        assert outcomes[0].block_kind is None

    def test_llm_raises_fails_task(self):
        wf = self._wf("llm")
        with patch(f"{_MOD}.run_llm_eval", side_effect=RuntimeError("boom")):
            outcomes = wf.run_tasks()
        assert outcomes[0].exit_code == 1

    def test_media_model_does_not_call_run_llm_eval(self):
        wf = self._wf("image")
        with patch(f"{_MOD}.run_llm_eval") as run, patch.object(
            wf, "_dispatch_task", return_value="media-outcome"
        ) as dispatch:
            outcomes = wf.run_tasks()
        run.assert_not_called()
        dispatch.assert_called_once()
        assert outcomes == ["media-outcome"]
