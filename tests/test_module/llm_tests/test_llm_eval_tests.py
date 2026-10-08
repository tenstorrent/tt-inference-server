# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for standard LLM evals: scoring, result-loading, orchestration, wiring.
These tests cover the copied scoring loop, the result-JSON loader, ``run_llm_eval``
orchestration, and the ``EvalsWorkflow`` LLM override.
"""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from llm_module.eval_command import build_eval_command
from llm_module.lm_eval_no_server_seed import _drop_server_seed, _patch_api_adapters
from reference_config.evals.eval_config import EvalTask, _eval_config_map
from test_module._test_common import ReportCheckTypes, TestStatus
from test_module.llm_tests import llm_eval_tests as mod
from workflows.workflow_types import EvalLimitMode, WorkflowVenvType

_MOD = "test_module.llm_tests.llm_eval_tests"


def tmp_output_path():
    import tempfile
    from pathlib import Path

    return Path(tempfile.mkdtemp())


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
    ctx.server_url = None
    ctx.remote_server = False
    ctx.server_port = 8000
    ctx.output_path = "/tmp/out"
    ctx.runtime_config = None
    return ctx


def _diffusiongemma_eval_task(task_name):
    tasks = _eval_config_map["google/diffusiongemma-26B-A4B-it"].tasks
    return next(task for task in tasks if task.task_name == task_name)


def _llama_1b_eval_task(task_name):
    tasks = _eval_config_map["meta-llama/Llama-3.2-1B-Instruct"].tasks
    return next(task for task in tasks if task.task_name == task_name)


def _build_eval_test_command(
    task,
    *,
    hf_model_repo="google/diffusiongemma-26B-A4B-it",
    model_id="diffusiongemma-26B-A4B-it",
    max_context=262144,
):
    model_spec = SimpleNamespace(
        model_id=model_id,
        model_name=model_id,
        hf_model_repo=hf_model_repo,
        device_model_spec=SimpleNamespace(
            max_context=max_context,
            max_concurrency=1,
            eval_max_retries=0,
        ),
    )
    return build_eval_command(task, model_spec, "P300x2", "/tmp/evals", 8000)


def _command_gen_kwargs(command):
    raw = command[command.index("--gen_kwargs") + 1]
    return dict(item.split("=", 1) for item in raw.split(","))


def _command_model_kwargs(command):
    raw = command[command.index("--model_args") + 1]
    return dict(item.split("=", 1) for item in raw.split(","))


# --- eval command and model-specific request contracts -----------------------


class TestEvalCommand:
    @pytest.mark.parametrize("field", ["wall_clock_timeout_seconds", "max_attempts"])
    @pytest.mark.parametrize("value", [0, -1, True, 1.5])
    def test_invalid_execution_policy(self, field, value):
        with pytest.raises(ValueError, match=field):
            EvalTask(task_name="aime25", **{field: value})

    def test_task_attempt_budget_overrides_device_default(self):
        # max_attempts is a TOTAL attempt count including the first, which is
        # what docs/eval_execution_budgets.md promises. lm-eval's max_retries
        # counts only the retries after the first attempt, so the total must be
        # converted rather than passed through.
        task = EvalTask(task_name="aime25", max_attempts=1)
        command = _build_eval_test_command(task)
        model_args = command[command.index("--model_args") + 1]
        assert "max_retries=0" in model_args

    def test_attempt_budget_is_a_total_not_a_retry_count(self):
        task = EvalTask(task_name="aime25", max_attempts=3)
        command = _build_eval_test_command(task)
        model_args = command[command.index("--model_args") + 1]
        assert "max_retries=2" in model_args

    def test_text_harness_context_comes_from_selected_device_spec(self):
        command = _build_eval_test_command(
            EvalTask(task_name="long_context", min_context_required=16384),
            max_context=32768,
        )

        assert _command_model_kwargs(command)["max_length"] == str(32768 - 64)

    @pytest.mark.parametrize(
        "task,max_context,error",
        [
            (
                EvalTask(task_name="undersized", min_context_required=16384),
                8192,
                "device max_context=8192",
            ),
        ],
    )
    def test_text_harness_rejects_context_contract_mismatch(
        self, task, max_context, error
    ):
        with pytest.raises(ValueError, match=error):
            _build_eval_test_command(task, max_context=max_context)

    def test_task_max_length_above_device_context_is_bound_to_the_device(self):
        task = EvalTask(task_name="oversized", model_kwargs={"max_length": 65536})

        command = _build_eval_test_command(task, max_context=40960)

        assert _command_model_kwargs(command)["max_length"] == str(40960 - 64)
        assert task.model_kwargs["max_length"] == 65536

    def test_explicit_task_max_length_may_be_below_device_minimum(self):
        task = EvalTask(
            task_name="task_specific_truncation",
            min_context_required=16384,
            model_kwargs={"max_length": 8192},
        )

        command = _build_eval_test_command(task, max_context=32768)

        assert _command_model_kwargs(command)["max_length"] == "8192"

    def test_text_harness_requires_a_declared_device_context(self):
        with pytest.raises(ValueError, match="requires device_model_spec.max_context"):
            _build_eval_test_command(EvalTask(task_name="unbounded"), max_context=None)

    def test_output_clamp_uses_explicit_harness_context_without_mutating_task(self):
        task = EvalTask(
            task_name="bounded_generation",
            model_kwargs={"max_length": 4096},
            gen_kwargs={"max_gen_toks": 8192},
        )

        command = _build_eval_test_command(task, max_context=32768)

        assert _command_gen_kwargs(command)["max_gen_toks"] == "3072"
        assert task.model_kwargs == {"max_length": 4096}
        assert task.gen_kwargs == {"max_gen_toks": 8192}

    def test_diffusiongemma_keeps_harness_seed_out_of_server_requests(self):
        task = _diffusiongemma_eval_task("gpqa_diamond_cot_zeroshot")
        command = _build_eval_test_command(task)

        gen_kwargs = _command_gen_kwargs(command)
        assert gen_kwargs["do_sample"] == "true"
        assert gen_kwargs["temperature"] == "1.0"
        assert "seed" not in gen_kwargs
        assert command[command.index("--seed") + 1] == "42"
        assert command[1].endswith("llm_module/lm_eval_no_server_seed.py")

    def test_eval_task_forwards_seed_by_default(self):
        task = EvalTask(
            task_name="seeded_sampling",
            gen_kwargs={"do_sample": "true", "temperature": 1.0},
        )
        command = _build_eval_test_command(task)

        assert _command_gen_kwargs(command)["seed"] == "42"
        assert command[command.index("--seed") + 1] == "42"
        assert command[0].endswith("/bin/lm_eval")

    def test_seed_filter_does_not_mutate_the_request(self):
        payload = {"model": "test-model", "seed": 42, "temperature": 1.0}

        assert _drop_server_seed(payload) == {
            "model": "test-model",
            "temperature": 1.0,
        }
        assert payload["seed"] == 42

    def test_completions_api_task_also_gets_no_server_seed_wrapper(self):
        task = EvalTask(
            task_name="model_owned_sampling",
            gen_kwargs={"do_sample": "true", "temperature": 1.0},
            propagate_seed_to_gen_kwargs=False,
        )
        command = _build_eval_test_command(task)

        assert "seed" not in _command_gen_kwargs(command)
        assert command[1].endswith("llm_module/lm_eval_no_server_seed.py")

    def test_no_server_seed_rejects_lmms_eval_tasks(self, monkeypatch):
        # build_eval_command exports OPENAI_API_BASE for vision tasks; keep the
        # write registered with monkeypatch so it is restored at teardown.
        monkeypatch.setenv("OPENAI_API_BASE", "restore-me")
        task = EvalTask(
            task_name="vision_task",
            workflow_venv_type=WorkflowVenvType.EVALS_VISION,
            propagate_seed_to_gen_kwargs=False,
        )

        with pytest.raises(ValueError, match="no-server-seed"):
            _build_eval_test_command(task)

    def test_seed_patch_covers_both_api_adapters(self, monkeypatch):
        class _Completions:
            def _create_payload(self, *args, **kwargs):
                return {"prompt": "p", "seed": 1234}

        class _Chat(_Completions):
            def _create_payload(self, *args, **kwargs):
                return {"messages": [], "seed": 1234}

        module = SimpleNamespace(
            LocalCompletionsAPI=_Completions, LocalChatCompletion=_Chat
        )
        models_pkg = SimpleNamespace(openai_completions=module)
        monkeypatch.setitem(sys.modules, "lm_eval", SimpleNamespace(models=models_pkg))
        monkeypatch.setitem(sys.modules, "lm_eval.models", models_pkg)
        monkeypatch.setitem(sys.modules, "lm_eval.models.openai_completions", module)

        _patch_api_adapters()

        assert "seed" not in _Completions()._create_payload()
        assert "seed" not in _Chat()._create_payload()


class TestDiffusionGemmaEvalContract:
    def test_gpqa_uses_canvas_aligned_model_owned_generation(self):
        task = _diffusiongemma_eval_task("gpqa_diamond_cot_zeroshot")

        assert (
            task.gen_kwargs["max_gen_toks"]
            == (task.model_kwargs["max_length"] - 2432) // 256 * 256
        )
        assert task.gen_kwargs["do_sample"] == "true"
        assert task.gen_kwargs["temperature"] == 1.0
        assert task.propagate_seed_to_gen_kwargs is False

    def test_terminal_bench_is_bounded_to_single_request_execution(self):
        task = _diffusiongemma_eval_task("terminal_bench_2_1")
        config = task.agentic_eval_config

        assert config.n_concurrent_trials == 1
        assert config.n_attempts == 1
        assert len(config.task_names_map[EvalLimitMode.CI_NIGHTLY]) == 3
        assert config.agent_timeout_sec == 45 * 60
        assert config.agent_kwargs["model_info"]["max_output_tokens"] == 4 * 1024
        assert config.agent_kwargs["llm_kwargs"]["max_tokens"] == 4 * 1024


class TestLlama1BLongBenchEvalContract:
    @pytest.mark.parametrize("task_name", ["longbench_code_e", "longbench_fewshot_e"])
    def test_code_and_fewshot_match_raw_completion_reference(self, task_name):
        task = _llama_1b_eval_task(task_name)
        command = _build_eval_test_command(
            task,
            hf_model_repo="meta-llama/Llama-3.2-1B-Instruct",
            model_id="Llama-3.2-1B-Instruct",
            max_context=32768,
        )

        assert task.use_chat_api is False
        assert task.apply_chat_template is False
        assert "--apply_chat_template" not in command
        assert "/v1/completions" in command[command.index("--model_args") + 1]
        # One paged-KV block below max_context: prompt + max_tokens stays inside.
        assert _command_model_kwargs(command)["max_length"] == str(32768 - 64)
        assert _command_gen_kwargs(command) == {
            "stream": "False",
            "temperature": "0",
            "max_gen_toks": "512",
            "seed": "42",
        }


# --- scoring -> Block (the copied logic) -------------------------------------


class TestBlocksForTask:
    def _blocks(self, task, results, **kwargs):
        with patch(f"{_MOD}.block_id", return_value=""):
            return mod.blocks_for_task(_ctx(), task, results, **kwargs)

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

    def test_mean_seconds_per_task_uses_wall_time_and_effective_samples(self):
        (block,) = self._blocks(
            _task(score=_score(published=90.5, reference=90.91)),
            {"gpqa": {"acc,none": 0.9}},
            sample_counts={"gpqa": 10},
            elapsed_seconds=900.0,
        )

        assert block.data["mean_seconds_per_task"] == 90.0

    def test_group_mean_seconds_uses_total_subtask_sample_count(self):
        blocks = self._blocks(
            _task("longbench", _score(published=50.0, reference=50.0)),
            {
                "longbench_2wikimqa": {"acc,none": 0.6},
                "longbench_hotpotqa": {"acc,none": 0.4},
            },
            sample_counts={
                "longbench_2wikimqa": 2,
                "longbench_hotpotqa": 3,
            },
            elapsed_seconds=50.0,
        )

        assert {block.data["mean_seconds_per_task"] for block in blocks} == {10.0}

    def test_mean_seconds_per_task_is_omitted_without_sample_count(self):
        (block,) = self._blocks(
            _task(score=_score(published=90.5, reference=90.91)),
            {"gpqa": {"acc,none": 0.9}},
            elapsed_seconds=900.0,
        )

        assert "mean_seconds_per_task" not in block.data

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

    def test_task_without_score_is_na(self):
        # A task that ran but has no score defined is not gradable -> NA (was
        # previously dropped, which made the caller mislabel it as a FAIL).
        (b,) = self._blocks(_task(score=None), {"gpqa": {"acc,none": 0.9}})
        assert b.data["status"] == TestStatus.NA.value
        assert b.data["reason"] == "no eval score defined"


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
        results, counts = mod.load_eval_results(
            [str(tmp_path / "results_1.json"), str(tmp_path / "results_2.json")]
        )
        assert set(results) == {"gpqa", "mmlu"}
        assert results["gpqa"]["acc,none"] == 0.9
        assert "alias" not in results["gpqa"]
        assert counts == {}

    @pytest.mark.parametrize("new_count", [10, None])
    def test_latest_metrics_and_count_come_from_the_same_file(
        self, tmp_path, new_count
    ):
        old = tmp_path / "results_old.json"
        new = tmp_path / "results_new.json"
        for path, metric, count, modified in (
            (old, 0.5, 100, 1),
            (new, 0.9, new_count, 2),
        ):
            self._write(path, "gpqa", metric)
            data = json.loads(path.read_text())
            if count is not None:
                data["n-samples"] = {"gpqa": {"effective": count}}
            path.write_text(json.dumps(data))
            os.utime(path, (modified, modified))

        with patch.object(mod.json, "load", wraps=json.load) as read:
            results, counts = mod.load_eval_results([str(old), str(new)])

        assert read.call_count == 2
        assert results == {"gpqa": {"acc,none": 0.9}}
        assert counts == ({"gpqa": new_count} if new_count is not None else {})

    def test_mmlu_pro_group_count_feeds_the_noise_rule(self, tmp_path):
        """lm-eval writes n-samples for mmlu_pro's 14 leaf subjects only. Its
        group exact_match is the size-weighted mean of the leaves (one binomial
        over every question), so the group's n is the leaf sum, which
        _entry_sample_counts derives through group_subtasks and the noise rule
        then uses for the single-key mmlu_pro score."""
        subjects = {  # the real MMLU-Pro test split sizes (12,032 questions)
            "biology": 717, "business": 789, "chemistry": 1132,
            "computer_science": 410, "economics": 844, "engineering": 969,
            "health": 818, "history": 381, "law": 1101, "math": 1351,
            "other": 924, "philosophy": 499, "physics": 1299, "psychology": 798,
        }  # fmt: skip
        leaves = {f"mmlu_pro_{name}": n for name, n in subjects.items()}
        correct = {leaf: n // 9 for leaf, n in leaves.items()}
        group = sum(correct.values()) / sum(leaves.values())
        results = {
            "mmlu_pro": {"exact_match,custom-extract": group, "alias": "mmlu_pro"}
        }
        results.update(
            {
                leaf: {"exact_match,custom-extract": correct[leaf] / n}
                for leaf, n in leaves.items()
            }
        )
        path = tmp_path / "results_mmlu_pro.json"
        path.write_text(
            json.dumps(
                {
                    "results": results,
                    "configs": {
                        leaf: {"task": leaf, "dataset_path": "TIGER-Lab/MMLU-Pro"}
                        for leaf in leaves
                    },
                    "group_subtasks": {"mmlu_pro": list(leaves)},
                    "n-samples": {
                        leaf: {"original": n, "effective": n}
                        for leaf, n in leaves.items()
                    },
                }
            )
        )

        _, counts = mod.load_eval_results([str(path)])

        assert counts["mmlu_pro"] == 12032
        # Size-weighted: the group score is pooled correct / pooled n.
        assert group == pytest.approx(
            sum(
                results[leaf]["exact_match,custom-extract"] * n
                for leaf, n in leaves.items()
            )
            / 12032
        )
        assert (
            mod.binomial_noise_n(["exact_match,custom-extract"], "mmlu_pro", counts)
            == 12032.0
        )


# --- orchestration -----------------------------------------------------------


class TestRunLLMEval:
    @pytest.mark.parametrize(
        "bad_result",
        [
            "{",
            "{}",
            "[]",
            '{"results": []}',
            '{"results": {"gpqa": null}}',
            json.dumps(
                {
                    "results": {"gpqa": {"acc,none": 0.9}},
                    "configs": {"gpqa": {"task": "wrong-task", "dataset_path": "d"}},
                }
            ),
            '{"results": {}, "n-samples": [1]}',
        ],
        ids=[
            "truncated",
            "empty",
            "list",
            "bad-results",
            "bad-metrics",
            "task-mismatch",
            "bad-counts",
        ],
    )
    def test_bad_result_does_not_reuse_old_scores_or_stop_later_tasks(
        self, tmp_path, bad_result, caplog
    ):
        from workflow_module import BlockAccumulator

        ctx = _ctx()
        ctx.model_spec.model_id = "test-llm"
        ctx.output_path = str(tmp_path)
        tasks = [_task(name, _score(reference=90.0)) for name in ("gpqa", "mmlu")]
        result_dir = tmp_path / "eval_test-llm" / "org__test-llm"
        result_dir.mkdir(parents=True)
        old_result = result_dir / "results_old.json"
        old_result.write_text(
            json.dumps(
                {
                    "results": {"gpqa": {"acc,none": 0.95}},
                    "configs": {"gpqa": {"task": "gpqa", "dataset_path": "d"}},
                }
            )
        )
        accumulator = BlockAccumulator()
        ran = []
        attempt_dirs = []

        def run_task(_ctx, task, _token, *, output_path):
            ran.append(task.task_name)
            attempt_dirs.append(output_path)
            result_dir = output_path / "eval_test-llm" / "org__test-llm"
            result_dir.mkdir(parents=True)
            if task.task_name == "gpqa":
                (result_dir / "results_gpqa.json").write_text(bad_result)
                return 1
            # The failed task must already be recorded before the next starts.
            assert len(accumulator.blocks) == 1
            assert accumulator.blocks[0].data["accuracy_check"] == ReportCheckTypes.FAIL
            (result_dir / "results_mmlu.json").write_text(
                json.dumps(
                    {
                        "results": {"mmlu": {"acc,none": 0.95}},
                        "configs": {"mmlu": {"task": "mmlu", "dataset_path": "d"}},
                        "n-samples": {"mmlu": {"effective": 10}},
                    }
                )
            )
            return 0

        server = MagicMock()
        server.wait_for_healthy.return_value = True
        server.get_health.return_value = SimpleNamespace(status_code=200)
        with patch(f"{_MOD}.get_llm_eval_tasks", return_value=tasks), patch(
            f"{_MOD}.HttpServerController", return_value=server
        ), patch(f"{_MOD}._run_eval_task", side_effect=run_task), patch(
            f"{_MOD}.accept_blocks", side_effect=accumulator.accept
        ):
            blocks = mod.run_llm_eval(ctx)

        assert ran == ["gpqa", "mmlu"]
        assert len(set(attempt_dirs)) == 2
        assert old_result.exists()
        assert blocks == accumulator.blocks
        assert blocks[0].data["accuracy_check"] == ReportCheckTypes.FAIL
        assert "no eval results parsed (rc=1)" in blocks[0].data["error"]
        assert blocks[1].data["accuracy_check"] == ReportCheckTypes.PASS
        assert blocks[1].data["score"] == 95.0
        assert "mean_seconds_per_task" in blocks[1].data
        assert "results_gpqa.json" in caplog.text

    def _run(self, tasks, *, healthy=True, blocks=None, run_rc=0, results=None):
        server = MagicMock()
        server.wait_for_healthy.return_value = healthy
        server.get_health.return_value = SimpleNamespace(status_code=200)
        with patch(f"{_MOD}.get_llm_eval_tasks", return_value=tasks), patch(
            f"{_MOD}.HttpServerController", return_value=server
        ), patch(f"{_MOD}._run_eval_task", return_value=run_rc) as run_task, patch(
            f"{_MOD}.discover_eval_results", return_value=["f.json"]
        ), patch(f"{_MOD}.load_eval_results", return_value=(results or {}, {})), patch(
            f"{_MOD}.blocks_for_task", return_value=blocks if blocks is not None else []
        ) as score_task, patch(f"{_MOD}.accept_blocks") as accept, patch(
            f"{_MOD}.block_id", return_value=""
        ):
            out = mod.run_llm_eval(_ctx())
        return out, run_task, score_task, accept

    def test_happy_path(self):
        blk = MagicMock()
        blk.kind = "evals"
        out, run_task, score_task, accept = self._run([_task()], blocks=[blk])
        assert out == [blk]
        run_task.assert_called_once()
        assert score_task.call_args.kwargs["elapsed_seconds"] >= 0
        accept.assert_called_once()

    def test_no_tasks(self):
        out, run_task, _score_task, accept = self._run([])
        assert out == []
        run_task.assert_not_called()
        accept.assert_not_called()

    def test_unhealthy_emits_fail_blocks(self):
        out, run_task, _score_task, _accept = self._run(
            [_task("a"), _task("b")], healthy=False
        )
        assert len(out) == 2
        assert all(b.data["accuracy_check"] == ReportCheckTypes.FAIL for b in out)
        run_task.assert_not_called()

    def test_ran_but_no_block_gets_fail_block(self):
        out, run_task, _score_task, _accept = self._run(
            [_task("gpqa")], blocks=[], run_rc=1
        )
        assert len(out) == 1
        assert out[0].data["accuracy_check"] == ReportCheckTypes.FAIL
        # main settled on this wording for a non-deadline subprocess failure
        # (see test_bad_result_does_not_reuse_old_scores_or_stop_later_tasks);
        # this PR only adds the rc=124 deadline case on top of it.
        assert "no eval results parsed (rc=1)" in out[0].data["error"]
        run_task.assert_called_once()

    def test_deadline_cannot_score_partial_results_as_pass(self):
        out, _, score_task, _ = self._run([_task()], blocks=[MagicMock()], run_rc=124)
        score_task.assert_not_called()
        assert out[0].data["subprocess_rc"] == 124
        assert out[0].data["accuracy_check"] == ReportCheckTypes.FAIL
        assert "incomplete" in out[0].data["error"]

    def test_min_context_skip(self):
        # Task needs more context than the device provides: not run, but now a
        # visible SKIP block instead of silently vanishing.
        out, run_task, _score_task, _accept = self._run(
            [_task("longctx", min_context_required=200000)]
        )
        run_task.assert_not_called()
        assert len(out) == 1
        assert out[0].data["status"] == TestStatus.SKIP.value
        assert "requires max_context >= 200000" in out[0].data["reason"]

    def _run_killed_on_second_task(self, tasks, blocks_by_task):
        """Run ``tasks``; the second ``_run_eval_task`` call is a GitHub cancel."""
        server = MagicMock()
        server.wait_for_healthy.return_value = True
        server.get_health.return_value = SimpleNamespace(status_code=200)
        run_calls = []

        def run_eval_task(_ctx, task, _token, *, output_path):
            run_calls.append(task.task_name)
            if len(run_calls) == 2:
                raise KeyboardInterrupt  # SIGINT from a GitHub cancel
            return 0

        with patch(f"{_MOD}.get_llm_eval_tasks", return_value=tasks), patch(
            f"{_MOD}.HttpServerController", return_value=server
        ), patch(f"{_MOD}._run_eval_task", side_effect=run_eval_task), patch(
            f"{_MOD}.discover_eval_results", return_value=["f.json"]
        ), patch(f"{_MOD}.load_eval_results", return_value=({}, {})), patch(
            f"{_MOD}.blocks_for_task",
            side_effect=lambda _ctx, task, *_a, **_k: blocks_by_task[task.task_name],
        ), patch(f"{_MOD}.accept_blocks") as accept, patch(
            f"{_MOD}.block_id", return_value=""
        ), pytest.raises(KeyboardInterrupt):
            mod.run_llm_eval(_ctx())
        return accept

    def test_each_task_is_scored_and_accepted_before_the_next_one_runs(self):
        """Accepting checkpoints the report, so a cancel during task N must find
        tasks 1..N-1 already scored and accepted -- not waiting on a post-loop
        parse that never happens."""
        gpqa = MagicMock()
        accept = self._run_killed_on_second_task(
            [_task("gpqa"), _task("mmlu")], {"gpqa": [gpqa], "mmlu": [MagicMock()]}
        )
        accept.assert_called_once()
        assert accept.call_args.args[0] == [gpqa]

    def test_skipped_task_is_accepted_as_it_is_skipped(self):
        accept = self._run_killed_on_second_task(
            [
                _task("longctx", min_context_required=200000),
                _task("gpqa"),
                _task("mmlu"),
            ],
            {"gpqa": [MagicMock()], "mmlu": [MagicMock()]},
        )
        accepted = [b for call in accept.call_args_list for b in call.args[0]]
        assert accepted[0].data["status"] == TestStatus.SKIP.value

    def test_skip_block_keeps_its_place_in_task_order(self):
        # blocks_for_task is mocked to [], so ran tasks become FAIL blocks; the
        # SKIP for "b" must sit between "a" and "c", in config order.
        out, _run_task, _score_task, _accept = self._run(
            [_task("a"), _task("b", min_context_required=200000), _task("c")]
        )
        assert [blk.data.get("status") for blk in out][1] == TestStatus.SKIP.value


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

    def test_incomplete_subprocess_is_nonzero_workflow(self):
        wf = self._wf("llm")
        block = SimpleNamespace(kind="evals", data={"subprocess_rc": 124})
        with patch(f"{_MOD}.run_llm_eval", return_value=[block]):
            assert wf.run_tasks()[0].exit_code == 1

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


class TestDeadlineReachesRunCommand:
    """Regression guard: a declared deadline must reach proc.run_command.

    wall_clock_timeout_seconds was previously declared, validated, documented
    and unit-tested for validation only, yet never passed to run_command -- so
    the bounded path in proc.py was unreachable and the field had no effect.
    """

    def _invoke(self, task):
        seen = {}

        def fake_run_command(**kwargs):
            seen.update(kwargs)
            return 0

        with patch(f"{_MOD}.build_eval_command", return_value=["echo", "x"]), patch(
            f"{_MOD}.run_command", side_effect=fake_run_command
        ):
            rc = mod._run_eval_task(_ctx(), task, "", output_path=tmp_output_path())
        return rc, seen

    def test_declared_deadline_is_passed_through(self):
        rc, seen = self._invoke(
            EvalTask(task_name="aime25", wall_clock_timeout_seconds=3600)
        )
        assert rc == 0
        assert seen.get("timeout_seconds") == 3600

    def test_absent_deadline_leaves_execution_unbounded(self):
        rc, seen = self._invoke(EvalTask(task_name="aime25"))
        assert rc == 0
        # Passing timeout_seconds=None would still select the bounded POSIX
        # path, so unbounded callers must omit the argument entirely.
        assert "timeout_seconds" not in seen


def test_harness_window_keeps_a_block_of_headroom_below_max_context():
    """A long prompt must land strictly inside the served context.

    lm-eval truncates to max_length - 1 - max_gen_toks, so max_length ==
    max_context puts prompt + max_tokens at max_context - 1 (the aime25 empty-
    response boundary on TT). The window keeps one paged-KV block of slack.
    """
    command = _build_eval_test_command(
        EvalTask(task_name="long_context", min_context_required=16384),
        max_context=131072,
    )
    max_length = int(_command_model_kwargs(command)["max_length"])
    assert max_length == 131072 - 64


def test_multilevel_result_keys_are_not_replaced_by_metric_autodetect():
    # leaderboard_math_hard is scored as the mean of per-subtask paths. The
    # group entry itself carries an aggregate exact_match, which the
    # metric-mismatch auto-detect used to substitute as a bare string --
    # score_multilevel_keys_mean then asserted and the task scored 0.
    from reference_config.evals.eval_utils import score_multilevel_keys_mean

    subtasks = [
        "leaderboard_math_algebra_hard",
        "leaderboard_math_counting_and_prob_hard",
        "leaderboard_math_geometry_hard",
        "leaderboard_math_intermediate_algebra_hard",
        "leaderboard_math_num_theory_hard",
        "leaderboard_math_prealgebra_hard",
        "leaderboard_math_precalculus_hard",
    ]
    task = SimpleNamespace(
        task_name="leaderboard_math_hard",
        score=SimpleNamespace(
            score_func=score_multilevel_keys_mean,
            score_func_kwargs={
                "result_keys": [(name, "exact_match,none") for name in subtasks],
                "unit": "percent",
            },
            published_score=1.87,
        ),
    )
    results = {"leaderboard_math_hard": {"exact_match,none": 0.5, "alias": "math"}}
    results.update({name: {"exact_match,none": 0.02} for name in subtasks})
    ref = {"reference_score": None, "tolerance": 0.05}
    score, ratio, _, check = mod._score_one(task, results, "leaderboard_math_hard", ref)
    assert score == pytest.approx(2.0)
    assert ratio == pytest.approx(2.0 / task.score.published_score)
    assert check == ReportCheckTypes.PASS


def test_group_task_subtasks_survive_loading_and_score(tmp_path):
    # leaderboard_math_hard's results file lists the group first and its seven
    # subtasks as siblings. The loader used to keep only the first entry, so
    # score_multilevel_keys_mean never saw the subtasks and the task scored 0.
    from reference_config.evals.eval_utils import score_multilevel_keys_mean

    subtasks = ["leaderboard_math_algebra_hard", "leaderboard_math_geometry_hard"]
    payload = {
        "results": {
            "leaderboard_math_hard": {"exact_match,none": 0.5, "alias": "math"},
            **{
                name: {"alias": f" - {name}", "exact_match,none": 0.02}
                for name in subtasks
            },
        },
        "configs": {
            name: {"task": name, "dataset_path": "math"}
            for name in ["leaderboard_math_hard", *subtasks]
        },
    }
    path = tmp_path / "results_2026-10-05T10-17-59.json"
    path.write_text(json.dumps(payload))
    results, _ = mod.load_eval_results([str(path)])
    assert set(subtasks) <= set(results)

    task = SimpleNamespace(
        task_name="leaderboard_math_hard",
        score=SimpleNamespace(
            score_func=score_multilevel_keys_mean,
            score_func_kwargs={
                "result_keys": [(name, "exact_match,none") for name in subtasks],
                "unit": "percent",
            },
            published_score=1.87,
        ),
    )
    ref = {"reference_score": None, "tolerance": 0.05}
    score, _, _, check = mod._score_one(task, results, "leaderboard_math_hard", ref)
    assert score == pytest.approx(2.0)
    assert check == ReportCheckTypes.PASS


@pytest.mark.parametrize("impl_id", ["tt_transformers", "llama31_8b_qb2"])
def test_llama31_longbench_preserves_generation_settings(
    impl_id, tmp_path, monkeypatch
):
    from llm_module.eval_configs import get_llm_eval_tasks
    from workflows.model_spec import load_templates_from_yaml
    from workflows.utils import get_repo_root_path
    from workflows.workflow_types import DeviceTypes

    templates = load_templates_from_yaml(
        get_repo_root_path() / "workflows/model_specs/dev/llm.yaml"
    )
    model_spec = next(
        spec
        for template in templates
        if template.impl.impl_id == impl_id
        for spec in template.expand_to_specs()
        if spec.hf_model_repo == "meta-llama/Llama-3.1-8B-Instruct"
        and spec.device_type == DeviceTypes.P300X2
    )
    tokenizer_calls = []

    def pinned_tokenizer(spec, output):
        tokenizer_calls.append((spec, output))
        return "/verified/tokenizer"

    monkeypatch.setattr(f"{_MOD}.resolve_tokenizer", pinned_tokenizer)
    tasks = get_llm_eval_tasks(model_spec)
    longbench = {t.task_name: t for t in tasks if t.task_name.startswith("longbench_")}
    references = {
        "longbench_code_e": 48.12,
        "longbench_fewshot_e": 63.34,
        "longbench_multi_e": 20.84,
        "longbench_single_e": 22.22,
        "longbench_summarization_e": 26.09,
        "longbench_synthetic_e": 14.86,
    }
    assert set(longbench) == set(references)
    for name, task in longbench.items():
        tokenizer = mod._prepare_eval_tokenizer(
            SimpleNamespace(model_spec=model_spec), task, tmp_path
        )
        command = build_eval_command(
            task,
            model_spec,
            DeviceTypes.P300X2,
            tmp_path,
            8000,
            tokenizer_path=tokenizer,
        )
        assert "--apply_chat_template" not in command
        assert "--limit" not in command
        assert command[command.index("--model") + 1] == "local-completions"
        assert (
            "base_url=http://127.0.0.1:8000/v1/completions"
            in command[command.index("--model_args") + 1]
        )
        args = dict(
            item.split("=", 1)
            for item in command[command.index("--model_args") + 1].split(",")
        )
        assert args["max_length"] == str(model_spec.device_model_spec.max_context - 64)
        if impl_id == "llama31_8b_qb2":
            assert args["tokenizer"] == "/verified/tokenizer"
            assert args["num_concurrent"] == "32"
            assert "max_length" not in task.model_kwargs  # Do not change shared tasks.
        else:
            assert "tokenizer" not in args
        gen_kwargs = _command_gen_kwargs(command)
        assert gen_kwargs["temperature"] == "0"
        assert gen_kwargs["max_gen_toks"] == "512"
        assert task.score.gpu_reference_score == references[name]
        assert task.score.tolerance == 0.05
    assert len(tokenizer_calls) == (6 if impl_id == "llama31_8b_qb2" else 0)


@pytest.mark.parametrize(
    "task_name,venv",
    [
        ("meta_ifeval", WorkflowVenvType.EVALS_META),
        ("other_task", WorkflowVenvType.EVALS_COMMON),
    ],
)
def test_qb2_tokenizer_pinning_preserves_upstream_other_eval_context(
    task_name, venv, tmp_path
):
    spec = SimpleNamespace(
        hf_model_repo="org/model",
        model_id="model",
        impl=SimpleNamespace(impl_id="llama31_8b_qb2"),
        device_model_spec=SimpleNamespace(max_context=131072, max_concurrency=32),
    )
    task = EvalTask(task_name=task_name, workflow_venv_type=venv, include_path=None)
    with patch(f"{_MOD}.resolve_tokenizer") as resolve:
        assert (
            mod._prepare_eval_tokenizer(
                SimpleNamespace(model_spec=spec), task, tmp_path
            )
            is None
        )
        command = build_eval_command(task, spec, None, tmp_path, 8000)
    resolve.assert_not_called()
    args = command[command.index("--model_args") + 1]
    assert "max_length=131008" in args
    assert ",tokenizer=" not in args


@pytest.mark.parametrize("max_context", [None, 0, "invalid", True])
def test_longbench_rejects_invalid_context(max_context, tmp_path):
    spec = SimpleNamespace(
        hf_model_repo="org/model",
        model_id="model",
        impl=SimpleNamespace(impl_id="llama31_8b_qb2"),
        device_model_spec=SimpleNamespace(max_context=max_context, max_concurrency=32),
    )
    task = EvalTask(task_name="longbench_single_e", gen_kwargs={})
    with pytest.raises(ValueError, match="(max_context|positive integer)"):
        build_eval_command(task, spec, None, tmp_path, 8000)


def test_qb2_longbench_rejects_wrong_tokenizer_before_launch(tmp_path):
    spec = SimpleNamespace(
        hf_model_repo="org/model",
        model_id="model",
        impl=SimpleNamespace(impl_id="llama31_8b_qb2"),
        device_model_spec=SimpleNamespace(
            max_context=131072,
            max_concurrency=32,
            vllm_args={"revision": "a" * 40, "tokenizer_revision": "a" * 40},
        ),
    )
    task = EvalTask(task_name="longbench_single_e")
    with patch(
        f"{_MOD}.resolve_tokenizer",
        side_effect=ValueError("Tokenizer files differ"),
    ):
        with pytest.raises(ValueError, match="Tokenizer files differ"):
            mod._prepare_eval_tokenizer(
                SimpleNamespace(model_spec=spec), task, tmp_path
            )
