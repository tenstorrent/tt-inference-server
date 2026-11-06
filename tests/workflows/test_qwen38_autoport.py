# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from pathlib import Path

from reference_config.benchmarking.benchmark_config import get_benchmark_config
from reference_config.evals.eval_config import _eval_config_map
from workflows.model_spec import load_templates_from_yaml
from workflows.workflow_types import EvalLimitMode


def test_autoport_uses_validated_serving_contract_and_exact_release_matrix():
    templates = load_templates_from_yaml(Path("workflows/model_specs/dev/llm.yaml"))
    spec = next(
        t for t in templates if t.impl.impl_id == "qwen38_autoport"
    ).expand_to_specs()[0]
    args = spec.device_model_spec.vllm_args
    assert args["max_num_seqs"] == 8
    assert args["block_size"] == 32
    assert args["no-enable-prefix-caching"] is True
    assert args["enable_auto_tool_choice"] is True
    assert args["tool_call_parser"] == "qwen3_coder"
    assert args["reasoning_parser"] == "qwen3"
    matrix = get_benchmark_config(spec)
    rows = matrix.tasks[0].param_map[spec.device_type]
    assert len(matrix.tasks) == 1
    assert len(rows) == 21
    assert {r.max_concurrency for r in rows} == {1, 8, 32}
    assert all(r.task_type == "text" for r in rows)
    assert any(
        (r.isl, r.osl, r.max_concurrency, r.num_prompts) == (4096, 128, 1, 4)
        for r in rows
    )


def test_smoke_cohorts_are_exact_and_limits_preserve_release_values():
    gpqa, terminal, swe = _eval_config_map["Qwen/Qwen3.8-27B"].tasks
    assert gpqa.max_concurrent == 2
    assert len(gpqa.doc_ids) == len(set(gpqa.doc_ids)) == 198
    assert gpqa.limit_samples_map[EvalLimitMode.CI_NIGHTLY] == 2
    assert gpqa.gen_kwargs["max_gen_toks"] == 81920
    assert gpqa.model_kwargs["timeout"] == 7200
    for task in (terminal, swe):
        cfg = task.agentic_eval_config
        assert cfg.n_concurrent_trials == 2
        assert len(cfg.task_names_map[EvalLimitMode.CI_NIGHTLY]) == 2
        assert (
            cfg.task_names_map[EvalLimitMode.SMOKE_TEST]
            == cfg.task_names_map[EvalLimitMode.CI_NIGHTLY]
        )
        assert cfg.agent_timeout_sec == 21600
        assert cfg.override_cpus is cfg.override_memory_mb is None
    assert (
        terminal.agentic_eval_config.agent_kwargs["llm_kwargs"]["max_tokens"] == 16384
    )
    assert swe.agentic_eval_config.agent_kwargs["max_tokens"] == 32768
    assert swe.agentic_eval_config.llm_timeout_sec == 3600
    assert swe.score.published_score is None
