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
    assert args["max_num_seqs"] == 32
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


def test_benchmark_entry_preserves_counts_and_uses_pinned_fixed_chat_protocol():
    import json

    from llm_module.benchmark_configs import get_llm_configs
    from llm_module.config import ServerConnection
    from llm_module.drivers.vllm import build_vllm_bench_serve_argv
    from utils.pinned_artifacts import get_pinned_revision

    templates = load_templates_from_yaml(Path("workflows/model_specs/dev/llm.yaml"))
    spec = next(
        t for t in templates if t.impl.impl_id == "qwen38_autoport"
    ).expand_to_specs()[0]
    assert spec.device_model_spec.max_tokens_all_users == 1049568
    assert (
        get_pinned_revision(spec, required=True)
        == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    )
    assert set(spec.metadata["benchmark_tokenizer_sha256"]) == {
        "tokenizer.json",
        "tokenizer_config.json",
    }
    configs = get_llm_configs(spec, spec.device_type)
    rows = get_benchmark_config(spec).tasks[0].param_map[spec.device_type]
    assert [(c.isl, c.osl, c.max_concurrency, c.num_prompts) for c in configs] == [
        (r.isl, r.osl, r.max_concurrency, r.num_prompts) for r in rows
    ]
    server = ServerConnection(
        "http://172.17.0.2", 8009, "Qwen/Qwen3.8-27B", tokenizer="/pinned/tokenizer"
    )
    for config in configs:
        assert config.token_timing and config.require_complete_metrics
        assert config.repetitions == 1 and not config.full_workload_warmup
        argv, _ = build_vllm_bench_serve_argv(
            vllm_binary="vllm",
            config=config,
            server=server,
            result_filename=Path("result.json"),
        )
        assert argv[argv.index("--backend") + 1] == "openai-chat"
        assert argv[argv.index("--endpoint") + 1] == "/v1/chat/completions"
        assert "--ignore-eos" in argv
        assert argv[argv.index("--random-range-ratio") + 1] == "0.0"
        assert argv[argv.index("--random-input-len") + 1] == str(config.isl)
        assert argv[argv.index("--random-output-len") + 1] == str(config.osl)
        assert argv[argv.index("--num-prompts") + 1] == str(config.num_prompts)
        body = json.loads(argv[argv.index("--extra-body") + 1])
        assert body["seed"] == 42 and body["truncate_prompt_tokens"] == config.isl
