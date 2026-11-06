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
    assert args["max_num_seqs"] == 16
    assert spec.device_model_spec.max_concurrency == 16
    assert args["max_num_batched_tokens"] == 65536
    assert args["num_gpu_blocks_override"] == 49153
    assert args["distributed_executor_backend"] == "mp"
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
    assert gpqa.capture_reasoning is True
    assert gpqa.log_samples is True
    assert gpqa.max_concurrent == 16
    assert len(gpqa.doc_ids) == len(set(gpqa.doc_ids)) == 198
    assert gpqa.limit_samples_map[EvalLimitMode.CI_NIGHTLY] == 16
    assert gpqa.gen_kwargs["max_gen_toks"] == 81920
    assert gpqa.model_kwargs["timeout"] == 14400
    for task in (terminal, swe):
        cfg = task.agentic_eval_config
        assert len(cfg.task_names_map[EvalLimitMode.SMOKE_TEST]) == 2
        assert cfg.agent_timeout_sec == (36000 if task is swe else 21600)
        assert cfg.override_cpus is cfg.override_memory_mb is None
    assert (
        terminal.agentic_eval_config.agent_kwargs["llm_kwargs"]["max_tokens"] == 32768
    )
    cfg = terminal.agentic_eval_config
    assert cfg.n_concurrent_trials == 2
    assert terminal.limit_samples_map[EvalLimitMode.CI_NIGHTLY] == 0
    assert (
        cfg.dataset_ref
        == "sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a"
    )
    assert cfg.agent_kwargs["model_info"]["max_output_tokens"] == 32768
    assert cfg.agent_kwargs["llm_kwargs"]["timeout"] == 7200
    assert cfg.agent_kwargs["interleaved_thinking"] is False
    assert cfg.agent_kwargs["llm_kwargs"]["extra_body"]["chat_template_kwargs"] == {
        "reasoning_effort": "medium"
    }
    assert cfg.task_names_map[EvalLimitMode.CI_NIGHTLY] == [
        "terminal-bench/break-filter-js-from-html",
        "terminal-bench/portfolio-optimization",
    ]
    assert swe.limit_samples_map[EvalLimitMode.CI_NIGHTLY] == 0
    assert swe.agentic_eval_config.n_concurrent_trials == 5
    assert swe.agentic_eval_config.agent_kwargs["max_tokens"] == 32768
    assert swe.agentic_eval_config.llm_timeout_sec == 7200
    assert swe.agentic_eval_config.task_names_map[EvalLimitMode.CI_NIGHTLY] == [
        "django__django-11299",
        "astropy__astropy-14096",
        "matplotlib__matplotlib-25332",
        "sympy__sympy-13551",
        "scikit-learn__scikit-learn-14629",
    ]
    assert swe.agentic_eval_config.dataset_ref is None
    assert swe.agentic_eval_config.agent_kwargs["version"] == "2.2.8"
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
    assert "default-chat-template-kwargs" not in spec.device_model_spec.vllm_args
    assert spec.device_model_spec.max_tokens_all_users == 1572864
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
