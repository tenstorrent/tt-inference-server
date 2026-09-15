import json
from dataclasses import replace

import pytest

from llm_module.config import LLMRunConfig, ServerConnection
from llm_module.custom_longbench import build_longbench_configs
from llm_module.drivers.vllm import build_vllm_bench_serve_argv


def test_longbench_selects_matching_points_without_changing_load(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(
            json.dumps({"prompt": f"Document {i}", "input_tokens": 8192}) + "\n"
            for i in range(16)
        )
    )
    baseline = [
        LLMRunConfig(128, 128, 1, 8),
        LLMRunConfig(8192, 128, 1, 2),
        LLMRunConfig(8192, 1024, 8, 16),
    ]
    selected = build_longbench_configs(baseline, source, tmp_path / "prepared")
    assert [(c.isl, c.osl, c.max_concurrency, c.num_prompts) for c in selected] == [
        (8192, 128, 1, 2),
        (8192, 1024, 8, 16),
    ]
    assert all(c.custom_dataset_path is None for c in baseline)
    assert all(
        json.loads(line).keys() == {"prompt"}
        for line in selected[0].custom_dataset_path.read_text().splitlines()
    )
    server = ServerConnection(
        "https://example.test", 443, "zai-org/GLM-5.3", is_remote=True
    )
    common = dict(
        vllm_binary="vllm", server=server, result_filename=tmp_path / "result.json"
    )
    random_argv, _ = build_vllm_bench_serve_argv(config=baseline[2], **common)
    custom_argv, _ = build_vllm_bench_serve_argv(config=selected[1], **common)
    assert random_argv[random_argv.index("--dataset-name") + 1] == "random"
    assert custom_argv[custom_argv.index("--dataset-name") + 1] == "custom"
    assert "--skip-chat-template" in custom_argv
    assert "--ignore-eos" in custom_argv
    assert "--ignore-eos" not in random_argv  # vLLM already enables it for random.
    assert custom_argv[custom_argv.index("--custom-output-len") + 1] == "1024"
    assert (
        replace(selected[1], custom_dataset_path=None, ignore_eos=False) == baseline[2]
    )


@pytest.mark.parametrize(
    "rows,message",
    [
        ([], "do not match"),
        ([{"prompt": "x", "input_tokens": 1024}], "do not match"),
        ([{"prompt": "x", "input_tokens": 8192}], "refusing implicit oversampling"),
        ([{"prompt": "x"}], "require positive input_tokens"),
        ([{"prompt": "x", "input_tokens": True}], "require positive input_tokens"),
    ],
)
def test_invalid_or_insufficient_data_fails_before_any_benchmark(
    tmp_path, rows, message
):
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match=message):
        build_longbench_configs(
            [LLMRunConfig(8192, 128, 8, 16)], source, tmp_path / "prepared"
        )
    assert not (tmp_path / "prepared").exists()


@pytest.mark.parametrize("workflow", ["evals", "agentic", "release", "spec_tests"])
def test_custom_selector_cannot_change_other_workflows(monkeypatch, capsys, workflow):
    import sys
    from types import SimpleNamespace

    import run
    from run import parse_arguments

    monkeypatch.setattr(
        run,
        "MODEL_SPECS",
        {
            "glm53": SimpleNamespace(
                model_name="GLM-5.3",
                hf_model_repo="zai-org/GLM-5.3",
                impl=SimpleNamespace(impl_name="tt-transformers"),
            )
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--model",
            "GLM-5.3",
            "--workflow",
            workflow,
            "--device",
            "super_cluster",
            "--benchmark",
            "custom-longbench",
            "--dataset-path",
            "dataset.jsonl",
        ],
    )
    with pytest.raises(SystemExit) as error:
        parse_arguments()
    assert error.value.code == 2
    assert "requires --workflow benchmarks" in capsys.readouterr().err


@pytest.mark.parametrize("entrypoint", ["run", "run_workflows"])
def test_custom_selector_rejects_other_model_tokenizers(
    monkeypatch, capsys, entrypoint
):
    import importlib
    import sys

    module = importlib.import_module(entrypoint)
    parse = module.parse_arguments if entrypoint == "run" else module.parse_args
    monkeypatch.setattr(
        sys,
        "argv",
        [
            entrypoint,
            "--model",
            "Llama-3.1-8B-Instruct",
            "--workflow",
            "benchmarks",
            "--device",
            "super_cluster",
            "--benchmark",
            "custom-longbench",
            "--dataset-path",
            "dataset.jsonl",
        ],
    )
    with pytest.raises(SystemExit) as error:
        parse()
    assert error.value.code == 2
    assert "requires --model GLM-5.3" in capsys.readouterr().err


def test_direct_custom_benchmark_rejects_other_model_tokenizers():
    from types import SimpleNamespace

    from test_module.llm_tests.llm_benchmark_tests import run_llm_bench

    context = SimpleNamespace(model_spec=SimpleNamespace(model_name="GLM-5.2"))
    with pytest.raises(ValueError, match="GLM-5.3"):
        run_llm_bench(
            context, benchmark="custom-longbench", dataset_path="dataset.jsonl"
        )


def test_existing_positional_benchmark_options_keep_their_meaning():
    from pathlib import Path

    from workflow_module.execution import LLMBenchOptions

    config = LLMRunConfig(
        128,
        128,
        1,
        8,
        {},
        1,
        Path("custom.jsonl"),
        "should",
        {"tput": "should"},
        "ttft:2000",
    )
    assert config.priority == "should"
    assert config.target_priorities == {"tput": "should"}
    assert config.goodput == "ttft:2000"
    assert config.ignore_eos is False
    options = LLMBenchOptions("aiperf", "token", "/python", "ttft:2000")
    assert options.auth_token == "token"
    assert options.venv_python == "/python"
    assert options.goodput == "ttft:2000"
    assert options.benchmark is None and options.dataset_path is None
