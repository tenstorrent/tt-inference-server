# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""vLLM's structured-output benchmark under v2: the run configs, the client
that renders reasoning models' prompts with thinking off and checks that the
grammar applied, the driver argv and the parser."""

from __future__ import annotations

import json
import logging
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_module import structured_output_client as client
from llm_module.benchmark_configs import (
    get_structured_output_configs,
    structured_output_chat_template_kwargs,
)
from llm_module.config import DriverContext, ServerConnection, StructuredOutputRunConfig
from llm_module.drivers import vllm_structured_output as driver_module
from llm_module.drivers.vllm_structured_output import (
    VLLMStructuredOutputDriver,
    build_structured_output_argv,
)
from llm_module.parsers.vllm_structured_output import VLLMStructuredOutputParser
from reference_config.benchmarking.benchmark_config import STRUCTURED_OUTPUT_PAIRS
from workflows.model_spec import MODEL_SPECS
from workflows.workflow_types import ModelType


def _spec(vllm_args):
    return SimpleNamespace(device_model_spec=SimpleNamespace(vllm_args=vllm_args))


@pytest.mark.parametrize(
    "vllm_args, expected",
    [
        ({}, None),
        ({"reasoning_parser": "qwen3"}, {"enable_thinking": False}),
        ({"reasoning-parser": "mimo"}, {"enable_thinking": False}),
        (
            {
                "reasoning-parser": "gemma4",
                "default-chat-template-kwargs": '{"enable_thinking": true, "x": 1}',
            },
            {"enable_thinking": False, "x": 1},
        ),
        (
            {
                "reasoning_parser": "qwen3",
                "default_chat_template_kwargs": {"enable_thinking": False},
            },
            {"enable_thinking": False},
        ),
    ],
)
def test_thinking_is_turned_off_only_for_a_server_with_a_reasoning_parser(
    vllm_args, expected
):
    assert structured_output_chat_template_kwargs(_spec(vllm_args)) == expected


def test_metadata_reasoning_parser_name_alone_keeps_the_prompts():
    spec = _spec({})
    spec.metadata = {"reasoning_parser_name": "qwen3"}
    assert structured_output_chat_template_kwargs(spec) is None


def test_every_llm_spec_runs_the_structured_output_pairs():
    llm_specs = [s for s in MODEL_SPECS.values() if s.model_type == ModelType.LLM]
    assert llm_specs
    for spec in llm_specs[:20]:
        configs = get_structured_output_configs(spec, spec.device_type)
        assert [(c.dataset, c.structured_output_ratio) for c in configs] == list(
            STRUCTURED_OUTPUT_PAIRS
        ), spec.model_id
        kwargs = structured_output_chat_template_kwargs(spec)
        assert all(c.chat_template_kwargs == kwargs for c in configs)
        assert all(c.isl == 0 and c.repetitions == 1 for c in configs)


class FakeTokenizer:
    """Renders a chat as ``<role>content`` turns plus the template arguments."""

    def __init__(self):
        self.renders = []

    def apply_chat_template(self, conversation, **kwargs):
        self.renders.append(kwargs)
        turns = "".join(f"<{m['role']}>{m['content']}" for m in conversation)
        thinking = kwargs.get("enable_thinking", True)
        return turns + ("<assistant>" if thinking else "<assistant><think></think>")

    def __call__(self, text):
        return SimpleNamespace(input_ids=text.split())


def _request(prompt, schema=None):
    return SimpleNamespace(prompt=prompt, prompt_len=0, schema=schema)


def test_raw_prompts_become_chat_turns_with_the_template_arguments():
    tokenizer = FakeTokenizer()

    def sample_requests(tok, args):
        return [_request("make a profile"), _request("make another")]

    wrapped = client.chat_sample_requests(sample_requests, {"enable_thinking": False})
    requests = wrapped(tokenizer, SimpleNamespace(dataset="json"))

    assert [r.prompt for r in requests] == [
        "<user>make a profile<assistant><think></think>",
        "<user>make another<assistant><think></think>",
    ]
    assert requests[0].prompt_len == 3
    assert all(r["enable_thinking"] is False for r in tokenizer.renders)


def test_chat_templated_prompts_get_the_template_arguments_added():
    tokenizer = FakeTokenizer()

    def sample_requests(tok, args):
        # As the script does for xgrammar_bench: it renders the dataset's chat.
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return [_request(prompt)]

    wrapped = client.chat_sample_requests(sample_requests, {"enable_thinking": False})
    (request,) = wrapped(tokenizer, SimpleNamespace(dataset="xgrammar_bench"))

    assert request.prompt == "<system>s<user>u<assistant><think></think>"
    assert tokenizer.renders == [
        {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    ]


SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "required": ["name"],
}


def test_output_checks_tell_an_unconstrained_output_from_a_constrained_one():
    outputs = [
        {"generated": '{"name": "a"}'},
        {"generated": ' {"name": "b"}'},
        {"generated": '{"name": 3}'},
        {"generated": '{"name": "trunc'},
        {"generated": 'Sure:\n```json\n{"name": "c"}\n```'},
        {"generated": "plain request"},
    ]
    checks = client.output_checks(
        outputs, [0, 1, 2, 3, 4], [json.dumps(SCHEMA)] * 6, "xgrammar_bench"
    )
    assert checks["structured_requests"] == 5
    assert checks["grammar_active_rate"] == pytest.approx(4 / 5)
    assert checks["complete_json_rate"] == pytest.approx(3 / 5)
    assert checks["schema_valid_rate"] == pytest.approx(2 / 5)


def test_output_checks_skip_runs_without_a_json_schema():
    outputs = [{"generated": "Positive"}]
    assert (
        client.output_checks(outputs, [0], [["Positive"]], "choice")[
            "grammar_active_rate"
        ]
        is None
    )
    assert (
        client.output_checks(outputs, [], [None], "json")["grammar_active_rate"] is None
    )


FAKE_SCRIPT = textwrap.dedent(
    """
    import argparse, json, os, random

    class Req:
        def __init__(self, prompt, schema):
            self.prompt, self.prompt_len, self.schema = prompt, 0, schema

    def create_argument_parser():
        p = argparse.ArgumentParser()
        p.add_argument("--dataset", default="json")
        p.add_argument("--num-prompts", type=int, default=4)
        p.add_argument("--structured-output-ratio", type=float, default=1.0)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--save-results", action="store_true")
        p.add_argument("--result-dir")
        p.add_argument("--result-filename")
        return p

    def sample_requests(tokenizer, args):
        schema = {"type": "object", "required": ["name"]}
        return [Req(f"prompt {i}", schema) for i in range(args.num_prompts)]

    def main(args):
        random.seed(args.seed)
        requests = sample_requests(Tok(), args)
        n = len(requests)
        picked = random.sample(range(n), int(n * args.structured_output_ratio))
        # A server whose grammar applies only to the picked requests that were
        # rendered with the reasoning already ended.
        outputs = [
            {"generated": '{"name": "x"}' if i in picked and r.prompt.endswith("</think>")
             else "Here is JSON: {}", "expected": None}
            for i, r in enumerate(requests)
        ]
        path = os.path.join(args.result_dir, args.result_filename)
        json.dump({"outputs": outputs, "correct_rate(%)": 100.0}, open(path, "w"))

    class Tok:
        def apply_chat_template(self, conversation, **kwargs):
            end = "</think>" if kwargs.get("enable_thinking") is False else ""
            return conversation[-1]["content"] + end

        def __call__(self, text):
            class R:
                input_ids = text.split()
            return R()
    """
)


def _run_client(tmp_path, *extra):
    script = tmp_path / "benchmark_serving_structured_output.py"
    script.write_text(FAKE_SCRIPT)
    sys.modules.pop(script.stem, None)
    argv = [
        "--script",
        str(script),
        "--num-prompts",
        "4",
        "--structured-output-ratio",
        "0.5",
        "--save-results",
        "--result-dir",
        str(tmp_path),
        "--result-filename",
        "out.json",
        *extra,
    ]
    assert client.main(argv) == 0
    return json.loads((tmp_path / "out.json").read_text())["tt_structured_output"]


def test_client_without_template_arguments_reports_the_grammar_inactive(tmp_path):
    checks = _run_client(tmp_path)
    assert checks["structured_requests"] == 2
    assert checks["grammar_active_rate"] == 0
    assert checks["chat_template_kwargs"] is None


def test_client_with_thinking_off_reports_the_grammar_active(tmp_path):
    checks = _run_client(
        tmp_path, "--chat-template-kwargs", '{"enable_thinking": false}'
    )
    assert checks["structured_requests"] == 2
    assert checks["grammar_active_rate"] == 1
    assert checks["schema_valid_rate"] == 1
    assert checks["chat_template_kwargs"] == {"enable_thinking": False}


def _config(**overrides):
    values = dict(
        dataset="xgrammar_bench",
        structured_output_ratio=1.0,
        osl=128,
        max_concurrency=4,
        num_prompts=100,
        chat_template_kwargs={"enable_thinking": False},
    )
    values.update(overrides)
    return StructuredOutputRunConfig(**values)


def _server(**overrides):
    values = dict(
        base_url="http://127.0.0.1",
        service_port=8000,
        model="XiaomiMiMo/MiMo-V2.6-Flash-MOPD",
        tokenizer_trust_remote_code=True,
    )
    values.update(overrides)
    return ServerConnection(**values)


def test_argv_carries_the_pair_the_server_and_the_template_arguments():
    cmd, token = build_structured_output_argv(
        venv_python=Path("/venv/bin/python"),
        script=Path("/venv/work_dir/benchmark_serving_structured_output.py"),
        config=_config(),
        server=_server(),
        result_path=Path("/out/run.json"),
    )
    assert cmd[:2] == ["/venv/bin/python", str(driver_module.CLIENT)]
    assert cmd[cmd.index("--dataset") + 1] == "xgrammar_bench"
    assert cmd[cmd.index("--structured-output-ratio") + 1] == "1.0"
    assert json.loads(cmd[cmd.index("--chat-template-kwargs") + 1]) == {
        "enable_thinking": False
    }
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert cmd[cmd.index("--port") + 1] == "8000"
    assert cmd[cmd.index("--result-dir") + 1] == "/out"
    assert "--trust-remote-code" in cmd
    assert token == ""


def test_argv_without_ratio_or_template_arguments():
    cmd, _ = build_structured_output_argv(
        venv_python=Path("/venv/bin/python"),
        script=Path("/s.py"),
        config=_config(structured_output_ratio=None, chat_template_kwargs=None),
        server=_server(tokenizer_trust_remote_code=False),
        result_path=Path("/out/run.json"),
    )
    assert "--no-structured-output" in cmd
    assert "--structured-output-ratio" not in cmd
    assert "--chat-template-kwargs" not in cmd
    assert "--trust-remote-code" not in cmd


def test_driver_records_the_run_and_flags_an_inactive_grammar(
    tmp_path, monkeypatch, caplog
):
    venv = tmp_path / "venv"
    (venv / "work_dir").mkdir(parents=True)
    (venv / "work_dir" / "benchmark_serving_structured_output.py").write_text("")

    def fake_run(cmd, env=None, timeout_s=None):
        out = (
            Path(cmd[cmd.index("--result-dir") + 1])
            / cmd[cmd.index("--result-filename") + 1]
        )
        out.write_text(
            json.dumps({"tt_structured_output": {"grammar_active_rate": 0.0}})
        )
        return 0

    monkeypatch.setattr(driver_module, "run_command", fake_run)
    driver = VLLMStructuredOutputDriver(venv_python=venv / "bin" / "python")
    with caplog.at_level(logging.ERROR):
        result = driver.run(
            _config(), _server(), DriverContext(output_dir=tmp_path / "out", device="x")
        )
    assert result.return_code == 0
    assert result.raw["tt_dataset"] == "xgrammar_bench"
    assert result.raw["tt_structured_output_ratio"] == 1.0
    assert "grammar did not constrain" in caplog.text


def test_driver_without_the_fetched_script_fails_loudly(tmp_path):
    driver = VLLMStructuredOutputDriver(venv_python=tmp_path / "bin" / "python")
    with pytest.raises(FileNotFoundError, match="fetch_structured_output_scripts"):
        driver.run(_config(), _server(), DriverContext(output_dir=tmp_path, device="x"))


def test_parser_makes_a_structured_output_block():
    raw = {
        "model_id": "XiaomiMiMo/MiMo-V2.6-Flash-MOPD",
        "max_concurrency": 4,
        "completed": 100,
        "mean_tpot_ms": 10.06,
        "output_throughput": 307.4,
        "correct_rate(%)": 100.0,
        "tt_dataset": "json",
        "tt_structured_output_ratio": 1.0,
        "tt_osl": 128,
        "tt_structured_output": {
            "structured_requests": 100,
            "grammar_active_rate": 1.0,
            "complete_json_rate": 0.98,
            "schema_valid_rate": 0.98,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }
    block = VLLMStructuredOutputParser().parse(raw, device="galaxy")
    assert block.kind == "structured_output"
    assert block.title == "vLLM Structured Output: json, ratio 1.0"
    assert block.data["grammar_active_pct"] == 100.0
    assert block.data["schema_valid_pct"] == 98.0
    assert block.data["mean_tpot_ms"] == 10.06
    assert block.targets["tool"] == "vllm_structured_output"


def test_vllm_bench_appends_the_structured_output_runs(monkeypatch, tmp_path):
    from llm_module.runner import RunnerResult
    from test_module.llm_tests import llm_benchmark_tests as bench

    calls = []

    def fake_run_llm_performance(ctx, *, driver, configs, output_subdir="llm", **kw):
        calls.append((driver.name, len(configs), output_subdir))
        return RunnerResult(blocks=[driver.name], return_codes=[0] * len(configs))

    monkeypatch.setattr(bench, "run_llm_performance", fake_run_llm_performance)
    spec = next(s for s in MODEL_SPECS.values() if s.model_type == ModelType.LLM)
    ctx = SimpleNamespace(
        model_spec=spec,
        device=spec.device_type,
        runtime_config=SimpleNamespace(limit_samples_mode=None),
    )
    text = RunnerResult(blocks=["vllm"], return_codes=[0])
    result = bench._with_structured_output(ctx, text, "", tmp_path / "bin" / "python")

    assert calls == [
        (
            "vllm_structured_output",
            len(STRUCTURED_OUTPUT_PAIRS),
            "llm_structured_output",
        )
    ]
    assert result.blocks == ["vllm", "vllm_structured_output"]
    assert result.return_codes == [0] * (1 + len(STRUCTURED_OUTPUT_PAIRS))
