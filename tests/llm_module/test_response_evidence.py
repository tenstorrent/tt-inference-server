import asyncio
import copy
import hashlib
import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from llm_module import lm_eval_request_overrides as wrapper


def install_adapter(monkeypatch):
    mod = ModuleType("lm_eval.models.openai_completions")

    class LocalChatCompletion:
        def _create_payload(self, message):
            return {
                "model": "m",
                "messages": [{"role": "user", "content": message}],
                "max_tokens": 126976,
                "seed": 42,
            }

        @staticmethod
        def parse_generations(outputs, **kwargs):
            return [item["message"]["content"] for item in outputs["choices"]]

    mod.LocalChatCompletion = LocalChatCompletion
    monkeypatch.setitem(sys.modules, "lm_eval.models.openai_completions", mod)
    monkeypatch.setitem(
        sys.modules, "lm_eval.models", SimpleNamespace(openai_completions=mod)
    )
    monkeypatch.setitem(
        sys.modules,
        "lm_eval",
        SimpleNamespace(models=SimpleNamespace(openai_completions=mod)),
    )
    return LocalChatCompletion


def test_observer_payload_scoring_identity_and_concurrent_correlation(
    monkeypatch, tmp_path
):
    adapter = install_adapter(monkeypatch)
    base = adapter()._create_payload("private question")
    body = {"chat_template_kwargs": {"enable_thinking": True}}
    expected = wrapper.merge_request_body(base, body)
    output = {
        "id": "private-id",
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 100,
            "total_tokens": 120,
            "completion_tokens_details": {"reasoning_tokens": 70},
        },
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "content": "private answer",
                    "reasoning": "private reasoning",
                },
            }
        ],
    }
    original = copy.deepcopy(output)
    target = tmp_path / "responses.jsonl"
    wrapper.patch_api_adapters(body, False, target)
    assert adapter()._create_payload("private question") == expected

    async def request(message):
        payload = adapter()._create_payload(message)
        await asyncio.sleep(0)
        assert adapter.parse_generations(output) == ["private answer"]
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    async def run():
        return await asyncio.gather(request("first"), request("second"))

    hashes = asyncio.run(run())
    rows = [json.loads(line) for line in target.read_text().splitlines()]
    assert [row["request"]["payload_sha256"] for row in rows] == hashes
    assert rows[0]["usage"]["reasoning_tokens"] == 70
    assert rows[0]["choices"] == [{"index": 0, "finish_reason": "stop"}]
    assert output == original
    assert "private" not in target.read_text()
    # Repatching does not duplicate response records.
    wrapper.patch_api_adapters(body, False, target)
    adapter()._create_payload("third")
    adapter.parse_generations(output)
    assert len(target.read_text().splitlines()) == 3


def test_missing_usage_retained_unknown_and_length_finish_visible(
    monkeypatch, tmp_path
):
    adapter = install_adapter(monkeypatch)
    target = tmp_path / "responses.jsonl"
    wrapper.patch_api_adapters({}, False, target)
    adapter()._create_payload("q")
    response = {
        "choices": [
            {"index": 0, "finish_reason": "length", "message": {"content": "a"}}
        ]
    }
    assert adapter.parse_generations(response) == ["a"]
    row = json.loads(target.read_text())
    assert all(value is None for value in row["usage"].values())
    assert row["choices"][0]["finish_reason"] == "length"


def test_pinned_common_eval_resolves_asset_identity(tmp_path):
    from test_module.llm_tests.llm_eval_tests import _prepare_eval_tokenizer
    from workflow_module.engine_types import WorkflowVenvType

    pin = "842da3794eaa0b77d5f08bae87a17459d91ff475"
    spec = SimpleNamespace(
        model_id="test",
        metadata={},
        device_model_spec=SimpleNamespace(
            vllm_args={"revision": pin, "tokenizer_revision": pin}
        ),
    )
    task = SimpleNamespace(
        task_name="r1_gpqa_diamond", workflow_venv_type=WorkflowVenvType.EVALS_COMMON
    )
    with patch(
        "test_module.llm_tests.llm_eval_tests.resolve_tokenizer",
        return_value="/cache/snapshots/" + pin,
    ) as resolve:
        assert _prepare_eval_tokenizer(
            SimpleNamespace(model_spec=spec), task, tmp_path
        ).endswith(pin)
    assert resolve.call_args.args[1] == tmp_path / "eval_test" / "tokenizer_protocol"


def test_partition_keeps_frozen_ids_and_release_limits():
    from reference_config.evals.eval_config import EVAL_CONFIGS

    (task,) = EVAL_CONFIGS["google/gemma-4-31B-it"].tasks
    assert task.doc_ids == [4, 5, 6, 7, 8, 9]
    assert not set(task.doc_ids) & {0, 1, 2, 3}
    assert task.gen_kwargs["max_gen_toks"] == 126976
    assert task.model_kwargs["timeout"] == 14400
    assert task.max_concurrent == 32


def test_command_pins_revision_and_keeps_exact_partition_payload_options(tmp_path):
    from reference_config.evals.eval_config import EVAL_CONFIGS
    from llm_module.eval_command import build_eval_command

    pin = "842da3794eaa0b77d5f08bae87a17459d91ff475"
    (task,) = EVAL_CONFIGS["google/gemma-4-31B-it"].tasks
    spec = SimpleNamespace(
        model_id="test",
        model_name="test",
        hf_model_repo="google/gemma-4-31B-it",
        metadata={},
        device_model_spec=SimpleNamespace(
            max_context=262144,
            max_concurrency=32,
            eval_max_retries=None,
            vllm_args={"revision": pin, "tokenizer_revision": pin},
        ),
    )
    command = build_eval_command(
        task, spec, "P300x2", tmp_path, 8000, tokenizer_path="/cache/snapshots/" + pin
    )
    args = command[command.index("--model_args") + 1]
    assert "revision=" + pin in args
    assert "tokenizer=/cache/snapshots/" + pin in args
    assert "timeout=14400" in args and "num_concurrent=32" in args
    assert command[command.index("--response-evidence") + 1] == str(
        tmp_path / "api_response_evidence.jsonl"
    )
    assert json.loads(command[command.index("--samples") + 1]) == {
        "r1_gpqa_diamond": [4, 5, 6, 7, 8, 9]
    }
    assert "max_gen_toks=126976" in command[command.index("--gen_kwargs") + 1]
