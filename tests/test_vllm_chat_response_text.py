import importlib.util
from pathlib import Path

import pytest


_SPEC = importlib.util.spec_from_file_location(
    "vllm_chat_completions",
    Path(__file__).parents[1] / "llm_module/test_vllm_chat_completions.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
message_text = _MODULE.message_text


def _response(**message):
    return {"choices": [{"message": message}]}


def test_message_text_prefers_the_final_channel():
    assert message_text(
        _response(content="final answer", reasoning_content="private work")
    ) == "final answer"


def test_message_text_keeps_reasoning_only_truncated_generation_testable():
    assert message_text(
        _response(content=None, reasoning_content="generated before max_tokens")
    ) == "generated before max_tokens"


def test_message_text_rejects_a_response_without_generated_text():
    with pytest.raises(AssertionError, match="no generated text"):
        message_text(_response(content=None, reasoning_content=None))
