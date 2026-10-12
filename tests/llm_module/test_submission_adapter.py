# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import json

from llm_module.agentic.submission_adapter import (
    SUBMISSION_COMMAND,
    normalize_submission_marker,
)


def _payload():
    return {
        "tool_choice": "auto",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                    },
                },
            }
        ],
    }


def _response(content=SUBMISSION_COMMAND):
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ]
    }


def test_exact_final_submission_marker_becomes_bash_tool_call():
    original = _response("Checked the patch.\n" + SUBMISSION_COMMAND)
    normalized = normalize_submission_marker(original, _payload())

    choice = normalized["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["function"]["name"] == "bash"
    assert json.loads(call["function"]["arguments"]) == {"command": SUBMISSION_COMMAND}
    assert "tool_calls" not in original["choices"][0]["message"]


def test_marker_is_not_normalized_without_exact_safe_protocol_shape():
    assert (
        normalize_submission_marker(
            _response("prefix " + SUBMISSION_COMMAND), _payload()
        )
        is None
    )
    assert (
        normalize_submission_marker(_response("```\n" + SUBMISSION_COMMAND), _payload())
        is None
    )
    payload = _payload()
    payload["tool_choice"] = "none"
    assert normalize_submission_marker(_response(), payload) is None
