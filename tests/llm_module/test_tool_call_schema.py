# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for the tool-call JSON-schema suite's request building."""

from __future__ import annotations

import pytest

from llm_module import tool_call_schema
from llm_module.tool_call_schema import (
    DEFAULT_SAMPLING_PARAMS,
    TOOL_DESCRIPTION,
    CaseResult,
    SuiteSettings,
    ValidatorCase,
    build_request,
    resolve_sampling_params,
    run_cases,
    select_cases,
)

SCHEMA = {"type": "object", "properties": {"value": {"type": "string"}}}


def test_request_has_no_sampling_params_or_bypass_by_default():
    payload = build_request(SCHEMA, SuiteSettings(), stream=False)

    assert not set(DEFAULT_SAMPLING_PARAMS) & set(payload)
    assert payload["tools"][0]["function"]["description"] == TOOL_DESCRIPTION


def test_default_sampling_params_are_sent():
    settings = SuiteSettings(sampling_params=resolve_sampling_params(True))

    payload = build_request(SCHEMA, settings, stream=True)

    assert {k: payload[k] for k in DEFAULT_SAMPLING_PARAMS} == DEFAULT_SAMPLING_PARAMS
    assert payload["stream"] is True


def test_overrides_merge_over_defaults_and_null_is_kept():
    params = resolve_sampling_params(True, '{"top_k": 0, "seed": null}')

    assert params == {**DEFAULT_SAMPLING_PARAMS, "top_k": 0, "seed": None}


def test_overrides_alone_without_default_sampling():
    assert resolve_sampling_params(False, {"temperature": 0.6}) == {"temperature": 0.6}


@pytest.mark.parametrize("overrides", [None, "", "  "])
def test_nothing_to_send_resolves_to_none(overrides):
    assert resolve_sampling_params(False, overrides) is None


@pytest.mark.parametrize("overrides", ["[1, 2]", '"top_k"', 3])
def test_overrides_must_be_a_json_object(overrides):
    with pytest.raises(ValueError):
        resolve_sampling_params(True, overrides)


def test_settings_reject_non_dict_sampling_params():
    with pytest.raises(ValueError):
        SuiteSettings(sampling_params=[("top_k", 0)])


def test_cache_bypass_prefixes_a_fresh_id_per_request():
    settings = SuiteSettings(cache_bypass=True)

    first, second = (
        build_request(SCHEMA, settings, stream=False)["tools"][0]["function"][
            "description"
        ]
        for _ in range(2)
    )

    assert first.startswith("[request ") and first.endswith(TOOL_DESCRIPTION)
    assert first != second


def test_thinking_fields_still_apply_with_sampling_params():
    settings = SuiteSettings(
        think_mode="kimi",
        thinking=True,
        sampling_params=resolve_sampling_params(True),
    )

    payload = build_request(SCHEMA, settings, stream=False)

    assert payload["thinking"] == {"type": "enabled"}
    assert payload["temperature"] == 1.0


def test_streamed_non_ascii_arguments_decode_as_utf8():
    """A bare ``text/event-stream`` (no charset) must not be read as ISO-8859-1."""
    import io
    import json

    import requests

    from llm_module.tool_call_schema import SelectedCase, ValidatorCase, evaluate_once

    schema = {
        "type": "object",
        "required": ["value"],
        "additionalProperties": False,
        "properties": {
            "value": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"abc😊": {"type": "string"}},
            }
        },
    }
    arguments = json.dumps({"value": {"abc😊": "x"}}, ensure_ascii=False)
    chunk = {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_0",
                            "type": "function",
                            "function": {
                                "name": "kvv_walle_case",
                                "arguments": arguments,
                            },
                        }
                    ]
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    body = f"data: {json.dumps(chunk, ensure_ascii=False)}\n\ndata: [DONE]\n\n"

    response = requests.Response()
    response.status_code = 200
    response.headers["Content-Type"] = "text/event-stream"
    response.encoding = requests.utils.get_encoding_from_headers(response.headers)
    response.raw = io.BytesIO(body.encode("utf-8"))
    selected = SelectedCase(ValidatorCase("TestEmoji", 1, "{}"), schema, "object")

    attempt = evaluate_once(
        lambda payload, stream: response, selected, "stream", SuiteSettings()
    )

    assert attempt.passed, attempt.message
    assert json.loads(attempt.arguments) == {"value": {"abc😊": "x"}}


def _case(line: int, schema: str) -> ValidatorCase:
    return ValidatorCase(suite="TestRefs", line=line, schema_text=schema)


INLINE_CASE = _case(1, '{"type": "integer"}')
REF_CASE = _case(
    2,
    '{"$defs": {"n": {"type": "integer"}}, "type": "array", "items": {"$ref": "#/$defs/n"}}',
)


def test_uses_ref_flags_only_schemas_typed_through_ref():
    inline, ref = select_cases([INLINE_CASE, REF_CASE])

    assert not inline.uses_ref
    assert ref.uses_ref


def test_exclude_ref_drops_ref_cases_before_max_cases():
    kept = select_cases([REF_CASE, INLINE_CASE], max_cases=1, exclude_ref=True)

    assert [c.case_id for c in kept] == [INLINE_CASE.case_id]


def test_run_cases_reports_each_result_and_keeps_task_order(monkeypatch):
    def fake_run_case(send, selected, mode, settings, *, retry_delay):
        return CaseResult(
            case_id=selected,
            suite="S",
            line=0,
            mode=mode,
            selection_reason="",
            status="passed",
            cause=None,
            message="",
            attempts=1,
        )

    monkeypatch.setattr(tool_call_schema, "run_case", fake_run_case)
    tasks = [(f"case{i}", "stream") for i in range(5)]
    seen = []

    results = run_cases(
        None,
        tasks,
        SuiteSettings(workers=3),
        on_result=lambda done, total, result: seen.append((done, total)),
    )

    assert [r.case_id for r in results] == [f"case{i}" for i in range(5)]
    assert seen == [(i, 5) for i in range(1, 6)]
