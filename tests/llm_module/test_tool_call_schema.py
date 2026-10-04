# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for the tool-call JSON-schema suite's request building."""

from __future__ import annotations

import pytest

from llm_module.tool_call_schema import (
    DEFAULT_SAMPLING_PARAMS,
    TOOL_DESCRIPTION,
    SuiteSettings,
    build_request,
    resolve_sampling_params,
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
