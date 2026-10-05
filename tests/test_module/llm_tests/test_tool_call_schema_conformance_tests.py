# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Tests for the tool-call JSON-schema spec-test wrapper's pytest arguments."""

from __future__ import annotations

import asyncio
import json
import logging
import sys

import pytest

from test_module._test_common import TestConfig
from test_module.llm_tests.tool_call_schema_conformance_test import (
    _PYTEST_OPTIONS,
    ENV_PREFIX,
    PROGRESS_PREFIX,
    ToolCallSchemaConformanceTest,
    grade_tool_call_schema_results,
)


@pytest.fixture(autouse=True)
def _no_env_overrides(monkeypatch):
    """Keep a caller's TOOL_CALL_SCHEMA_* environment out of these tests."""
    for key, _option, _kind in _PYTEST_OPTIONS:
        monkeypatch.delenv(ENV_PREFIX + key.upper(), raising=False)


def _args(config) -> list:
    return ToolCallSchemaConformanceTest(TestConfig(config), {})._extra_pytest_args()


def test_config_values_map_to_pytest_options():
    args = _args({"case_retries": 3, "workers": 16, "tool_choice": "auto"})

    assert args == [
        "--schema-tool-choice",
        "auto",
        "--schema-retries",
        "3",
        "--schema-workers",
        "16",
        "--tb=line",
    ]


def test_sampling_and_cache_bypass_from_config():
    args = _args(
        {
            "default_sampling": True,
            "sampling_params": {"top_k": 0, "seed": None},
            "cache_bypass": True,
        }
    )

    assert "--schema-default-sampling" in args
    assert "--schema-cache-bypass" in args
    params = args[args.index("--schema-sampling-params") + 1]
    assert json.loads(params) == {"top_k": 0, "seed": None}


def test_exclude_ref_schemas_maps_to_its_flag():
    assert "--schema-exclude-ref" in _args({"exclude_ref_schemas": True})
    assert "--schema-exclude-ref" not in _args({"exclude_ref_schemas": False})


def test_false_flags_add_nothing():
    assert _args({"default_sampling": False, "cache_bypass": False}) == ["--tb=line"]


def test_environment_overrides_test_config(monkeypatch):
    monkeypatch.setenv("TOOL_CALL_SCHEMA_CASE_RETRIES", "0")
    monkeypatch.setenv("TOOL_CALL_SCHEMA_DEFAULT_SAMPLING", "yes")
    monkeypatch.setenv("TOOL_CALL_SCHEMA_CACHE_BYPASS", "false")
    monkeypatch.setenv("TOOL_CALL_SCHEMA_SAMPLING_PARAMS", '{"top_k":0}')

    args = _args({"case_retries": 3, "default_sampling": False, "cache_bypass": True})

    assert args[args.index("--schema-retries") + 1] == "0"
    assert "--schema-default-sampling" in args
    assert "--schema-cache-bypass" not in args
    assert json.loads(args[args.index("--schema-sampling-params") + 1]) == {"top_k": 0}


@pytest.mark.parametrize("value", ["[1]", "not json"])
def test_invalid_sampling_params_fail_before_the_suite_runs(monkeypatch, value):
    monkeypatch.setenv("TOOL_CALL_SCHEMA_SAMPLING_PARAMS", value)

    with pytest.raises(ValueError):
        _args({})


def _cases(passed: int, failed: int) -> list:
    return [{"status": "passed", "mode": "stream"}] * passed + [
        {"status": "failed", "mode": "stream", "cause": "invalid_json"}
    ] * failed


def test_default_threshold_requires_every_case():
    test = ToolCallSchemaConformanceTest(TestConfig({}), {})
    threshold = test._resolve_threshold()

    assert threshold == 1.0
    assert grade_tool_call_schema_results(_cases(408, 0), threshold)["success"]
    assert not grade_tool_call_schema_results(_cases(407, 1), threshold)["success"]


def test_targets_can_lower_the_threshold():
    test = ToolCallSchemaConformanceTest(TestConfig({}), {"pass_rate_threshold": 0.995})

    assert grade_tool_call_schema_results(_cases(406, 2), test._resolve_threshold())[
        "success"
    ]


def test_child_output_is_streamed_line_by_line():
    """Complete lines reach the hook as they arrive, including an unterminated
    last line and one longer than asyncio's default 64 KiB readline limit."""
    long_line = "x" * 200_000
    script = f"import sys; sys.stdout.write('a\\nb\\n{long_line}\\ntail')"
    test = ToolCallSchemaConformanceTest(TestConfig({}), {})
    lines = []
    test._on_pytest_output_line = lines.append

    async def run():
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        return await test._stream_pytest_output(process)

    output = asyncio.run(run())

    assert lines == ["a", "b", long_line, "tail"]
    assert output == f"a\nb\n{long_line}\ntail".encode()


def test_only_progress_lines_are_logged(caplog):
    test = ToolCallSchemaConformanceTest(TestConfig({}), {})
    progress = f"{PROGRESS_PREFIX} 17/338 done (5%): 17 passed, 0 failed, 0m42s"

    with caplog.at_level(logging.INFO):
        test._on_pytest_output_line("..F..")
        test._on_pytest_output_line(progress)

    assert [r.getMessage() for r in caplog.records] == [progress]
