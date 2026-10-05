# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tool-call JSON-schema conformance suite (ported from the Kimi Vendor Verifier).

Each walle JSON Schema case is offered as the ``parameters`` of a ``strict``
tool, in non-streaming and streaming mode, and the model's
``function.arguments`` must validate against it. With the default ``all``
selection that is 204 cases x 2 modes = 408 tests.

Requests for every collected test run up front on a thread pool
(``--schema-workers``), each case retried up to ``--schema-retries`` times,
optionally with sampling params (``--schema-default-sampling``,
``--schema-sampling-params``) and a per-request prefix-cache bypass
(``--schema-cache-bypass``), and with ``--schema-exclude-ref`` the cases
whose schema uses ``$ref`` left out (listed under ``excluded_cases``);
the per-test functions then assert their own result. Per-case records, with
untruncated failure messages, go to
``<output-path>/parameter_report_<task-name>.json`` under
``results["test_tool_call_json_schema"]``.

Run as a child process by
``test_module.llm_tests.tool_call_schema_conformance_test``, which grades the
pass rate against a threshold. Options are declared in ``llm_module/conftest.py``.
"""

from __future__ import annotations

from typing import Dict, Tuple

import pytest

from llm_module.tool_call_schema import (
    REQUEST_MODES,
    TOOL_NAME,
    CaseResult,
    ChatCompletionsClient,
    SuiteSettings,
    load_cases,
    resolve_sampling_params,
    run_cases,
    select_cases,
)
from test_fixtures.conftest import _get_bearer_token

# Key under the report's "results" (the test function's name, as for the other
# llm_module suites).
REPORT_RESULTS_KEY = "test_tool_call_json_schema"
CASE_PARAM = "tool_schema_case"


def _settings(config: pytest.Config) -> SuiteSettings:
    return SuiteSettings(
        tool_choice=config.getoption("--schema-tool-choice"),
        think_mode=config.getoption("--schema-think-mode"),
        thinking=config.getoption("--schema-thinking"),
        selection=config.getoption("--schema-selection"),
        max_cases=config.getoption("--schema-max-cases"),
        max_tokens=config.getoption("--schema-max-tokens"),
        case_retries=config.getoption("--schema-retries"),
        workers=config.getoption("--schema-workers"),
        request_timeout=config.getoption("--schema-request-timeout"),
        sampling_params=resolve_sampling_params(
            config.getoption("--schema-default-sampling"),
            config.getoption("--schema-sampling-params"),
        ),
        cache_bypass=config.getoption("--schema-cache-bypass"),
        exclude_ref_schemas=config.getoption("--schema-exclude-ref"),
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if CASE_PARAM not in metafunc.fixturenames:
        return
    settings = _settings(metafunc.config)
    selected = select_cases(
        load_cases(metafunc.config.getoption("--schema-case-dir")),
        selection=settings.selection,
        max_cases=settings.max_cases,
        exclude_ref=settings.exclude_ref_schemas,
    )
    metafunc.parametrize(
        CASE_PARAM,
        [
            pytest.param((case, mode), id=f"{case.case_id}:{mode}")
            for case in selected
            for mode in REQUEST_MODES
        ],
    )


@pytest.fixture(scope="session")
def tool_schema_results(
    request, endpoint_url, results_report
) -> Dict[Tuple[str, str], CaseResult]:
    """Run every collected (case, mode) concurrently; record them in the report."""
    config = request.config
    settings = _settings(config)
    items = [
        item
        for item in request.session.items
        if CASE_PARAM in getattr(getattr(item, "callspec", None), "params", {})
    ]
    tasks = [item.callspec.params[CASE_PARAM] for item in items]
    client = ChatCompletionsClient(
        endpoint_url,
        model_name=config.getoption("--model-name"),
        bearer_token=_get_bearer_token(),
        timeout=settings.request_timeout,
    )
    results = run_cases(client, tasks, settings)

    results_report["tool_name"] = TOOL_NAME
    results_report["settings"] = settings.to_dict()
    results_report["excluded_cases"] = _excluded_ref_cases(config, settings)
    results_report["results"][REPORT_RESULTS_KEY] = [
        {
            "test_id": item.nodeid,
            "test_node_name": item.name,
            **result.to_dict(),
        }
        for item, result in zip(items, results)
    ]
    return {(result.case_id, result.mode): result for result in results}


def _excluded_ref_cases(config: pytest.Config, settings: SuiteSettings) -> list:
    """The $ref cases ``--schema-exclude-ref`` kept out of the run, so the
    report names them (``max_cases`` is ignored: it is a debugging cap)."""
    if not settings.exclude_ref_schemas:
        return []
    return [
        {"case_id": case.case_id, "suite": case.case.suite, "reason": "uses_ref"}
        for case in select_cases(
            load_cases(config.getoption("--schema-case-dir")),
            selection=settings.selection,
        )
        if case.uses_ref
    ]


def test_tool_call_json_schema(tool_schema_case, tool_schema_results):
    """The model's tool-call arguments must validate against the case schema."""
    case, mode = tool_schema_case
    result = tool_schema_results[(case.case_id, mode)]
    assert result.passed, (
        f"{case.case_id} [{mode}] ({case.selection_reason}) {result.cause} "
        f"after {result.attempts} attempt(s): {result.message}"
    )
