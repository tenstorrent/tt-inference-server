# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Pytest fixtures for the vLLM parameter-conformance suites.

These suites (``test_vllm_chat_completions.py`` / ``test_vllm_responses.py``
/ ``test_tool_call_json_schema.py`` / ...) are run as a child pytest process
by the wrappers in ``test_module.llm_tests``.
"""

from llm_module.tool_call_schema import (
    DEFAULT_CASE_DIR,
    DEFAULT_CASE_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_REQUEST_TIMEOUT_S,
    DEFAULT_SELECTION,
    DEFAULT_THINK_MODE,
    DEFAULT_TOOL_CHOICE,
    DEFAULT_WORKERS,
    SELECTIONS,
    THINK_MODES,
    TOOL_CHOICES,
)
from test_fixtures.conftest import (  # noqa: F401
    api_client,
    endpoint_url,
    max_context,
    output_path,
    pytest_runtest_makereport,
    report_test,
    results_report,
)
from test_fixtures.conftest import pytest_addoption as _base_pytest_addoption


def pytest_addoption(parser):
    _base_pytest_addoption(parser)

    # test_tool_call_json_schema.py; ToolCallSchemaConformanceTest maps its
    # test_config onto these.
    group = parser.getgroup("tool-call JSON schema suite")
    group.addoption(
        "--schema-tool-choice",
        choices=TOOL_CHOICES,
        default=DEFAULT_TOOL_CHOICE,
        help="tool_choice sent with every case (default: %(default)s)",
    )
    group.addoption(
        "--schema-think-mode",
        choices=THINK_MODES,
        default=DEFAULT_THINK_MODE,
        help="thinking request format; none sends nothing (default: %(default)s)",
    )
    group.addoption(
        "--schema-thinking",
        action="store_true",
        default=False,
        help="enable thinking (ignored with --schema-think-mode none)",
    )
    group.addoption(
        "--schema-selection",
        choices=SELECTIONS,
        default=DEFAULT_SELECTION,
        help="which walle cases to run (default: %(default)s)",
    )
    group.addoption(
        "--schema-max-cases",
        type=int,
        default=None,
        help="run at most this many selected cases (default: all)",
    )
    group.addoption(
        "--schema-max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="max_tokens per request (default: %(default)s)",
    )
    group.addoption(
        "--schema-retries",
        type=int,
        default=DEFAULT_CASE_RETRIES,
        help="extra attempts for a failing case (default: %(default)s)",
    )
    group.addoption(
        "--schema-workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="concurrent requests (default: %(default)s)",
    )
    group.addoption(
        "--schema-request-timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_S,
        help="per-request timeout in seconds (default: %(default)s)",
    )
    group.addoption(
        "--schema-default-sampling",
        action="store_true",
        default=False,
        help="send DEFAULT_SAMPLING_PARAMS (temperature 1.0, top_p 0.95, top_k 50, "
        "n 1, presence/frequency_penalty 0) with every request",
    )
    group.addoption(
        "--schema-sampling-params",
        default="",
        help="JSON object of sampling params merged over the defaults (or sent on "
        "its own without --schema-default-sampling), e.g. '{\"top_k\": 0}'",
    )
    group.addoption(
        "--schema-cache-bypass",
        action="store_true",
        default=False,
        help="start every request's tool description with a unique id so no "
        "request can be served from the server's prefix cache",
    )
    group.addoption(
        "--schema-exclude-ref",
        action="store_true",
        default=False,
        help="do not send cases whose schema uses $ref; for servers whose "
        "tool-call parser cannot resolve $ref (MiniMax, see "
        "SuiteSettings.exclude_ref_schemas). Listed under excluded_cases",
    )
    group.addoption(
        "--schema-case-dir",
        default=str(DEFAULT_CASE_DIR),
        help="walle validator_cases directory (default: bundled snapshot)",
    )
