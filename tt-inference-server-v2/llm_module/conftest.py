# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Pytest fixtures for the vLLM parameter-conformance suites.

These suites (``test_vllm_chat_completions.py`` / ``test_vllm_responses.py``)
are run as a child pytest process by
``test_module.llm_tests.vllm_param_conformance_test``.
"""

from server_tests.conftest import (  # noqa: F401
    api_client,
    endpoint_url,
    max_context,
    output_path,
    pytest_addoption,
    pytest_runtest_makereport,
    report_test,
    results_report,
)
