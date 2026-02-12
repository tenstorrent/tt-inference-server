# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from pathlib import Path
import pytest
import requests
import json
from datetime import datetime
from utils.prompt_client import PromptClient
from utils.prompt_configs import EnvironmentConfig


# 1. Add command-line options for endpoint and metadata
def pytest_addoption(parser):
    parser.addoption(
        "--output-path",
        type=str,
        action="store",
        help="Directory to write results report to",
    )
    parser.addoption(
        "--endpoint-url",
        action="store",
        default="http://127.0.0.1:8000/v1/chat/completions",
        help="The URL of the API endpoint to test",
    )
    parser.addoption(
        "--model-name",
        action="store",
        default="unknown-model",
        help="Name of the model being tested",
    )
    parser.addoption(
        "--model-impl",
        action="store",
        default="unknown-impl",
        help="Implementation serving the model (e.g., tt-transformers)",
    )


# 2. A fixture to make the URL available to tests
@pytest.fixture(scope="session")
def endpoint_url(request):
    return request.config.getoption("--endpoint-url")


@pytest.fixture(scope="session")
def output_path(request):
    output_path = request.config.getoption("--output-path")
    output_path = Path(output_path)
    # ensure path exists
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


# 3. Fixture for the session-wide report dictionary (now with metadata)
@pytest.fixture(scope="session")
def results_report(request, output_path):
    report_data = {
        "endpoint_url": request.config.getoption("--endpoint-url"),
        "model_name": request.config.getoption("--model-name"),
        "model_impl": request.config.getoption("--model-impl"),
        "results": {},
    }
    yield report_data

    # 4. This code runs after the session finishes
    print("Generating parameter_report.json...")
    filename = output_path / "parameter_report.json"
    with open(filename, "w") as f:
        json.dump(report_data, f, indent=2)
    print("parameter_report.json generated.")


# 5. Helper fixture to make API calls (unchanged, it's already clean)
@pytest.fixture
def api_client(endpoint_url):
    """A simple client to make requests."""

    def _make_request(json_payload, timeout=30):
        env_config = EnvironmentConfig()
        prompt_client = PromptClient(env_config)
        authorization = prompt_client._get_authorization()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {authorization}",
        }
        try:
            # Added a timeout for robustness
            response = requests.post(
                endpoint_url, headers=headers, json=json_payload, timeout=timeout
            )
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            return response.json()
        except requests.exceptions.HTTPError as e:
            # Return the JSON body of the error if possible
            try:
                error_json = e.response.json()
            except json.JSONDecodeError:
                error_json = {
                    "error": "Non-JSON error response",
                    "status_code": e.response.status_code,
                    "text": e.response.text,
                }
            raise requests.exceptions.HTTPError(
                f"API Error: {str(e)}. Response: {error_json}"
            )

    return _make_request


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    This hook captures the result of each test phase (setup, call, teardown).
    We store the 'call' phase report on the test item itself.
    """
    # Execute all other hooks to obtain the report object
    outcome = yield
    report = outcome.get_result()

    # We only care about the 'call' phase, which is the test execution
    if report.when == "call":
        # Store the report on the item (test node)
        # This allows fixtures to access the test outcome
        setattr(item, "rep_call", report)


@pytest.fixture
def report_test(results_report, request):
    """
    Fixture to report test result after execution.

    Usage:
        def test_example(report_test):
            ...

    Features:
        - Automatically knows test pass/fail
        - Access to parametrization via request.node.callspec.params
        - Captures exception traceback
        - Integrates cleanly with pytest
    """
    test_start_ts = datetime.now().isoformat()
    yield  # Run the test
    test_end_ts = datetime.now().isoformat()

    # Get the report object we stored in the hook
    report = getattr(request.node, "rep_call", None)
    assert report is not None, "Test did not report its call phase."
    outcome = report.outcome

    # --- Build message ---
    # Include traceback if failed
    tb = ""
    if not report.passed and report.longrepr:
        if hasattr(report.longrepr, "reprcrash"):
            # pytest ReprExceptionInfo object
            tb = f"\nTraceback:\n{report.longrepr.reprcrash.message}"
        else:
            tb = f"\nTraceback:\n{report.longrepr}"

    message = tb.strip() if tb else ""

    # --- Add the test outcome to the report ---

    # Use the base function name (e.g., "test_n") as the group key
    test_func_name = request.node.originalname or request.node.name.split("[")[0]
    # Use the full node name (e.g., "test_n[2]") for specific context
    test_node_name = request.node.name

    if test_func_name not in results_report["results"]:
        results_report["results"][test_func_name] = []

    results_report["results"][test_func_name].append(
        {
            "test_start_ts": test_start_ts,
            "test_end_ts": test_end_ts,
            "test_id": request.node.nodeid,
            "status": outcome,
            "message": message,
            "test_node_name": test_node_name,
        }
    )
