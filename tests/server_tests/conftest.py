import pytest
import requests
import json
import time
from datetime import datetime
from utils.prompt_client import PromptClient
from utils.prompt_configs import EnvironmentConfig

# 1. Add command-line options for endpoint and metadata
def pytest_addoption(parser):
    parser.addoption(
        "--endpoint_url", 
        action="store", 
        default="http://127.0.0.1:8000/v1/chat/completions",
        help="The URL of the API endpoint to test"
    )
    parser.addoption(
        "--model_name",
        action="store",
        default="unknown-model",
        help="Name of the model being tested"
    )
    parser.addoption(
        "--model_backend",
        action="store",
        default="unknown-backend",
        help="Backend serving the model (e.g., vLLM, TGI, Pytorch)"
    )

# 2. A fixture to make the URL available to tests
@pytest.fixture(scope="session")
def endpoint_url(request):
    return request.config.getoption("--endpoint_url")

# 3. Fixture for the session-wide report dictionary (now with metadata)
@pytest.fixture(scope="session")
def results_report(request):
    report_data = {
        "endpoint_url": request.config.getoption("--endpoint_url"),
        "model_name": request.config.getoption("--model_name"),
        "model_backend": request.config.getoption("--model_backend"),
        "test_run_timestamp_utc": datetime.utcnow().isoformat(),
        "parameter_support": {}
    }
    yield report_data
    
    # 4. This code runs after the session finishes
    print("\nGenerating report.json...")
    with open("report.json", "w") as f:
        json.dump(report_data, f, indent=2)
    print("report.json generated.")

# 5. Helper fixture to make API calls (unchanged, it's already clean)
@pytest.fixture
def api_client(endpoint_url):
    """A simple client to make requests."""
    def _make_request(json_payload, timeout=30):
        env_config = EnvironmentConfig()
        prompt_client = PromptClient(env_config)
        authorization = prompt_client._get_authorization()
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {authorization}"}
        try:
            # Added a timeout for robustness
            response = requests.post(endpoint_url, headers=headers, json=json_payload, timeout=timeout)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            return response.json(), None
        except requests.exceptions.HTTPError as e:
            # Return the JSON body of the error if possible
            try:
                error_json = e.response.json()
            except json.JSONDecodeError:
                error_json = {"error": "Non-JSON error response", "status_code": e.response.status_code, "text": e.response.text}
            return error_json, e
        except Exception as e:
            return None, e
            
    return _make_request

# 6. REFACTORED helper fixture to add test results to the report
@pytest.fixture
def add_to_report(results_report, request):
    """
    Helper to add results to our custom report.
    This now abstracts the 'pass'/'fail' logic and automatically
    uses the test function name as the key.
    """
    def _add(succeeded, message, test_value="N/A"):
        """
        Args:
            succeeded (bool): Whether the test assertion passed.
            message (str): A human-readable message about the result.
            test_value (any, optional): The specific value being tested (e.g., 2, 5, "temperature=0.0").
        """
        status = "pass" if succeeded else "fail"
        
        # Use the base function name (e.g., "test_n") as the group key
        test_func_name = request.node.originalname or request.node.name.split('[')[0]
        # Use the full node name (e.g., "test_n[2]") for specific context
        test_node_name = request.node.name
        
        if test_func_name not in results_report["parameter_support"]:
            results_report["parameter_support"][test_func_name] = []
        
        results_report["parameter_support"][test_func_name].append({
            "status": status,
            "message": message,
            "test_node_name": test_node_name,
            "test_value": test_value
        })
    return _add
