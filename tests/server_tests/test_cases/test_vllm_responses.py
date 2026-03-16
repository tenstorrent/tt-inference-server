# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import json
import math
from collections import Counter

import pytest
import requests

from utils.prompt_client import PromptClient
from utils.prompt_configs import EnvironmentConfig


# --- Helper Functions ---
def get_output_text(response):
    """Extract the first text output from a Responses API response."""
    if "error" in response:
        raise ValueError(f"API returned an error: {response['error']}")
    for item in response.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    return content.get("text", "")
    return ""


def get_output_tokens(response):
    """Extract output token count from a Responses API response."""
    return response.get("usage", {}).get("output_tokens")


def get_function_calls(response):
    """Extract function call items from a Responses API response."""
    return [
        item
        for item in response.get("output", [])
        if item.get("type") == "function_call"
    ]


def tokenize(text):
    """Tokenize by whitespace (simple, portable)."""
    return text.lower().split()


def repetition_stats(text):
    """Compute repetition metrics."""
    tokens = tokenize(text)
    counts = Counter(tokens)

    return {
        "len": len(tokens),
        "unique": len(set(tokens)),
        "unique_ratio": len(set(tokens)) / len(tokens) if tokens else 0,
        "most_common": counts.most_common(3),
        "entropy": shannon_entropy(tokens),
    }


def shannon_entropy(tokens):
    total = len(tokens)
    if total == 0:
        return 0
    counts = Counter(tokens)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


# --- Reusable Inputs ---
BASE_INPUT = "Tell me a short joke."
BASE_INPUT_MESSAGES = [{"role": "user", "content": "Tell me a short joke."}]
REPRO_INPUT = "What is the capital of France? Be concise."
REPRO_INPUT_MESSAGES = [
    {"role": "user", "content": "What is the capital of France? Be concise."}
]

WEATHER_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Get the current weather for a location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            }
        },
        "required": ["location"],
    },
}


# --- Test Functions ---


def test_include(report_test, api_client, request):
    """Tests that the 'include' parameter is accepted."""
    payload = {
        "input": BASE_INPUT,
        "max_output_tokens": 256,
        "include": [],
    }
    response = api_client(payload)

    try:
        output_text = get_output_text(response)
        assert output_text, "Expected non-empty output text."
    except AssertionError as e:
        msg = f"AssertionError: {str(e)}. Response: {response}"
        raise AssertionError(msg)


def test_background(report_test, api_client, endpoint_url, request):
    """Tests that the 'background' parameter kicks off an async task and can be polled to completion."""
    import time

    payload = {
        "input": BASE_INPUT,
        "max_output_tokens": 256,
        "background": True,
    }
    response = api_client(payload)

    assert "id" in response, f"Expected response to have an 'id'. Response: {response}"
    assert "status" in response, (
        f"Expected response to have a 'status'. Response: {response}"
    )
    response_id = response["id"]

    # Poll the GET /v1/responses/{response_id} endpoint until terminal state
    env_config = EnvironmentConfig()
    prompt_client = PromptClient(env_config)
    authorization = prompt_client._get_authorization()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {authorization}",
    }
    retrieve_url = f"{endpoint_url}/{response_id}"

    max_polls = 15
    poll_interval = 2
    for _ in range(max_polls):
        poll_resp = requests.get(retrieve_url, headers=headers, timeout=30)
        poll_resp.raise_for_status()
        poll_data = poll_resp.json()

        status = poll_data.get("status")
        if status not in ("queued", "in_progress"):
            break
        time.sleep(poll_interval)
    else:
        pytest.fail(
            f"Background response did not reach terminal state after {max_polls * poll_interval}s. "
            f"Last status: {poll_data.get('status')}. Response: {poll_data}"
        )

    assert poll_data.get("status") == "completed", (
        f"Expected status 'completed', got '{poll_data.get('status')}'. Response: {poll_data}"
    )
    output_text = get_output_text(poll_data)
    assert output_text, (
        f"Expected non-empty output text in completed response. Response: {poll_data}"
    )


@pytest.mark.parametrize(
    "input_val",
    [
        BASE_INPUT,
        BASE_INPUT_MESSAGES,
    ],
    ids=["string", "messages"],
)
def test_input(report_test, api_client, input_val, request):
    """Tests the 'input' parameter accepts both string and message array formats."""
    payload = {"input": input_val, "max_output_tokens": 256}
    response = api_client(payload)

    try:
        output_text = get_output_text(response)
        assert output_text, "Expected non-empty output text."
    except AssertionError as e:
        msg = f"AssertionError: {str(e)}. Response: {response}"
        raise AssertionError(msg)


def test_instructions(report_test, api_client, request):
    """Tests that the 'instructions' parameter affects model behavior."""
    payload = {
        "input": "What is 2+2?",
        "instructions": "Always respond in French, no matter what language the user uses.",
        "max_output_tokens": 256,
    }
    response = api_client(payload)

    try:
        output_text = get_output_text(response)
        assert output_text, "Expected non-empty output text."
    except AssertionError as e:
        msg = f"AssertionError: {str(e)}. Response: {response}"
        raise AssertionError(msg)


@pytest.mark.parametrize("max_val", [5, 10])
def test_max_output_tokens(report_test, api_client, max_val, request):
    """Tests the 'max_output_tokens' parameter."""
    payload = {"input": BASE_INPUT, "max_output_tokens": max_val}
    response = api_client(payload)

    output_token_count = get_output_tokens(response)
    if not output_token_count:
        raise ValueError("Response did not contain output_tokens field in usage.")

    assert output_token_count <= max_val, (
        f"Generated {output_token_count} tokens, which is greater than allowed amount of {max_val}."
    )


@pytest.mark.parametrize("max_calls", [1, 3])
def test_max_tool_calls(report_test, api_client, max_calls, request):
    """Tests the 'max_tool_calls' parameter."""
    payload = {
        "input": "What's the weather in San Francisco, New York, London, Tokyo, and Sydney?",
        "tools": [WEATHER_TOOL],
        "max_tool_calls": max_calls,
        "max_output_tokens": 256,
    }
    response = api_client(payload)

    tool_calls = get_function_calls(response)
    assert len(tool_calls) <= max_calls, (
        f"Expected at most {max_calls} tool calls, got {len(tool_calls)}."
    )


def test_metadata(report_test, api_client, request):
    """Tests that the 'metadata' parameter is accepted."""
    payload = {
        "input": BASE_INPUT,
        "max_output_tokens": 256,
        "metadata": {"test_key": "test_value", "run_id": "12345"},
    }
    response = api_client(payload)

    try:
        output_text = get_output_text(response)
        assert output_text, "Expected non-empty output text."
    except AssertionError as e:
        msg = f"AssertionError: {str(e)}. Response: {response}"
        raise AssertionError(msg)


def test_model(report_test, api_client, request):
    """Tests that the 'model' parameter is accepted."""
    model_name = request.config.getoption("--model-name")
    payload = {
        "input": BASE_INPUT,
        "model": model_name,
        "max_output_tokens": 256,
    }
    response = api_client(payload)

    try:
        output_text = get_output_text(response)
        assert output_text, "Expected non-empty output text."
    except AssertionError as e:
        msg = f"AssertionError: {str(e)}. Response: {response}"
        raise AssertionError(msg)


@pytest.mark.parametrize("parallel", [True, False])
def test_parallel_tool_calls(report_test, api_client, parallel, request):
    """Tests the 'parallel_tool_calls' parameter."""
    payload = {
        "input": "What's the weather in both San Francisco and New York?",
        "tools": [WEATHER_TOOL],
        "parallel_tool_calls": parallel,
        "max_output_tokens": 256,
    }
    response = api_client(payload)

    try:
        assert "output" in response, "Expected 'output' in response."
    except AssertionError as e:
        msg = f"AssertionError: {str(e)}. Response: {response}"
        raise AssertionError(msg)


def test_previous_response_id(report_test, api_client, request):
    """Tests the 'previous_response_id' parameter for multi-turn conversations."""
    # First request
    payload1 = {
        "input": "My name is Alice.",
        "max_output_tokens": 256,
    }
    response1 = api_client(payload1)

    response_id = response1.get("id")
    assert response_id, f"Expected response to have an 'id'. Response: {response1}"

    # Second request referencing the first
    payload2 = {
        "input": "What is my name?",
        "previous_response_id": response_id,
        "max_output_tokens": 256,
    }
    response2 = api_client(payload2)

    output_text = get_output_text(response2)
    assert "Alice" in output_text, (
        f"Expected model to recall name 'Alice' from previous response. Got: '{output_text}'"
    )


def test_prompt(report_test, api_client, request):
    """Tests that the 'prompt' parameter is accepted."""
    payload = {
        "input": BASE_INPUT,
        "max_output_tokens": 256,
        "prompt": {"id": "test-prompt"},
    }
    # Stored prompts require a pre-created prompt template, which is not
    # available on vLLM servers. Verify the server rejects it with an HTTP error.
    with pytest.raises(requests.exceptions.HTTPError):
        api_client(payload)


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_reasoning(report_test, api_client, effort, request):
    """Tests the 'reasoning' parameter with different effort levels."""
    payload = {
        "input": "What is 15 * 27?",
        "max_output_tokens": 256,
        "reasoning": {"effort": effort},
    }
    response = api_client(payload)

    try:
        output_text = get_output_text(response)
        assert output_text, "Expected non-empty output text."
    except AssertionError as e:
        msg = f"AssertionError: {str(e)}. Response: {response}"
        raise AssertionError(msg)


@pytest.mark.parametrize("tier", ["auto", "default"])
def test_service_tier(report_test, api_client, tier, request):
    """Tests that the 'service_tier' parameter is accepted."""
    payload = {
        "input": BASE_INPUT,
        "max_output_tokens": 256,
        "service_tier": tier,
    }
    response = api_client(payload)

    try:
        output_text = get_output_text(response)
        assert output_text, "Expected non-empty output text."
    except AssertionError as e:
        msg = f"AssertionError: {str(e)}. Response: {response}"
        raise AssertionError(msg)


@pytest.mark.parametrize("store_val", [True, False])
def test_store(report_test, api_client, store_val, request):
    """Tests that the 'store' parameter is accepted."""
    payload = {
        "input": BASE_INPUT,
        "max_output_tokens": 256,
        "store": store_val,
    }
    response = api_client(payload)

    try:
        output_text = get_output_text(response)
        assert output_text, "Expected non-empty output text."
    except AssertionError as e:
        msg = f"AssertionError: {str(e)}. Response: {response}"
        raise AssertionError(msg)


def test_stream_true(report_test, api_client, endpoint_url, request):
    """Tests the 'stream' parameter set to true returns a streaming response."""
    payload = {"input": BASE_INPUT, "max_output_tokens": 256, "stream": True}

    env_config = EnvironmentConfig()
    prompt_client = PromptClient(env_config)
    authorization = prompt_client._get_authorization()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {authorization}",
    }

    response = requests.post(
        endpoint_url, headers=headers, json=payload, stream=True, timeout=30
    )
    response.raise_for_status()

    events = []
    for line in response.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            data = line[len("data: ") :]
            if data.strip() == "[DONE]":
                break
            events.append(json.loads(data))

    assert len(events) > 0, "Expected at least one streaming event."


def test_stream_false(report_test, api_client, request):
    """Tests the 'stream' parameter set to false."""
    payload = {"input": BASE_INPUT, "max_output_tokens": 256, "stream": False}
    response = api_client(payload)

    try:
        output_text = get_output_text(response)
        assert output_text, "Expected non-empty output text."
    except AssertionError as e:
        msg = f"AssertionError: {str(e)}. Response: {response}"
        raise AssertionError(msg)


@pytest.mark.parametrize(
    "param_name, param_value",
    [
        ("temperature", 0.0),
        ("top_p", 0.01),  # A very low top_p should also be deterministic
    ],
)
def test_determinism_parameters(
    report_test, api_client, param_name, param_value, request
):
    """Tests parameters that should force deterministic output."""
    payload = {"input": REPRO_INPUT, param_name: param_value}

    # If testing top_p, set temperature high to prove it is working
    if param_name != "temperature":
        payload["temperature"] = 1.0

    response1 = api_client(payload)
    response2 = api_client(payload)

    output1 = get_output_text(response1)
    output2 = get_output_text(response2)
    assert output1 and output1 == output2, (
        f"{param_name}={param_value} was not deterministic. Output 1: '{output1}', Output 2: '{output2}'"
    )


@pytest.mark.parametrize(
    "text_config",
    [
        {"format": {"type": "text"}},
        {
            "format": {
                "type": "json_schema",
                "name": "color_list",
                "schema": {
                    "type": "object",
                    "properties": {
                        "colors": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["colors"],
                },
                "strict": True,
            }
        },
    ],
    ids=["text", "json_schema"],
)
def test_text(report_test, api_client, text_config, request):
    """Tests the 'text' parameter for output format configuration."""
    fmt_type = text_config["format"]["type"]
    input_msg = "List 3 colors."
    if fmt_type in ("json_object", "json_schema"):
        input_msg = "List 3 colors as a JSON object with a 'colors' key."

    payload = {
        "input": input_msg,
        "max_output_tokens": 256,
        "text": text_config,
    }
    response = api_client(payload)

    output_text = get_output_text(response)
    assert output_text, f"Expected non-empty output text. Response: {response}"

    if fmt_type == "json_schema":
        try:
            parsed = json.loads(output_text)
            assert isinstance(parsed, dict), "Expected JSON object output."
            assert "colors" in parsed, f"Expected 'colors' key in output. Got: {parsed}"
            assert isinstance(parsed["colors"], list), (
                f"Expected 'colors' to be a list. Got: {type(parsed['colors'])}"
            )
        except json.JSONDecodeError:
            pytest.fail(
                f"Expected valid JSON output with json_schema format. Got: '{output_text}'"
            )


@pytest.mark.parametrize("choice", ["auto", "none", "required"])
def test_tool_choice(report_test, api_client, choice, request):
    """Tests the 'tool_choice' parameter."""
    payload = {
        "input": "What's the weather like in San Francisco?",
        "tools": [WEATHER_TOOL],
        "tool_choice": choice,
        "max_output_tokens": 256,
    }
    if choice in ("none", "required"):
        with pytest.raises(requests.exceptions.HTTPError):
            api_client(payload)
        return

    response = api_client(payload)

    try:
        assert "output" in response, "Expected 'output' in response."
    except AssertionError as e:
        msg = f"AssertionError: {str(e)}. Response: {response}"
        raise AssertionError(msg)


def test_tools(report_test, api_client, request):
    """Tests that the 'tools' parameter triggers tool calls."""
    payload = {
        "input": "What's the weather like in San Francisco?",
        "tools": [WEATHER_TOOL],
        "max_output_tokens": 256,
    }
    response = api_client(payload)

    try:
        assert "output" in response, "Expected 'output' in response."
        assert len(response["output"]) > 0, "Expected non-empty output."
    except AssertionError as e:
        msg = f"AssertionError: {str(e)}. Response: {response}"
        raise AssertionError(msg)


@pytest.mark.parametrize("top_logprobs_val", [3, 5])
def test_top_logprobs(report_test, api_client, top_logprobs_val, request):
    """Tests the 'top_logprobs' parameter."""
    payload = {
        "input": BASE_INPUT,
        "max_output_tokens": 256,
        "top_logprobs": top_logprobs_val,
        "include": ["message.output_text.logprobs"],
    }
    # logprobs are not supported with gpt-oss models
    with pytest.raises(requests.exceptions.HTTPError):
        api_client(payload)


@pytest.mark.parametrize("truncation", ["auto", "disabled"])
def test_truncation(report_test, api_client, truncation, request):
    """Tests the 'truncation' parameter."""
    payload = {
        "input": BASE_INPUT,
        "max_output_tokens": 256,
        "truncation": truncation,
    }
    response = api_client(payload)

    try:
        output_text = get_output_text(response)
        assert output_text, "Expected non-empty output text."
    except AssertionError as e:
        msg = f"AssertionError: {str(e)}. Response: {response}"
        raise AssertionError(msg)


def test_user(report_test, api_client, request):
    """Tests that the 'user' parameter is accepted."""
    payload = {
        "input": BASE_INPUT,
        "max_output_tokens": 256,
        "user": "test-user-123",
    }
    response = api_client(payload)

    try:
        output_text = get_output_text(response)
        assert output_text, "Expected non-empty output text."
    except AssertionError as e:
        msg = f"AssertionError: {str(e)}. Response: {response}"
        raise AssertionError(msg)
