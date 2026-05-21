# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import json

import pytest
import requests


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


# --- Reusable Inputs ---
BASE_INPUT = "Tell me a short joke."
BASE_INPUT_MESSAGES = [{"role": "user", "content": f"{BASE_INPUT}"}]
REPRO_INPUT = "What is the capital of France? Be concise."
WEATHER_TOOL = {
    "type": "function",
    "name": "get_current_weather",
    "description": "Get the current weather in a given location",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The city to find the weather for, e.g. 'San Francisco'",
            },
            "state": {
                "type": "string",
                "description": "the two-letter abbreviation for the state "
                "that the city is in, e.g. 'CA' which would "
                "mean 'California'",
            },
            "unit": {
                "type": "string",
                "description": "The unit to fetch the temperature in",
                "enum": ["celsius", "fahrenheit"],
            },
        },
    },
}

SEARCH_TOOL = {
    "type": "function",
    "name": "web_search",
    "description": "Search the internet and get a summary of the top "
    "10 webpages. Should only be used if you don't know "
    "the answer to a user query, and the results are likely"
    "to be able to be found with a web search",
    "parameters": {
        "type": "object",
        "properties": {
            "search_term": {
                "type": "string",
                "description": "The term to use in the search. This should"
                "ideally be keywords to search for, not a"
                "natural-language question",
            }
        },
        "required": ["search_term"],
    },
}
# --- Test Functions ---


def test_include(report_test, api_client, request):
    """Tests that the 'include' parameter is accepted."""
    payload = {
        "input": BASE_INPUT,
        "max_output_tokens": 2048,
        "include": [],
    }
    response = api_client(payload)

    try:
        output_text = get_output_text(response)
        assert output_text, "Expected non-empty output text."
    except AssertionError as e:
        msg = f"AssertionError: {str(e)}. Response: {response}"
        raise AssertionError(msg)


def test_background(report_test, api_client, request):
    """Tests that the 'background' parameter kicks off an async task and can be polled to completion."""
    import time

    payload = {
        "input": BASE_INPUT,
        "max_output_tokens": 2048,
        "background": True,
        "temperature": 0,
    }
    response = api_client(payload)

    assert "id" in response, f"Expected response to have an 'id'. Response: {response}"
    assert "status" in response, (
        f"Expected response to have a 'status'. Response: {response}"
    )
    response_id = response["id"]

    # Poll the GET /v1/responses/{response_id} endpoint until terminal state
    max_polls = 15
    poll_interval = 2
    for _ in range(max_polls):
        poll_data = api_client(url_suffix=response_id, method=requests.get)

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
    payload = {"input": input_val, "max_output_tokens": 2048}
    response = api_client(payload)

    try:
        output_text = get_output_text(response)
        assert output_text, "Expected non-empty output text."
    except AssertionError as e:
        msg = f"AssertionError: {str(e)}. Response: {response}"
        raise AssertionError(msg)


def test_instructions(report_test, api_client, request):
    """Tests that the 'instructions' parameter affects model behavior."""
    tag = "XYZZY_ALPHA_7829"

    payload1 = {
        "input": "What is 2+2?",
        "instructions": f"You must include the string {tag} in every response.",
        "max_output_tokens": 4096,
    }

    response1 = api_client(payload1)
    response_id = response1.get("id")
    assert response_id, f"Expected response to have an 'id'. Response: {response1}"

    output1 = get_output_text(response1)
    assert tag in output1, (
        f"Expected '{tag}' in first response to confirm instructions work. Got: '{output1}'"
    )


@pytest.mark.xfail(
    reason="vLLM does not strip prior instructions when using previous_response_id https://github.com/vllm-project/vllm/issues/37697"
)
def test_instructions_not_carried_over(report_test, api_client, request):
    """Tests that instructions from a previous response are not carried over when using previous_response_id.
    from api: 'When using along with previous_response_id, the instructions from a previous response will not be carried over to the next response.'
    """
    tag = "XYZZY_ALPHA_7829"

    # First request: instructions tell the model to include a unique tag

    payload1 = {
        "input": "What is 2+2?",
        "instructions": f"You must include the string {tag} in every response.",
        "max_output_tokens": 4096,
    }

    response1 = api_client(payload1)
    response_id = response1.get("id")
    assert response_id, f"Expected response to have an 'id'. Response: {response1}"

    output1 = get_output_text(response1)
    assert tag in output1, (
        f"Expected '{tag}' in first response to confirm instructions work. Got: '{output1}'"
    )

    # Second request: use previous_response_id but provide NO instructions.
    # If old instructions carried over, the tag would still appear.
    payload2 = {
        "input": "What is 3+3?",
        "instructions": "Answer the question explicitly",
        "previous_response_id": response_id,
        "max_output_tokens": 4096,
    }
    response2 = api_client(payload2)

    output2 = get_output_text(response2)
    assert output2, f"Expected non-empty output text. Response: {response2}"
    assert tag not in output2, (
        f"'{tag}' from previous instructions should not appear without instructions. Got: '{output2}'"
    )


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
        "max_output_tokens": 4096,
        "temperature": 0,
    }
    response = api_client(payload)

    assert response.get("status") == "completed", (
        f"Expected response status 'completed', got '{response.get('status')}'. Response: {response}"
    )

    tool_calls = get_function_calls(response)
    assert len(tool_calls) <= max_calls, (
        f"Expected at most {max_calls} tool calls, got {len(tool_calls)}."
    )
    for call in tool_calls:
        assert call.get("name") == WEATHER_TOOL["name"], (
            f"Expected tool call name '{WEATHER_TOOL['name']}', got '{call.get('name')}'. Tool call: {call}"
        )


def test_metadata(report_test, api_client, request):
    """Tests that the 'metadata' parameter is stored and returned in the response."""
    metadata = {"test_key": "test_value", "run_id": "12345"}
    payload = {
        "input": BASE_INPUT,
        "max_output_tokens": 2048,
        "metadata": metadata,
    }
    response = api_client(payload)

    output_text = get_output_text(response)
    assert output_text, f"Expected non-empty output text. Response: {response}"

    # Verify metadata is returned in the response
    assert response.get("metadata") == metadata, (
        f"Expected metadata {metadata} in response, got {response.get('metadata')}."
    )

    # Verify metadata persists when retrieving the response by ID
    response_id = response.get("id")
    assert response_id, f"Expected response to have an 'id'. Response: {response}"
    retrieved = api_client(None, url_suffix=response_id, method=requests.get)
    assert retrieved.get("metadata") == metadata, (
        f"Expected metadata {metadata} in retrieved response, got {retrieved.get('metadata')}."
    )


def test_model(report_test, api_client, request):
    """Tests that the 'model' parameter is accepted."""
    model_name = request.config.getoption("--model-name")
    payload = {
        "input": BASE_INPUT,
        "model": model_name,
        "max_output_tokens": 2048,
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
    # lets try to use different function tool calls
    """Tests the 'parallel_tool_calls' parameter."""
    payload = {
        "input": "What is the weather in Dallas, Texas and Orlando, Florida in Fahrenheit?",
        "tools": [WEATHER_TOOL, SEARCH_TOOL],
        "parallel_tool_calls": parallel,
        "temperature": 0,
        "max_output_tokens": 200,
    }

    response = api_client(payload)

    # The model should respond with function_call items (tool use), not text output
    function_calls = [
        item
        for item in response.get("output", [])
        if item.get("type") == "function_call"
    ]
    assert function_calls, (
        f"Expected at least one function_call in output. Response: {response}"
    )
    # Verify the tool call targets one of the provided tools
    valid_tool_names = {tool["name"] for tool in payload["tools"]}
    for fc in function_calls:
        assert fc["name"] in valid_tool_names, (
            f"Unexpected tool call '{fc['name']}', expected one of {valid_tool_names}. Response: {response}"
        )


def test_previous_response_id(report_test, api_client, request):
    """Tests the 'previous_response_id' parameter for multi-turn conversations."""
    # First request
    payload1 = {
        "input": "My name is Alice.",
        "max_output_tokens": 2048,
    }
    response1 = api_client(payload1)

    response_id = response1.get("id")
    assert response_id, f"Expected response to have an 'id'. Response: {response1}"

    # Second request referencing the first
    payload2 = {
        "input": "What is my name?",
        "previous_response_id": response_id,
        "max_output_tokens": 2048,
    }
    response2 = api_client(payload2)

    output_text = get_output_text(response2)
    assert "Alice" in output_text, (
        f"Expected model to recall name 'Alice' from previous response. Got: '{output_text}'"
    )


def test_prompt(report_test, api_client, request):
    """Tests that the 'prompt' parameter references a stored prompt template.

    Per the API spec, the 'prompt' parameter is a ResponsePrompt object:
    - id (string): The unique identifier of the prompt template to use
    - variables (optional map): Values to substitute in the template
    - version (optional string): Version of the prompt template

    After creation, the response is retrieved via GET /responses/{response_id}
    and validated against the Response object schema.
    """
    payload = {
        "input": BASE_INPUT,
        "max_output_tokens": 256,
        "prompt": {"id": "test-prompt"},
    }
    # Stored prompts require a pre-created prompt template, which is not
    # available on vLLM servers. Verify the server rejects it with an HTTP error.
    with pytest.raises(requests.exceptions.HTTPError):
        api_client(payload)


def test_reasoning(report_test, api_client, request):
    """Tests that the 'reasoning' parameter effort levels affect reasoning token usage.

    Sends the same prompt at low, medium, and high effort and verifies that
    reasoning_tokens increases (or stays equal) as effort increases:
    low <= medium <= high.
    """
    prompt = (
        "A farmer has 3 fields. The first field is twice the size of the second. "
        "The third field is 50 acres more than the first. Together they total 750 acres. "
        "What is the size of each field?"
    )
    efforts = ["low", "medium", "high"]
    reasoning_tokens = {}

    for effort in efforts:
        payload = {
            "input": prompt,
            "max_output_tokens": 4096,
            "temperature": 0,
            "reasoning": {"effort": effort},
        }
        response = api_client(payload, timeout=120)

        output_text = get_output_text(response)
        assert output_text, (
            f"Expected non-empty output text for effort={effort}. Response: {response}"
        )
        tokens = (
            response.get("usage", {})
            .get("output_tokens_details", {})
            .get("reasoning_tokens", 0)
        )
        reasoning_tokens[effort] = tokens
    assert reasoning_tokens["low"] > 0, (
        f"Expected non-negative reasoning tokens for low effort. Got: {reasoning_tokens['low']}"
    )
    assert reasoning_tokens["low"] <= reasoning_tokens["medium"], (
        f"Expected low ({reasoning_tokens['low']}) <= medium ({reasoning_tokens['medium']}) reasoning tokens."
    )
    assert reasoning_tokens["medium"] <= reasoning_tokens["high"], (
        f"Expected medium ({reasoning_tokens['medium']}) <= high ({reasoning_tokens['high']}) reasoning tokens."
    )


@pytest.mark.parametrize("tier", ["auto", "default"])
def test_service_tier(report_test, api_client, tier, request):
    """Tests that the 'service_tier' parameter is accepted."""
    payload = {
        "input": BASE_INPUT,
        "max_output_tokens": 2048,
        "service_tier": tier,
        "temperature": 0,
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
    """Tests that the 'store' parameter is accepted.
    When store=True, also verifies the response can be retrieved and matches.
    """
    payload = {
        "input": BASE_INPUT,
        "max_output_tokens": 1024,
        "store": store_val,
    }
    response = api_client(payload)

    try:
        output_text = get_output_text(response)
        assert output_text, "Expected non-empty output text."
    except AssertionError as e:
        msg = f"AssertionError: {str(e)}. Response: {response}"
        raise AssertionError(msg)

    if store_val:
        response_id = response.get("id")
        assert response_id, f"Expected response to have an 'id'. Response: {response}"

        retrieved = api_client(
            None,
            url_suffix=response_id,
            method=requests.get,
        )
        assert retrieved == response, (
            f"Retrieved response does not match original.\n"
            f"Original:  {response}\n"
            f"Retrieved: {retrieved}"
        )


def test_stream_true(report_test, api_client, request):
    """Tests the 'stream' parameter set to true returns a streaming response."""
    payload = {"input": BASE_INPUT, "max_output_tokens": 2048, "stream": True}

    response = api_client(payload, stream=True)

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
    payload = {"input": BASE_INPUT, "max_output_tokens": 2048, "stream": False}
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
    if fmt_type == "json_schema":
        input_msg = "List 3 colors as a JSON object with a 'colors' key."

    payload = {
        "input": input_msg,
        "max_output_tokens": 2048,
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
            assert all(isinstance(c, str) for c in parsed["colors"]), (
                f"Expected all items in 'colors' to be strings. Got: {parsed['colors']}"
            )
        except json.JSONDecodeError:
            pytest.fail(
                f"Expected valid JSON output with json_schema format. Got: '{output_text}'"
            )


@pytest.mark.parametrize(
    "choice",
    [
        "auto",
        pytest.param(
            "none",
            marks=pytest.mark.xfail(
                reason="vLLM does not support tool_choice='none' https://github.com/vllm-project/vllm/issues/33966"
            ),
        ),
        pytest.param(
            "required",
            marks=pytest.mark.xfail(
                reason="vLLM does not support tool_choice='required' https://github.com/vllm-project/vllm/issues/33966"
            ),
        ),
    ],
)
def test_tool_choice(report_test, api_client, choice, request):
    """Tests the 'tool_choice' parameter."""
    payload = {
        "input": "What's the weather like in San Francisco?",
        "tools": [WEATHER_TOOL],
        "tool_choice": choice,
        "max_output_tokens": 256,
    }

    response = api_client(payload)

    try:
        assert "output" in response, "Expected 'output' in response."
    except AssertionError as e:
        msg = f"AssertionError: {str(e)}. Response: {response}"
        raise AssertionError(msg)


def test_tools(report_test, api_client, request):
    """Tests that the 'tools' parameter triggers tool calls with valid structure."""
    payload = {
        "input": "What's the weather like in San Francisco?",
        "tools": [WEATHER_TOOL],
        "temperature": 0,
        "max_output_tokens": 256,
    }
    response = api_client(payload)

    tool_calls = get_function_calls(response)
    assert tool_calls, (
        f"Expected at least one function_call in output. Response: {response}"
    )

    fc = tool_calls[0]
    assert fc["name"] == WEATHER_TOOL["name"], (
        f"Expected tool call name '{WEATHER_TOOL['name']}', got '{fc['name']}'."
    )

    args = json.loads(fc["arguments"])
    assert "city" in args, f"Expected 'city' in tool call arguments. Got: {args}"
    assert args["city"].lower() == "san francisco", (
        f"Expected city 'san francisco', got '{args['city']}'."
    )

    # Verify the call has a call_id
    assert fc.get("call_id"), f"Expected function_call to have a 'call_id'. Got: {fc}"


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


@pytest.mark.parametrize(
    "truncation",
    [
        pytest.param(
            "auto",
            marks=pytest.mark.xfail(
                reason="vLLM does not support truncation=auto https://github.com/vllm-project/vllm/issues/38132"
            ),
        ),
        "disabled",
    ],
)
def test_truncation(report_test, api_client, truncation, max_context, request):
    """Tests the 'truncation' parameter.

    Sends an input that exceeds the model's context window.
    - auto: the server should truncate and return a response with input_tokens <= max_context.
    - disabled: the server should not truncate; input_tokens should reflect the full input.
    """
    # Build an input large enough to exceed the context window.
    filler_message = "This is filler text to consume tokens. " * 50
    num_messages = (
        max_context // 40
    ) + 1  # rough estimate: ~40 tokens per filler message
    oversized_input = [
        {"role": "user", "content": filler_message} for _ in range(num_messages)
    ]

    payload = {
        "input": oversized_input,
        "max_output_tokens": 1024,
        "truncation": truncation,
    }

    if truncation == "auto":
        response = api_client(payload, timeout=120)
        assert response.get("status") in ("completed", "incomplete")
        input_tokens = response.get("usage", {}).get("input_tokens", 0)
        assert 0 < input_tokens <= max_context
    else:
        # disabled truncation should reject oversized input
        with pytest.raises(requests.exceptions.HTTPError, match="400"):
            api_client(payload, timeout=120)


def test_user(report_test, api_client, request):
    """Tests that the 'user' parameter is accepted."""
    payload = {
        "input": BASE_INPUT,
        "max_output_tokens": 2048,
        "user": "test-user-123",
    }
    response = api_client(payload)

    try:
        output_text = get_output_text(response)
        assert output_text, "Expected non-empty output text."
    except AssertionError as e:
        msg = f"AssertionError: {str(e)}. Response: {response}"
        raise AssertionError(msg)
