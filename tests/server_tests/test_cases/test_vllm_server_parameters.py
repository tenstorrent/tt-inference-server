import pytest
import tiktoken

# A simple tokenizer to test max_tokens
# Using 'cl100k_base' which is used by gpt-4 and gpt-3.5-turbo
try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception:
    tokenizer = tiktoken.get_encoding("gpt2") # Fallback

# --- Reusable Prompts ---
BASE_PROMPT = [{"role": "user", "content": "Tell me a short joke."}]
REPRO_PROMPT = [{"role": "user", "content": "What is the capital of France? Be concise."}]
REPETITION_PROMPT = [{"role": "user", "content": "Repeat this exact phrase five times: 'The quick brown fox jumps over the lazy dog.'"}]

# --- Test Functions (using simplified add_to_report) ---

@pytest.mark.xfail(reason="Currently only supporting n=1 on tt.", strict=True)
@pytest.mark.parametrize("n_val", [2, 3])
def test_n(api_client, add_to_report, n_val, request):
    """Tests the 'n' parameter (number of choices)."""
    payload = {"messages": BASE_PROMPT, "n": n_val}
    response, error = api_client(payload)
    
    if error:
        msg = f"API Error: {str(error)}. Response: {response}"
        add_to_report(False, msg, test_value=n_val)
        pytest.fail(msg)

    try:
        assert "choices" in response
        assert len(response["choices"]) == n_val
        add_to_report(True, f"Responded with {n_val} choices as requested.", test_value=n_val)
    except AssertionError as e:
        msg = f"AssertionError: {str(e)}. Response: {response}"
        add_to_report(False, msg, test_value=n_val)
        pytest.fail(f"Failed 'n' test for n={n_val}: {msg}")

@pytest.mark.parametrize("max_val", [5, 10])
def test_max_tokens(api_client, add_to_report, max_val, request):
    """Tests the 'max_tokens' parameter."""
    payload = {"messages": BASE_PROMPT, "max_tokens": max_val}
    response, error = api_client(payload)

    if error:
        msg = f"API Error: {str(error)}. Response: {response}"
        add_to_report(False, msg, test_value=max_val)
        pytest.fail(msg)

    try:
        output_text = response["choices"][0]["message"]["content"]
        token_count = len(tokenizer.encode(output_text))
        
        # Allow for a small buffer (e.g., 2 tokens)
        assert token_count <= max_val + 2
        add_to_report(True, f"Generated {token_count} tokens (<= {max_val}).", test_value=max_val)
    except Exception as e:
        msg = f"Test failed: {str(e)}. Response: {response}"
        add_to_report(False, msg, test_value=max_val)
        pytest.fail(f"Failed 'max_tokens' test: {msg}")

@pytest.mark.parametrize("stop_seq", [["Stop"], ["END", "DONE"]])
def test_stop(api_client, add_to_report, stop_seq, request):
    """Tests the 'stop' parameter."""
    prompt = [{"role": "user", "content": "Count to 5 and then say 'StopIt'."}]
    payload = {"messages": prompt, "stop": stop_seq, "max_tokens": 20}
    response, error = api_client(payload)

    if error:
        msg = f"API Error: {str(error)}. Response: {response}"
        add_to_report(False, msg, test_value=stop_seq)
        pytest.fail(msg)

    try:
        output_text = response["choices"][0]["message"]["content"]
        for seq in stop_seq:
            assert seq not in output_text
        add_to_report(True, f"Correctly stopped before sequence {stop_seq}.", test_value=stop_seq)
    except AssertionError as e:
        msg = f"Stop sequence '{seq}' was found in output: {e}"
        add_to_report(False, msg, test_value=stop_seq)
        pytest.fail(f"Failed 'stop' test: {msg}")

def test_seed_reproducibility(api_client, add_to_report, request):
    """Tests the 'seed' parameter for reproducibility."""
    payload = {"messages": REPRO_PROMPT, "seed": 42, "temperature": 0.5}
    
    response1, error1 = api_client(payload)
    if error1:
        msg = f"API Error on first call: {str(error1)}. Response: {response1}"
        add_to_report(False, msg, test_value=payload["seed"])
        pytest.fail(msg)
        
    response2, error2 = api_client(payload)
    if error2:
        msg = f"API Error on second call: {str(error2)}. Response: {response2}"
        add_to_report(False, msg, test_value=payload["seed"])
        pytest.fail(msg)

    try:
        output1 = response1["choices"][0]["message"]["content"]
        output2 = response2["choices"][0]["message"]["content"]
        assert output1 and output1 == output2
        add_to_report(True, "Same seed produced identical output.", test_value=payload["seed"])
    except AssertionError as e:
        msg = f"Seed did not produce reproducible results. Output 1: '{output1}', Output 2: '{output2}'"
        add_to_report(False, msg, test_value=payload["seed"])
        pytest.fail(f"Failed 'seed' test: {msg}")

def test_logprobs(api_client, add_to_report, request):
    """Tests the 'logprobs' parameter."""
    payload = {"messages": BASE_PROMPT, "logprobs": True, "top_logprobs": 2, "max_tokens": 100}
    response, error = api_client(payload)
    
    if error:
        msg = f"API Error: {str(error)}. Response: {response}"
        add_to_report(False, msg, test_value=payload["logprobs"])
        pytest.fail(msg)

    try:
        assert "logprobs" in response["choices"][0]
        assert response["choices"][0]["logprobs"] is not None
        assert "content" in response["choices"][0]["logprobs"]
        assert len(response["choices"][0]["logprobs"]["content"]) > 0
        add_to_report(True, "Logprobs content was returned.", test_value=payload["logprobs"])
    except Exception as e:
        msg = f"Logprobs data was missing or malformed: {str(e)}. Response: {response}"
        add_to_report(False, msg, test_value=payload["logprobs"])
        pytest.fail(f"Failed 'logprobs' test: {msg}")

@pytest.mark.parametrize("param_name, param_value", [
    ("temperature", 0.0),
    ("top_k", 1),
    ("top_p", 0.01)  # A very low top_p should also be deterministic
])
def test_determinism_parameters(api_client, add_to_report, param_name, param_value, request):
    """Tests parameters that should force deterministic output."""
    payload = {"messages": REPRO_PROMPT, param_name: param_value}
    
    # If testing top_k or top_p, set temperature high to prove they are working
    if param_name != "temperature":
        payload["temperature"] = 1.0

    response1, error1 = api_client(payload)
    response2, error2 = api_client(payload)
    
    if error1 or error2:
        msg = f"API Error: {error1 or error2}. Response: {response1 or response2}"
        add_to_report(False, msg, test_value=f"{param_name}={param_value}")
        pytest.fail(msg)

    try:
        output1 = response1["choices"][0]["message"]["content"]
        output2 = response2["choices"][0]["message"]["content"]
        assert output1 and output1 == output2
        add_to_report(True, f"{param_name}={param_value} produced deterministic output.", test_value=f"{param_name}={param_value}")
    except AssertionError as e:
        msg = f"{param_name}={param_value} was not deterministic. Output 1: '{output1}', Output 2: '{output2}'"
        add_to_report(False, msg, test_value=f"{param_name}={param_value}")
        pytest.fail(msg)

@pytest.mark.parametrize("penalty_param, penalty_val", [
    ("presence_penalty", 2.0),
    ("frequency_penalty", 2.0),
    ("repetition_penalty", 1.5)  # vLLM implements this, OpenAI uses the other two
])
def test_penalties(api_client, add_to_report, penalty_param, penalty_val, request):
    """Tests repetition, presence, and frequency penalties."""
    
    # Baseline run (no penalty)
    payload_base = {"messages": REPETITION_PROMPT, "temperature": 0.1}
    response_base, error_base = api_client(payload_base)
    if error_base:
        pytest.skip(f"Skipping penalty test; baseline call failed: {error_base}")
    
    # Test run (with penalty)
    payload_test = payload_base.copy()
    payload_test[penalty_param] = penalty_val
    response_test, error_test = api_client(payload_test, timeout=None)
    
    if error_test:
        msg = f"API Error when applying penalty: {error_test}. Response: {response_test}"
        add_to_report(False, msg, test_value=f"{penalty_param}={penalty_val}")
        pytest.fail(msg)
        
    try:
        text_base = response_base["choices"][0]["message"]["content"]
        text_test = response_test["choices"][0]["message"]["content"]
        
        # Simple heuristic: count the word "fox"
        base_count = text_base.lower().count("fox")
        test_count = text_test.lower().count("fox")
        
        # We expect the penalty to reduce repetitions.
        # It's possible it reduces it to 0, or just to a lower number.
        assert test_count < base_count
        add_to_report(True, f"Penalty reduced repetitions (Base count: {base_count}, Test count: {test_count}).", test_value=f"{penalty_param}={penalty_val}")
    except AssertionError as e:
        msg = f"Penalty did not reduce repetitions (Base count: {base_count}, Test count: {test_count})."
        add_to_report(False, msg, test_value=f"{penalty_param}={penalty_val}")
        pytest.fail(f"Failed '{penalty_param}' test: {msg}")
    except Exception as e:
        msg = f"Test failed: {str(e)}. Base: {text_base}, Test: {text_test}"
        add_to_report(False, msg, test_value=f"{penalty_param}={penalty_val}")
        pytest.fail(f"Failed '{penalty_param}' test: {msg}")
