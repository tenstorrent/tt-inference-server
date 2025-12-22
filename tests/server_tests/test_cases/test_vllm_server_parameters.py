import asyncio
import math
from collections import Counter

import pytest


# --- Helper Functions ---
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


# --- Reusable Prompts ---
BASE_PROMPT = [{"role": "user", "content": "Tell me a short joke."}]
REPRO_PROMPT = [
    {"role": "user", "content": "What is the capital of France? Be concise."}
]
PENALTY_PROMPTS = {
    "repeat_trap": [{"role": "user", "content": "Write a very repetitive story."}],
    "natural_repetition": [
        {
            "role": "user",
            "content": "Write a paragraph about bananas using simple language.",
        }
    ],
    "semantic_repetition": [
        {
            "role": "user",
            "content": "Write a creative story about a dragon named Blaze.",
        }
    ],
}


# --- Test Functions ---
@pytest.mark.parametrize("n_val", [2, 3])
def test_n(report_test, api_client, n_val, request):
    """Tests the 'n' parameter (number of choices)."""
    payload = {"messages": BASE_PROMPT, "n": n_val, "max_tokens": 32}
    response = api_client(payload)

    try:
        assert "choices" in response, "choices field is not in response"
        assert len(response["choices"]) == n_val, (
            f"length of choices field does not match n_val: {n_val}"
        )
    except AssertionError as e:
        msg = f"AssertionError: {str(e)}. Response: {response}"
        raise AssertionError(msg)


@pytest.mark.parametrize("max_val", [5, 10])
def test_max_tokens(report_test, api_client, max_val, request):
    """Tests the 'max_tokens' parameter."""
    payload = {"messages": BASE_PROMPT, "max_tokens": max_val}
    response = api_client(payload)

    completion_token_count = response["usage"].get("completion_tokens")
    if not completion_token_count:
        raise ValueError("Response did not contain completion_tokens field in usage.")

    assert completion_token_count <= max_val, (
        f"Generated {completion_token_count} tokens, which is greater than allowed amount of {max_val}."
    )


@pytest.mark.parametrize("stop_seq", [["Stop"], ["END", "DONE"]])
def test_stop(report_test, api_client, stop_seq, request):
    """Tests the 'stop' parameter."""
    prompt = [{"role": "user", "content": "Count to 5 and then say 'StopIt'."}]
    payload = {"messages": prompt, "stop": stop_seq, "max_tokens": 20}
    response = api_client(payload)

    output_text = response["choices"][0]["message"]["content"]
    for seq in stop_seq:
        assert seq not in output_text, f"Sequence {seq} was in {output_text}"


def test_seed_reproducibility(report_test, api_client, request):
    """Tests the 'seed' parameter for reproducibility."""
    payload = {"messages": REPRO_PROMPT, "seed": 42, "temperature": 0.5}

    response1 = api_client(payload)
    response2 = api_client(payload)

    output1 = response1["choices"][0]["message"]["content"]
    output2 = response2["choices"][0]["message"]["content"]
    assert output1 and output1 == output2, (
        f"Seed did not produce reproducible results. Output 1: '{output1}', Output 2: '{output2}'"
    )


@pytest.mark.asyncio
async def test_non_uniform_seeding(report_test, api_client, request):
    """
    Tests that vLLM Chat endpoint respects per-request seeding
    under concurrent load.

    Pattern:
    - 32 concurrent requests.
    - Seeds: [1, 0, 2, 0, 3, 0 ...]
    - Expectation:
        - All seed=0 requests -> Identical content.
        - All seed!=0 requests -> Unique content.
    """

    total_requests = 32
    seeds = []

    # Generate pattern: 1, 0, 2, 0, 3, 0...
    for i in range(1, (total_requests // 2) + 1):
        seeds.append(i)
        seeds.append(0)

    async def get_chat_response(seed_val):
        try:
            # We use a temperature > 0 to prove that the seed is
            # actually forcing the determinism.
            payload = {
                "messages": [
                    {"role": "user", "content": "Generate a list of 10 random colors."}
                ],
                "max_tokens": 50,
                "temperature": 0.9,
                "seed": seed_val,
            }
            response = await asyncio.to_thread(api_client, payload)
            return {
                "seed": seed_val,
                "content": response["choices"][0]["message"]["content"].strip(),
                "id": response["id"],
            }
        except Exception as e:
            pytest.fail(f"Request failed for seed {seed_val}: {str(e)}")

    # Fire all requests concurrently
    # This floods the server, forcing dynamic batching if enabled on vLLM
    tasks = [get_chat_response(s) for s in seeds]
    results = await asyncio.gather(*tasks)

    # Separate results
    zero_seed_contents = [r["content"] for r in results if r["seed"] == 0]
    unique_seed_contents = [r["content"] for r in results if r["seed"] != 0]
    assert len(zero_seed_contents) == total_requests / 2
    assert len(unique_seed_contents) == total_requests / 2

    # --- ASSERTIONS ---

    # 1. Verify Determinism (Seed = 0)
    # Using a set to find unique values. Length should be exactly 1.
    unique_zero_outputs = set(zero_seed_contents)
    assert len(unique_zero_outputs) == 1, (
        f"Determinism Failed for seed=0.\n"
        f"Expected 1 unique output, found {len(unique_zero_outputs)}.\n"
        f"Outputs: {unique_zero_outputs}"
    )

    # 2. Verify Entropy (Seed != 0)
    # The set length should equal the list length (all unique).
    unique_varied_outputs = set(unique_seed_contents)
    assert len(unique_varied_outputs) == len(unique_seed_contents), (
        f"Entropy Failed for non-zero seeds.\n"
        f"Expected {len(unique_seed_contents)} unique outputs, found {len(unique_varied_outputs)}.\n"
        f"Collisions detected in: {unique_seed_contents}"
    )


def test_logprobs(report_test, api_client, request):
    """Tests the 'logprobs' parameter."""
    payload = {"messages": BASE_PROMPT, "logprobs": True, "max_tokens": 100}
    response = api_client(payload)

    try:
        assert "logprobs" in response["choices"][0]
        assert response["choices"][0]["logprobs"] is not None
        assert "content" in response["choices"][0]["logprobs"]
        assert len(response["choices"][0]["logprobs"]["content"]) > 0
    except AssertionError as e:
        msg = f"Logprobs data was missing or malformed: {str(e)}. Response: {response}"
        raise AssertionError(msg)


@pytest.mark.parametrize(
    "param_name, param_value",
    [
        ("temperature", 0.0),
        ("top_k", 1),
        ("top_p", 0.01),  # A very low top_p should also be deterministic
    ],
)
def test_determinism_parameters(
    report_test, api_client, param_name, param_value, request
):
    """Tests parameters that should force deterministic output."""
    payload = {"messages": REPRO_PROMPT, param_name: param_value}

    # If testing top_k or top_p, set temperature high to prove they are working
    if param_name != "temperature":
        payload["temperature"] = 1.0

    response1 = api_client(payload)
    response2 = api_client(payload)

    output1 = response1["choices"][0]["message"]["content"]
    output2 = response2["choices"][0]["message"]["content"]
    assert output1 and output1 == output2, (
        f"{param_name}={param_value} was not deterministic. Output 1: '{output1}', Output 2: '{output2}'"
    )


@pytest.mark.parametrize("prompt_name, messages", PENALTY_PROMPTS.items())
@pytest.mark.parametrize(
    "penalty_param, penalty_val",
    [
        ("presence_penalty", 1.2),
        ("frequency_penalty", 1.2),
        ("repetition_penalty", 1.5),  # vLLM implements this, OpenAI uses the other two
    ],
)
def test_penalties(
    report_test, api_client, prompt_name, messages, penalty_param, penalty_val, request
):
    """Tests repetition, presence, and frequency penalties."""

    # Baseline run (no penalty)
    payload_base = {"messages": messages, "temperature": 0.1, "max_tokens": 1024}
    response_base = api_client(payload_base, timeout=None)

    # Test run (with penalty)
    payload_test = payload_base.copy()
    payload_test[penalty_param] = penalty_val
    response_test = api_client(payload_test, timeout=None)

    # Compute baseline and test statistics
    text_base = response_base["choices"][0]["message"]["content"]
    text_test = response_test["choices"][0]["message"]["content"]

    base_stats = repetition_stats(text_base)
    test_stats = repetition_stats(text_test)

    try:
        # 1. Diversity should not decrease (penalties push diversity up)
        assert test_stats["unique_ratio"] >= base_stats["unique_ratio"] * 0.90, (
            "Penalty unexpectedly reduced diversity."
        )

        # 2. Heavy repetition should decrease
        #    For repetition-heavy prompts, penalties reduce top-token dominance
        if prompt_name == "repeat_trap" and penalty_param != "presence_penalty":
            most_common_penalty = test_stats["most_common"][0][1]
            most_common_baseline = base_stats["most_common"][0][1]
            assert most_common_penalty <= most_common_baseline, (
                "Penalty didn't reduce repetition on repetition-trap prompt."
            )

        # 3. Length differences accepted but should not be identical
        assert test_stats["len"] != base_stats["len"], (
            "Penalty had no measurable effect on output length."
        )

        # For vLLM-specific repetition_penalty, check more aggressive behavior
        if penalty_param == "repetition_penalty":
            assert test_stats["unique_ratio"] > base_stats["unique_ratio"], (
                "vLLM repetition_penalty did not increase diversity enough."
            )
    except AssertionError as e:
        msg = f"Test failed: {str(e)}. Base: {text_base}, Test: {text_test}"
        raise AssertionError(msg)
