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
    "repeat_trap": [{"role": "user", "content": "Write a story where you reuse the same words as much as possible."}],
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
    payload_base = {"messages": messages, "max_tokens": 1024, "temperature": 0.1, "seed": 2002}
    print(payload_base)
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
        if prompt_name == "repeat_trap":
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
