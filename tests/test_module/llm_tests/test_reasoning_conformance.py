# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

import asyncio
from types import SimpleNamespace

import pytest

from llm_module import test_vllm_chat_completions as suite


def response(content=None, reasoning=None):
    return {
        "id": "test",
        "choices": [{"message": {"content": content, "reasoning": reasoning}}],
    }


@pytest.mark.parametrize(
    "value",
    [response(), response(" ", "\n"), response([], "text"), response("text", {})],
)
def test_empty_or_invalid_generated_channels_fail(value):
    with pytest.raises(AssertionError):
        suite.generated_text(value)


def test_plain_final_and_reasoning_channels_are_both_preserved():
    assert suite.generated_text(response("final")) == "final"
    assert suite.generated_text(response(reasoning="thinking")) == "thinking"
    assert suite.generated_text(response("final", "thinking")) == "thinking\nfinal"


@pytest.mark.parametrize("value", [response("Stop"), response(reasoning="Stop")])
def test_stop_leakage_fails_in_either_channel(value):
    with pytest.raises(AssertionError, match="Sequence Stop"):
        suite.test_stop(None, lambda _: value, ["Stop"], None)


def test_reasoning_stop_without_final_content_is_valid():
    suite.test_stop(
        None, lambda _: response(reasoning="Count then say '"), ["Stop"], None
    )


def test_coherence_never_accepts_target_in_reasoning_only():
    sentence = "The quick brown fox jumps over the lazy dog."
    requests = []

    def client(payload, **kwargs):
        assert kwargs == {"timeout": 3600}
        requests.append(payload)
        return response(reasoning=sentence)

    with pytest.raises(AssertionError, match="Coherence guard failed"):
        suite.test_coherence_verbatim_echo(
            None,
            client,
            SimpleNamespace(
                config=SimpleNamespace(
                    getoption=lambda _: "ibm-granite/granite-4.2-30b"
                )
            ),
        )
    assert requests[0]["max_tokens"] == 4096
    assert "chat_template_kwargs" not in requests[0]
    suite.test_coherence_verbatim_echo(
        None,
        lambda *a, **kw: response(sentence, "thinking"),
        SimpleNamespace(
            config=SimpleNamespace(getoption=lambda _: "ibm-granite/granite-4.2-30b")
        ),
    )


def test_seed_mismatch_fails_over_reasoning_channel():
    outputs = iter([response(reasoning="one"), response(reasoning="two")])
    with pytest.raises(AssertionError, match="Seed did not produce reproducible"):
        suite.test_seed_reproducibility(None, lambda _: next(outputs), None)


def test_concurrent_seeding_requires_nonempty_distinct_output():
    def client(payload):
        return response(reasoning=f"seed {payload['seed']}")

    asyncio.run(suite.test_non_uniform_seeding(None, client, None))
    with pytest.raises(pytest.fail.Exception):
        asyncio.run(suite.test_non_uniform_seeding(None, lambda _: response(), None))
    with pytest.raises(AssertionError, match="Entropy Failed"):
        asyncio.run(
            suite.test_non_uniform_seeding(
                None, lambda _: response(reasoning="same"), None
            )
        )


def test_penalty_no_effect_remains_failure():
    with pytest.raises(AssertionError, match="Penalty had no measurable effect"):
        suite.test_penalties(
            None,
            lambda *a, **kw: response("a b c", "thinking"),
            "natural_repetition",
            suite.PENALTY_PROMPTS["natural_repetition"],
            "repetition_penalty",
            1.5,
            None,
        )


def test_successful_final_echo_evidence_is_retained():
    from types import SimpleNamespace
    from test_fixtures import conftest as fixtures

    evidence = [
        {
            "request": {"max_tokens": 4096},
            "response": response(
                "The quick brown fox jumps over the lazy dog.", "thinking"
            ),
        }
    ]
    node = SimpleNamespace(
        originalname="test_coherence_verbatim_echo",
        name="test_coherence_verbatim_echo",
        nodeid="echo",
        api_responses=evidence,
    )
    request = SimpleNamespace(node=node)
    data = {"results": {}}
    report = fixtures.report_test.__wrapped__(data, request)
    next(report)
    node.rep_call = SimpleNamespace(
        passed=True, failed=False, outcome="passed", longrepr=None
    )
    with pytest.raises(StopIteration):
        next(report)
    assert data["coherence_api_responses"] == evidence
    assert "failed_api_responses" not in data


def test_legacy_reasoning_survives_null_new_channel():
    value = response()
    value["choices"][0]["message"]["reasoning_content"] = "legacy thinking"
    assert suite.generated_text(value) == "legacy thinking"


def test_penalty_reasoning_only_remains_quality_failure():
    outputs = iter(
        [
            response("final answer", "thinking"),
            response(reasoning="diverse but unfinished"),
        ]
    )
    with pytest.raises(AssertionError, match="penalty response has no final answer"):
        suite.test_penalties(
            None,
            lambda *a, **kw: next(outputs),
            "natural_repetition",
            suite.PENALTY_PROMPTS["natural_repetition"],
            "repetition_penalty",
            1.5,
            None,
        )


def test_other_models_keep_original_coherence_budget():
    def client(payload, **kwargs):
        assert payload["max_tokens"] == 32
        assert kwargs == {"timeout": 30}
        return response("The quick brown fox jumps over the lazy dog.")

    suite.test_coherence_verbatim_echo(
        None,
        client,
        SimpleNamespace(config=SimpleNamespace(getoption=lambda _: "other/model")),
    )
