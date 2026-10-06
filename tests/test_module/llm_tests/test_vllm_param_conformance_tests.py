# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Tests for the vLLM parameter-conformance spec-test wrappers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from test_module._test_common import TestConfig
from test_module.llm_tests.vllm_param_conformance_test import (
    DEFAULT_MODEL_NAME,
    VLLMDiffusionGemmaParamConformanceTest,
    VLLMParamConformanceTest,
)


def _make_test(ctx) -> VLLMParamConformanceTest:
    return VLLMParamConformanceTest(TestConfig({}), {}, ctx=ctx)


def _fake_ctx(*, hf_model_repo=None, model_name=None):
    """Minimal ctx stub carrying only what BaseTest.__init__ touches."""
    model_spec = SimpleNamespace(hf_model_repo=hf_model_repo, model_name=model_name)
    return SimpleNamespace(
        model_spec=model_spec,
        service_port=8000,
        base_url="http://127.0.0.1:8000",
    )


def test_resolve_model_name_uses_hf_model_repo_not_short_name():
    """Regression for #4489: the suite must send the full served name.

    The local vLLM server registers the model under hf_model_repo, so
    returning the short model_name causes a 404 "model does not exist".
    """
    ctx = _fake_ctx(
        hf_model_repo="meta-llama/Llama-3.1-8B-Instruct",
        model_name="Llama-3.1-8B-Instruct",
    )
    test = _make_test(ctx)

    assert test._resolve_model_name() == "meta-llama/Llama-3.1-8B-Instruct"


def test_resolve_model_name_falls_back_to_config_without_ctx():
    test = VLLMParamConformanceTest(TestConfig({"model": "org/some-model"}), {})

    assert test._resolve_model_name() == "org/some-model"


def test_resolve_model_name_defaults_when_unresolved():
    ctx = _fake_ctx(hf_model_repo=None, model_name=None)
    test = _make_test(ctx)

    assert test._resolve_model_name() == DEFAULT_MODEL_NAME


def test_diffusiongemma_suite_receives_catalog_max_context():
    ctx = _fake_ctx(
        hf_model_repo="google/diffusiongemma-26B-A4B-it",
        model_name="diffusiongemma-26B-A4B-it",
    )
    ctx.model_spec.device_model_spec = SimpleNamespace(max_context=262144)
    test = VLLMDiffusionGemmaParamConformanceTest(
        TestConfig({}),
        {},
        ctx=ctx,
    )

    assert test.PYTEST_FILENAME == "test_vllm_diffusiongemma.py"
    assert test._extra_pytest_args() == ["--max-context", "262144"]


def test_diffusiongemma_suite_fails_loudly_without_max_context():
    # A silently missing --max-context would pytest.skip the admission gates.
    ctx = _fake_ctx(
        hf_model_repo="google/diffusiongemma-26B-A4B-it",
        model_name="diffusiongemma-26B-A4B-it",
    )
    test = VLLMDiffusionGemmaParamConformanceTest(TestConfig({}), {}, ctx=ctx)

    with pytest.raises(RuntimeError, match="max_context"):
        test._extra_pytest_args()


def test_response_capture_is_scoped_to_chat_suite():
    from test_module.llm_tests.vllm_param_conformance_test import (
        VLLMResponsesParamConformanceTest,
        VLLMQwen3StreamingParamConformanceTest,
    )

    assert _make_test(None)._extra_pytest_args() == ["--capture-api-responses"]
    for cls in (
        VLLMResponsesParamConformanceTest,
        VLLMQwen3StreamingParamConformanceTest,
    ):
        assert cls(TestConfig({}), {})._extra_pytest_args() == []


@pytest.mark.parametrize("capture", [False, True])
def test_api_capture_preserves_payload_and_failed_report(monkeypatch, capture):
    from test_fixtures import conftest as fixtures

    monkeypatch.setattr(fixtures, "_get_bearer_token", lambda: "secret-not-recorded")
    payload = {"messages": [{"role": "user", "content": "Echo"}], "max_tokens": 32}
    body = {
        "choices": [
            {
                "message": {"content": None, "reasoning": "Thinking"},
                "finish_reason": "length",
            }
        ],
        "usage": {"completion_tokens": 32},
    }
    options = {"--model-name": "model", "--capture-api-responses": capture}
    node = SimpleNamespace(
        nodeid="public_test", originalname="test_echo", name="test_echo"
    )
    request = SimpleNamespace(
        node=node,
        config=SimpleNamespace(
            getoption=lambda key, default=None: options.get(key, default)
        ),
    )
    response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: body)
    client = fixtures.api_client.__wrapped__("http://localhost", request)
    assert client(payload, method=lambda *a, **kw: response) is body
    assert "model" not in payload
    report_data = {"results": {}}
    report = fixtures.report_test.__wrapped__(report_data, request)
    next(report)
    node.rep_call = SimpleNamespace(
        passed=False, failed=True, longrepr="original failure", outcome="failed"
    )
    with pytest.raises(StopIteration):
        next(report)
    assert report_data["results"]["test_echo"][0]["status"] == "failed"
    if capture:
        evidence = report_data["failed_api_responses"]["public_test"]
        assert evidence == [
            {"request": {**payload, "model": "model"}, "response": body}
        ]
        assert "secret-not-recorded" not in str(evidence)
    else:
        assert "failed_api_responses" not in report_data


def test_api_capture_preserves_http_errors(monkeypatch):
    import requests
    from test_fixtures import conftest as fixtures

    monkeypatch.setattr(fixtures, "_get_bearer_token", lambda: None)
    node = SimpleNamespace()
    request = SimpleNamespace(
        node=node,
        config=SimpleNamespace(
            getoption=lambda key, default=None: (
                True if key == "--capture-api-responses" else None
            )
        ),
    )
    response = SimpleNamespace(json=lambda: {"error": "server failed"})

    def fail():
        raise requests.exceptions.HTTPError("500", response=response)

    response.raise_for_status = fail
    client = fixtures.api_client.__wrapped__("http://localhost", request)
    with pytest.raises(requests.exceptions.HTTPError, match="server failed"):
        client({}, method=lambda *a, **kw: response)
    assert node.api_responses == []


def test_failed_raw_responses_survive_wrapper(monkeypatch):
    import asyncio

    test = _make_test(None)
    evidence = {
        "test_echo": [
            {
                "request": {"max_tokens": 32},
                "response": {
                    "choices": [
                        {
                            "message": {"content": None, "reasoning": "x" * 1000},
                            "finish_reason": "length",
                        }
                    ]
                },
            }
        ]
    }

    async def fake_suite(*args):
        return {
            "results": {"echo": [{"status": "failed", "message": "original failure"}]},
            "failed_api_responses": evidence,
        }

    monkeypatch.setattr(test, "_run_pytest_suite", fake_suite)
    result = asyncio.run(test._run_specific_test_async())
    assert result["success"] is False
    assert result["failed_api_responses"] == evidence
