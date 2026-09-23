# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Tests for the vLLM parameter-conformance spec-test wrappers."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from test_module._test_common import TestConfig
from test_module.llm_tests.vllm_param_conformance_test import (
    DEFAULT_MODEL_NAME,
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


def test_conformance_waits_for_local_vllm_health(monkeypatch):
    ctx = _fake_ctx(hf_model_repo="org/model")
    ctx.model_spec.device_model_spec = SimpleNamespace(tensor_cache_timeout=123)
    test = _make_test(ctx)
    observed = {}

    class FakeController:
        health_url = "http://127.0.0.1:8000/health"

        def __init__(self, **kwargs):
            observed["kwargs"] = kwargs

        def wait_for_healthy(self, timeout):
            observed["timeout"] = timeout
            return True

    monkeypatch.setattr(
        "test_module.llm_tests.vllm_param_conformance_test.HttpServerController",
        FakeController,
    )

    asyncio.run(test._wait_for_server_healthy())

    assert observed == {
        "kwargs": {
            "base_url": "http://127.0.0.1:8000",
            "service_port": 8000,
            "auth_token": "",
        },
        "timeout": 123.0,
    }


def test_conformance_health_failure_is_explicit(monkeypatch):
    test = _make_test(_fake_ctx(hf_model_repo="org/model"))

    class FakeController:
        health_url = "http://127.0.0.1:8000/health"

        def __init__(self, **_kwargs):
            pass

        def wait_for_healthy(self, _timeout):
            return False

    monkeypatch.setattr(
        "test_module.llm_tests.vllm_param_conformance_test.HttpServerController",
        FakeController,
    )

    try:
        asyncio.run(test._wait_for_server_healthy())
    except RuntimeError as exc:
        assert "did not become healthy" in str(exc)
    else:
        raise AssertionError("expected an explicit readiness failure")
