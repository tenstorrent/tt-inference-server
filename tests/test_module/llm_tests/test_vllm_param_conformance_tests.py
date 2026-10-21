# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Tests for the vLLM parameter-conformance spec-test wrappers."""

from __future__ import annotations

import json
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


def _make_test_with_config(config: dict) -> VLLMParamConformanceTest:
    return VLLMParamConformanceTest(
        TestConfig(config), {}, ctx=_fake_ctx(hf_model_repo="org/m", model_name="m")
    )


def test_suite_passes_the_test_cases_chat_template_kwargs_verbatim():
    """The checks read message.content, so a test case states the kwargs that
    switch its model's thought channel off; the key is model-specific."""
    for kwargs in ({"enable_thinking": False}, {"thinking": False}):
        test = _make_test_with_config({"chat_template_kwargs": kwargs})
        assert test._extra_pytest_args() == [
            "--chat-template-kwargs",
            json.dumps(kwargs),
        ]


def test_suite_accepts_chat_template_kwargs_as_a_json_string():
    test = _make_test_with_config(
        {"chat_template_kwargs": '{"enable_thinking": false}'}
    )
    assert test._extra_pytest_args() == [
        "--chat-template-kwargs",
        '{"enable_thinking": false}',
    ]


@pytest.mark.parametrize(
    "config", [{}, {"chat_template_kwargs": {}}, {"chat_template_kwargs": ""}]
)
def test_suite_adds_no_template_args_without_chat_template_kwargs(config):
    assert _make_test_with_config(config)._extra_pytest_args() == []
    assert VLLMParamConformanceTest(TestConfig(config), {})._extra_pytest_args() == []


def test_suite_rejects_chat_template_kwargs_that_are_not_an_object():
    with pytest.raises(ValueError, match="chat_template_kwargs"):
        _make_test_with_config(
            {"chat_template_kwargs": ["enable_thinking"]}
        )._extra_pytest_args()
