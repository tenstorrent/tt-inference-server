# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The key the agentic harnesses send: the literal API key to a media server, a JWT to vLLM."""

from types import SimpleNamespace

import pytest

from test_module.llm_tests import agentic_eval_tests
from workflows.workflow_types import InferenceEngine


@pytest.fixture
def env(monkeypatch):
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "VLLM_API_KEY",
    ):
        # Set first so teardown restores the original (often unset) value.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.setenv("API_KEY", "shield-key")
    monkeypatch.setenv("JWT_SECRET", "shield-secret")
    return monkeypatch


def _ctx(engine, remote=False):
    return SimpleNamespace(
        base_url="http://127.0.0.1:8000",
        remote_server=remote,
        runtime_config=SimpleNamespace(jwt_secret=None),
        model_spec=SimpleNamespace(inference_engine=engine),
    )


def test_media_server_gets_the_literal_key(env):
    agentic_eval_tests._configure_openai_env(_ctx(InferenceEngine.MEDIA))
    assert agentic_eval_tests.os.environ["OPENAI_API_KEY"] == "shield-key"


def test_vllm_server_gets_the_jwt(env):
    agentic_eval_tests._configure_openai_env(_ctx(InferenceEngine.VLLM))
    key = agentic_eval_tests.os.environ["OPENAI_API_KEY"]
    assert key != "shield-key" and key.count(".") == 2
