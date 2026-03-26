# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os

import pytest

from utils.prompt_configs import EnvironmentConfig, resolve_authorization_bearer
from utils.vllm_run_utils import configure_vllm_api_key_env, get_encoded_api_key


@pytest.fixture
def clean_vllm_auth_env():
    keys = ("VLLM_API_KEY", "API_KEY", "JWT_SECRET")
    saved = {k: os.environ.pop(k, None) for k in keys}
    yield
    for k in keys:
        if saved[k] is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = saved[k]


def test_configure_no_auth_removes_vllm_key(clean_vllm_auth_env):
    os.environ["VLLM_API_KEY"] = "secret"
    configure_vllm_api_key_env(no_auth=True)
    assert "VLLM_API_KEY" not in os.environ


def test_configure_prefers_vllm_api_key_over_api_key(clean_vllm_auth_env):
    os.environ["VLLM_API_KEY"] = "vllm-val"
    os.environ["API_KEY"] = "api-val"
    configure_vllm_api_key_env(no_auth=False)
    assert os.environ["VLLM_API_KEY"] == "vllm-val"


def test_configure_api_key_copied_to_vllm_api_key(clean_vllm_auth_env):
    os.environ["API_KEY"] = "shared-secret"
    configure_vllm_api_key_env(no_auth=False)
    assert os.environ["VLLM_API_KEY"] == "shared-secret"


def test_configure_jwt_secret_when_no_vllm_or_api_key(clean_vllm_auth_env):
    os.environ["JWT_SECRET"] = "jwt-secret-value"
    configure_vllm_api_key_env(no_auth=False)
    assert os.environ["VLLM_API_KEY"] == get_encoded_api_key("jwt-secret-value")


def test_configure_no_keys_no_vllm_env_var(clean_vllm_auth_env):
    configure_vllm_api_key_env(no_auth=False)
    assert "VLLM_API_KEY" not in os.environ


def test_configure_api_key_skips_jwt(clean_vllm_auth_env):
    os.environ["API_KEY"] = "from-api-key"
    os.environ["JWT_SECRET"] = "ignored"
    configure_vllm_api_key_env(no_auth=False)
    assert os.environ["VLLM_API_KEY"] == "from-api-key"


def test_resolve_authorization_prefers_vllm_key_over_api_key():
    env_config = EnvironmentConfig()
    env_config.vllm_api_key = "v1"
    env_config.api_key = "a1"
    assert resolve_authorization_bearer(env_config) == "v1"


def test_resolve_authorization_uses_api_key_when_no_vllm_key():
    env_config = EnvironmentConfig()
    env_config.vllm_api_key = None
    env_config.api_key = "media-style-key"
    assert resolve_authorization_bearer(env_config) == "media-style-key"
