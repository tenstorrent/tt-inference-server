#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from types import SimpleNamespace

from utils.prompt_client import PromptClient
from utils.prompt_configs import EnvironmentConfig


def test_prompt_client_uses_model_spec_tensor_cache_timeout(tmp_path):
    model_spec = SimpleNamespace(
        device_model_spec=SimpleNamespace(tensor_cache_timeout=3600.0)
    )

    prompt_client = PromptClient(
        EnvironmentConfig(), model_spec=model_spec, cache_dir=tmp_path
    )

    assert prompt_client.cache_monitor.model_spec is model_spec
    assert prompt_client.cache_monitor.get_tensor_cache_timeout() == 3600.0
