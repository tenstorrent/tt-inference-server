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


def test_prompt_client_accepts_cache_dir_as_string(tmp_path):
    prompt_client = PromptClient(
        EnvironmentConfig(), cache_dir=str(tmp_path / "cache_monitor")
    )

    assert prompt_client.cache_monitor.cache_dir == tmp_path / "cache_monitor"


def test_prompt_client_resolves_host_cache_dir_from_runtime_config(tmp_path):
    model_spec = SimpleNamespace(
        impl=SimpleNamespace(impl_id="tt-transformers"),
        model_name="TestModel",
        version="1.0.0",
        device_type=SimpleNamespace(name="N150"),
        subdevice_type=None,
        uses_tensor_model_cache=True,
        device_model_spec=SimpleNamespace(tensor_cache_timeout=3600.0),
    )
    runtime_config = SimpleNamespace(
        host_volume=str(tmp_path / "persistent_volume"),
        local_server=False,
        device="n150",
    )

    prompt_client = PromptClient(
        EnvironmentConfig(),
        model_spec=model_spec,
        runtime_config=runtime_config,
    )

    assert prompt_client.cache_monitor.cache_dir == (
        tmp_path
        / "persistent_volume"
        / "volume_id_tt-transformers-TestModel-v1.0.0"
        / "tt_metal_cache"
        / "cache_TestModel"
        / "N150"
    )
