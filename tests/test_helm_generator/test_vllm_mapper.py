# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from workflows.helm_generator.vllm.mapper import VllmMapper


def test_vllm_engine_and_destination(vllm_spec):
    mapped = VllmMapper().map(vllm_spec)
    assert mapped.engine == "vllm"
    assert mapped.model_name == "Llama-3.1-8B-Instruct"
    assert mapped.device_name == "galaxy"
    assert mapped.impl_id == "test_impl"
    assert mapped.is_default  # fixture sets default_impl=True


def test_vllm_config_omits_serverType(vllm_spec):
    cfg = VllmMapper().map(vllm_spec).config
    assert "serverType" not in cfg.to_yaml_dict()


def test_vllm_image_split(vllm_spec):
    cfg = VllmMapper().map(vllm_spec).config
    assert cfg.image.repository == "ghcr.io/acme/vllm-server"
    assert cfg.image.tag == "0.11.1-bac8b34"


def test_vllm_resources_from_min_ram_gb(vllm_spec):
    cfg = VllmMapper().map(vllm_spec).config
    assert cfg.resources.requests_memory == f"{int(vllm_spec.min_ram_gb)}Gi"


def test_omits_requests_memory_when_min_ram_gb_none(spec_no_ram):
    cfg = VllmMapper().map(spec_no_ram).config
    assert cfg.resources.requests_memory is None
    assert "resources" not in cfg.to_yaml_dict()


def test_progress_deadline_seconds(vllm_spec):
    cfg = VllmMapper().map(vllm_spec).config
    expected = int(vllm_spec.device_model_spec.tensor_cache_timeout) + 1800
    assert cfg.progress_deadline_seconds == expected


def test_probes_initial_delay(vllm_spec):
    cfg = VllmMapper().map(vllm_spec).config
    expected = int(vllm_spec.device_model_spec.tensor_cache_timeout * 2 / 3)
    assert cfg.probes.liveness.initial_delay_seconds == expected
    assert cfg.probes.readiness.initial_delay_seconds == expected
    assert cfg.probes.liveness.path is None
    assert cfg.probes.readiness.path is None


def test_env_list_sorted_and_includes_spec_env_vars(vllm_spec):
    cfg = VllmMapper().map(vllm_spec).config
    names = [e.name for e in cfg.env]
    assert names == sorted(names)
    assert "VLLM_CONFIGURE_LOGGING" in names
    assert "MESH_DEVICE" in names


def test_owned_paths_include_env_and_image():
    paths = VllmMapper().owned_leaf_paths()
    assert ("image", "tag") in paths
    assert ("image", "repository") in paths
    assert ("env",) in paths
    assert ("progressDeadlineSeconds",) in paths
    assert ("resources", "requests", "memory") in paths
    assert ("resources", "limits", "memory") not in paths
