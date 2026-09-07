# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from workflows.helm_generator.schema import (
    HelmEnvVar,
    HelmImage,
    HelmImplConfig,
    HelmProbe,
    HelmProbes,
    HelmResources,
    HelmModelSpec,
)


def _impl_config(**overrides):
    defaults = dict(
        image=HelmImage(repository="ghcr.io/org/image", tag="1.0.0"),
        progress_deadline_seconds=5400,
        probes=HelmProbes(
            liveness=HelmProbe(),
            readiness=HelmProbe(),
        ),
    )
    defaults.update(overrides)
    return HelmImplConfig(**defaults)


def test_image_yaml_dict():
    img = HelmImage(repository="repo", tag="t")
    assert img.to_yaml_dict() == {"repository": "repo", "tag": "t"}


def test_probe_omits_block_when_path_none():
    assert HelmProbe().to_yaml_dict() == {}


def test_probe_includes_path_when_set():
    assert HelmProbe(path="/tt-liveness").to_yaml_dict() == {"path": "/tt-liveness"}


def test_probes_omitted_when_empty():
    cfg = _impl_config()
    assert "probes" not in cfg.to_yaml_dict()


def test_probes_emits_only_liveness_path():
    cfg = _impl_config(
        probes=HelmProbes(
            liveness=HelmProbe(path="/tt-liveness"), readiness=HelmProbe()
        )
    )
    assert cfg.to_yaml_dict()["probes"] == {"liveness": {"path": "/tt-liveness"}}


def test_resources_omits_block_when_no_memory():
    assert HelmResources().to_yaml_dict() is None


def test_resources_emits_only_requests_memory():
    assert HelmResources(requests_memory="20Gi").to_yaml_dict() == {
        "requests": {"memory": "20Gi"}
    }


def test_impl_config_omits_resources_and_env_when_empty():
    cfg = _impl_config()
    out = cfg.to_yaml_dict()
    assert "resources" not in out
    assert "env" not in out
    assert "serverType" not in out, "serverType lives in path, not in impl block"
    assert out["progressDeadlineSeconds"] == 5400


def test_impl_config_emits_env_list_in_order():
    cfg = _impl_config(
        env=[HelmEnvVar(name="A", value="1"), HelmEnvVar(name="B", value="2")]
    )
    assert cfg.to_yaml_dict()["env"] == [
        {"name": "A", "value": "1"},
        {"name": "B", "value": "2"},
    ]


def test_mapped_spec_fields():
    spec = HelmModelSpec(
        model_name="m",
        engine="vllm",
        device_name="galaxy",
        impl_id="impl_x",
        is_default=True,
        config=_impl_config(),
    )
    assert spec.model_name == "m"
    assert spec.engine == "vllm"
    assert spec.device_name == "galaxy"
    assert spec.impl_id == "impl_x"
    assert spec.is_default
