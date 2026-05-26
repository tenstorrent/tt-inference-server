# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from pathlib import Path

from workflows.model_spec import (
    DeviceModelSpec,
    KnownIssue,
    ModelSpecTemplate,
    SystemRequirements,
    VersionRequirement,
    _IMPL_REGISTRY,
    _build_device_model_spec,
    _build_system_requirements,
    _build_template,
    load_templates_from_yaml,
    tt_transformers_impl,
)
from workflows.workflow_types import (
    DeviceTypes,
    InferenceEngine,
    ModelStatusTypes,
    ModelType,
    VersionMode,
    WorkflowType,
)


def test_impl_registry_is_populated():
    """Every ImplSpec instance defined at module scope must be in _IMPL_REGISTRY."""
    assert _IMPL_REGISTRY["tt_transformers"] is tt_transformers_impl
    # impl_id of each registry entry must match its key
    for impl_id, impl in _IMPL_REGISTRY.items():
        assert impl.impl_id == impl_id


def test_build_system_requirements_full():
    out = _build_system_requirements(
        {
            "firmware": {"specifier": ">=19.2.0", "mode": "STRICT"},
            "kmd": {"specifier": ">=2.5.0", "mode": "SUGGESTED"},
        }
    )
    assert isinstance(out, SystemRequirements)
    assert out.firmware == VersionRequirement(
        specifier=">=19.2.0", mode=VersionMode.STRICT
    )
    assert out.kmd == VersionRequirement(
        specifier=">=2.5.0", mode=VersionMode.SUGGESTED
    )


def test_build_system_requirements_none_returns_none():
    assert _build_system_requirements(None) is None


def test_build_device_model_spec_with_known_issues_and_overrides():
    spec = _build_device_model_spec(
        {
            "device": "T3K",
            "max_concurrency": 32,
            "max_context": 32768,
            "default_impl": True,
            "vllm_args": {
                "data_parallel_size": 4,
                "limit-mm-per-prompt": '{"image": 1}',
            },
            "override_tt_config": {"trace_region_size": 90000000},
            "env_vars": {
                "TT_MM_THROTTLE_PERF": 5,
                "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
            },
            "known_issues": [
                {
                    "workflow_type": "EVALS",
                    "reason": "broken on this device",
                    "task_name": "ifeval",
                },
            ],
        }
    )
    assert isinstance(spec, DeviceModelSpec)
    assert spec.device == DeviceTypes.T3K
    assert spec.vllm_args["data_parallel_size"] == 4
    assert spec.override_tt_config["trace_region_size"] == 90000000
    assert spec.known_issues == [
        KnownIssue(
            workflow_type=WorkflowType.EVALS,
            reason="broken on this device",
            task_name="ifeval",
        ),
    ]


def test_build_template_resolves_all_enum_and_impl_references():
    template = _build_template(
        {
            "weights": ["Qwen/Qwen3-8B"],
            "impl": "tt_transformers",
            "version": "0.10.0",
            "tt_metal_commit": "abc1234",
            "vllm_commit": "def5678",
            "inference_engine": "VLLM",
            "device_model_specs": [
                {
                    "device": "N150",
                    "max_concurrency": 32,
                    "max_context": 32768,
                    "default_impl": True,
                },
            ],
            "status": "FUNCTIONAL",
            "model_type": "LLM",
            "supported_modalities": ["text"],
            "env_vars": {"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"},
            "metadata": {"Qwen/Qwen3-8B": {"reasoning_parser_name": "qwen3"}},
        }
    )
    assert isinstance(template, ModelSpecTemplate)
    assert template.impl is _IMPL_REGISTRY["tt_transformers"]
    assert template.inference_engine == InferenceEngine.VLLM.value
    assert template.status == ModelStatusTypes.FUNCTIONAL
    assert template.model_type == ModelType.LLM
    assert template.device_model_specs[0].device == DeviceTypes.N150


def test_load_templates_from_yaml_roundtrip(tmp_path):
    yaml_path = tmp_path / "tiny.yaml"
    yaml_path.write_text(
        """
templates:
  - weights: [Qwen/Qwen3-8B]
    impl: tt_transformers
    version: "0.10.0"
    tt_metal_commit: abc1234
    vllm_commit: def5678
    inference_engine: VLLM
    device_model_specs:
      - device: N150
        max_concurrency: 32
        max_context: 32768
        default_impl: true
    status: FUNCTIONAL
""".strip()
    )
    templates = load_templates_from_yaml(yaml_path)
    assert len(templates) == 1
    assert templates[0].weights == ["Qwen/Qwen3-8B"]
    assert templates[0].impl is _IMPL_REGISTRY["tt_transformers"]


import pytest

MODEL_SPECS_DIR = Path(__file__).resolve().parent.parent / "workflows" / "model_specs"
EXPECTED_CATALOG_ENVS = ("prod", "dev")
EXPECTED_CATALOG_FILES = (
    "llm.yaml",
    "vlm.yaml",
    "video.yaml",
    "image.yaml",
    "audio_tts.yaml",
    "embedding.yaml",
    "cnn.yaml",
)


@pytest.mark.parametrize("env", EXPECTED_CATALOG_ENVS)
def test_all_expected_catalog_files_exist(env):
    found = {p.name for p in (MODEL_SPECS_DIR / env).glob("*.yaml")}
    missing = set(EXPECTED_CATALOG_FILES) - found
    assert not missing, f"Missing catalog YAML files in {env}/: {missing}"


@pytest.mark.parametrize(
    "env,yaml_name",
    [(env, name) for env in EXPECTED_CATALOG_ENVS for name in EXPECTED_CATALOG_FILES],
)
def test_catalog_yaml_loads_and_every_template_expands(env, yaml_name):
    """Each per-category catalog YAML (in each env) must load and every
    template must expand to >=1 spec. Surfaces typos and missing-field errors
    with a per-env, per-file, per-template assertion message instead of one
    opaque import-time exception.
    """
    templates = load_templates_from_yaml(MODEL_SPECS_DIR / env / yaml_name)
    assert templates, f"{env}/{yaml_name} produced zero templates"
    for t in templates:
        specs = t.expand_to_specs()
        assert specs, (
            f"{env}/{yaml_name}: template {t.weights} expanded to zero specs"
        )
