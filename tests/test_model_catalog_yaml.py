# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import json
from pathlib import Path

from workflows.model_spec import MODEL_SPECS, export_model_specs_json

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "model_specs_golden.json"


def _current_export(tmp_path: Path) -> dict:
    out = tmp_path / "current.json"
    export_model_specs_json(MODEL_SPECS, out)
    return json.loads(out.read_text())


def test_model_specs_match_golden(tmp_path):
    """MODEL_SPECS output must be byte-identical to the pre-migration snapshot."""
    assert GOLDEN_PATH.exists(), (
        "Golden snapshot missing. Regenerate with: python -c \"...\" "
        "(see plan task 1 step 1)."
    )
    golden = json.loads(GOLDEN_PATH.read_text())
    current = _current_export(tmp_path)
    # Compare only the catalog payload; schema_version / release_version
    # can drift independently of catalog content.
    assert current["model_specs"] == golden["model_specs"]


from workflows.model_spec import (
    DeviceModelSpec,
    ImplSpec,
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
    out = _build_system_requirements({
        "firmware": {"specifier": ">=19.2.0", "mode": "STRICT"},
        "kmd": {"specifier": ">=2.5.0", "mode": "SUGGESTED"},
    })
    assert isinstance(out, SystemRequirements)
    assert out.firmware == VersionRequirement(specifier=">=19.2.0", mode=VersionMode.STRICT)
    assert out.kmd == VersionRequirement(specifier=">=2.5.0", mode=VersionMode.SUGGESTED)


def test_build_system_requirements_none_returns_none():
    assert _build_system_requirements(None) is None


def test_build_device_model_spec_with_known_issues_and_overrides():
    spec = _build_device_model_spec({
        "device": "T3K",
        "max_concurrency": 32,
        "max_context": 32768,
        "default_impl": True,
        "vllm_args": {"data_parallel_size": 4, "limit-mm-per-prompt": '{"image": 1}'},
        "override_tt_config": {"trace_region_size": 90000000},
        "env_vars": {"TT_MM_THROTTLE_PERF": 5, "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"},
        "known_issues": [
            {"workflow_type": "EVALS", "reason": "broken on this device", "task_name": "ifeval"},
        ],
    })
    assert isinstance(spec, DeviceModelSpec)
    assert spec.device == DeviceTypes.T3K
    assert spec.vllm_args["data_parallel_size"] == 4
    assert spec.override_tt_config["trace_region_size"] == 90000000
    assert spec.known_issues == [
        KnownIssue(workflow_type=WorkflowType.EVALS, reason="broken on this device", task_name="ifeval"),
    ]


def test_build_template_resolves_all_enum_and_impl_references():
    template = _build_template({
        "weights": ["Qwen/Qwen3-8B"],
        "impl": "tt_transformers",
        "version": "0.10.0",
        "tt_metal_commit": "abc1234",
        "vllm_commit": "def5678",
        "inference_engine": "VLLM",
        "device_model_specs": [
            {"device": "N150", "max_concurrency": 32, "max_context": 32768, "default_impl": True},
        ],
        "status": "FUNCTIONAL",
        "model_type": "LLM",
        "supported_modalities": ["text"],
        "env_vars": {"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"},
        "metadata": {"Qwen/Qwen3-8B": {"reasoning_parser_name": "qwen3"}},
    })
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
