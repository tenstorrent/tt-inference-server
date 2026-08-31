# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import json
import re
from dataclasses import replace

import workflows.model_spec as model_spec_module
from workflows.utils import get_repo_root_path
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
    get_model_spec_map,
    get_runtime_model_spec,
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


def test_catalog_device_spec_accepts_perf_targets_map():
    """The tiers are settable per device; the reference numbers are not.

    `perf_targets_map` says what fraction of theoretical counts as passing, which
    is a property of the stack on that board. `perf_reference` is the theoretical
    number itself, a property of the model and the hardware, so it is derived and
    rejected here. Keeping both readable in one place because the pair is easy to
    confuse.
    """
    spec = _build_device_model_spec(
        {
            "device": "P300X2",
            "max_concurrency": 16,
            "max_context": 1024,
            "perf_targets_map": {"complete": 0.30},
        }
    )
    assert spec.perf_targets_map == {"complete": 0.30}


def test_catalog_device_spec_rejects_explicit_performance_reference():
    with pytest.raises(ValueError, match="must not define perf_reference"):
        _build_device_model_spec(
            {
                "device": "N150",
                "max_concurrency": 1,
                "max_context": 1024,
                "perf_reference": [],
            }
        )


def test_expansion_carries_every_catalog_device_field_but_perf_reference():
    """Expansion may substitute the derived field and nothing else.

    ``image_benchmark_num_batches`` was configurable in the catalog for a month
    without reaching the runtime spec, because expansion rebuilt the device spec
    from a hand-written field list. Compare against the whole catalog spec so a
    field added later cannot be dropped the same way.
    """
    template = _build_template(
        {
            "weights": ["Qwen/Qwen3-8B"],
            "impl": "tt_transformers",
            "inference_engine": "VLLM",
            "device_model_specs": [
                {
                    "device": "N150",
                    "max_concurrency": 32,
                    "max_context": 32768,
                    "default_impl": True,
                    "image_benchmark_num_batches": 7,
                    "eval_max_retries": 1,
                    "tensor_cache_timeout": 60.0,
                },
            ],
        },
        "dev",
    )
    catalog_spec = template.device_model_specs[0]

    runtime_spec = template.expand_to_specs()[0].device_model_spec

    assert runtime_spec.image_benchmark_num_batches == 7
    # ModelSpec.__post_init__ adds the weight to the runtime spec's vllm_args,
    # so check that one separately. Blanking it on both sides lets __post_init__
    # rebuild the same derived args and leaves every other field comparable.
    assert runtime_spec.vllm_args == {
        **catalog_spec.vllm_args,
        "model": "Qwen/Qwen3-8B",
    }
    assert replace(runtime_spec, vllm_args={}, perf_reference=[]) == replace(
        catalog_spec, vllm_args={}, perf_reference=[]
    )


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
    # prod/ dir so the pins below are accepted (env selects ProdModelSpecTemplate)
    prod_dir = tmp_path / "prod"
    prod_dir.mkdir()
    yaml_path = prod_dir / "tiny.yaml"
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


def _write_catalog(tmp_path, env, extra_fields):
    """Write a one-template catalog under tmp_path/<env>/x.yaml.

    extra_fields is rendered as additional 4-space-indented template keys.
    """
    d = tmp_path / env
    d.mkdir(parents=True)
    p = d / "x.yaml"
    lines = "".join(f"    {k}: {v}\n" for k, v in extra_fields.items())
    p.write_text(
        "templates:\n"
        "  - weights: [Qwen/Qwen3-8B]\n"
        "    impl: tt_transformers\n"
        "    inference_engine: VLLM\n"
        f"{lines}"
        "    device_model_specs:\n"
        "      - {device: N150, max_concurrency: 1, max_context: 1024, default_impl: true}\n"
    )
    return p


def test_prod_catalog_requires_tt_metal_commit_and_version(tmp_path):
    # Missing both -> ProdModelSpecTemplate construction fails for both.
    p = _write_catalog(tmp_path, "prod", {})
    with pytest.raises(ValueError, match=r"tt_metal_commit.*version"):
        load_templates_from_yaml(p)
    # Both present -> loads.
    p2 = _write_catalog(
        tmp_path / "ok", "prod", {"tt_metal_commit": "abc1234", "version": '"0.10.0"'}
    )
    assert len(load_templates_from_yaml(p2)) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("tt_metal_commit", "abc1234"),
        ("vllm_commit", "def5678"),
        ("version", '"0.10.0"'),
        ("docker_image", '"ghcr.io/x/y:1.0"'),
    ],
)
def test_dev_catalog_forbids_pinning_fields(tmp_path, field, value):
    # The dev base template has no pin fields, so setting one is an unexpected kwarg.
    p = _write_catalog(tmp_path, "dev", {field: value})
    with pytest.raises(ValueError, match=rf"unexpected keyword argument '{field}'"):
        load_templates_from_yaml(p)


def test_dev_catalog_without_pinning_fields_loads(tmp_path):
    p = _write_catalog(tmp_path, "dev", {})
    templates = load_templates_from_yaml(p)
    assert len(templates) == 1
    # dev (base) template carries no pin fields at all.
    assert not hasattr(templates[0], "tt_metal_commit")
    assert not hasattr(templates[0], "version")
    # expanded spec has them as None and skips docker/code-link synthesis.
    spec = templates[0].expand_to_specs()[0]
    assert spec.tt_metal_commit is None
    assert spec.version is None
    assert spec.docker_image is None
    assert spec.code_link is None


MODEL_SPECS_DIR = get_repo_root_path() / "workflows" / "model_specs"
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
        assert specs, f"{env}/{yaml_name}: template {t.weights} expanded to zero specs"


def test_multiweight_dev_templates_define_stable_display_name():
    for yaml_name in EXPECTED_CATALOG_FILES:
        templates = load_templates_from_yaml(MODEL_SPECS_DIR / "dev" / yaml_name)
        for template in templates:
            if len(template.weights) > 1:
                expected = template.weights[0].split("/")[-1]
                assert template.model_display_name == expected, (
                    f"dev/{yaml_name}: multiweight template {template.weights!r} "
                    f"must define model_display_name={expected!r}"
                )


@pytest.mark.parametrize("env", EXPECTED_CATALOG_ENVS)
def test_catalog_environment_has_unambiguous_expanded_identities(env):
    """Validate identities across category-file boundaries within one environment."""
    templates = [
        template
        for yaml_name in EXPECTED_CATALOG_FILES
        for template in load_templates_from_yaml(MODEL_SPECS_DIR / env / yaml_name)
    ]
    expanded_count = sum(len(template.expand_to_specs()) for template in templates)

    assert len(get_model_spec_map(templates)) == expanded_count


def _dev_llm_spec_map():
    templates = load_templates_from_yaml(MODEL_SPECS_DIR / "dev" / "llm.yaml")
    return get_model_spec_map(templates)


@pytest.mark.parametrize(
    "model_name,expected_context,expected_native_impl",
    [
        ("Qwen3.6-27B", 8192, "qwen36-blackhole"),
        ("gemma-4-31B-it", 4096, "tt-transformers"),
    ],
)
def test_quetzal_dev_specs_are_explicit_and_preserve_native_default(
    monkeypatch, model_name, expected_context, expected_native_impl
):
    specs = _dev_llm_spec_map()
    monkeypatch.setattr(model_spec_module, "MODEL_SPECS", specs)
    monkeypatch.setattr(model_spec_module, "_MODEL_SPECS_ENV", "dev")

    native, native_impl, _ = get_runtime_model_spec(model_name, "p300x2")
    assert native_impl == expected_native_impl
    assert native.device_model_spec.default_impl is True

    quetzal, resolved_impl, _ = get_runtime_model_spec(
        model_name, "p300x2", impl="quetzal"
    )
    assert resolved_impl == "quetzal"
    assert quetzal.impl.impl_id == "quetzal"
    assert quetzal.device_model_spec.default_impl is False
    assert quetzal.device_model_spec.max_concurrency == 1
    assert quetzal.device_model_spec.max_context == expected_context


@pytest.mark.parametrize("model_name", ["Qwen3.6-27B", "gemma-4-31B-it"])
def test_quetzal_dev_specs_use_content_store_contract(monkeypatch, model_name):
    specs = _dev_llm_spec_map()
    monkeypatch.setattr(model_spec_module, "MODEL_SPECS", specs)
    monkeypatch.setattr(model_spec_module, "_MODEL_SPECS_ENV", "dev")
    quetzal, _, _ = get_runtime_model_spec(model_name, "p300x2", impl="quetzal")
    env = quetzal.env_vars

    package_root = env["QUETZAL_PACKAGE_ROOT"]
    assert package_root.startswith("/home/container_app_user/quetzal/packages/sha256-")
    assert env["QZ_MODELS_ROOT"] == package_root
    assert env["QUETZAL_PACKAGE_ID"] == package_root.rsplit("/", 1)[-1]
    assert re.fullmatch(r"[0-9a-f]{64}", env["QUETZAL_BUNDLE_MANIFEST_SHA256"])
    for key in (
        "QZ_QUALIFICATION_MANIFEST",
        "QUETZAL_PREFILL_GENERATED_PY",
        "QUETZAL_DECODE_GENERATED_PY",
        "QUETZAL_PREFILL_METADATA_JSON",
        "QUETZAL_DECODE_METADATA_JSON",
        "QUETZAL_WEIGHTS",
    ):
        assert env[key].startswith(f"{package_root}/")

    serialized_env = "\n".join(f"{key}={value}" for key, value in env.items())
    assert "/home/ttuser" not in serialized_env
    assert "/mnt/nas" not in serialized_env


def test_qwen_quetzal_dev_spec_uses_portable_qb2_fabric_contract(monkeypatch):
    specs = _dev_llm_spec_map()
    monkeypatch.setattr(model_spec_module, "MODEL_SPECS", specs)
    monkeypatch.setattr(model_spec_module, "_MODEL_SPECS_ENV", "dev")
    quetzal, _, _ = get_runtime_model_spec("Qwen3.6-27B", "p300x2", impl="quetzal")
    env = quetzal.env_vars

    assert "TT_MESH_GRAPH_DESC_PATH" not in env
    assert env["TTQ_ROW_ALL_REDUCE_TOPOLOGY"] == "Linear"
    assert env["TTQ_TUNED_ROW_ALL_REDUCE_LINKS"] == "1"
    assert env["TTQ_TUNED_ROW_ALL_REDUCE"] == "1"


def test_qwen_quetzal_dev_spec_binds_canonical_package_identity(monkeypatch):
    specs = _dev_llm_spec_map()
    monkeypatch.setattr(model_spec_module, "MODEL_SPECS", specs)
    monkeypatch.setattr(model_spec_module, "_MODEL_SPECS_ENV", "dev")
    quetzal, _, _ = get_runtime_model_spec("Qwen3.6-27B", "p300x2", impl="quetzal")
    env = quetzal.env_vars

    identity_path = MODEL_SPECS_DIR / "dev" / "quetzal_package_identities.json"
    identities = json.loads(identity_path.read_text())
    source = identities["source"]
    identity = identities["models"]["Qwen/Qwen3.6-27B"]

    assert source == {
        "repository": "tenstorrent/tt-quetzalcoatlus",
        "revision": "9fb41112535ee87140e91c1bba6f831e62c30d42",
        "path": "productization/release_matrix.json",
        "sha256": "914cdb2ec37e31938a7ef2ed55801758b81eb9b03d7552b7808f3b1b4d851967",
    }
    assert env["QUETZAL_PACKAGE_ID"] == identity["package_id"]
    assert (
        env["QUETZAL_BUNDLE_MANIFEST_SHA256"]
        == identity["bundle_manifest_sha256"]
    )
    assert env["QUETZAL_HF_REVISION"] == identity["checkpoint_revision"]
    assert env["QUETZAL_REQUIRED_SOURCE_REVISION"] == source["revision"]
    assert env["QUETZAL_REQUIRED_TT_METAL_PATCHSET_SHA256"] == (
        "e240fa3880ea0c2597dd7df8ab657a69aca9fe215de58220ae96e47a48a29910"
    )


def test_gemma_quetzal_dev_spec_binds_exact_s4096_ring_candidate(monkeypatch):
    specs = _dev_llm_spec_map()
    monkeypatch.setattr(model_spec_module, "MODEL_SPECS", specs)
    monkeypatch.setattr(model_spec_module, "_MODEL_SPECS_ENV", "dev")
    quetzal, _, _ = get_runtime_model_spec("gemma-4-31B-it", "p300x2", impl="quetzal")
    env = quetzal.env_vars

    assert quetzal.device_model_spec.max_context == 4096
    assert env["QUETZAL_BUNDLE_MANIFEST_SHA256"] == (
        "e3ecc5557a84955bf0b95615e4b8e9fa83bcc431c9755e969ba5c441fc8d94cf"
    )
    assert env["QUETZAL_REQUIRED_SOURCE_REVISION"] == (
        "76a15d4cdd0c2b400ef9b89499a334a6b748e56b"
    )
    assert env["QUETZAL_REQUIRED_TT_METAL_PATCHSET_SHA256"] == (
        "22fb0bd2523b8a5c63fa20c3c8a1586dc9ead5150449d0eb02231fa8173a7edd"
    )
    assert env["TTQ_ROW_ALL_REDUCE_TOPOLOGY"] == "Ring"
    assert env["TTQ_TUNED_ROW_ALL_REDUCE_LINKS"] == "2"
    assert env["QZ_LM_HEAD_UPLOAD_CHUNK_COLS"] == "8192"
    assert "p300_x2_mesh_graph_descriptor.textproto" in env["TT_MESH_GRAPH_DESC_PATH"]


def test_diffusiongemma_dev_spec_matches_validated_256k_contract():
    templates = load_templates_from_yaml(MODEL_SPECS_DIR / "dev" / "llm.yaml")
    template = next(
        t for t in templates if t.weights == ["google/diffusiongemma-26B-A4B-it"]
    )
    spec = template.expand_to_specs()[0]
    device_spec = spec.device_model_spec

    assert device_spec.max_context == 262144
    assert device_spec.max_concurrency == 1
    assert device_spec.vllm_args["block_size"] == "64"
    assert device_spec.vllm_args["max_model_len"] == "262144"
    assert device_spec.vllm_args["max_num_batched_tokens"] == "262144"
    assert device_spec.vllm_args["max_num_seqs"] == "1"
    assert (
        device_spec.vllm_args["default-chat-template-kwargs"]
        == '{"enable_thinking": true}'
    )
    # vLLM 0.24 makes the DiffusionGemma parser effective, but lm-eval only
    # scores message.content. Keep scored serving parser-off so a final boxed
    # answer cannot be moved exclusively into message.reasoning.
    assert "reasoning-parser" not in device_spec.vllm_args
    assert "reasoning_parser_name" not in spec.metadata
    assert spec.metadata["output_block_size"] == 256
    assert (
        spec.metadata["max_effective_input_tokens"]
        == device_spec.max_context - spec.metadata["output_block_size"]
    )
    assert spec.has_builtin_warmup is True

    env = device_spec.env_vars
    assert env["DG_UPFRONT_CAPTURE"] == "1"
    assert env["DG_MODEL_OWNED_HYBRID_KV"] == "1"
    assert env["DG_UPFRONT_COARSE_PREFILL_BUCKETS"] == "1"
    assert env["DG_UPFRONT_LAZY_PREFILL_RECAPTURE"] == "1"
    assert env["DG_PREFILL_FIXED_CHUNKS"] == "1"
    assert env["DG_PREFILL_CHUNK_SIZE"] == "4096"
    assert env["DG_PREFILL_RAGGED_CHUNK"] == "1024"
    assert env["DG_DENOISE_REVEAL_PMAX"] == "262144"
    assert env["DG_UPFRONT_PREFILL_WARMUP_LENS"] == "32,64,96"
    assert env["DISABLE_METAL_OP_TIMEOUT"] == "1"
    assert int(env["DG_TRACE_REGION_SIZE"]) == 3758096384
    additional_config = json.loads(device_spec.vllm_args["additional_config"])
    assert additional_config == {
        "tt": {
            "sample_on_device_mode": "all",
            "enable_model_warmup": True,
            "trace_mode": "all",
            "trace_region_size": 3758096384,
        }
    }
    assert (
        int(env["DG_TRACE_REGION_SIZE"]) == additional_config["tt"]["trace_region_size"]
    )
