# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy

import pytest

from scripts.release.plan_gpt120_f0b_quetzal_enrollment import (
    AUXILIARY_TREE_SHA256,
    CHECKPOINT,
    COMPILER_COMMIT,
    CONTAINER_AUXILIARY_PARENT,
    CONTAINER_PACKAGE_PARENT,
    DESCRIPTOR_CONTAINER_PATH,
    DESCRIPTOR_SHA256,
    EXPECTED_ARTIFACTS,
    EXPECTED_RELATIVE_PATHS,
    GPT_RELEASE_EVALS,
    GPT_SWEBENCH_INPUT_TOKENS,
    GPT_SWEBENCH_MIN_CONTEXT,
    GPT_SWEBENCH_OUTPUT_TOKENS,
    LOCAL_ON_DISPATCH_PROFILE,
    LOCAL_TTFT_MS_RANGE,
    MODEL_ID,
    QUETZAL_COMMIT,
    RUNNER_LABEL,
    SCHEMA,
    SHIELD_REQUIRED_ANCESTOR,
    STALE_5CAB_BUNDLE_MANIFEST_SHA256,
    STALE_5CAB_PACKAGE_ID,
    TT_METAL_COMMIT,
    TT_METAL_PATCHSET,
    TTFT_TARGET_MS,
    TTIS_REQUIRED_ANCESTOR,
    TOPOLOGY_SELECTION_SHA256,
    ContractError,
    render_contract,
)
from workflows import quetzal_topology_admission


def publication_response():
    generation = "tt-ci-models-private-generation-314159"
    core_host = "/mnt/models/quetzal/immutable/v1/gpt120-f0b-core"
    aux_host = "/mnt/models/quetzal/immutable/v1/gpt120-f0b-aux"
    package_id = "sha256-v2-" + "a" * 64
    core_container = f"{CONTAINER_PACKAGE_PARENT}/{package_id}"
    aux_container = (
        f"{CONTAINER_AUXILIARY_PARENT}/"
        "openai_gpt-oss-120b-streamed-cache/gpt120-f0b-aux"
    )
    return {
        "schema": SCHEMA,
        "model_id": MODEL_ID,
        "identity": {
            "checkpoint_revision": CHECKPOINT,
            "compiler_source_commit": COMPILER_COMMIT,
            "quetzal_source_commit": QUETZAL_COMMIT,
        },
        "integration": {
            "ttis_source_commit": TTIS_REQUIRED_ANCESTOR,
            "shield_source_commit": SHIELD_REQUIRED_ANCESTOR,
            "ttis_required_ancestor": TTIS_REQUIRED_ANCESTOR,
            "ttis_required_ancestor_verified": True,
            "shield_required_ancestor": SHIELD_REQUIRED_ANCESTOR,
            "shield_required_ancestor_verified": True,
            "runner_label": RUNNER_LABEL,
            "per_implementation_image_selection": True,
            "additional_args_forwarded_losslessly": True,
        },
        "publication": {
            "package_id": package_id,
            "bundle_manifest_sha256": "b" * 64,
            "immutable_generation_id": generation,
            "attestation_path": "/mnt/models/quetzal/immutable/v1/attestations/gpt120-f0b.json",
            "attestation_sha256": "c" * 64,
            "administrator_owned": True,
            "read_only": True,
            "runtime_principal_can_mutate": False,
            "no_writable_aliases": True,
            "revoked": False,
            "generated_model_tree": {
                "host_root": core_host,
                "container_root": core_container,
                "tree_sha256": "d" * 64,
                "immutable_generation_id": generation,
                "administrator_owned": True,
                "read_only": True,
            },
            "streamed_cache": {
                "name": "openai_gpt-oss-120b-streamed-cache",
                "host_root": aux_host,
                "container_root": aux_container,
                "tree_sha256": AUXILIARY_TREE_SHA256,
                "immutable_generation_id": generation,
                "administrator_owned": True,
                "read_only": True,
            },
            "full_streaming_verification": {
                "status": "pass",
                "receipt_sha256": "e" * 64,
                "package_id": package_id,
                "auxiliary_tree_sha256": AUXILIARY_TREE_SHA256,
            },
        },
        "runtime": {
            "image": "ghcr.io/tenstorrent/ttis-quetzal@sha256:" + "f" * 64,
            "quetzal_source_commit": QUETZAL_COMMIT,
            "ttis_source_commit": TTIS_REQUIRED_ANCESTOR,
            "tt_metal_commit": TT_METAL_COMMIT,
            "tt_metal_patchset_sha256": TT_METAL_PATCHSET,
            "server_boundary": "official_ttis",
            "platform_provider": "vllm-tt-plugin",
            "serving_backend": "generated_quetzal",
            "provider_policy": "generated_quetzal_only",
            "native_fallback_allowed": False,
            "plugin_entrypoint": "quetzal_model_registry",
            "vllm_plugins": "quetzal_model_registry,tt",
            "tt_vllm_builtin_models": "0",
            "descriptor_container_path": DESCRIPTOR_CONTAINER_PATH,
            "descriptor_sha256": DESCRIPTOR_SHA256,
        },
        "topology": {
            "runner_label": RUNNER_LABEL,
            "slurm_backed": True,
            "chip_count": 4,
            "mesh_shape": [2, 2],
            "logical_degree_histogram": {"2": 4},
            "physical_degree_histogram": {"2": 4},
            "descriptor_sha256": DESCRIPTOR_SHA256,
            "collective_topology_selected_not_measured": "Ring",
            "collective_links_selected_not_measured": 2,
            "preweight_fresh_admission_required": True,
        },
        "artifacts": {
            "batch_size": 1,
            "max_context": 8192,
            "prefill_buckets": [128, 1024],
            **EXPECTED_ARTIFACTS,
            "relative_paths": EXPECTED_RELATIVE_PATHS,
        },
    }


def test_enrollment_and_runtime_topology_gate_bind_the_same_f0b_bytes():
    """A planner/runtime mismatch creates a contract no image can satisfy."""
    assert quetzal_topology_admission._DESCRIPTOR_SHA256 == DESCRIPTOR_SHA256
    assert (
        quetzal_topology_admission._EMIT_SHA256
        == (EXPECTED_ARTIFACTS["candidate_pair_emit_sha256"])
    )
    assert quetzal_topology_admission._SELECTION_SHA256 == TOPOLOGY_SELECTION_SHA256


def test_exact_response_renders_catalogue_ci_and_ring2_contract():
    response = publication_response()
    rendered = render_contract(response)

    assert rendered["status"] == "validated_patch_contract_not_applied"
    spec = rendered["ttis_dev_catalogue_fragment"]
    device = spec["device_model_specs"][0]
    assert spec["impl"] == "quetzal"
    assert QUETZAL_COMMIT == "76a15d4cdd0c2b400ef9b89499a334a6b748e56b"
    assert device["max_context"] == 8192
    assert device["max_concurrency"] == 1
    assert device["default_impl"] is False
    assert device["vllm_args"]["enable-auto-tool-choice"] is True
    assert device["vllm_args"]["tool-call-parser"] == "openai"
    assert device["vllm_args"]["reasoning-parser"] == "openai_gptoss"
    env = device["env_vars"]
    assert env["QUETZAL_REQUIRED_SOURCE_REVISION"] == QUETZAL_COMMIT
    assert env["QUETZAL_GENERATOR_SOURCE_REVISION"] == COMPILER_COMMIT
    assert len(env["QUETZAL_RUNTIME_ATTESTATION_SHA256"]) == 64
    assert env["QUETZAL_SERVE_PROFILE"] == "gpt_oss_120b.serve"
    assert env["TT_VLLM_BUILTIN_MODELS"] == "0"
    assert env["VLLM_PLUGINS"] == "quetzal_model_registry,tt"
    assert env["TTQ_ROW_ALL_REDUCE_TOPOLOGY"] == "Ring"
    assert env["TTQ_TUNED_ROW_ALL_REDUCE_LINKS"] == "2"
    package_root = f"{CONTAINER_PACKAGE_PARENT}/{response['publication']['package_id']}"
    assert env["QUETZAL_PACKAGE_ROOT"] == package_root
    assert env["QZ_MODELS_ROOT"] == package_root
    for name in (
        "QZ_QUALIFICATION_MANIFEST",
        "QUETZAL_PREFILL_GENERATED_PY",
        "QUETZAL_PREFILL_METADATA_JSON",
        "QUETZAL_DECODE_GENERATED_PY",
        "QUETZAL_DECODE_METADATA_JSON",
        "QUETZAL_WEIGHTS",
    ):
        assert env[name].startswith(package_root + "/")
    assert env["QUETZAL_DECODE_GENERATED_PY"].endswith(
        "/compiled/openai_gpt-oss-120b-s1024/full/decode/generated.py"
    )
    assert env["QUETZAL_WEIGHTS"].endswith(
        "/compiled_weights/openai_gpt-oss-120b-s1024/full/weights.pt"
    )
    assert env["QUETZAL_REQUIRED_AUXILIARY_NAMES"] == (
        "openai_gpt-oss-120b-streamed-cache"
    )
    assert env["QUETZAL_REQUIRED_PREFILL_BUCKETS"] == "128,1024"
    assert env["QUETZAL_AUXILIARY_ROOTS_JSON"] == (
        '{"openai_gpt-oss-120b-streamed-cache":'
        '"/home/container_app_user/quetzal/auxiliary/'
        'openai_gpt-oss-120b-streamed-cache/gpt120-f0b-aux"}'
    )
    assert env["TT_MESH_GRAPH_DESC_PATH"] == DESCRIPTOR_CONTAINER_PATH

    ci = rendered["models_ci_implementation_fragment"]
    assert ci["impl"] == "quetzal"
    assert ci["image"].startswith("ghcr.io/tenstorrent/ttis-quetzal@sha256:")
    assert set(ci["ci"]) == {"nightly", "release"}
    for schedule in ci["ci"].values():
        assert schedule["devices"] == ["P300X2"]
        args = schedule["device-args"]["P300X2"]["additional-args"]
        assert "--quetzal-models-root /mnt/models/quetzal/immutable/" in args
        assert "--quetzal-runtime-attestation /mnt/models/quetzal/immutable/" in args
        assert "--quetzal-auxiliary-root openai_gpt-oss-120b-streamed-cache=" in args

    shield = rendered["shield_required_contract"]
    assert shield["runner_label"] == RUNNER_LABEL
    assert shield["image_must_be_selected_per_model_implementation"] is True
    assert shield["forbid_shared_generic_quetzal_image"] is True
    assert shield["required_quetzal_source_commit"] == QUETZAL_COMMIT
    assert (
        shield["expected_topology"]["qualified_selection_sha256"]
        == TOPOLOGY_SELECTION_SHA256
    )
    assert shield["source_commit"] == SHIELD_REQUIRED_ANCESTOR
    assert shield["required_ancestor"] == SHIELD_REQUIRED_ANCESTOR
    assert rendered["exact_identity"]["ttis_source_commit"] == TTIS_REQUIRED_ANCESTOR


def test_gpt_pending_row_defines_all_ci_entries_without_weakening_release():
    rendered = render_contract(publication_response())
    frontier = rendered["qualification_frontier"]

    assert frontier["catalogue_activation"] == (
        "disabled_until_validated_publication_response"
    )
    assert frontier["initial_on_dispatch"]["status_after_publication"] == (
        "dispatchable_not_certified"
    )
    assert frontier["initial_on_dispatch"]["profile"] == LOCAL_ON_DISPATCH_PROFILE
    entries = frontier["nightly_and_release_entries"]
    assert entries["rendered"] is True
    assert entries["activate_nightly"] is False
    assert entries["activate_release"] is False
    assert "workflow=release" in entries["reason"]
    assert frontier["defined_evals"] == list(GPT_RELEASE_EVALS)

    agentic = frontier["agentic_release_context"]
    assert agentic["max_input_tokens"] == GPT_SWEBENCH_INPUT_TOKENS
    assert agentic["max_output_tokens"] == GPT_SWEBENCH_OUTPUT_TOKENS
    assert agentic["required_context"] == GPT_SWEBENCH_MIN_CONTEXT == 8192
    assert agentic["available_context"] == 8192
    assert agentic["status"] == "admitted_bounded_collection_report_only"

    performance = frontier["performance"]
    assert performance["local_ttft_ms_range"] == list(LOCAL_TTFT_MS_RANGE)
    assert performance["ttft_target_ms"] == TTFT_TARGET_MS
    assert performance["status"] == "miss"
    assert "not full release" in frontier["claim_boundary"]


def test_reviewed_descendant_sources_do_not_self_block_enrollment():
    response = publication_response()
    response["integration"]["ttis_source_commit"] = "1" * 40
    response["runtime"]["ttis_source_commit"] = "1" * 40
    response["integration"]["shield_source_commit"] = "2" * 40

    rendered = render_contract(response)

    assert rendered["exact_identity"]["ttis_source_commit"] == "1" * 40
    assert rendered["exact_identity"]["shield_source_commit"] == "2" * 40
    assert rendered["exact_identity"]["ttis_required_ancestor"] == (
        TTIS_REQUIRED_ANCESTOR
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("runtime.quetzal_source_commit", "8a3bebe4afdd58068d4190248c3f7b82cc27ae9f"),
        ("runtime.tt_metal_commit", "0" * 40),
        ("runtime.tt_metal_patchset_sha256", "0" * 64),
        ("runtime.server_boundary", "standalone_quetzal"),
        ("runtime.platform_provider", "native_model"),
        ("runtime.native_fallback_allowed", True),
        ("runtime.vllm_plugins", "quetzal_model_registry"),
        ("runtime.tt_vllm_builtin_models", "1"),
        ("runtime.image", "ghcr.io/tenstorrent/ttis-quetzal:latest"),
        ("integration.ttis_source_commit", "not-a-commit"),
        ("integration.shield_source_commit", "not-a-commit"),
        ("integration.ttis_required_ancestor", "0" * 40),
        ("integration.ttis_required_ancestor_verified", False),
        ("integration.shield_required_ancestor", "0" * 40),
        ("integration.shield_required_ancestor_verified", False),
        ("runtime.ttis_source_commit", "0" * 40),
        ("integration.runner_label", "bh-qb-ae"),
        ("integration.per_implementation_image_selection", False),
        ("integration.additional_args_forwarded_losslessly", False),
        ("topology.runner_label", "bh-qb-ae"),
        ("topology.mesh_shape", [1, 4]),
        ("topology.collective_topology_selected_not_measured", "Linear"),
        ("topology.collective_links_selected_not_measured", 1),
        ("publication.streamed_cache.tree_sha256", "0" * 64),
        (
            "publication.generated_model_tree.container_root",
            "/home/container_app_user/cache_root/quetzal/immutable/v1/gpt120-f0b-core",
        ),
        (
            "publication.streamed_cache.container_root",
            "/home/container_app_user/cache_root/quetzal/immutable/v1/gpt120-f0b-aux",
        ),
        ("publication.read_only", False),
        ("publication.runtime_principal_can_mutate", True),
        ("publication.full_streaming_verification.status", "not_run"),
        ("artifacts.max_context", 1024),
        ("artifacts.decode_generated_sha256", "0" * 64),
        (
            "artifacts.relative_paths.prefill_s1024_generated",
            "compiled/gpt120/full/prefill/generated.py",
        ),
    ],
)
def test_identity_topology_and_publication_mismatches_fail_closed(path, value):
    response = copy.deepcopy(publication_response())
    target = response
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value

    with pytest.raises(ContractError):
        render_contract(response)


def test_missing_role_and_unsafe_path_fail_closed():
    missing = publication_response()
    del missing["publication"]["streamed_cache"]
    with pytest.raises(ContractError, match="streamed_cache"):
        render_contract(missing)

    unsafe = publication_response()
    unsafe["artifacts"]["relative_paths"]["weights"] = "../weights.pt"
    with pytest.raises(ContractError, match="unsafe relative path"):
        render_contract(unsafe)


def test_adversarial_oci_reference_fails_closed_without_regex_backtracking():
    response = publication_response()
    response["runtime"]["image"] = "!/" * 10_000 + "!:@sha256:" + "f" * 64
    with pytest.raises(ContractError, match="runtime.image"):
        render_contract(response)


def test_placeholders_are_never_rendered():
    response = publication_response()
    response["publication"]["package_id"] = "${TTQ_GPT120_F0B_PACKAGE_ID}"
    with pytest.raises(ContractError, match="placeholder"):
        render_contract(response)


@pytest.mark.parametrize(
    ("field", "stale_value"),
    [
        ("package_id", STALE_5CAB_PACKAGE_ID),
        ("bundle_manifest_sha256", STALE_5CAB_BUNDLE_MANIFEST_SHA256),
    ],
)
def test_stale_5cab_v2_package_identity_is_never_rendered(field, stale_value):
    response = publication_response()
    response["publication"][field] = stale_value

    with pytest.raises(ContractError, match="stale 5cab"):
        render_contract(response)
