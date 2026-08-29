# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy

import pytest

from scripts.release.plan_gpt120_f0b_quetzal_enrollment import (
    AUXILIARY_TREE_SHA256,
    CHECKPOINT,
    COMPILER_COMMIT,
    DESCRIPTOR_SHA256,
    EXPECTED_ARTIFACTS,
    MODEL_ID,
    QUETZAL_COMMIT,
    RUNNER_LABEL,
    SCHEMA,
    TT_METAL_COMMIT,
    TT_METAL_PATCHSET,
    ContractError,
    render_contract,
)


def publication_response():
    generation = "tt-ci-models-private-generation-314159"
    core_host = "/mnt/models/quetzal/immutable/v1/gpt120-f0b-core"
    core_container = (
        "/home/container_app_user/cache_root/quetzal/immutable/v1/gpt120-f0b-core"
    )
    aux_host = "/mnt/models/quetzal/immutable/v1/gpt120-f0b-aux"
    aux_container = (
        "/home/container_app_user/cache_root/quetzal/immutable/v1/gpt120-f0b-aux"
    )
    package_id = "sha256-v2-" + "a" * 64
    return {
        "schema": SCHEMA,
        "model_id": MODEL_ID,
        "identity": {
            "checkpoint_revision": CHECKPOINT,
            "compiler_source_commit": COMPILER_COMMIT,
            "quetzal_source_commit": QUETZAL_COMMIT,
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
            "tt_metal_commit": TT_METAL_COMMIT,
            "tt_metal_patchset_sha256": TT_METAL_PATCHSET,
            "serving_backend": "generated_quetzal",
            "provider_policy": "generated_quetzal_only",
            "native_fallback_allowed": False,
            "plugin_entrypoint": "quetzal_model_registry",
            "descriptor_container_path": "/home/container_app_user/cache_root/quetzal/immutable/v1/descriptors/p150x4.textproto",
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
            "relative_paths": {
                "qualification_manifest": "qualification_manifest.yaml",
                "prefill_s1024_generated": "compiled/gpt120/full/prefill/generated.py",
                "prefill_s1024_metadata": "compiled/gpt120/full/prefill/metadata.json",
                "decode_generated": "compiled/gpt120/full/decode/generated.py",
                "decode_metadata": "compiled/gpt120/full/decode/metadata.json",
                "weights": "compiled_weights/gpt120/full/weights.pt",
            },
        },
    }


def test_exact_response_renders_catalogue_ci_and_ring2_contract():
    response = publication_response()
    rendered = render_contract(response)

    assert rendered["status"] == "validated_patch_contract_not_applied"
    spec = rendered["ttis_dev_catalogue_fragment"]
    device = spec["device_model_specs"][0]
    assert spec["impl"] == "quetzal"
    assert device["max_context"] == 8192
    assert device["max_concurrency"] == 1
    assert device["default_impl"] is False
    env = device["env_vars"]
    assert env["TT_VLLM_BUILTIN_MODELS"] == "0"
    assert env["TTQ_ROW_ALL_REDUCE_TOPOLOGY"] == "Ring"
    assert env["TTQ_TUNED_ROW_ALL_REDUCE_LINKS"] == "2"
    assert env["QUETZAL_DECODE_GENERATED_PY"].endswith(
        "/compiled/gpt120/full/decode/generated.py"
    )

    ci = rendered["models_ci_implementation_fragment"]
    assert ci["impl"] == "quetzal"
    assert ci["image"].startswith("ghcr.io/tenstorrent/ttis-quetzal@sha256:")
    assert set(ci["ci"]) == {"nightly", "release"}
    for schedule in ci["ci"].values():
        assert schedule["devices"] == ["P300X2"]
        args = schedule["device-args"]["P300X2"]["additional-args"]
        assert "--quetzal-models-root /mnt/models/quetzal/immutable/" in args
        assert "--quetzal-auxiliary-root openai_gpt-oss-120b-streamed-cache=" in args

    shield = rendered["shield_required_contract"]
    assert shield["runner_label"] == RUNNER_LABEL
    assert shield["image_must_be_selected_per_model_implementation"] is True
    assert shield["forbid_shared_generic_quetzal_image"] is True
    assert shield["required_quetzal_source_commit"] == QUETZAL_COMMIT


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("runtime.quetzal_source_commit", "8a3bebe4afdd58068d4190248c3f7b82cc27ae9f"),
        ("runtime.tt_metal_commit", "0" * 40),
        ("runtime.tt_metal_patchset_sha256", "0" * 64),
        ("runtime.native_fallback_allowed", True),
        ("runtime.image", "ghcr.io/tenstorrent/ttis-quetzal:latest"),
        ("topology.runner_label", "bh-qb-ge"),
        ("topology.mesh_shape", [1, 4]),
        ("topology.collective_topology_selected_not_measured", "Linear"),
        ("topology.collective_links_selected_not_measured", 1),
        ("publication.streamed_cache.tree_sha256", "0" * 64),
        ("publication.read_only", False),
        ("publication.runtime_principal_can_mutate", True),
        ("publication.full_streaming_verification.status", "not_run"),
        ("artifacts.max_context", 1024),
        ("artifacts.decode_generated_sha256", "0" * 64),
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


def test_placeholders_are_never_rendered():
    response = publication_response()
    response["publication"]["package_id"] = "${TTQ_GPT120_F0B_PACKAGE_ID}"
    with pytest.raises(ContractError, match="placeholder"):
        render_contract(response)
