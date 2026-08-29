#!/usr/bin/env python3
"""Validate and render the exact GPT-OSS-120B f0b Models CI contract.

This tool deliberately does not edit the development catalogue or Models CI
configuration.  The historical f0b package is not published in an administered
immutable namespace yet.  Once storage and runner administrators return the
response described here, this tool turns it into reviewable, deterministic
fragments.  It is a patch planner, not a signature or storage-attestation
validator; the supplied response must itself come from the reviewed
administrative publication process.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "ttq.gpt120-f0b-publication-response/v1"
MODEL_ID = "openai/gpt-oss-120b"
CHECKPOINT = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
COMPILER_COMMIT = "3750c4872bcaf0c0c9404a4c99edcefb9e6d103d"
QUETZAL_COMMIT = "f0b58c692a0a04af7631b6e8753af021fb08b746"
TT_METAL_COMMIT = "b534549300fe2af11e6ee828675294bc0e359555"
TT_METAL_PATCHSET = "22fb0bd2523b8a5c63fa20c3c8a1586dc9ead5150449d0eb02231fa8173a7edd"
RUNNER_LABEL = "qb2-p300x2-physical-2x2-ring-links2"
DESCRIPTOR_SHA256 = "f4c9fb5acf307e1b320525007035ed9e75039f793e4350120365243682e37792"
AUXILIARY_NAME = "openai_gpt-oss-120b-streamed-cache"
AUXILIARY_TREE_SHA256 = (
    "2b2e528a75cae51a53db4a3e309f075553fe5f5f7fec7d2a29480f6572f2e416"
)

EXPECTED_ARTIFACTS = {
    "bucket_set_emit_sha256": "6fc8be3dd87a8e31d0d86af454f013acc39bf4d00accda637f739b0eec04a1fd",
    "prefill_s128_generated_sha256": "567e94efaabf5469f569cee45def92402d8ecb96dab61fadba73379dab7f31a7",
    "prefill_s1024_generated_sha256": "aa18fe6bef6756995307a4bce55d8e4278b7415dcd6d154024728297497e7922",
    "prefill_s1024_metadata_sha256": "d479d2588ee20e810b0efa565778912a32ed08f5ff417bc5fa7c91f3206d566a",
    "decode_generated_sha256": "19804ee667a47117e15be5ca1118c666d85c499063867020b0d8639001459b7e",
    "decode_metadata_sha256": "25cb98544cb38c571092f8a69c89c66ed7a6f7e6d4afd1c043139e8d6f5e117c",
    "candidate_pair_emit_sha256": "f296b7049ad6c9bfb3876f51c5cd1e717b19ebb0a667585907779ef45019370d",
    "codegen_fingerprint": "31ce9a154bebca1edf0ccb159024dc52645eb055fc05ae277f2ad897e5dcfd60",
    "weights_fingerprint": "f75cbe891ecfe72c29c395cee959e350cae0eecdfa02353f739c6ff281ccabb3",
    "weights_file_sha256": "03756cbcd27540b80576f839b32b23b9648fc2623d4130246a785902a84a4dd8",
}

REQUIRED_RELATIVE_PATHS = (
    "qualification_manifest",
    "prefill_s1024_generated",
    "prefill_s1024_metadata",
    "decode_generated",
    "decode_metadata",
    "weights",
)
HEX64 = re.compile(r"[0-9a-f]{64}")
OCI_DIGEST = re.compile(r"[^\s@:]+(?:/[^\s@:]+)+@sha256:[0-9a-f]{64}")


class ContractError(ValueError):
    pass


def _at(data: dict[str, Any], path: str) -> Any:
    value: Any = data
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ContractError(f"missing required field: {path}")
        value = value[part]
    return value


def _exact(data: dict[str, Any], path: str, expected: Any) -> Any:
    value = _at(data, path)
    if value != expected:
        raise ContractError(f"{path}: expected {expected!r}, got {value!r}")
    return value


def _non_placeholder(data: dict[str, Any], path: str) -> str:
    value = _at(data, path)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{path}: expected a non-empty string")
    lowered = value.lower()
    if (
        "${" in value
        or "<" in value
        or "placeholder" in lowered
        or "pending" in lowered
    ):
        raise ContractError(f"{path}: unresolved placeholder is not admissible")
    return value


def _sha256(data: dict[str, Any], path: str) -> str:
    value = _non_placeholder(data, path)
    if not HEX64.fullmatch(value):
        raise ContractError(f"{path}: expected 64 lowercase hexadecimal characters")
    return value


def _immutable_root(data: dict[str, Any], path: str) -> str:
    value = _non_placeholder(data, path)
    pure = PurePosixPath(value)
    if not pure.is_absolute() or ".." in pure.parts:
        raise ContractError(f"{path}: expected a contained absolute path")
    allowed = (
        "/mnt/models/quetzal/immutable/",
        "/home/container_app_user/cache_root/quetzal/immutable/",
    )
    if not any(value.startswith(prefix) for prefix in allowed):
        raise ContractError(
            f"{path}: path is outside an administered immutable namespace"
        )
    return value.rstrip("/")


def _relative_path(data: dict[str, Any], name: str) -> str:
    path = _non_placeholder(data, f"artifacts.relative_paths.{name}")
    pure = PurePosixPath(path)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ContractError(f"artifacts.relative_paths.{name}: unsafe relative path")
    return path


def validate_response(data: dict[str, Any]) -> None:
    _exact(data, "schema", SCHEMA)
    _exact(data, "model_id", MODEL_ID)
    _exact(data, "identity.checkpoint_revision", CHECKPOINT)
    _exact(data, "identity.compiler_source_commit", COMPILER_COMMIT)
    _exact(data, "identity.quetzal_source_commit", QUETZAL_COMMIT)

    _at(data, "publication")
    package_id = _non_placeholder(data, "publication.package_id")
    if not package_id.startswith("sha256-"):
        raise ContractError(
            "publication.package_id: expected a digest-addressed package ID"
        )
    _sha256(data, "publication.bundle_manifest_sha256")
    generation = _non_placeholder(data, "publication.immutable_generation_id")
    _sha256(data, "publication.attestation_sha256")
    _non_placeholder(data, "publication.attestation_path")
    _exact(data, "publication.administrator_owned", True)
    _exact(data, "publication.read_only", True)
    _exact(data, "publication.runtime_principal_can_mutate", False)
    _exact(data, "publication.no_writable_aliases", True)
    _exact(data, "publication.revoked", False)

    core = _at(data, "publication.generated_model_tree")
    auxiliary = _at(data, "publication.streamed_cache")
    for role, entry in (("generated_model_tree", core), ("streamed_cache", auxiliary)):
        if not isinstance(entry, dict):
            raise ContractError(f"publication.{role}: expected an object")
        _immutable_root(data, f"publication.{role}.host_root")
        _immutable_root(data, f"publication.{role}.container_root")
        _sha256(data, f"publication.{role}.tree_sha256")
        _exact(data, f"publication.{role}.immutable_generation_id", generation)
        _exact(data, f"publication.{role}.administrator_owned", True)
        _exact(data, f"publication.{role}.read_only", True)
    _exact(data, "publication.streamed_cache.name", AUXILIARY_NAME)
    _exact(data, "publication.streamed_cache.tree_sha256", AUXILIARY_TREE_SHA256)
    _exact(data, "publication.full_streaming_verification.status", "pass")
    _sha256(data, "publication.full_streaming_verification.receipt_sha256")
    _exact(data, "publication.full_streaming_verification.package_id", package_id)
    _exact(
        data,
        "publication.full_streaming_verification.auxiliary_tree_sha256",
        AUXILIARY_TREE_SHA256,
    )

    image = _non_placeholder(data, "runtime.image")
    if not OCI_DIGEST.fullmatch(image):
        raise ContractError(
            "runtime.image: expected registry/path@sha256:<64 lowercase hex>"
        )
    _exact(data, "runtime.quetzal_source_commit", QUETZAL_COMMIT)
    _exact(data, "runtime.tt_metal_commit", TT_METAL_COMMIT)
    _exact(data, "runtime.tt_metal_patchset_sha256", TT_METAL_PATCHSET)
    _exact(data, "runtime.serving_backend", "generated_quetzal")
    _exact(data, "runtime.provider_policy", "generated_quetzal_only")
    _exact(data, "runtime.native_fallback_allowed", False)
    _exact(data, "runtime.plugin_entrypoint", "quetzal_model_registry")
    _immutable_root(data, "runtime.descriptor_container_path")
    _exact(data, "runtime.descriptor_sha256", DESCRIPTOR_SHA256)

    _exact(data, "topology.runner_label", RUNNER_LABEL)
    _exact(data, "topology.slurm_backed", True)
    _exact(data, "topology.chip_count", 4)
    _exact(data, "topology.mesh_shape", [2, 2])
    _exact(data, "topology.logical_degree_histogram", {"2": 4})
    _exact(data, "topology.physical_degree_histogram", {"2": 4})
    _exact(data, "topology.descriptor_sha256", DESCRIPTOR_SHA256)
    _exact(data, "topology.collective_topology_selected_not_measured", "Ring")
    _exact(data, "topology.collective_links_selected_not_measured", 2)
    _exact(data, "topology.preweight_fresh_admission_required", True)

    _exact(data, "artifacts.batch_size", 1)
    _exact(data, "artifacts.max_context", 8192)
    _exact(data, "artifacts.prefill_buckets", [128, 1024])
    for key, expected in EXPECTED_ARTIFACTS.items():
        _exact(data, f"artifacts.{key}", expected)
    for name in REQUIRED_RELATIVE_PATHS:
        _relative_path(data, name)


def render_contract(data: dict[str, Any]) -> dict[str, Any]:
    validate_response(data)
    publication = data["publication"]
    runtime = data["runtime"]
    artifacts = data["artifacts"]
    relative = artifacts["relative_paths"]
    package_id = publication["package_id"]
    host_root = publication["generated_model_tree"]["host_root"].rstrip("/")
    container_root = publication["generated_model_tree"]["container_root"].rstrip("/")
    aux_host = publication["streamed_cache"]["host_root"].rstrip("/")
    aux_container = publication["streamed_cache"]["container_root"].rstrip("/")

    env = {
        "ARCH_NAME": "blackhole",
        "MESH_DEVICE": "P150x4",
        "QUETZAL_VLLM": "1",
        "QUETZAL_MODEL": MODEL_ID,
        "QUETZAL_HF_REVISION": CHECKPOINT,
        "QUETZAL_PACKAGE_ID": package_id,
        "QUETZAL_BUNDLE_MANIFEST_SHA256": publication["bundle_manifest_sha256"],
        "QUETZAL_REQUIRED_TT_METAL_PATCHSET_SHA256": TT_METAL_PATCHSET,
        "QUETZAL_PACKAGE_ROOT": container_root,
        "QZ_MODELS_ROOT": container_root,
        "QZ_QUALIFICATION_MANIFEST": f"{container_root}/{relative['qualification_manifest']}",
        "QUETZAL_PREFILL_GENERATED_PY": f"{container_root}/{relative['prefill_s1024_generated']}",
        "QUETZAL_PREFILL_METADATA_JSON": f"{container_root}/{relative['prefill_s1024_metadata']}",
        "QUETZAL_DECODE_GENERATED_PY": f"{container_root}/{relative['decode_generated']}",
        "QUETZAL_DECODE_METADATA_JSON": f"{container_root}/{relative['decode_metadata']}",
        "QUETZAL_WEIGHTS": f"{container_root}/{relative['weights']}",
        "QUETZAL_AUXILIARY_ROOTS_JSON": json.dumps(
            {AUXILIARY_NAME: aux_container}, separators=(",", ":")
        ),
        "QZ_MMAP_WEIGHTS": "1",
        "TTQ_STREAM_WEIGHTS": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_PLUGINS": "quetzal_model_registry,tt",
        "TT_VLLM_BUILTIN_MODELS": "0",
        "TTQ_ROW_ALL_REDUCE_TOPOLOGY": "Ring",
        "TTQ_TUNED_ROW_ALL_REDUCE": "1",
        "TTQ_TUNED_ROW_ALL_REDUCE_LINKS": "2",
        "TT_MESH_GRAPH_DESC_PATH": runtime["descriptor_container_path"],
    }
    extra_args = (
        f"--quetzal-models-root {host_root} "
        f"--quetzal-auxiliary-root {AUXILIARY_NAME}={aux_host}"
    )
    schedule = {
        "devices": ["P300X2"],
        "device-args": {"P300X2": {"additional-args": extra_args}},
    }
    return {
        "schema": "ttis.gpt120-f0b-enrollment-patch-contract/v1",
        "status": "validated_patch_contract_not_applied",
        "claim_boundary": (
            "Fragments are review inputs only. The publication response still requires "
            "independent administrative authenticity review; no catalogue row, workflow, "
            "or dispatch is changed by this command."
        ),
        "exact_identity": {
            "model_id": MODEL_ID,
            "checkpoint_revision": CHECKPOINT,
            "quetzal_source_commit": QUETZAL_COMMIT,
            "compiler_source_commit": COMPILER_COMMIT,
            "tt_metal_commit": TT_METAL_COMMIT,
            "tt_metal_patchset_sha256": TT_METAL_PATCHSET,
            "image": runtime["image"],
            "package_id": package_id,
            "bundle_manifest_sha256": publication["bundle_manifest_sha256"],
            "immutable_generation_id": publication["immutable_generation_id"],
        },
        "ttis_dev_catalogue_fragment": {
            "weights": [MODEL_ID],
            "impl": "quetzal",
            "inference_engine": "VLLM",
            "model_type": "LLM",
            "supported_modalities": ["text"],
            "device_model_specs": [
                {
                    "device": "P300X2",
                    "max_concurrency": 1,
                    "max_context": 8192,
                    "default_impl": False,
                    "env_vars": env,
                    "vllm_args": {
                        "block_size": 64,
                        "max_model_len": 8192,
                        "max_num_seqs": 1,
                        "revision": CHECKPOINT,
                        "tokenizer_revision": CHECKPOINT,
                    },
                    "override_tt_config": {
                        "fabric_config": "FABRIC_1D",
                        "l1_small_size": 16384,
                        "trace_region_size": 90000000,
                    },
                },
            ],
            "status": "EXPERIMENTAL",
            "has_builtin_warmup": False,
            "metadata": {
                MODEL_ID: {
                    "reasoning_parser_name": "openai_gptoss",
                    "tool_call_parser_name": "openai",
                }
            },
        },
        "models_ci_implementation_fragment": {
            "inference_engine": "vLLM",
            "impl": "quetzal",
            "image": runtime["image"],
            "ci": {"nightly": schedule, "release": schedule},
        },
        "shield_required_contract": {
            "runner_label": RUNNER_LABEL,
            "device_type": "p300x2",
            "image": runtime["image"],
            "image_must_be_selected_per_model_implementation": True,
            "forbid_shared_generic_quetzal_image": True,
            "required_quetzal_source_commit": QUETZAL_COMMIT,
            "required_tt_metal_commit": TT_METAL_COMMIT,
            "required_tt_metal_patchset_sha256": TT_METAL_PATCHSET,
            "require_fresh_slurm_bound_preweight_topology_admission": True,
            "expected_topology": {
                "chip_count": 4,
                "mesh_shape": [2, 2],
                "logical_degree_histogram": {"2": 4},
                "physical_degree_histogram": {"2": 4},
                "descriptor_sha256": DESCRIPTOR_SHA256,
                "selected_collective_topology_not_measured": "Ring",
                "selected_collective_links_not_measured": 2,
            },
            "required_integration": (
                "Shield must preserve this per-model+impl image and prefer it over "
                "the workflow-wide Quetzal image before enabling f0b"
            ),
        },
        "enablement_order": [
            "review administrator attestation authenticity and lifetime",
            "install exact digest image and immutable core+aux roots on the Ring/2 runner class",
            "verify Shield per-model+impl immutable image selection is deployed",
            "apply and schema-validate the TTIS dev catalogue and Models CI fragments",
            "run one guarded on-dispatch qualification",
            "enable nightly only after on-dispatch passes",
            "enable release only after CS-owned acceptance policy is recorded",
        ],
        "forbidden_claims": [
            "portable P300X2",
            "native or TTNN fallback",
            "historical PCC transfers to the new publication/runtime identity",
            "Ring/2 is discovered physics rather than selected qualified-artifact configuration",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication-response", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        data = json.loads(args.publication_response.read_text())
        contract = render_contract(data)
    except (OSError, json.JSONDecodeError, ContractError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(contract, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
