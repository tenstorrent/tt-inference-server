# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#!/usr/bin/env python3
"""Render a Gemma Quetzal Models CI enrollment only from approved evidence.

This is intentionally not wired into a workflow.  It reads a small control-plane
publication response, validates every dispatch-critical identity, and writes
reviewable catalogue/config/handoff fragments to a new directory.  It never
reads a model payload or edits the active catalogue.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

import jsonschema
import yaml

from scripts.validate_models_ci_config import validate_implementation_identities


SCHEMA = "quetzal.gemma-models-ci-enrollment-evidence.v1"
MODEL = "google/gemma-4-31B-it"
MODEL_KEY = "gemma-4-31B-it"
HF_REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
QUETZAL_SOURCE = "bb02e1975437ee210578fd008721a7acff3f2dba"
TT_METAL = "b534549300fe2af11e6ee828675294bc0e359555"
PATCHSET = "22fb0bd2523b8a5c63fa20c3c8a1586dc9ead5150449d0eb02231fa8173a7edd"
INIT_SHA256 = "b0073851fe9142c62d1ff488c40b8e5a9307040d4e6e93d1ebcb365440a5a218"
RUNNER = "qb2-p300x2-physical-2x2-ring-links2"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
GIT_REVISION = re.compile(r"^[0-9a-f]{40}$")
PACKAGE = re.compile(r"^sha256(?:-v[0-9]+)?(?:-[0-9a-f]{64}){2,3}$")


class EnrollmentError(ValueError):
    pass


def _need(value: Any, expected: Any, field: str) -> None:
    if value != expected:
        raise EnrollmentError(f"{field} must be {expected!r}, got {value!r}")


def _git_revision(value: Any, field: str) -> str:
    if not isinstance(value, str) or not GIT_REVISION.fullmatch(value):
        raise EnrollmentError(f"{field} must be a full lowercase git revision")
    return value


def _current_repository_revision(repo_root: Path, field: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--verify", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise EnrollmentError(f"cannot resolve the exact {field} checkout revision") from error
    return _git_revision(result.stdout.strip(), f"current {field} checkout revision")


def _absolute(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise EnrollmentError(f"{field} must be a non-empty absolute path")
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts:
        raise EnrollmentError(f"{field} must be a contained absolute path")
    return value.rstrip("/")


def _relative(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise EnrollmentError(f"{field} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise EnrollmentError(f"{field} must be a contained relative path")
    return value


def validate_evidence(
    data: dict[str, Any], *, ttis_revision: str, shield_revision: str
) -> dict[str, Any]:
    ttis_revision = _git_revision(ttis_revision, "expected TTIS revision")
    shield_revision = _git_revision(shield_revision, "expected Shield revision")
    _need(data.get("schema_version"), SCHEMA, "schema_version")
    _need(data.get("decision"), "approved", "decision")
    _need(data.get("administrator_owned"), True, "administrator_owned")
    _need(data.get("read_only"), True, "read_only")
    _need(data.get("no_writable_aliases"), True, "no_writable_aliases")
    _need(data.get("revocation_status"), "active", "revocation_status")

    identity = data.get("identity", {})
    _need(identity.get("model_id"), MODEL, "identity.model_id")
    _need(identity.get("hf_revision"), HF_REVISION, "identity.hf_revision")
    _need(identity.get("quetzal_source_revision"), QUETZAL_SOURCE, "identity.quetzal_source_revision")
    _need(identity.get("ttis_revision"), ttis_revision, "identity.ttis_revision")
    _need(identity.get("shield_revision"), shield_revision, "identity.shield_revision")
    _need(identity.get("tt_metal_revision"), TT_METAL, "identity.tt_metal_revision")
    _need(identity.get("tt_metal_patchset_sha256"), PATCHSET, "identity.tt_metal_patchset_sha256")
    _need(identity.get("patchset_applied_manifest_matches"), True, "identity.patchset_applied_manifest_matches")
    _need(identity.get("initialization_milestones_sha256"), INIT_SHA256, "identity.initialization_milestones_sha256")

    package_id = data.get("package_id")
    if not isinstance(package_id, str) or not PACKAGE.fullmatch(package_id):
        raise EnrollmentError("package_id must be a complete content-addressed package id")
    manifest_sha = data.get("package_manifest_sha256")
    if not isinstance(manifest_sha, str) or not SHA256.fullmatch(manifest_sha):
        raise EnrollmentError("package_manifest_sha256 must be an exact lowercase SHA-256")
    host_root = _absolute(data.get("host_package_root"), "host_package_root")
    container_root = _absolute(data.get("container_package_root"), "container_package_root")
    if PurePosixPath(host_root).name != package_id or PurePosixPath(container_root).name != package_id:
        raise EnrollmentError("host/container package roots must end in the exact package_id")

    profile = data.get("profile", {})
    expected_profile = {
        "batch_size": 1,
        "concurrency": 1,
        "prefill_capacity": 1024,
        "decode_capacity": 2048,
        "precision": "BFP8",
    }
    _need(profile, expected_profile, "profile")
    topology = data.get("topology", {})
    _need(topology.get("chip_count"), 4, "topology.chip_count")
    _need(topology.get("mesh_shape"), [2, 2], "topology.mesh_shape")
    _need(topology.get("collective"), "Ring", "topology.collective")
    _need(topology.get("links"), 2, "topology.links")
    _need(topology.get("runner_label"), RUNNER, "topology.runner_label")

    roles = data.get("roles", {})
    if set(roles) != {"compiled_weights", "generated_prefill", "generated_decode", "qualification_manifest"}:
        raise EnrollmentError("roles must bind exactly compiled_weights/generated_prefill/generated_decode/qualification_manifest")
    role_paths = {name: _relative(value, f"roles.{name}") for name, value in roles.items()}

    qualification = data.get("qualification", {})
    pcc = qualification.get("pcc")
    if not isinstance(pcc, (int, float)) or isinstance(pcc, bool) or pcc < 0.99:
        raise EnrollmentError("qualification.pcc must be a fresh exact-package result >= 0.99")
    _need(qualification.get("fresh"), True, "qualification.fresh")
    _need(qualification.get("exact_package_identity"), package_id, "qualification.exact_package_identity")
    _need(qualification.get("endpoint_isl"), 1024, "qualification.endpoint_isl")
    _need(qualification.get("endpoint_osl"), 512, "qualification.endpoint_osl")
    _need(qualification.get("http_200"), True, "qualification.http_200")
    _need(qualification.get("clean_unload"), True, "qualification.clean_unload")
    _need(qualification.get("zero_device_holders_after"), True, "qualification.zero_device_holders_after")
    _need(
        qualification.get("initialization_terminal"),
        {"event": "engine_ready", "state": "complete"},
        "qualification.initialization_terminal",
    )

    return {
        "package_id": package_id,
        "manifest_sha": manifest_sha,
        "host_root": host_root,
        "container_root": container_root,
        "roles": role_paths,
        "pcc": float(pcc),
        "ttis_revision": ttis_revision,
        "shield_revision": shield_revision,
    }


def render_fragments(
    data: dict[str, Any], repo_root: Path, *, shield_repo_root: Path
) -> dict[str, Any]:
    exact = validate_evidence(
        data,
        ttis_revision=_current_repository_revision(repo_root, "TTIS"),
        shield_revision=_current_repository_revision(shield_repo_root, "Shield"),
    )
    root = exact["container_root"]
    roles = exact["roles"]
    env = {
        "ARCH_NAME": "blackhole",
        "MESH_DEVICE": "P150x4",
        "QUETZAL_VLLM": "1",
        "QUETZAL_MODEL": MODEL,
        "QUETZAL_HF_REVISION": HF_REVISION,
        "QUETZAL_PACKAGE_ID": exact["package_id"],
        "QUETZAL_BUNDLE_MANIFEST_SHA256": exact["manifest_sha"],
        "QUETZAL_REQUIRED_TT_METAL_PATCHSET_SHA256": PATCHSET,
        "QUETZAL_PACKAGE_ROOT": root,
        "QZ_MODELS_ROOT": root,
        "QZ_QUALIFICATION_MANIFEST": f"{root}/{roles['qualification_manifest']}",
        "QUETZAL_PREFILL_GENERATED_PY": f"{root}/{roles['generated_prefill']}",
        "QUETZAL_DECODE_GENERATED_PY": f"{root}/{roles['generated_decode']}",
        "QUETZAL_WEIGHTS": f"{root}/{roles['compiled_weights']}",
        "QZ_MMAP_WEIGHTS": "1",
        "TTQ_STREAM_WEIGHTS": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_PLUGINS": "quetzal_model_registry,tt",
        "TT_VLLM_BUILTIN_MODELS": "0",
        "TTQ_ROW_ALL_REDUCE_TOPOLOGY": "Ring",
        "TTQ_TUNED_ROW_ALL_REDUCE_LINKS": "2",
        "TTQ_TUNED_ROW_ALL_REDUCE": "1",
    }
    catalogue = {
        "templates": [{
            "weights": [MODEL], "impl": "quetzal", "inference_engine": "VLLM",
            "model_type": "LLM", "supported_modalities": ["text"],
            "device_model_specs": [{
                "device": "P300X2", "max_concurrency": 1, "max_context": 2048,
                "default_impl": False, "env_vars": env,
                "vllm_args": {"block_size": 64, "max_model_len": 2048, "max_num_seqs": 1,
                              "revision": HF_REVISION, "tokenizer_revision": HF_REVISION},
                "override_tt_config": {"fabric_config": "FABRIC_1D", "l1_small_size": 16384,
                                       "trace_region_size": 90000000},
            }],
            "status": "EXPERIMENTAL", "has_builtin_warmup": False,
            "metadata": {MODEL: {"reasoning_parser_name": "gemma4", "tool_call_parser_name": "gemma4"}},
        }]
    }
    args = f"--quetzal-models-root {exact['host_root']}"
    implementation = {
        "inference_engine": "vLLM", "impl": "quetzal",
        "ci": {schedule: {"devices": ["P300X2"], "device-args": {"P300X2": {"additional-args": args}}}
               for schedule in ("nightly", "release")},
    }
    config_path = repo_root / ".github/workflows/models-ci-config.json"
    schema_path = repo_root / ".github/workflows/models-ci-config-schema.json"
    config = json.loads(config_path.read_text())
    rows = config["models"][MODEL_KEY]["implementations"]
    if any(row.get("impl") == "quetzal" for row in rows):
        raise EnrollmentError("active config already has a Gemma Quetzal row; refuse duplicate promotion")
    candidate = json.loads(json.dumps(config))
    candidate["models"][MODEL_KEY]["implementations"].append(implementation)
    jsonschema.validate(candidate, json.loads(schema_path.read_text()))
    errors = validate_implementation_identities(candidate)
    if errors:
        raise EnrollmentError("; ".join(errors))
    handoff = {
        "schema_version": "quetzal.shield-enrollment-handoff.v1",
        "model": MODEL_KEY,
        "impl": "quetzal",
        "device": "P300X2",
        "runner_label": RUNNER,
        "quetzal_source_revision": QUETZAL_SOURCE,
        "ttis_revision": exact["ttis_revision"],
        "shield_revision": exact["shield_revision"],
        "tt_metal_revision": TT_METAL,
        "tt_metal_patchset_sha256": PATCHSET,
        "package_id": exact["package_id"],
        "host_package_root": exact["host_root"],
        "package_manifest_sha256": exact["manifest_sha"],
        "schedules": ["nightly", "release"],
        "generated_only": True,
        "fallback_allowed": False,
    }
    return {"catalogue": catalogue, "implementation": implementation, "candidate_config": candidate, "handoff": handoff}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--shield-repo-root",
        required=True,
        type=Path,
        help="Exact tt-shield checkout that will consume the handoff",
    )
    args = parser.parse_args()
    if args.out_dir.exists():
        raise EnrollmentError("out-dir must not already exist")
    data = json.loads(args.evidence.read_text())
    rendered = render_fragments(
        data,
        args.repo_root,
        shield_repo_root=args.shield_repo_root,
    )
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "gemma-dev-catalogue-entry.yaml").write_text(yaml.safe_dump(rendered["catalogue"], sort_keys=False))
    for name, key in (("gemma-models-ci-implementation.json", "implementation"),
                      ("models-ci-config.candidate.json", "candidate_config"),
                      ("shield-handoff.json", "handoff")):
        (args.out_dir / name).write_text(json.dumps(rendered[key], indent=2) + "\n")
    print(f"rendered review-only enrollment artifacts in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
