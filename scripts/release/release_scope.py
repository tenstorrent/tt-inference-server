# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Exact release-scope evidence shared by release tooling."""

from __future__ import annotations

import io
import json
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from scripts.release.model_spec_resolver import LeafIdentity
from workflows.model_spec import (
    MODEL_SPEC_CATALOG_FILES,
    get_model_spec_map,
    load_templates_from_yaml,
    model_spec_leaf_identity,
)
from workflows.workflow_types import DeviceTypes, InferenceEngine


NOTE_COLUMNS = (
    "Impl ID",
    "HF Repository",
    "Device",
    "Engine",
    "Version",
    "TT-Metal Commit",
    "vLLM Commit",
    "Status Change",
    "CI Job Link",
)
UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ProdPin:
    version: str
    tt_metal_commit: str
    vllm_commit: str | None
    docker_image: str | None


@dataclass(frozen=True)
class ProdLeaf:
    identity: LeafIdentity
    pin: ProdPin
    status: str


def _required_string(value, field: str, identity: LeafIdentity) -> str:
    if value is None or not str(value).strip():
        raise ValueError(f"Prod leaf {identity!r} is missing {field}")
    return str(value)


def load_prod_leaves(prod_dir: Path):
    """Load and index exact prod leaves from the runtime catalog file set.

    One weight and one device per prod entry is a permanent invariant of the
    generated catalog, not an option: every release consumer indexes prod by
    exact leaf identity and a coarse entry has no single one.
    """
    prod_dir = Path(prod_dir)
    templates = []
    for filename in MODEL_SPEC_CATALOG_FILES:
        path = prod_dir / filename
        try:
            file_templates = load_templates_from_yaml(path, env="prod")
        except KeyError as exc:
            raise ValueError(
                f"Prod catalog {path} contains an invalid enum value: {exc}"
            ) from exc
        for template in file_templates:
            if len(template.weights) != 1 or len(template.device_model_specs) != 1:
                raise ValueError(
                    f"Prod template in {path} is not leaf-granular: "
                    f"weights={template.weights!r}, "
                    f"devices={[item.device.name for item in template.device_model_specs]!r}"
                )
        templates.extend(file_templates)

    model_specs = get_model_spec_map(templates)
    leaves = {}
    for spec in model_specs.values():
        identity = model_spec_leaf_identity(spec)
        version = _required_string(spec.version, "version", identity)
        tt_metal_commit = _required_string(
            spec.tt_metal_commit, "tt_metal_commit", identity
        )
        vllm_commit = spec.vllm_commit
        if identity[2] == InferenceEngine.VLLM.value:
            vllm_commit = _required_string(vllm_commit, "vllm_commit", identity)
        leaf = ProdLeaf(
            identity=identity,
            pin=ProdPin(
                version=version,
                tt_metal_commit=tt_metal_commit,
                vllm_commit=str(vllm_commit) if vllm_commit else None,
                docker_image=spec.docker_image,
            ),
            status=spec.status.name,
        )
        # get_model_spec_map() runs validate_model_specs(), which already
        # rejects a repeated leaf identity, so this index cannot collide.
        leaves[identity] = leaf
    return leaves


def load_prod_leaves_from_ref(ref: str):
    """Index the prod catalogue at a git ref by exact leaf identity.

    The base side of a release comparison is read through the same loader as the
    working tree, so both sides are validated identically and a change is only
    ever reported between two leaves that exist. This requires the ref to carry
    the leaf-granular catalogue; a coarse ref fails here rather than being
    widened on the fly into leaves the catalogue never actually pinned.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        prod_dir = Path(temp_dir) / "prod"
        prod_dir.mkdir()
        for filename in MODEL_SPEC_CATALOG_FILES:
            try:
                text = subprocess.run(
                    ["git", "show", f"{ref}:workflows/model_specs/prod/{filename}"],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout
            except subprocess.CalledProcessError as exc:
                raise ValueError(
                    f"Could not read prod catalog {filename!r} from ref {ref!r}"
                ) from exc
            (prod_dir / filename).write_text(text)
        return load_prod_leaves(prod_dir)


def identity_from_model_spec_document(document: dict) -> LeafIdentity:
    """Extract exact identity from a runtime-spec JSON document."""
    if not isinstance(document, dict):
        raise ValueError("Runtime model spec must be a JSON object")
    if isinstance(document.get("runtime_model_spec"), dict):
        document = document["runtime_model_spec"]

    try:
        hf_repo = document["hf_model_repo"]
        raw_device = document["device_type"]
        raw_engine = document["inference_engine"]
        implementation = document["impl"]
        impl_id = (
            implementation["impl_id"] if isinstance(implementation, dict) else None
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Runtime model spec is missing exact identity fields "
            "(hf_model_repo, device_type, inference_engine, impl.impl_id)"
        ) from exc
    raw_values = {
        "hf_model_repo": hf_repo,
        "device_type": raw_device,
        "inference_engine": raw_engine,
        "impl.impl_id": impl_id,
    }
    invalid = [
        field
        for field, value in raw_values.items()
        if not isinstance(value, str) or not value.strip()
    ]
    if invalid:
        raise ValueError(
            f"Runtime model spec has invalid exact identity fields: {invalid!r}"
        )
    hf_repo = hf_repo.strip()
    impl_id = impl_id.strip()
    try:
        device = DeviceTypes.from_string(raw_device.strip()).to_string()
        engine = InferenceEngine.from_string(raw_engine.strip()).value
    except (KeyError, ValueError) as exc:
        raise ValueError("Runtime model spec has an invalid device or engine") from exc
    return (hf_repo, device, engine, impl_id)


def _is_runtime_spec_path(name: str) -> bool:
    path = PurePosixPath(name)
    return path.suffix.lower() == ".json" and any(
        part == "runtime_model_specs" for part in path.parts
    )


def extract_bundle_identity(bundle: Path | bytes) -> LeafIdentity:
    """Require one consistent exact identity across a workflow-log bundle."""
    source = io.BytesIO(bundle) if isinstance(bundle, bytes) else Path(bundle)
    identities = set()
    candidate_count = 0
    try:
        with zipfile.ZipFile(source) as archive:
            runtime_paths = set()
            for info in archive.infolist():
                name = info.filename
                if not _is_runtime_spec_path(name):
                    continue
                if name in runtime_paths:
                    raise ValueError(
                        f"Workflow artifact contains duplicate runtime model spec "
                        f"path {name!r}"
                    )
                runtime_paths.add(name)
                candidate_count += 1
                try:
                    document = json.loads(archive.read(info))
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise ValueError(
                        f"Invalid runtime model spec JSON {name!r}"
                    ) from exc
                identities.add(identity_from_model_spec_document(document))
    except zipfile.BadZipFile as exc:
        raise ValueError("Workflow artifact is not a valid ZIP") from exc

    if candidate_count == 0:
        raise ValueError("Workflow artifact contains no runtime model spec JSON")
    if len(identities) != 1:
        raise ValueError(
            f"Workflow artifact contains conflicting runtime identities: "
            f"{sorted(identities)!r}"
        )
    return next(iter(identities))
