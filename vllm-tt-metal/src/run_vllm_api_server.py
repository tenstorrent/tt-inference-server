# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import logging
import multiprocessing
import os
import re
import runpy
import shlex
import sys
from pathlib import Path
from typing import Optional

import yaml
from huggingface_hub import snapshot_download

from utils.cache_monitor import get_container_cache_dir
from utils.device_utils import get_mesh_device_name
from utils.logging_utils import set_vllm_logging_config
from utils.prompt_client import run_background_trace_capture
from utils.vllm_run_utils import (
    create_model_symlink,
    get_encoded_api_key,
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_VLLM_SERVER_PORT = "8000"
QUETZAL_IMPL_ID = "quetzal"
QUETZAL_BACKEND = "generated_quetzal"
QUETZAL_PLUGIN_NAME = "quetzal_model_registry"
QUETZAL_PLUGIN_VALUE = "tt_quetzalcoatlus.vllm_plugin:register"
QUETZAL_PLUGIN_ALLOWLIST = {QUETZAL_PLUGIN_NAME, "tt"}
QUETZAL_ARTIFACT_ENV_VARS = (
    "QZ_QUALIFICATION_MANIFEST",
    "QUETZAL_PREFILL_GENERATED_PY",
    "QUETZAL_DECODE_GENERATED_PY",
    "QUETZAL_PREFILL_METADATA_JSON",
    "QUETZAL_DECODE_METADATA_JSON",
    "QUETZAL_WEIGHTS",
)
QUETZAL_BUNDLE_MANIFESTS_DIR = ".quetzal-bundle-manifests"
QUETZAL_BUNDLE_SCHEMAS = {"ttq.artifact_bundle/v1", "ttq.artifact_bundle/v2"}


def parse_args():
    """Parse wrapper CLI args and return remaining vLLM passthrough args."""
    parser = argparse.ArgumentParser(description="TT vLLM API Server")
    parser.add_argument(
        "--model",
        type=str,
        help="HuggingFace model repo (e.g., meta-llama/Llama-3.1-8B)",
    )
    parser.add_argument(
        "--tt-device",
        type=str,
        required=True,
        help="Device type (e.g., n300, t3k, galaxy)",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device type (e.g., n300, t3k, galaxy)",
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["vllm", "media", "forge"],
        help="Inference engine override (vllm/media/forge).",
    )
    parser.add_argument(
        "--impl",
        type=str,
        help="Implementation name override (e.g. tt-transformers).",
    )
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="Disable vLLM API key authorization (skips JWT_SECRET requirement)",
    )
    parser.add_argument(
        "--disable-trace-capture",
        action="store_true",
        help="Disable automatic trace capture requests on server startup",
    )
    parser.add_argument(
        "--service-port",
        type=int,
        default=None,
        help="Service port for vLLM server and trace capture client",
    )
    # Parse known args to allow vLLM args to pass through
    args, remaining_args = parser.parse_known_args()

    return args, remaining_args


def normalize_device_type(device_arg: str) -> str:
    """Convert user-provided device string to canonical device type name.

    Args:
        device_arg: User-provided device type (e.g., "n300", "galaxy", "T3K")

    Returns:
        Canonical device type name (e.g., "N300", "GALAXY", "T3K")
    """
    return device_arg.upper()


def normalize_engine_type(engine_arg: Optional[str]) -> Optional[str]:
    if not engine_arg:
        return None
    engine_map = {
        "vllm": "vLLM",
        "media": "media",
        "forge": "forge",
    }
    return engine_map[engine_arg.lower()]


def unwrap_model_specs_catalog(model_specs: dict) -> dict:
    """Return the nested model specs catalog from wrapped or legacy JSON."""
    if "model_specs" in model_specs and isinstance(model_specs["model_specs"], dict):
        return model_specs["model_specs"]
    return model_specs


def load_model_spec(
    model_arg: Optional[str],
    device_arg: Optional[str],
    engine_arg: Optional[str] = None,
    impl_arg: Optional[str] = None,
) -> dict:
    """Load and resolve a single model spec.

    Resolution order:
    1. Runtime mode: RUNTIME_MODEL_SPEC_JSON_PATH points to a pre-resolved spec
       (produced by run.py --docker-server)
    2. Catalog mode: MODEL_SPECS_JSON_PATH + --model/--tt-device/--device (+ optional
       --engine/--impl) are used to resolve one spec from the built-in catalog.

    Returns:
        dict: The resolved single model spec.

    Raises:
        RuntimeError: If runtime path is not available and required CLI args are missing.
    """
    runtime_path = os.getenv("RUNTIME_MODEL_SPEC_JSON_PATH")
    if runtime_path:
        runtime_path = Path(runtime_path)
        if runtime_path.exists():
            logger.info(
                "Using pre-resolved runtime model spec from "
                f"RUNTIME_MODEL_SPEC_JSON_PATH={runtime_path}"
            )
            logger.info(f"Loading runtime model spec from: {runtime_path}")
            with open(runtime_path, "r") as f:
                data = json.load(f)
            return data.get("runtime_model_spec", data)
        logger.warning(
            f"RUNTIME_MODEL_SPEC_JSON_PATH={runtime_path} does not exist, "
            "falling back to default model spec catalog."
        )

    if not model_arg or not device_arg:
        raise RuntimeError(
            "Either set RUNTIME_MODEL_SPEC_JSON_PATH env var "
            "(for 'python run.py --docker-server' workflow), or provide --model and "
            "--tt-device/--device for direct docker run. "
            "Example: docker run <image> --model meta-llama/Llama-3.1-8B --tt-device n300"
        )

    # Catalog mode (model_spec.json built into image)
    specs_path = os.getenv(
        "MODEL_SPECS_JSON_PATH",
        "/home/container_app_user/model_specs/model_spec.json",
    )
    logger.info(f"Loading all model specs from MODEL_SPECS_JSON_PATH: {specs_path}")
    with open(specs_path, "r") as f:
        model_specs = unwrap_model_specs_catalog(json.load(f))

    device_type = normalize_device_type(device_arg)
    model_spec = find_default_impl(
        model_specs,
        model_arg,
        device_type,
        engine_arg=engine_arg,
        impl_arg=impl_arg,
    )
    logger.info(
        f"Using default interface: found model spec for --model={model_arg}, "
        f"--device={device_type}, --engine={engine_arg}, --impl={impl_arg}"
    )
    return model_spec


def _resolve_hf_repo(model_specs: dict, model_arg: str) -> str:
    """Resolve model_arg to an hf_model_repo key in model_specs.

    Tries exact match first, then falls back to matching the short model name
    (last path segment) against all hf_model_repo keys.

    Args:
        model_specs: Nested model specs dict keyed by hf_model_repo at top level
        model_arg: The --model argument (HuggingFace repo or model name)

    Returns:
        The matching hf_model_repo key

    Raises:
        ValueError: If no matching hf_model_repo is found
    """
    if model_arg in model_specs:
        return model_arg

    short_name = model_arg.split("/")[-1]
    for hf_repo in model_specs:
        if hf_repo.split("/")[-1] == short_name:
            return hf_repo

    raise ValueError(
        f"No model spec found for model={model_arg}. "
        f"Available models: {list(model_specs.keys())[:10]}..."
    )


def find_default_impl(
    model_specs: dict,
    model_arg: str,
    device_type: str,
    engine_arg: Optional[str] = None,
    impl_arg: Optional[str] = None,
) -> dict:
    """Find the default implementation spec for a given model and device.

    Navigates the nested model spec structure to find the spec with
    default_impl=True for the given hf_model_repo and device_type.

    Args:
        model_specs: Nested dict: hf_model_repo > device_type > engine > impl_id > spec
        model_arg: The --model argument (HuggingFace repo or model name)
        device_type: Canonical device type name (e.g., "N300", "GALAXY")

    Returns:
        dict: The matching model spec with default_impl=True

    Raises:
        ValueError: If no matching spec is found
    """
    hf_repo = _resolve_hf_repo(model_specs, model_arg)
    device_specs = model_specs[hf_repo].get(device_type)
    if not device_specs:
        available_devices = list(model_specs[hf_repo].keys())
        raise ValueError(
            f"No model spec found for model={model_arg}, device={device_type}. "
            f"Available devices for {hf_repo}: {available_devices}"
        )

    if engine_arg:
        device_specs = {engine_arg: device_specs.get(engine_arg, {})}

    for engine_specs in device_specs.values():
        for spec in engine_specs.values():
            spec_impl_name = spec.get("impl", {}).get("impl_name")
            if impl_arg and spec_impl_name != impl_arg:
                continue
            if spec.get("device_model_spec", {}).get("default_impl"):
                return spec

    for engine_specs in device_specs.values():
        for spec in engine_specs.values():
            spec_impl_name = spec.get("impl", {}).get("impl_name")
            if impl_arg and spec_impl_name != impl_arg:
                continue
            return spec

    raise ValueError(
        f"No default_impl found for model={model_arg}, device={device_type}, "
        f"engine={engine_arg}, impl={impl_arg}. "
        f"Check that at least one impl has default_impl=True."
    )


def ensure_weights_available(model_spec: dict) -> Path:
    """Ensure model weights are available, downloading if necessary.

    If MODEL_WEIGHTS_DIR is already set (e.g. from --host-weights-dir bind mount),
    uses that directory directly and skips downloading.

    Args:
        model_spec: The model specification dictionary

    Returns:
        Path: Path to the model weights directory
    """
    # If MODEL_WEIGHTS_DIR is already set, use it directly and skip downloading
    model_weights_dir = os.getenv("MODEL_WEIGHTS_DIR")
    if model_weights_dir:
        weights_path = Path(model_weights_dir)
        if not weights_path.exists():
            raise RuntimeError(
                f"MODEL_WEIGHTS_DIR={model_weights_dir} does not exist. "
                "Ensure the host directory is correctly bind-mounted."
            )
        if not any(weights_path.iterdir()):
            raise RuntimeError(
                f"MODEL_WEIGHTS_DIR={model_weights_dir} is empty. "
                "Ensure the host directory contains model weight files."
            )
        logger.info(f"Using pre-mounted weights from MODEL_WEIGHTS_DIR: {weights_path}")
        return weights_path

    # Default: download weights into cache_root.
    # snapshot_download resumes partial downloads and skips files already present, so
    # always invoke it: a partially-downloaded directory looks non-empty but would crash
    # the server at load time if treated as complete. Fall back to existing weights only
    # when the hub is unreachable, preserving offline startup with complete weights.
    cache_root = Path(os.getenv("CACHE_ROOT", "/home/container_app_user/cache_root"))
    model_name = model_spec["model_name"]
    weights_path = cache_root / "weights" / model_name
    hf_repo = model_spec.get("hf_weights_repo") or model_spec["hf_model_repo"]
    revision = (
        model_spec.get("device_model_spec", {}).get("vllm_args", {}).get("revision")
    )

    weights_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading weights from {hf_repo} to {weights_path}")
    try:
        snapshot_download(repo_id=hf_repo, revision=revision, local_dir=weights_path)
    except Exception as e:
        if any(weights_path.iterdir()):
            logger.warning(
                f"Could not reach Hugging Face to verify weights ({e}); "
                f"using existing weights at {weights_path}"
            )
        else:
            raise

    os.environ["MODEL_WEIGHTS_DIR"] = str(weights_path)
    return weights_path


def set_cache_paths(model_spec: dict, device_type: str):
    """Set TT_CACHE_PATH and MESH_DEVICE for model-specific cache directory.

    Args:
        model_spec: The model specification dictionary
        device_type: Canonical device type name (e.g., "N300", "GALAXY")
    """
    mesh_device = get_mesh_device_name(device=device_type)
    tt_cache_path = get_container_cache_dir(model_spec, device=device_type)
    if tt_cache_path is None:
        raise RuntimeError("Could not resolve TT cache path from model spec.")

    # Set MESH_DEVICE env var for other components that need it
    os.environ["MESH_DEVICE"] = mesh_device
    logger.info(f"Set MESH_DEVICE to {mesh_device}")

    tt_cache_path.mkdir(parents=True, exist_ok=True)
    os.environ["TT_CACHE_PATH"] = str(tt_cache_path)
    logger.info(f"Set TT_CACHE_PATH to {tt_cache_path}")


def register_tt_models(impl_id=None):
    """Configure vLLM ModelRegistry according to ModelSpec.impl.impl_id.

    Args:
        impl_id: Implementation ID from ModelSpec JSON (e.g., "tt_transformers",
                 "llama3_70b_galaxy", "qwen3_32b_galaxy"). If None, defaults to
                 "tt_transformers".
    """
    impl_id = impl_id or "tt_transformers"
    if impl_id == QUETZAL_IMPL_ID:
        raise RuntimeError("native TT model registration is forbidden for impl=quetzal")

    # Delay importing vLLM until model-spec environment and Quetzal admission
    # have been applied. VLLM discovers plugins during import in some versions.
    from vllm import ModelRegistry

    # Llama path selection based on impl_id
    if impl_id == "llama3_70b_galaxy":
        os.environ["TT_LLAMA_TEXT_VER"] = "llama3_70b_galaxy"
    else:  # default: tt_transformers
        os.environ["TT_LLAMA_TEXT_VER"] = "tt_transformers"

    # Qwen3 env var setting based on impl_id
    if impl_id == "qwen3_32b_galaxy":
        os.environ["TT_QWEN3_TEXT_VER"] = "qwen3_32b_galaxy"
    else:
        os.environ["TT_QWEN3_TEXT_VER"] = "tt_transformers"

    # Arcee AFM-4.5B - Text
    ModelRegistry.register_model(
        "TTArceeForCausalLM",
        "models.tt_transformers.tt.generator_vllm:TTArceeForCausalLM",
    )


def _general_plugin_entry_points():
    """Return installed vLLM general plugins across Python metadata APIs."""
    try:
        return list(importlib.metadata.entry_points(group="vllm.general_plugins"))
    except TypeError:  # Python/importlib_metadata compatibility
        return list(
            importlib.metadata.entry_points().select(group="vllm.general_plugins")
        )


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _canonical_json(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _tree_digest(files: list[dict]) -> str:
    identity = [
        {"path": row.get("path"), "size": row.get("size"), "sha256": row.get("sha256")}
        for row in files
    ]
    return hashlib.sha256(_canonical_json(identity)).hexdigest()


def _safe_auxiliary_relative(value, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise RuntimeError(f"{label} is not a portable relative path")
    path = Path(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise RuntimeError(f"{label} is not a canonical relative path")
    if path.as_posix() != value:
        raise RuntimeError(f"{label} is not a canonical relative path")
    return path


def _require_read_only_path(path: Path, *, directory: bool, label: str) -> None:
    if path.is_symlink():
        raise RuntimeError(f"{label} may not be a symlink: {path}")
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise RuntimeError(f"{label} is missing: {path}") from exc
    if directory and not path.is_dir():
        raise RuntimeError(f"{label} is not a directory: {path}")
    if not directory and not path.is_file():
        raise RuntimeError(f"{label} is not a regular file: {path}")
    if metadata.st_mode & 0o222:
        raise RuntimeError(f"{label} is mutable: {path}")


def _require_read_only_package_member(
    package_root: Path, value: str, *, label: str
) -> tuple[Path, str]:
    """Resolve one package member without hiding symlink or mutable-parent state."""
    member = Path(value)
    try:
        relative = member.relative_to(package_root)
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes QUETZAL_PACKAGE_ROOT: {member}") from exc
    if not relative.parts or any(part in ("", ".", "..") for part in relative.parts):
        raise RuntimeError(f"{label} is not a canonical package path: {member}")

    current = package_root
    for part in relative.parts[:-1]:
        current = current / part
        _require_read_only_path(current, directory=True, label=f"{label} parent")
    _require_read_only_path(member, directory=False, label=label)
    try:
        resolved = member.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"{label} is missing: {member}") from exc
    if not resolved.is_relative_to(package_root):
        raise RuntimeError(f"{label} escapes QUETZAL_PACKAGE_ROOT: {member}")
    return member, relative.as_posix()


def _validate_quetzal_auxiliary_references(
    bundle_manifest: dict, package_id: str
) -> None:
    schema = bundle_manifest.get("schema")
    roots_value = os.getenv("QUETZAL_AUXILIARY_ROOTS_JSON", "")
    if schema == "ttq.artifact_bundle/v1":
        if roots_value:
            raise RuntimeError("v1 Quetzal package must not declare auxiliary roots")
        return
    if schema != "ttq.artifact_bundle/v2":
        raise RuntimeError("Quetzal trusted-root proof has an invalid schema")

    references = bundle_manifest.get("auxiliary_references")
    if not isinstance(references, list) or not references:
        raise RuntimeError("Quetzal v2 package requires auxiliary references")
    try:
        roots = json.loads(roots_value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("QUETZAL_AUXILIARY_ROOTS_JSON is invalid JSON") from exc
    if not isinstance(roots, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in roots.items()
    ):
        raise RuntimeError("QUETZAL_AUXILIARY_ROOTS_JSON must map names to paths")

    names = []
    external_files = 0
    external_bytes = 0
    for index, reference in enumerate(references):
        label = f"auxiliary_references[{index}]"
        if (
            not isinstance(reference, dict)
            or set(reference) != {"role", "name", "sha256", "files"}
            or reference.get("role") != "streamed_cache"
        ):
            raise RuntimeError(f"{label} is not a streamed_cache reference")
        name = reference.get("name")
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z0-9._@+-]+", name) is None
        ):
            raise RuntimeError(f"{label}.name is invalid")
        files = reference.get("files")
        if not isinstance(files, list) or not files:
            raise RuntimeError(f"{label}.files must be non-empty")
        paths = []
        for row_index, row in enumerate(files):
            row_label = f"{label}.files[{row_index}]"
            if (
                not isinstance(row, dict)
                or set(row) != {"path", "size", "sha256"}
                or not isinstance(row.get("size"), int)
                or isinstance(row.get("size"), bool)
                or row["size"] < 0
                or re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256"))) is None
            ):
                raise RuntimeError(f"{row_label} is invalid")
            _safe_auxiliary_relative(row["path"], f"{row_label}.path")
            paths.append(row["path"])
            external_files += 1
            external_bytes += row["size"]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise RuntimeError(f"{label}.files must be sorted and unique")
        if reference.get("sha256") != _tree_digest(files):
            raise RuntimeError(f"{label} inventory digest mismatch")
        names.append(name)
    if names != sorted(names) or len(names) != len(set(names)):
        raise RuntimeError("Quetzal auxiliary references must be sorted and unique")
    if set(roots) != set(names):
        raise RuntimeError(
            "Quetzal auxiliary roots do not match the trusted manifest: "
            f"expected {sorted(names)}, got {sorted(roots)}"
        )
    if (
        bundle_manifest.get("external_total_files") != external_files
        or bundle_manifest.get("external_total_bytes") != external_bytes
    ):
        raise RuntimeError("Quetzal auxiliary totals do not match the trusted manifest")

    trees = bundle_manifest.get("trees")
    if not isinstance(trees, list):
        raise RuntimeError("Quetzal v2 package trees are invalid")
    tree_digests = {
        tree.get("role"): tree.get("sha256") for tree in trees if isinstance(tree, dict)
    }
    if set(tree_digests) != {"compiled", "compiled_weights"} or any(
        re.fullmatch(r"[0-9a-f]{64}", str(value)) is None
        for value in tree_digests.values()
    ):
        raise RuntimeError("Quetzal v2 package tree identities are invalid")
    auxiliary_identity = hashlib.sha256(
        _canonical_json(
            [
                {key: reference[key] for key in ("role", "name", "sha256")}
                for reference in references
            ]
        )
    ).hexdigest()
    expected_package_id = (
        f"sha256-v2-{tree_digests['compiled']}-"
        f"{tree_digests['compiled_weights']}-{auxiliary_identity}"
    )
    if package_id != expected_package_id:
        raise RuntimeError("Quetzal v2 package ID does not bind its auxiliary identity")

    for reference in references:
        name = reference["name"]
        root = Path(roots[name])
        _require_read_only_path(root, directory=True, label=f"auxiliary root {name}")
        if root.name != f"sha256-{reference['sha256']}":
            raise RuntimeError(f"auxiliary root {name} is not digest-addressed")
        resolved_root = root.resolve(strict=True)
        checked_directories = {resolved_root}
        for row in reference["files"]:
            relative = _safe_auxiliary_relative(row["path"], f"auxiliary object {name}")
            current = root
            for part in relative.parts[:-1]:
                current = current / part
                if current.is_symlink():
                    raise RuntimeError(f"auxiliary path contains a symlink: {current}")
                try:
                    resolved = current.resolve(strict=True)
                except OSError as exc:
                    raise RuntimeError(
                        f"auxiliary directory is missing: {current}"
                    ) from exc
                if not resolved.is_relative_to(resolved_root):
                    raise RuntimeError(
                        f"auxiliary directory escapes its root: {current}"
                    )
                if resolved not in checked_directories:
                    _require_read_only_path(
                        current,
                        directory=True,
                        label=f"auxiliary directory {name}/{relative.parent}",
                    )
                    checked_directories.add(resolved)
            payload = root / relative
            if payload.is_symlink():
                raise RuntimeError(f"auxiliary path contains a symlink: {payload}")
            try:
                resolved_payload = payload.resolve(strict=True)
            except OSError as exc:
                raise RuntimeError(f"auxiliary object is missing: {payload}") from exc
            if not resolved_payload.is_relative_to(resolved_root):
                raise RuntimeError(f"auxiliary object escapes its root: {payload}")
            _require_read_only_path(
                payload, directory=False, label=f"auxiliary object {name}/{row['path']}"
            )
            digest, size = _sha256_file(payload)
            if size != row["size"] or digest != row["sha256"]:
                raise RuntimeError(
                    f"auxiliary object digest mismatch: {name}/{row['path']}"
                )


def _validate_quetzal_package_and_runtime(root: Path, model_id: str) -> None:
    package_id = os.getenv("QUETZAL_PACKAGE_ID")
    package_root_value = os.getenv("QUETZAL_PACKAGE_ROOT")
    if not package_id or not package_root_value:
        raise RuntimeError(
            "impl=quetzal requires QUETZAL_PACKAGE_ID and QUETZAL_PACKAGE_ROOT"
        )
    package_root_input = Path(package_root_value)
    if package_root_input.is_symlink():
        raise RuntimeError(
            f"QUETZAL_PACKAGE_ROOT may not be a symlink: {package_root_input}"
        )
    try:
        package_root = package_root_input.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(
            f"QUETZAL_PACKAGE_ROOT is missing: {package_root_input}"
        ) from exc
    if root != package_root or package_root.name != package_id:
        raise RuntimeError(
            "QZ_MODELS_ROOT must equal QUETZAL_PACKAGE_ROOT and its basename "
            "must equal QUETZAL_PACKAGE_ID"
        )
    _require_read_only_path(package_root, directory=True, label="Quetzal package root")

    manifest_sha256 = os.getenv("QUETZAL_BUNDLE_MANIFEST_SHA256", "")
    if re.fullmatch(r"[0-9a-f]{64}", manifest_sha256) is None:
        raise RuntimeError(
            "impl=quetzal requires QUETZAL_BUNDLE_MANIFEST_SHA256 as lowercase SHA-256"
        )
    trusted_manifest_dir = package_root / QUETZAL_BUNDLE_MANIFESTS_DIR
    _require_read_only_path(
        trusted_manifest_dir,
        directory=True,
        label="Quetzal trusted-root proof directory",
    )
    trusted_manifest = trusted_manifest_dir / f"{manifest_sha256}.json"
    try:
        _require_read_only_path(
            trusted_manifest,
            directory=False,
            label="Quetzal trusted-root proof",
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"Quetzal package has a missing trusted-root proof or an invalid one: "
            f"{trusted_manifest}: {exc}"
        ) from exc
    actual_manifest_sha256, manifest_size = _sha256_file(trusted_manifest)
    if manifest_size > 16 * 1024 * 1024 or actual_manifest_sha256 != manifest_sha256:
        raise RuntimeError("Quetzal trusted-root proof digest mismatch")
    try:
        bundle_manifest = json.loads(trusted_manifest.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Quetzal trusted-root proof is invalid JSON") from exc
    if bundle_manifest.get("schema") not in QUETZAL_BUNDLE_SCHEMAS:
        raise RuntimeError("Quetzal trusted-root proof has an invalid schema")
    _validate_quetzal_auxiliary_references(bundle_manifest, package_id)

    inventory = {}
    for tree in bundle_manifest.get("trees", []):
        role = tree.get("role")
        name = tree.get("name")
        for row in tree.get("files", []):
            inventory[f"{role}/{name}/{row.get('path')}"] = row
    inventory["qualification_manifest.yaml"] = bundle_manifest.get(
        "qualification_manifest", {}
    )
    for env_name in QUETZAL_ARTIFACT_ENV_VARS:
        value = os.getenv(env_name)
        if not value:
            raise RuntimeError(f"impl=quetzal requires {env_name}")
        artifact, relative = _require_read_only_package_member(
            package_root, value, label=env_name
        )
        row = inventory.get(relative)
        digest, size = _sha256_file(artifact)
        if (
            not isinstance(row, dict)
            or row.get("size") != size
            or row.get("sha256") != digest
        ):
            raise RuntimeError(f"{env_name} failed trusted-root verification")

    qualification_path = Path(os.environ["QZ_QUALIFICATION_MANIFEST"])
    try:
        qualification = yaml.safe_load(qualification_path.read_text())
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise RuntimeError("Quetzal qualification manifest is invalid YAML") from exc
    rows = qualification.get("models", []) if isinstance(qualification, dict) else []
    matches = [
        row for row in rows if isinstance(row, dict) and row.get("model_id") == model_id
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "Quetzal qualification manifest must contain exactly one model contract"
        )
    required_runtime = (
        matches[0].get("charter_pcc", {}).get("required_runtime_tt_metal_commit")
    )
    actual_runtime = os.getenv("TT_METAL_COMMIT_SHA_OR_TAG")
    if not required_runtime or actual_runtime != required_runtime:
        raise RuntimeError(
            "Quetzal TT-Metal runtime mismatch: package requires "
            f"{required_runtime!r}, image provides {actual_runtime!r}"
        )

    required_patchset = os.getenv("QUETZAL_REQUIRED_TT_METAL_PATCHSET_SHA256")
    actual_patchset = os.getenv("TT_METAL_PATCHSET_SHA256")
    if not required_patchset or actual_patchset != required_patchset:
        raise RuntimeError(
            "Quetzal TT-Metal patchset mismatch: catalog requires "
            f"{required_patchset!r}, image provides {actual_patchset!r}"
        )
    identity_path = Path(os.getenv("TT_METAL_HOME", "")) / ".ttq-runtime-identity.json"
    if identity_path.is_symlink() or not identity_path.is_file():
        raise RuntimeError("Quetzal image is missing TT-Metal runtime identity")
    try:
        identity = json.loads(identity_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Quetzal TT-Metal runtime identity is invalid") from exc
    if (
        identity.get("base_revision") != actual_runtime
        or identity.get("patchset_sha256") != actual_patchset
    ):
        raise RuntimeError("Quetzal TT-Metal runtime identity mismatch")


def validate_quetzal_runtime(model_spec: dict) -> dict | None:
    """Fail closed before vLLM import for a generated Quetzal launch.

    The installed Quetzal plugin performs the definitive architecture
    registration. This preflight proves the selected model spec cannot fall
    through to the handcrafted registry while startup is still host-only.
    """
    impl_id = model_spec.get("impl", {}).get("impl_id")
    if impl_id != QUETZAL_IMPL_ID:
        return None

    model_id = model_spec.get("hf_model_repo")
    env_model = os.getenv("QUETZAL_MODEL")
    if os.getenv("QUETZAL_VLLM") != "1":
        raise RuntimeError("impl=quetzal requires QUETZAL_VLLM=1")
    if not model_id or env_model != model_id:
        raise RuntimeError(
            "Quetzal model identity mismatch: catalog hf_model_repo must equal "
            "QUETZAL_MODEL"
        )

    plugins = [item.strip() for item in os.getenv("VLLM_PLUGINS", "").split(",")]
    plugins = [item for item in plugins if item]
    if set(plugins) != QUETZAL_PLUGIN_ALLOWLIST or len(plugins) != 2:
        raise RuntimeError(
            "impl=quetzal requires exactly "
            "VLLM_PLUGINS=quetzal_model_registry,tt; native model registries "
            "are forbidden"
        )
    if os.getenv("TT_VLLM_BUILTIN_MODELS") != "0":
        raise RuntimeError("impl=quetzal requires TT_VLLM_BUILTIN_MODELS=0")

    vllm_args = model_spec.get("device_model_spec", {}).get("vllm_args", {})
    revision = os.getenv("QUETZAL_HF_REVISION")
    if not revision or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise RuntimeError(
            "Quetzal HF revision must be an immutable lowercase 40-hex commit"
        )
    if vllm_args.get("revision") != revision:
        raise RuntimeError("Quetzal config revision does not match artifact revision")
    if vllm_args.get("tokenizer_revision") != revision:
        raise RuntimeError(
            "Quetzal tokenizer revision does not match artifact revision"
        )

    root_value = os.getenv("QZ_MODELS_ROOT")
    if not root_value:
        raise RuntimeError("impl=quetzal requires QZ_MODELS_ROOT")
    root = Path(root_value)
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"QZ_MODELS_ROOT must be a real directory: {root}")
    root = root.resolve()

    manifest_value = os.getenv("QZ_QUALIFICATION_MANIFEST")
    if not manifest_value:
        raise RuntimeError("impl=quetzal requires QZ_QUALIFICATION_MANIFEST")
    manifest = Path(manifest_value)
    if manifest.is_symlink() or not manifest.is_file():
        raise RuntimeError(
            f"QZ_QUALIFICATION_MANIFEST must be a regular file: {manifest}"
        )
    manifest = manifest.resolve()
    if not manifest.is_relative_to(root):
        raise RuntimeError(
            "Quetzal qualification manifest must be inside QZ_MODELS_ROOT"
        )

    _validate_quetzal_package_and_runtime(root, model_id)

    entry_points = [
        entry
        for entry in _general_plugin_entry_points()
        if entry.name == QUETZAL_PLUGIN_NAME
    ]
    if len(entry_points) != 1:
        raise RuntimeError(
            "impl=quetzal requires exactly one installed "
            f"{QUETZAL_PLUGIN_NAME!r} vLLM entry point; found {len(entry_points)}"
        )
    if entry_points[0].value != QUETZAL_PLUGIN_VALUE:
        raise RuntimeError(
            f"{QUETZAL_PLUGIN_NAME!r} must resolve to {QUETZAL_PLUGIN_VALUE!r}; "
            f"found {entry_points[0].value!r}"
        )

    # Discovery is generated-only by default. It validates the qualification
    # policy, prefill/decode pair, weights, runtime ABI, and backend identity.
    quetzal_server = importlib.import_module("serving.quetzal_server")
    entries = quetzal_server.discover_models(str(root))
    required_context = model_spec.get("device_model_spec", {}).get("max_context")
    matching = []
    for entry in entries.values():
        bucket_lengths = {
            bucket.get("seq_len")
            for bucket in entry.get("prefill_buckets", [])
            if isinstance(bucket, dict)
        }
        if (
            entry.get("model_id") == model_id
            and entry.get("backend") == QUETZAL_BACKEND
            and entry.get("batch_size") in (None, 1)
            and entry.get("target_mesh") in ("p150x4", "4-chip")
            and required_context in bucket_lengths
        ):
            matching.append(entry)
    if not matching:
        raise RuntimeError(
            "no qualified generated_quetzal p150x4/B1 artifact with the "
            f"catalog context {required_context} was discovered for {model_id} "
            f"under {root}; refusing native fallback"
        )
    logger.info(
        "Quetzal preflight admitted model=%s revision=%s backend=%s emit_hash=%s",
        model_id,
        revision,
        matching[0].get("backend"),
        matching[0].get("emit_hash"),
    )
    return matching[0]


def model_setup(model_spec_json):
    # step 1: validate env vars passed in
    cache_root = Path(os.getenv("CACHE_ROOT"))
    assert cache_root.exists(), f"CACHE_ROOT: {cache_root} does not exist"
    symlinks_dir = cache_root / "model_file_symlinks_map"
    symlinks_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"MODEL_WEIGHTS_DIR: {os.getenv('MODEL_WEIGHTS_DIR')}")
    assert os.getenv("MODEL_WEIGHTS_DIR") is not None, "MODEL_WEIGHTS_DIR must be set"
    weights_dir = Path(os.getenv("MODEL_WEIGHTS_DIR"))
    assert weights_dir.exists(), f"MODEL_WEIGHTS_DIR: {weights_dir} does not exist"

    logging.info(f"TT_CACHE_PATH: {os.getenv('TT_CACHE_PATH')}")
    assert os.getenv("TT_CACHE_PATH") is not None, "TT_CACHE_PATH must be set"

    # step 2: set default runtime env vars
    # set up logging
    config_path, log_path = set_vllm_logging_config(level="DEBUG")
    logger.info(f"setting vllm logging config at: {config_path}")
    logger.info(f"setting vllm logging file at: {log_path}")

    # set HF_MODEL environment variable for loading
    logging.info(f"HF model setup for {model_spec_json['hf_model_repo']}")
    model_dir_name = model_spec_json["hf_model_repo"].split("/")[-1]
    hf_dir = create_model_symlink(symlinks_dir, model_dir_name, weights_dir)

    dynamic_env_vars = {
        "VLLM_LOGGING_CONFIG_PATH": str(config_path),
        "HF_MODEL": hf_dir,
    }

    # Set dynamic environment variables
    logger.info("setting dynamic runtime environment variables:")
    for key, value in dynamic_env_vars.items():
        if value is not None:
            logger.info(f"setting env var: {key}={value}")
            os.environ[key] = str(value)
        elif key in os.environ:
            logger.warning(
                f"removing env var: {key} from os.environ, previous value={os.environ[key]}"
            )
            del os.environ[key]


def handle_secrets(no_auth=False):
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        logger.info("HF_TOKEN is set")
    else:
        logger.warning(
            "HF_TOKEN is not set - this may cause issues accessing private models or models requiring authorization"
        )

    if no_auth:
        # Remove VLLM_API_KEY if present to disable authorization
        if "VLLM_API_KEY" in os.environ:
            del os.environ["VLLM_API_KEY"]
        logger.info(
            "--no-auth is set: requests to vLLM API will not require authorization. "
            "HTTP Authorization header will not be checked."
        )
        return

    # Check for VLLM_API_KEY first, then fall back to JWT_SECRET
    vllm_api_key = os.getenv("VLLM_API_KEY")
    if vllm_api_key:
        logger.info("VLLM_API_KEY is already set, using existing value")
        return

    # VLLM_API_KEY is not set, check if JWT_SECRET is available
    jwt_secret = os.getenv("JWT_SECRET")
    if not jwt_secret:
        logger.warning(
            "Neither VLLM_API_KEY nor JWT_SECRET are set: HTTP requests to vLLM API will not require authorization"
        )
        return

    encoded_api_key = get_encoded_api_key(jwt_secret)
    if encoded_api_key is not None:
        os.environ["VLLM_API_KEY"] = encoded_api_key
        logger.info(
            "JWT_SECRET is set: HTTP requests to vLLM API require bearer token in 'Authorization' header. See docs for how to get bearer token."
        )


def runtime_settings(model_spec_json, no_auth=False):
    logger.info(f"using model: {model_spec_json['model_id']}")
    handle_secrets(no_auth=no_auth)

    # In multihost deployments, model weights are on shared storage and accessed
    # via model-specific environment variables (e.g., DEEPSEEK_V3_HF_MODEL).
    # Skip model_setup() which requires MODEL_WEIGHTS_DIR and creates symlinks.
    # TODO(tt-metal): Update DeepSeek model impl to use standard HF_MODEL env var
    # so we can reuse existing model setup and standard weight/cache mounting.
    if os.getenv("MULTIHOST_ROLE"):
        logger.info(
            "Multihost mode detected, skipping model_setup() - "
            "weights accessed via model-specific env vars on shared storage"
        )
        return

    # TODO: check HF repo access with HF_TOKEN supplied
    model_setup(model_spec_json)


def set_metal_timeout_env_vars():
    """Set tt-metal operation timeout env vars for automatic hang detection.

    When enabled (default), configures TT_METAL_OPERATION_TIMEOUT_SECONDS and
    TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE so that tt-triage runs
    automatically when an op dispatch hangs.

    Disabled when DISABLE_METAL_OP_TIMEOUT=1 is set by either the model spec or
    ``run.py --disable-metal-timeout``.
    """
    if os.getenv("DISABLE_METAL_OP_TIMEOUT") == "1":
        # DISABLE_METAL_OP_TIMEOUT is an inference-server control flag; tt-metal
        # itself only reads these two variables. Clear inherited values so an
        # explicit model/CLI disable cannot leave a previously configured timeout
        # active in this process.
        os.environ.pop("TT_METAL_OPERATION_TIMEOUT_SECONDS", None)
        os.environ.pop("TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE", None)
        logger.info("Metal op timeout disabled via DISABLE_METAL_OP_TIMEOUT=1")
        return

    tt_metal_home = os.getenv("TT_METAL_HOME", "/home/container_app_user/tt-metal")
    python_env_dir = os.getenv("PYTHON_ENV_DIR", f"{tt_metal_home}/python_env")
    # Triage report dir: TT_TRIAGE_LOGS_PATH (the cache_root volume in CI) if set,
    # else the tt-metal logs dir. Separate from TT_METAL_LOGS_PATH so tt-metal's
    # high-churn Inspector/watcher logs stay off the host-owned volume. See #4255.
    log_dir = os.getenv("TT_TRIAGE_LOGS_PATH") or os.getenv(
        "TT_METAL_LOGS_PATH", "/home/container_app_user/logs"
    )

    triage_new = Path(tt_metal_home) / "tools" / "triage" / "triage.py"
    triage_old = Path(tt_metal_home) / "scripts" / "debugging_scripts" / "triage.py"
    triage_script = str(triage_new if triage_new.exists() else triage_old)

    # mkdir -p so the redirect succeeds when log_dir doesn't exist yet (in CI it
    # points at the cache_root volume, which has no pre-created logs/ dir). See #2670.
    # Tee rather than redirect: the triage report names the stalled core/kernel and
    # is the only artifact that explains a dispatch hang. Writing it solely into
    # log_dir loses it whenever that directory is a container volume CI does not
    # upload (the Galaxy release job is exactly this case), so a hang reproduces as
    # an unexplained TT_THROW with the diagnosis stranded inside the dead container.
    # Teeing keeps the on-disk copy and also puts the report on stdout, where it is
    # captured in the server log and therefore in the CI job log.
    timeout_cmd = (
        f"mkdir -p {log_dir} && "
        f"{python_env_dir}/bin/python {triage_script} "
        f"--disable-progress 2>&1 | "
        f"tee {log_dir}/tt-triage-$(date +%Y%m%d-%H%M%S).log"
    )

    os.environ["TT_METAL_OPERATION_TIMEOUT_SECONDS"] = "5.0"
    os.environ["TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE"] = timeout_cmd
    logger.info("Set TT_METAL_OPERATION_TIMEOUT_SECONDS=5.0")
    logger.info(f"Set TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE={timeout_cmd}")


def set_runtime_env_vars(model_spec_json):
    """Set runtime environment variables from model spec.

    Handles env_vars in two possible locations:
    1. Top level: model_spec_json["env_vars"] (from ModelSpec.__post_init__ merge)
    2. Nested: model_spec_json["device_model_spec"]["env_vars"] (raw JSON)

    Both locations are checked and merged, with top-level taking precedence.
    """
    env_vars = {}

    # Check nested location first (device_model_spec.env_vars)
    device_model_spec = model_spec_json.get("device_model_spec", {})
    if isinstance(device_model_spec, dict):
        nested_env_vars = device_model_spec.get("env_vars", {})
        if nested_env_vars:
            env_vars.update(nested_env_vars)

    # Check top-level location (takes precedence)
    top_level_env_vars = model_spec_json.get("env_vars", {})
    if top_level_env_vars:
        env_vars.update(top_level_env_vars)

    if not env_vars:
        logger.info("No env_vars found in model spec")
        return

    for key, value in env_vars.items():
        if not isinstance(key, str):
            key = str(key)
            logger.warning(
                f"env var key:={key} is not a string, converting to string: {key}"
            )
        if not isinstance(value, str):
            logger.warning(
                f"env var value:={value} is not a string, converting to string: {value}"
            )
            value = str(value)

        original_value = os.getenv(key)
        if original_value is not None:
            logger.warning(
                f"env var {key} is already set to {original_value}, overriding with {value}"
            )
        logger.info(f"setting env var: {key}={value}")
        os.environ[key] = value


def start_trace_capture(
    model_spec_json, service_port: int, disable_trace_capture: bool = False
):
    # Models with builtin warmup handle their own trace capture internally
    if not disable_trace_capture and model_spec_json.get("has_builtin_warmup", False):
        disable_trace_capture = True
        logger.info(
            "Model has builtin warmup (has_builtin_warmup=True), "
            "skipping background trace capture"
        )

    if disable_trace_capture:
        logger.info("Trace capture is disabled via --disable-trace-capture")
        return

    supported_modalities = model_spec_json.get("supported_modalities", ["text"])

    # Get max_context from device_model_spec for trace calculation
    max_context = model_spec_json.get("device_model_spec", {}).get("max_context")
    if max_context is None:
        # Fallback to vllm_args if not in device_model_spec
        max_model_len_str = (
            model_spec_json.get("device_model_spec", {})
            .get("vllm_args", {})
            .get("max_model_len")
        )
        if max_model_len_str:
            max_context = int(max_model_len_str)

    logger.info("Starting background trace capture process...")
    trace_process = multiprocessing.Process(
        target=run_background_trace_capture,
        args=(
            model_spec_json["hf_model_repo"],
            service_port,
            supported_modalities,
            max_context,
        ),
        daemon=True,
        name="trace_capture",
    )
    trace_process.start()
    logger.info(
        f"Background trace capture process started (PID: {trace_process.pid}, "
        f"max_context: {max_context})"
    )


def _normalize_vllm_arg_name(arg_name: str) -> str:
    return arg_name.lstrip("-").split("=", 1)[0].replace("-", "_")


def _append_vllm_arg(argv: list[str], arg_name: str, value) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            argv.append(arg_name)
        return
    argv.extend([arg_name, str(value)])


def _extract_cli_arg_value(argv: list[str], arg_name: str) -> Optional[str]:
    for index, token in enumerate(argv):
        if token == arg_name:
            if index + 1 < len(argv):
                return argv[index + 1]
            return None
        if token.startswith(f"{arg_name}="):
            return token.split("=", 1)[1]
    return None


def resolve_service_port() -> int:
    port_value = _extract_cli_arg_value(sys.argv[1:], "--port")
    if port_value is not None:
        return int(port_value)
    return int(DEFAULT_VLLM_SERVER_PORT)


def format_vllm_serve_command(argv) -> str:
    """Render the normalized argv as a multi-line bash command."""
    command_lines = ["vllm serve"]
    index = 1
    while index < len(argv):
        token = argv[index]
        rendered_tokens = [shlex.quote(token)]
        has_separate_value = (
            token.startswith("--")
            and "=" not in token
            and index + 1 < len(argv)
            and not argv[index + 1].startswith("--")
        )
        if has_separate_value:
            rendered_tokens.append(shlex.quote(argv[index + 1]))
            index += 1

        command_lines.append(" ".join(rendered_tokens))
        index += 1

    return " \\\n  ".join(command_lines)


def set_vllm_sys_argv(args, remaining_sys_argv, default_vllm_args):
    # runpy uses sys.argv, rebuild it with the merged vLLM args.
    vllm_argv = [sys.argv[0]]
    remaining_default_vllm_args = dict(default_vllm_args)
    default_arg_name_by_normalized_name = {
        _normalize_vllm_arg_name(arg_name): arg_name
        for arg_name in remaining_default_vllm_args
    }
    input_vllm_argv = list(remaining_sys_argv)
    if args.service_port is not None:
        already_set_port = _extract_cli_arg_value(input_vllm_argv, "--port")
        if already_set_port is not None:
            logger.warning(
                f"vLLM server --port={already_set_port} already set direcly, ignoring --service-port={args.service_port}"
            )
        else:
            # Remap wrapper --service-port to vLLM's --port.
            input_vllm_argv.extend(["--port", str(args.service_port)])

    index = 0
    while index < len(input_vllm_argv):
        token = input_vllm_argv[index]
        if not token.startswith("--"):
            vllm_argv.append(token)
            index += 1
            continue

        cli_arg_name, separator, inline_value = token.partition("=")
        overridden_default_arg_name = default_arg_name_by_normalized_name.pop(
            _normalize_vllm_arg_name(cli_arg_name), None
        )
        if overridden_default_arg_name is not None:
            remaining_default_vllm_args.pop(overridden_default_arg_name, None)

        if separator:
            vllm_argv.append(f"{cli_arg_name}={inline_value}")
            index += 1
            continue

        vllm_argv.append(cli_arg_name)
        next_token_is_value = index + 1 < len(input_vllm_argv) and not input_vllm_argv[
            index + 1
        ].startswith("--")
        if next_token_is_value:
            value = input_vllm_argv[index + 1]
            vllm_argv.append(value)
            index += 2
            continue

        index += 1

    for key, value in remaining_default_vllm_args.items():
        cli_arg_name = f"--{key}"
        _append_vllm_arg(vllm_argv, cli_arg_name, value)

    # finally set sys.argv to the vllm server args
    sys.argv = vllm_argv
    logger.info(f"vLLM command:\n{format_vllm_serve_command(sys.argv)}")


def main():
    # Step 1: Parse --model argument (if provided)
    args, remaining_sys_argv = parse_args()
    args.device = args.tt_device or args.device
    args.engine = normalize_engine_type(args.engine)

    # Step 2: Load model spec
    model_spec = load_model_spec(
        model_arg=args.model,
        device_arg=args.device,
        engine_arg=args.engine,
        impl_arg=args.impl,
    )
    # Apply the catalog environment before any vLLM import. Quetzal's plugin
    # allowlist is an admission boundary, not a setting that can be changed
    # safely after vLLM has discovered model registries.
    set_runtime_env_vars(model_spec)
    quetzal_artifact = validate_quetzal_runtime(model_spec)
    impl_id = model_spec.get("impl", {}).get("impl_id")
    device_type = model_spec.get("device_type")
    if device_type:
        device_type = normalize_device_type(device_type)
    elif args.device:
        device_type = normalize_device_type(args.device)

    if device_type and not os.getenv("TT_CACHE_PATH"):
        set_cache_paths(model_spec, device_type)
    # NOTE: In multihost deployments, model weights are expected to reside on shared
    # storage (e.g., NFS) and are read directly by each worker via model-specific
    # environment variables (e.g., DEEPSEEK_V3_HF_MODEL). Users are responsible for
    # downloading weights to a location on shared storage beforehand. Therefore,
    # automatic weight download is skipped when MULTIHOST_ROLE is set.
    if (
        impl_id != QUETZAL_IMPL_ID
        and not os.getenv("MODEL_WEIGHTS_DIR")
        and not os.getenv("MULTIHOST_ROLE")
    ):
        ensure_weights_available(model_spec)

    logger.info(f"Using model spec: {model_spec['model_id']}")

    # Step 3: Register handcrafted TT models only for native implementations.
    # Generated Quetzal is registered exclusively by its installed vLLM entry
    # point; calling this function would create the forbidden fallback lane.
    if impl_id != QUETZAL_IMPL_ID:
        register_tt_models(impl_id)
    else:
        logger.info(
            "Skipping native TT model registration for Quetzal artifact %s",
            quetzal_artifact.get("emit_hash") if quetzal_artifact else None,
        )

    # Step 4: Set runtime environment variables and vLLM server args
    set_metal_timeout_env_vars()
    runtime_settings(model_spec, no_auth=args.no_auth)
    default_vllm_args = model_spec["device_model_spec"]["vllm_args"]
    set_vllm_sys_argv(args, remaining_sys_argv, default_vllm_args)

    # Step 5: Start trace capture if needed
    start_trace_capture(
        model_spec,
        service_port=resolve_service_port(),
        disable_trace_capture=args.disable_trace_capture,
    )

    # Step 6: Launch vLLM server
    # runpy uses the same process and environment so the registered models are available
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
