# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Host-to-runtime path contract for immutable Quetzal packages."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


QUETZAL_IMPL_ID = "quetzal"
QUETZAL_PACKAGE_ROOT_ENV = "QUETZAL_PACKAGE_ROOT"
QUETZAL_MANIFEST_SHA256_ENV = "QUETZAL_BUNDLE_MANIFEST_SHA256"
QUETZAL_INSTALLED_MANIFEST_DIR = ".quetzal-bundle-manifests"
QUETZAL_PACKAGE_PATH_ENV_VARS = (
    QUETZAL_PACKAGE_ROOT_ENV,
    "QZ_MODELS_ROOT",
    "QZ_QUALIFICATION_MANIFEST",
    "QUETZAL_PREFILL_GENERATED_PY",
    "QUETZAL_DECODE_GENERATED_PY",
    "QUETZAL_PREFILL_METADATA_JSON",
    "QUETZAL_DECODE_METADATA_JSON",
    "QUETZAL_WEIGHTS",
)


@dataclass(frozen=True)
class QuetzalPackageMount:
    host_root: Path
    runtime_root: Path


def _is_quetzal(model_spec) -> bool:
    return (
        getattr(getattr(model_spec, "impl", None), "impl_id", None) == QUETZAL_IMPL_ID
    )


def resolve_quetzal_package_mount(
    model_spec, runtime_config
) -> Optional[QuetzalPackageMount]:
    """Validate and resolve a host package and its catalog runtime location."""
    configured = getattr(runtime_config, "quetzal_models_root", None)
    is_quetzal = _is_quetzal(model_spec)
    launches_server = bool(
        getattr(runtime_config, "docker_server", False)
        or getattr(runtime_config, "local_server", False)
    )

    if configured and not is_quetzal:
        raise ValueError("--quetzal-models-root is only valid with --impl quetzal")
    if configured and not launches_server:
        raise ValueError(
            "--quetzal-models-root requires --docker-server or --local-server"
        )
    if is_quetzal and launches_server and not configured:
        raise ValueError("--impl quetzal requires --quetzal-models-root")
    if not configured:
        return None

    supplied = Path(configured).expanduser()
    if supplied.is_symlink() or not supplied.is_dir():
        raise ValueError(
            f"--quetzal-models-root must be an existing real directory: {supplied}"
        )
    host_root = supplied.resolve()

    env_vars = model_spec.env_vars
    runtime_root_value = env_vars.get(QUETZAL_PACKAGE_ROOT_ENV)
    if not runtime_root_value:
        raise ValueError(
            "impl=quetzal model spec must define an absolute QUETZAL_PACKAGE_ROOT"
        )
    runtime_root = Path(runtime_root_value)
    if not runtime_root.is_absolute() or runtime_root == Path("/"):
        raise ValueError(
            "impl=quetzal model spec must define a non-root absolute "
            f"QUETZAL_PACKAGE_ROOT: {runtime_root_value}"
        )

    expected_sha256 = env_vars.get(QUETZAL_MANIFEST_SHA256_ENV, "")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError(
            "impl=quetzal model spec must define "
            "QUETZAL_BUNDLE_MANIFEST_SHA256 as a lowercase SHA-256"
        )
    portable_manifest = host_root / "manifest.json"
    installed_proof = (
        host_root / QUETZAL_INSTALLED_MANIFEST_DIR / f"{expected_sha256}.json"
    )
    has_portable_manifest = (
        not portable_manifest.is_symlink() and portable_manifest.is_file()
    )
    has_installed_proof = not installed_proof.is_symlink() and installed_proof.is_file()
    if not (has_portable_manifest or has_installed_proof):
        raise ValueError(
            "--quetzal-models-root does not contain the catalog-pinned portable "
            f"manifest or installed proof: {host_root}"
        )

    return QuetzalPackageMount(host_root=host_root, runtime_root=runtime_root)


def rebase_quetzal_package_env(
    env_vars: Dict[str, str], runtime_root: Path, replacement_root: Path
) -> Dict[str, str]:
    """Rebase catalog package paths without changing non-path environment values."""
    rebased = dict(env_vars)
    for name in QUETZAL_PACKAGE_PATH_ENV_VARS:
        value = env_vars.get(name)
        if not value:
            continue
        try:
            relative = Path(value).relative_to(runtime_root)
        except ValueError as error:
            raise ValueError(
                f"impl=quetzal {name} must be inside QUETZAL_PACKAGE_ROOT: {value}"
            ) from error
        rebased[name] = str(replacement_root / relative)
    return rebased
