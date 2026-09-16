# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Host-to-container mount contract for an immutable Quetzal package."""

from __future__ import annotations

import hashlib
import json
import re
import stat
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Optional

QUETZAL_IMPL_ID = "quetzal"
QUETZAL_PACKAGE_ROOT_ENV = "QUETZAL_PACKAGE_ROOT"
QUETZAL_MANIFEST_SHA256_ENV = "QUETZAL_BUNDLE_MANIFEST_SHA256"
QUETZAL_AUXILIARY_ROOTS_ENV = "QUETZAL_AUXILIARY_ROOTS_JSON"
QUETZAL_AUXILIARY_RUNTIME_ROOT = Path(
    "/home/container_app_user/quetzal/auxiliary"
)
MAX_MANIFEST_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class QuetzalAuxiliaryMount:
    name: str
    host_root: Path
    runtime_root: Path


@dataclass(frozen=True)
class QuetzalPackageMount:
    host_root: Path
    runtime_root: Path
    manifest_sha256: str
    auxiliary: tuple[QuetzalAuxiliaryMount, ...] = ()


def _safe_auxiliary_name(value: object) -> str:
    """Accept Quetzal's canonical single-segment names, not a smaller regex.

    Package admission already treats the manifest as untrusted input. Keep the
    same portable-path contract as Quetzal's ``_safe_name`` while explicitly
    rejecting control characters before a name reaches Docker's mount syntax.
    """
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or any(unicodedata.category(character) == "Cc" for character in value)
    ):
        raise ValueError(f"Quetzal auxiliary reference has unsafe name: {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or len(path.parts) != 1
        or path.parts[0] in (".", "..")
        or path.as_posix() != value
    ):
        raise ValueError(f"Quetzal auxiliary reference has unsafe name: {value!r}")
    return value


def _manifest_auxiliary_mounts(
    host_root: Path, env_vars: dict[str, object], expected_sha256: str
) -> tuple[QuetzalAuxiliaryMount, ...]:
    manifest_path = host_root / "manifest.json"
    configured = env_vars.get(QUETZAL_AUXILIARY_ROOTS_ENV)
    if not manifest_path.exists() and not configured:
        return ()
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(
            "Quetzal auxiliary mounts require a regular package manifest: "
            f"{manifest_path}"
        )
    size = manifest_path.stat(follow_symlinks=False).st_size
    if size > MAX_MANIFEST_BYTES:
        raise ValueError(f"Quetzal package manifest exceeds {MAX_MANIFEST_BYTES} bytes")
    raw = manifest_path.read_bytes()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "Quetzal package manifest SHA-256 differs from the catalog: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    try:
        manifest = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Quetzal package manifest is invalid JSON: {error}") from error
    references = manifest.get("auxiliary_references") if isinstance(manifest, dict) else None
    if references is None:
        if configured:
            raise ValueError(
                "QUETZAL_AUXILIARY_ROOTS_JSON is set for a package with no "
                "auxiliary references"
            )
        return ()
    if not isinstance(references, list) or not references:
        raise ValueError("Quetzal V2 auxiliary_references must be a non-empty list")
    if not isinstance(configured, str) or not configured.strip():
        raise ValueError(
            "Quetzal V2 package requires generated QUETZAL_AUXILIARY_ROOTS_JSON"
        )
    try:
        runtime_roots = json.loads(configured)
    except json.JSONDecodeError as error:
        raise ValueError("QUETZAL_AUXILIARY_ROOTS_JSON is invalid JSON") from error
    if not isinstance(runtime_roots, dict):
        raise ValueError("QUETZAL_AUXILIARY_ROOTS_JSON must be a name-to-path map")

    expected_names = set()
    mounts = []
    for reference in references:
        if not isinstance(reference, dict):
            raise ValueError("Quetzal auxiliary reference must be a mapping")
        name = _safe_auxiliary_name(reference.get("name"))
        digest = reference.get("sha256")
        if name in expected_names:
            raise ValueError(f"Quetzal auxiliary reference name is duplicated: {name}")
        expected_names.add(name)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Quetzal auxiliary reference {name} has invalid SHA-256")
        runtime_value = runtime_roots.get(name)
        expected_runtime = (
            QUETZAL_AUXILIARY_RUNTIME_ROOT / name / f"sha256-{digest}"
        )
        if runtime_value != str(expected_runtime):
            raise ValueError(
                f"Quetzal auxiliary runtime path for {name} must be "
                f"{expected_runtime}, got {runtime_value!r}"
            )
        host_path = host_root.parent / "auxiliary" / name / f"sha256-{digest}"
        if host_path.is_symlink() or not host_path.is_dir():
            raise ValueError(
                f"Quetzal auxiliary root {name} is absent or unpublished: {host_path}"
            )
        mode = host_path.stat(follow_symlinks=False).st_mode
        if not stat.S_ISDIR(mode) or mode & 0o222:
            raise ValueError(
                f"Quetzal auxiliary root {name} must be a read-only real directory: "
                f"{host_path}"
            )
        mounts.append(
            QuetzalAuxiliaryMount(
                name=name,
                host_root=host_path.resolve(strict=True),
                runtime_root=expected_runtime,
            )
        )
    if set(runtime_roots) != expected_names:
        raise ValueError(
            "QUETZAL_AUXILIARY_ROOTS_JSON names differ from the package manifest"
        )
    return tuple(mounts)


def _is_quetzal(model_spec) -> bool:
    return (
        getattr(getattr(model_spec, "impl", None), "impl_id", None) == QUETZAL_IMPL_ID
    )


def resolve_quetzal_package_mount(
    model_spec, runtime_config
) -> Optional[QuetzalPackageMount]:
    """Validate and resolve a host package and its catalog runtime location."""
    configured = getattr(runtime_config, "quetzal_package_root", None)
    is_quetzal = _is_quetzal(model_spec)
    launches_server = bool(
        getattr(runtime_config, "docker_server", False)
        or getattr(runtime_config, "local_server", False)
    )

    if configured and not is_quetzal:
        raise ValueError("--quetzal-package-root is only valid with --impl quetzal")
    if configured and not launches_server:
        raise ValueError(
            "--quetzal-package-root requires --docker-server or --local-server"
        )
    if is_quetzal and launches_server and not configured:
        raise ValueError("--impl quetzal requires --quetzal-package-root")
    if not configured:
        return None

    supplied = Path(configured).expanduser()
    if supplied.is_symlink() or not supplied.is_dir():
        raise ValueError(
            f"--quetzal-package-root must be an existing real directory: {supplied}"
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
    auxiliary = _manifest_auxiliary_mounts(
        host_root, env_vars, expected_sha256
    )
    return QuetzalPackageMount(
        host_root=host_root,
        runtime_root=runtime_root,
        manifest_sha256=expected_sha256,
        auxiliary=auxiliary,
    )


def quetzal_package_env(
    package: QuetzalPackageMount, *, local_server: bool
) -> dict[str, str]:
    """Return only the two values that select the admitted package."""
    package_root = package.host_root if local_server else package.runtime_root
    result = {
        QUETZAL_PACKAGE_ROOT_ENV: str(package_root),
        QUETZAL_MANIFEST_SHA256_ENV: package.manifest_sha256,
    }
    if package.auxiliary:
        result[QUETZAL_AUXILIARY_ROOTS_ENV] = json.dumps(
            {
                mount.name: str(
                    mount.host_root if local_server else mount.runtime_root
                )
                for mount in package.auxiliary
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    return result
