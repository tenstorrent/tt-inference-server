# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Host-to-container mount contract for an immutable Quetzal package."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

QUETZAL_IMPL_ID = "quetzal"
QUETZAL_PACKAGE_ROOT_ENV = "QUETZAL_PACKAGE_ROOT"
QUETZAL_MANIFEST_SHA256_ENV = "QUETZAL_BUNDLE_MANIFEST_SHA256"


@dataclass(frozen=True)
class QuetzalPackageMount:
    host_root: Path
    runtime_root: Path
    manifest_sha256: str


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
    return QuetzalPackageMount(
        host_root=host_root,
        runtime_root=runtime_root,
        manifest_sha256=expected_sha256,
    )


def quetzal_package_env(
    package: QuetzalPackageMount, *, local_server: bool
) -> dict[str, str]:
    """Return only the two values that select the admitted package."""
    package_root = package.host_root if local_server else package.runtime_root
    return {
        QUETZAL_PACKAGE_ROOT_ENV: str(package_root),
        QUETZAL_MANIFEST_SHA256_ENV: package.manifest_sha256,
    }
