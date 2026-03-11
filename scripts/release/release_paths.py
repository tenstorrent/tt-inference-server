#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from pathlib import Path
from typing import Optional, Union
import sys

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from workflows.utils import get_version


PathLike = Union[str, Path]

DEFAULT_RELEASE_LOG_ROOT = Path("release_logs")


def get_release_version_tag(version: Optional[str] = None) -> str:
    """Return the `vX.Y.Z` tag used for versioned release output directories."""
    resolved_version = version or get_version()
    return f"v{resolved_version}"


def get_versioned_release_logs_dir(
    version: Optional[str] = None, base_dir: PathLike = DEFAULT_RELEASE_LOG_ROOT
) -> Path:
    """Return the default versioned release log directory."""
    return Path(base_dir) / get_release_version_tag(version)


def resolve_release_output_dir(
    output_dir: Optional[PathLike] = None,
    version: Optional[str] = None,
    base_dir: PathLike = DEFAULT_RELEASE_LOG_ROOT,
) -> Path:
    """Resolve an explicit output path or the default versioned release log directory."""
    if output_dir:
        return Path(output_dir).resolve()
    return get_versioned_release_logs_dir(version=version, base_dir=base_dir).resolve()
