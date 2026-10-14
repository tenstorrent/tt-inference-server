# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Explicit dev-only file overlays for immutable tt-metal source images."""

import hashlib
import logging
import re
from pathlib import Path, PurePosixPath

logger = logging.getLogger("run_log")


def tt_metal_file_overlay_args(model_spec, runtime_config, repo_root):
    overlays = (getattr(model_spec, "metadata", None) or {}).get(
        "tt_metal_file_overlays"
    )
    if overlays is None:
        return []
    if not runtime_config.dev_mode or not isinstance(overlays, list) or not overlays:
        raise ValueError("tt-metal file overlays require explicit dev mode")

    allowed_sources = (Path(repo_root) / "reference_config").resolve()
    args = []
    for overlay in overlays:
        if not isinstance(overlay, dict) or set(overlay) != {
            "source",
            "target",
            "sha256",
        }:
            raise ValueError("Invalid tt-metal file overlay declaration")
        source_name = overlay["source"]
        target_name = overlay["target"]
        expected_digest = overlay["sha256"]
        if not all(isinstance(value, str) for value in overlay.values()):
            raise ValueError("tt-metal file overlay fields must be strings")
        if Path(source_name).is_absolute():
            raise ValueError("tt-metal file overlay source must be repository-relative")

        source = (Path(repo_root) / source_name).resolve()
        target = PurePosixPath(target_name)
        if (
            not source.is_relative_to(allowed_sources)
            or not source.is_file()
            or any(char in str(source) for char in ",\n\r")
        ):
            raise ValueError("Overlay source must be a file inside reference_config")
        if (
            target.is_absolute()
            or ".." in target.parts
            or not re.fullmatch(
                r"models/(?:autoports|demos)/[a-z0-9_]+/(?:config/precision\.json|doc/datatype_sweep/selected_precision_config\.json)",
                target.as_posix(),
            )
        ):
            raise ValueError("Overlay target is not an allowlisted precision policy")

        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != expected_digest:
            raise ValueError("tt-metal file overlay hash mismatch")
        destination = f"/home/container_app_user/tt-metal/{target.as_posix()}"
        logger.info(
            "Experimental tt-metal file overlay sha256=%s target=%s",
            digest,
            destination,
        )
        args.extend(
            ["--mount", f"type=bind,src={source},dst={destination},readonly"]
        )
    return args
