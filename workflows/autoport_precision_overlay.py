# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Explicit dev-only precision-policy overlays for immutable autoport images."""

import hashlib
import json
import logging
import re
from pathlib import Path, PurePosixPath

logger = logging.getLogger("run_log")


def precision_overlay_args(model_spec, runtime_config, repo_root):
    relative = (getattr(model_spec, "metadata", None) or {}).get(
        "autoport_precision_config"
    )
    if relative is None:
        return []
    if not runtime_config.dev_mode:
        raise ValueError(
            "autoport_precision_config is restricted to explicit dev-mode experiments"
        )
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError(
            "Precision overlay must name a repository-relative configuration"
        )
    code_path = PurePosixPath(model_spec.impl.code_path)
    if (
        len(code_path.parts) != 3
        or code_path.parts[:2] != ("models", "autoports")
        or not re.fullmatch(r"[a-z0-9_]+", code_path.parts[2])
    ):
        raise ValueError(
            "Precision overlay requires one models/autoports implementation"
        )
    source = (Path(repo_root) / relative).resolve()
    allowed = (Path(repo_root) / "reference_config").resolve()
    if (
        not source.is_relative_to(allowed)
        or not source.is_file()
        or any(c in str(source) for c in ",\n\r")
    ):
        raise ValueError(
            "Precision overlay must be a regular file inside reference_config"
        )
    raw = source.read_bytes()
    policy = json.loads(raw)
    if (
        not isinstance(policy, dict)
        or type(policy.get("schema_version")) is not int
        or policy["schema_version"] != 1
        or not isinstance(policy.get("config_id"), str)
        or not policy["config_id"]
    ):
        raise ValueError("Precision overlay requires a named schema_version=1 policy")
    destination = f"/home/container_app_user/tt-metal/{code_path}/doc/datatype_sweep/selected_precision_config.json"
    logger.info(
        "Experimental precision overlay config_id=%s sha256=%s target=%s",
        policy["config_id"],
        hashlib.sha256(raw).hexdigest(),
        destination,
    )
    return ["--mount", f"type=bind,src={source},dst={destination},readonly"]
