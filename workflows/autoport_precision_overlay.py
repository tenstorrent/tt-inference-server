# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Explicit dev-only precision-policy overlays for immutable autoport images."""

import hashlib
import json
import logging
import re
from pathlib import Path, PurePosixPath

logger = logging.getLogger("run_log")


def pinned_hf_cache_args(model_spec):
    """Set cache locations before Python imports Hugging Face's constants."""
    if not (getattr(model_spec, "metadata", None) or {}).get(
        "autoport_pinned_hf_cache"
    ):
        return []
    if not str(model_spec.impl.code_path).startswith("models/autoports/"):
        raise ValueError("Pinned autoport cache requires an autoport implementation")
    return [
        "--env",
        "HF_HOME=/home/container_app_user/cache_root/huggingface",
        "--env",
        "HF_HUB_CACHE=/home/container_app_user/cache_root/huggingface/hub",
    ]


def source_overlay_args(model_spec, runtime_config, repo_root):
    """Mount explicitly attested Python files; native image contents stay unchanged."""
    overlay = (getattr(model_spec, "metadata", None) or {}).get(
        "autoport_source_overlay"
    )
    if overlay is None:
        return []
    if not runtime_config.dev_mode or not isinstance(overlay, dict):
        raise ValueError("Source overlays require explicit dev-mode metadata")
    revision, files = overlay.get("tt_metal_revision"), overlay.get("files")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Source overlay requires an immutable tt-metal revision")
    code = str(model_spec.impl.code_path)
    if not re.fullmatch(r"models/autoports/[a-z0-9_]+", code):
        raise ValueError("Source overlay requires one autoport implementation")
    allowed_targets = {"tt/precision_policy.py", "tt/multichip_decoder.py"}
    if not isinstance(files, dict) or set(files) != allowed_targets:
        raise ValueError(
            "Source overlay requires exactly the two prefill-policy Python files"
        )
    allowed = (Path(repo_root) / "reference_config/autoport_sources").resolve()
    args = []
    for target, entry in sorted(files.items()):
        if not isinstance(entry, dict) or not isinstance(entry.get("source"), str):
            raise ValueError("Invalid source overlay entry")
        source = (Path(repo_root) / entry["source"]).resolve()
        if (
            not source.is_relative_to(allowed)
            or not source.is_file()
            or any(c in str(source) for c in ",\n\r")
        ):
            raise ValueError(
                "Source overlay must be a regular file inside reference_config/autoport_sources"
            )
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != entry.get("sha256"):
            raise ValueError("Source overlay hash mismatch")
        destination = f"/home/container_app_user/tt-metal/{code}/{target}"
        logger.info(
            "Experimental Python overlay tt_metal_revision=%s sha256=%s target=%s",
            revision,
            digest,
            destination,
        )
        args.extend(["--mount", f"type=bind,src={source},dst={destination},readonly"])
    return args


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
        or policy["schema_version"] not in (1, 2)
        or not isinstance(policy.get("config_id"), str)
        or not policy["config_id"]
    ):
        raise ValueError("Precision overlay requires a named supported policy")
    if policy["schema_version"] == 2 and not source_overlay_args(
        model_spec, runtime_config, repo_root
    ):
        raise ValueError(
            "Schema-2 prefill policy requires attested Python source overlays"
        )
    destination = f"/home/container_app_user/tt-metal/{code_path}/doc/datatype_sweep/selected_precision_config.json"
    logger.info(
        "Experimental precision overlay config_id=%s sha256=%s target=%s",
        policy["config_id"],
        hashlib.sha256(raw).hexdigest(),
        destination,
    )
    return ["--mount", f"type=bind,src={source},dst={destination},readonly"]
