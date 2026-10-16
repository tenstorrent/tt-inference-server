# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Install the immutable Python-only Gemma prefix candidate in a pinned image."""
import hashlib
import json
import logging
from pathlib import Path
import subprocess


def _install(repo, manifest_name, patch_name, marker):
    bundle = Path(__file__).with_name("gemma_prefix_overlay")
    manifest = json.loads((bundle / manifest_name).read_text())
    patch = bundle / patch_name
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    if digest(patch) != manifest["patch_sha256"]:
        raise RuntimeError("Gemma prefix patch digest mismatch")
    current = {name: digest(repo / name) for name in manifest["files"]}
    for name, expected in manifest["files"].items():
        if current[name] not in (expected["base"], expected["target"]):
            raise RuntimeError(f"Gemma prefix overlay refuses unexpected model source: {name}")
    if not all(current[name] == expected["target"] for name, expected in manifest["files"].items()):
        if not all(current[name] == expected["base"] for name, expected in manifest["files"].items()):
            raise RuntimeError("Gemma prefix overlay found partially patched model source")
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=repo, check=True)
        subprocess.run(["git", "apply", str(patch)], cwd=repo, check=True)
    for name, expected in manifest["files"].items():
        if digest(repo / name) != expected["target"]:
            raise RuntimeError(f"Gemma prefix installed source digest mismatch: {name}")
    logging.getLogger(__name__).info("%s %s", marker, json.dumps(manifest, sort_keys=True))


def install_prefix_overlay(metal):
    _install(metal, "manifest.json", "model.patch", "GEMMA_PREFIX_MODEL")
    _install(metal, "safety-manifest.json", "safety.patch", "GEMMA_HISTORY_LIFETIME_REPAIR")
    _install(metal, "memory-manifest.json", "memory.patch", "GEMMA_COMBINED_MEMORY")
    _install(metal.parent / "vllm-tt-plugin", "plugin-manifest.json", "plugin.patch", "GEMMA_PREFIX_PLUGIN")
