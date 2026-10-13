# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Install the immutable Python-only Gemma prefix candidate in a pinned image."""
import hashlib
import json
import logging
from pathlib import Path
import subprocess


def install_prefix_overlay(metal):
    bundle = Path(__file__).with_name("gemma_prefix_overlay")
    manifest = json.loads((bundle / "manifest.json").read_text())
    patch = bundle / "model.patch"
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if digest(patch) != manifest["patch_sha256"]:
        raise RuntimeError("Gemma prefix patch digest mismatch")
    current = {name: digest(metal / name) for name in manifest["files"]}
    for name, expected in manifest["files"].items():
        if current[name] not in (expected["base"], expected["target"]):
            raise RuntimeError(f"Gemma prefix overlay refuses unexpected model source: {name}")
    if not all(current[name] == expected["target"] for name, expected in manifest["files"].items()):
        if not all(current[name] == expected["base"] for name, expected in manifest["files"].items()):
            raise RuntimeError("Gemma prefix overlay found partially patched model source")
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=metal, check=True)
        subprocess.run(["git", "apply", str(patch)], cwd=metal, check=True)
    for name, expected in manifest["files"].items():
        if digest(metal / name) != expected["target"]:
            raise RuntimeError(f"Gemma prefix installed source digest mismatch: {name}")
    logging.getLogger(__name__).info("GEMMA_PREFIX_MODEL %s", json.dumps(manifest, sort_keys=True))
