# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Install the immutable Python-only Gemma prefix candidate in a pinned image."""

import hashlib
import json
import logging
import subprocess
from pathlib import Path


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
            raise RuntimeError(
                f"Gemma prefix overlay refuses unexpected model source: {name}"
            )
    if not all(
        current[name] == expected["target"]
        for name, expected in manifest["files"].items()
    ):
        if not all(
            current[name] == expected["base"]
            for name, expected in manifest["files"].items()
        ):
            raise RuntimeError(
                "Gemma prefix overlay found partially patched model source"
            )
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=repo, check=True)
        subprocess.run(["git", "apply", str(patch)], cwd=repo, check=True)
    for name, expected in manifest["files"].items():
        if digest(repo / name) != expected["target"]:
            raise RuntimeError(f"Gemma prefix installed source digest mismatch: {name}")
    logging.getLogger(__name__).info(
        "%s %s", marker, json.dumps(manifest, sort_keys=True)
    )


def install_prefix_overlay(metal):
    # A complete later overlay is an accepted final stack state. Re-running
    # the older installer on those files would reject its own successor.
    bundle = Path(__file__).with_name("gemma_prefix_overlay")
    later = json.loads((bundle / "serial-coordinates-manifest.json").read_text())
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    installed = all(digest(metal / name) == value["target"] for name, value in later["files"].items())
    if installed:
        earlier = json.loads((bundle / "batched-prefill-manifest.json").read_text())
        for name, value in earlier["files"].items():
            if name not in later["files"] and digest(metal / name) != value["target"]:
                raise RuntimeError("Gemma earlier overlay member changed: " + name)
    else:
        # Partial/unknown later states are rejected by the original strict gate.
        _install(metal, "batched-prefill-manifest.json", "batched-prefill.patch", "GEMMA_BATCHED_PREFILL_MODEL")
    _install(metal, "serial-coordinates-manifest.json", "serial-coordinates.patch", "GEMMA_SERIAL_COORDINATES_MODEL")
    _install(
        metal.parent / "vllm-tt-plugin",
        "plugin-manifest.json",
        "plugin.patch",
        "GEMMA_PREFIX_PLUGIN",
    )
