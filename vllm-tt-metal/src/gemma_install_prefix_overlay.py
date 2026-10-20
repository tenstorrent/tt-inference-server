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
    _install(metal, "serial-serving-hook-manifest.json", "serial-serving-hook.patch", "GEMMA_SERIAL_SERVING_HOOK_MODEL")
    _install(
        metal.parent / "vllm-tt-plugin",
        "plugin-manifest.json",
        "plugin.patch",
        "GEMMA_PREFIX_PLUGIN",
    )


def install_anchor_overlay(root):
    """Fail closed before vLLM imports; source copied into ephemeral image only."""
    import sys
    bundle = Path(__file__).with_name("gemma_prefix_overlay") / "anchor"
    manifest = json.loads((bundle / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        module = "vllm." + name[:-3].replace("/", ".")
        if module in sys.modules:
            raise RuntimeError("Anchor overlay must precede vLLM import: " + module)
        source = bundle / Path(name).name
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected["target"]:
            raise RuntimeError("Anchor bundle corruption: " + name)
    current = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
               for name in manifest["files"]}
    if not (all(current[n] == v["base"] for n,v in manifest["files"].items())
            or all(current[n] == v["target"] for n,v in manifest["files"].items())):
        raise RuntimeError("Anchor overlay refuses unexpected or mixed core sources: " + json.dumps(current))
    for name, expected in manifest["files"].items():
        destination = root / name
        temporary = destination.with_suffix(".anchor-tmp")
        temporary.write_bytes((bundle / Path(name).name).read_bytes())
        temporary.replace(destination)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != expected["target"]:
            raise RuntimeError("Anchor installed source mismatch: " + name)
    logging.getLogger(__name__).info("GEMMA_SW_ANCHOR_SOURCE %s", json.dumps(manifest, sort_keys=True))
    subprocess.run([sys.executable, str(bundle / "real_pool_check.py")], check=True, timeout=90)
