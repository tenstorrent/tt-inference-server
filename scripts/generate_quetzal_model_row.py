#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
"""Generate a Quetzal serve row for workflows/model_specs/<env>/llm.yaml.

The single source of truth for a Quetzal serving profile is the catalog fragment
shipped by the Quetzal package at ``serving/catalog/dev_llm_quetzal.yaml``. Every
``QZ_*`` / ``QUETZAL_*`` path in a row is derived from one content-addressed
package id, so the rows must never be edited by hand (a stale copy of the id in
one path is the classic Quetzal mis-serve, documented in AUDIT.md).

This tool loads a model's profile from the catalog fragment, re-points the
content-addressed package paths at a given package id, and prints the resulting
ttis ``llm.yaml`` row. It is deliberately mechanical: it does not invent env
vars, only rewrites the ones the fragment already declares.

Example
-------
  python3 scripts/generate_quetzal_model_row.py \
      --catalog /path/to/tt-quetzalcoatlus/serving/catalog/dev_llm_quetzal.yaml \
      --model meta-llama/Llama-3.2-1B-Instruct \
      --package-id sha256-<tree>-<manifest> \
      --manifest-sha256 <hex> \
      >> workflows/model_specs/dev/llm.yaml
"""

from __future__ import annotations

import argparse
import re
import sys

import yaml

# Content prefix the catalog bakes into QUETZAL_PACKAGE_ROOT / QZ_* paths for a
# container serve. Matches serving/serve_quetzal.py's CONTAINER_PKG_PREFIX.
CONTAINER_PKG_PREFIX = "/home/container_app_user/cache_root/quetzal/packages"

# A content-addressed Quetzal package id: sha256-<64hex>-<64hex> (v1) or the
# v2 triple form. Used to swap one package id for another inside baked paths.
_PACKAGE_ID_RE = re.compile(
    r"sha256(?:-v2)?-[0-9a-f]{64}-[0-9a-f]{64}(?:-[0-9a-f]{64})?"
)


def _load_profiles(catalog_path: str) -> list[dict]:
    with open(catalog_path) as stream:
        doc = yaml.safe_load(stream)
    # The fragment is a bare YAML list of profile dicts (each a `- weights:` row).
    if isinstance(doc, dict) and "templates" in doc:
        doc = doc["templates"]
    if not isinstance(doc, list):
        raise SystemExit(f"{catalog_path}: expected a list of profiles")
    return [row for row in doc if isinstance(row, dict) and "weights" in row]


def _select_profile(profiles: list[dict], model: str) -> dict:
    """Match by full HF id or by basename (the ttis CLI resolves by basename)."""
    base = model.rsplit("/", 1)[-1]
    matches = [
        row
        for row in profiles
        if any(
            w == model or w.rsplit("/", 1)[-1] == base for w in row.get("weights", [])
        )
    ]
    if not matches:
        available = sorted(w for row in profiles for w in row.get("weights", []))
        raise SystemExit(f"no profile for {model!r} in catalog; available: {available}")
    if len(matches) > 1:
        raise SystemExit(f"ambiguous: {len(matches)} profiles match {model!r}")
    return matches[0]


def _repoint(value, package_id: str):
    """Rewrite any baked package id inside a string to the target package id."""
    if isinstance(value, str) and _PACKAGE_ID_RE.search(value):
        return _PACKAGE_ID_RE.sub(package_id, value)
    return value


def retarget_profile(
    profile: dict,
    package_id: str,
    manifest_sha256: str | None,
    hf_revision: str | None,
) -> dict:
    row = yaml.safe_load(yaml.safe_dump(profile))  # deep copy via round-trip
    for dms in row.get("device_model_specs", []):
        env = dms.get("env_vars", {})
        for key, val in list(env.items()):
            env[key] = _repoint(val, package_id)
        env["QUETZAL_PACKAGE_ID"] = package_id
        env["QUETZAL_PACKAGE_ROOT"] = f"{CONTAINER_PKG_PREFIX}/{package_id}"
        env["QZ_MODELS_ROOT"] = f"{CONTAINER_PKG_PREFIX}/{package_id}"
        if manifest_sha256:
            env["QUETZAL_BUNDLE_MANIFEST_SHA256"] = manifest_sha256
        if hf_revision:
            env["QUETZAL_HF_REVISION"] = hf_revision
            vllm_args = dms.setdefault("vllm_args", {})
            vllm_args["revision"] = hf_revision
            vllm_args["tokenizer_revision"] = hf_revision
    return row


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        required=True,
        help="Path to tt-quetzalcoatlus/serving/catalog/dev_llm_quetzal.yaml",
    )
    parser.add_argument("--model", required=True, help="HF model id or basename")
    parser.add_argument(
        "--package-id",
        required=True,
        help="Content-addressed package id (sha256-<tree>-<manifest>[-<aux>])",
    )
    parser.add_argument(
        "--manifest-sha256",
        default=None,
        help="QUETZAL_BUNDLE_MANIFEST_SHA256 for this package (recommended)",
    )
    parser.add_argument(
        "--hf-revision",
        default=None,
        help="Override the checkpoint revision baked in the profile",
    )
    args = parser.parse_args(argv)

    if not _PACKAGE_ID_RE.fullmatch(args.package_id):
        raise SystemExit(
            f"--package-id is not a content-addressed id: {args.package_id}"
        )

    profiles = _load_profiles(args.catalog)
    profile = _select_profile(profiles, args.model)
    row = retarget_profile(
        profile, args.package_id, args.manifest_sha256, args.hf_revision
    )

    sys.stdout.write(
        "# Generated by scripts/generate_quetzal_model_row.py from the Quetzal "
        "catalog fragment.\n"
        "# Do not edit the package paths by hand; regenerate with a new "
        "--package-id.\n"
    )
    yaml.safe_dump([row], sys.stdout, sort_keys=False, default_flow_style=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
