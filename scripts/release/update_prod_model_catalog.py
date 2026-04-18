#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""
Update models-catalog-prod.json with model definitions from models-catalog-dev.json
for models that have a release configuration in models-ci-config.json.

Static input files (paths relative to repo root):
  - models-catalog-dev.json          source catalog
  - models-catalog-prod.json         target catalog (updated in-place)
  - .github/workflows/models-ci-config.json  release model filter

Usage:
    python3 scripts/release/update_prod_model_catalog.py
    python3 scripts/release/update_prod_model_catalog.py --dry-run
    python3 scripts/release/update_prod_model_catalog.py --test
"""

import argparse
import json
import sys
from pathlib import Path

# Paths are relative to the repository root (where the script is invoked from)
DEV_CATALOG_PATH = ".github/workflows/models-catalog-dev.json"
PROD_CATALOG_PATH = ".github/workflows/models-catalog-prod.json"
CI_CONFIG_PATH = ".github/workflows/models-ci-config.json"


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def save_json(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def strip_org_prefix(key: str) -> str:
    """Return the model name portion of a key, stripping any org/ prefix."""
    return key.split("/")[-1] if "/" in key else key


def build_dev_index(model_specs: dict) -> dict:
    """Build a lookup from bare model name -> full models-catalog-dev.json key."""
    index = {}
    for key in model_specs:
        bare = strip_org_prefix(key)
        # Lower-case for case-insensitive matching
        index[bare.lower()] = key
    return index


def find_dev_key(model_name: str, dev_index: dict) -> str | None:
    """Find the models-catalog-dev.json key for a models-ci-config model name."""
    return dev_index.get(model_name.lower())


def run(dry_run: bool = False) -> None:
    dev_data = load_json(DEV_CATALOG_PATH)
    prod_data = load_json(PROD_CATALOG_PATH)
    ci_config = load_json(CI_CONFIG_PATH)

    dev_specs = dev_data["model_specs"]
    prod_specs = prod_data["model_specs"]
    ci_models = ci_config["models"]

    dev_index = build_dev_index(dev_specs)

    # ------------------------------------------------------------------ #
    # Phase 1 — classify every model in models-ci-config.json             #
    # ------------------------------------------------------------------ #
    skipped = []       # no release config
    to_update = []     # in models-catalog-prod.json → will be replaced
    to_add = []        # NOT in models-catalog-prod.json → will be appended

    print("=" * 70)
    print("PHASE 1 — Analysing models-ci-config.json")
    print("=" * 70)

    for model_name, model_cfg in ci_models.items():
        ci_block = model_cfg.get("ci", {})

        if "release" not in ci_block:
            skipped.append(model_name)
            continue

        release_devices = ci_block["release"].get("devices", [])
        dev_key = find_dev_key(model_name, dev_index)

        if dev_key is None:
            print(
                f"  [WARNING] {model_name} has release config but no match "
                f"in models-catalog-dev.json — skipping"
            )
            skipped.append(model_name)
            continue

        # Report per-device match for devices that exist in models-catalog-dev.json entry
        dev_entry = dev_specs[dev_key]
        matched_devices = []
        for device in release_devices:
            if device in dev_entry:
                print(
                    f"  [MATCH] {model_name}/ci/release/devices/{device} "
                    f"-> {dev_key}/{device}"
                )
                matched_devices.append(device)
            else:
                print(
                    f"  [MISSING] {model_name}/ci/release/devices/{device} "
                    f"not found in models-catalog-dev.json entry {dev_key}"
                )

        if dev_key in prod_specs:
            to_update.append((model_name, dev_key))
        else:
            to_add.append((model_name, dev_key))

    # ------------------------------------------------------------------ #
    # Phase 2 — summary before making changes                             #
    # ------------------------------------------------------------------ #
    print()
    print("=" * 70)
    print("PHASE 2 — Changes to be applied to models-catalog-prod.json")
    print("=" * 70)

    print(f"\nModels to UPDATE ({len(to_update)}):")
    for model_name, dev_key in to_update:
        print(f"  {model_name}  [{dev_key}]")

    print(f"\nModels to ADD at end ({len(to_add)}):")
    for model_name, dev_key in to_add:
        print(f"  {model_name}  [{dev_key}]")

    print(
        f"\nModels SKIPPED (no release config or no models-catalog-dev.json match) ({len(skipped)}):"
    )
    for model_name in skipped:
        print(f"  {model_name}")

    if dry_run:
        print("\n[DRY RUN] No changes written to models-catalog-prod.json.")
        return

    # ------------------------------------------------------------------ #
    # Phase 3 — apply changes                                             #
    # ------------------------------------------------------------------ #
    print()
    print("=" * 70)
    print("PHASE 3 — Applying changes")
    print("=" * 70)

    for model_name, dev_key in to_update:
        prod_specs[dev_key] = dev_specs[dev_key]
        print(f"  UPDATED  {dev_key}")

    for model_name, dev_key in to_add:
        prod_specs[dev_key] = dev_specs[dev_key]
        print(f"  ADDED    {dev_key}")

    save_json(PROD_CATALOG_PATH, prod_data)
    print(
        f"\nmodels-catalog-prod.json written to {PROD_CATALOG_PATH}"
        f"  ({len(to_update)} updated, {len(to_add)} added)"
    )


# ------------------------------------------------------------------ #
# Built-in tests                                                       #
# ------------------------------------------------------------------ #
def run_tests() -> None:
    dev_data = load_json(DEV_CATALOG_PATH)
    prod_data = load_json(PROD_CATALOG_PATH)
    ci_config = load_json(CI_CONFIG_PATH)

    dev_specs = dev_data["model_specs"]
    prod_specs = prod_data["model_specs"]
    ci_models = ci_config["models"]
    dev_index = build_dev_index(dev_specs)

    passed = 0
    failed = 0

    def check(label: str, condition: bool) -> None:
        nonlocal passed, failed
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {label}")
        if condition:
            passed += 1
        else:
            failed += 1

    print()
    print("=" * 70)
    print("TESTS")
    print("=" * 70)

    # -------------------------------------------------------------- #
    # Test 1 — a model WITHOUT release config should be skipped       #
    # -------------------------------------------------------------- #
    print("\nTest 1 — Models without release config (should be skipped)")
    nightly_only = [
        name for name, cfg in ci_models.items()
        if "ci" in cfg and "release" not in cfg["ci"]
    ]
    print(f"  Found {len(nightly_only)} nightly-only models:")
    for name in nightly_only[:5]:
        print(f"    {name}")
    if len(nightly_only) > 5:
        print(f"    ... and {len(nightly_only) - 5} more")
    check(
        "At least one nightly-only model exists and has no release key",
        len(nightly_only) > 0 and all(
            "release" not in ci_models[n].get("ci", {})
            for n in nightly_only
        ),
    )
    check(
        "DeepSeek-R1-0528 is nightly-only",
        "DeepSeek-R1-0528" in nightly_only,
    )

    # -------------------------------------------------------------- #
    # Test 2 — a model WITH release config is matched in dev catalog  #
    # -------------------------------------------------------------- #
    print("\nTest 2 — Model with release config matched in models-catalog-dev.json (whisper-large-v3)")
    whisper_cfg = ci_models.get("whisper-large-v3", {})
    whisper_has_release = "release" in whisper_cfg.get("ci", {})
    whisper_dev_key = find_dev_key("whisper-large-v3", dev_index)
    whisper_in_prod = whisper_dev_key in prod_specs if whisper_dev_key else False
    check("whisper-large-v3 has release config", whisper_has_release)
    check(
        f"whisper-large-v3 matched in models-catalog-dev.json as '{whisper_dev_key}'",
        whisper_dev_key is not None,
    )
    check("whisper-large-v3 exists in models-catalog-prod.json (will be updated)", whisper_in_prod)
    if whisper_dev_key and whisper_has_release:
        for device in whisper_cfg["ci"]["release"].get("devices", []):
            in_dev = device in dev_specs.get(whisper_dev_key, {})
            check(f"  device {device} present in models-catalog-dev.json entry", in_dev)

    # -------------------------------------------------------------- #
    # Test 3 — Qwen3-32B missing from prod catalog and will be added  #
    # -------------------------------------------------------------- #
    print("\nTest 3 — Qwen3-32B missing from models-catalog-prod.json (should be added at end)")
    qwen_cfg = ci_models.get("Qwen3-32B", {})
    qwen_has_release = "release" in qwen_cfg.get("ci", {})
    qwen_dev_key = find_dev_key("Qwen3-32B", dev_index)
    qwen_in_prod = qwen_dev_key in prod_specs if qwen_dev_key else False
    qwen_in_dev = qwen_dev_key in dev_specs if qwen_dev_key else False
    check("Qwen3-32B has release config", qwen_has_release)
    check(
        f"Qwen3-32B matched in models-catalog-dev.json as '{qwen_dev_key}'",
        qwen_dev_key is not None,
    )
    check("Qwen3-32B is NOT in models-catalog-prod.json (will be added)", not qwen_in_prod)
    check("Qwen3-32B IS in models-catalog-dev.json (source for add)", qwen_in_dev)

    print(f"\nTests complete: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Update models-catalog-prod.json with model specs from "
            "models-catalog-dev.json for release models defined in "
            "models-ci-config.json."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes without writing models-catalog-prod.json",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run built-in validation tests and exit (read-only)",
    )
    args = parser.parse_args()

    for p in [DEV_CATALOG_PATH, PROD_CATALOG_PATH, CI_CONFIG_PATH]:
        if not Path(p).exists():
            print(f"Error: file not found: {p}", file=sys.stderr)
            sys.exit(1)

    if args.test:
        run_tests()
    else:
        run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
