# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#!/usr/bin/env python3
"""Validate Models CI schema plus implementation-aware job identity.

The matrix generator lives in tenstorrent/tt-shield and consumes this file.
This repository owns the input boundary: entries with the same model and
inference engine must name distinct implementations so the external generator
can produce independent native and generated jobs/artifacts.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

IMMUTABLE_OCI_IMAGE_RE = re.compile(r"^[^\s@:]+(?:/[^\s@:]+)+@sha256:[0-9a-f]{64}$")


def iter_implementations(model_entry: dict):
    yield from model_entry.get("implementations", [model_entry])


def validate_implementation_identities(config: dict) -> list[str]:
    errors: list[str] = []
    for model, model_entry in config.get("models", {}).items():
        implementations = list(iter_implementations(model_entry))
        engines = Counter(
            str(item.get("inference_engine", "")).lower() for item in implementations
        )
        identities: set[tuple[str, str | None]] = set()
        for index, item in enumerate(implementations):
            engine = str(item.get("inference_engine", "")).lower()
            impl = item.get("impl")
            image = item.get("image")
            if engines[engine] > 1 and not impl:
                errors.append(
                    f"{model}.implementations[{index}]: impl is required because "
                    f"inference_engine={item.get('inference_engine')!r} is duplicated"
                )
            identity = (engine, impl)
            if identity in identities:
                errors.append(
                    f"{model}.implementations[{index}]: duplicate Models CI identity "
                    f"(inference_engine={item.get('inference_engine')!r}, impl={impl!r})"
                )
            identities.add(identity)
            if image is not None and not (
                isinstance(image, str) and IMMUTABLE_OCI_IMAGE_RE.fullmatch(image)
            ):
                errors.append(
                    f"{model}.implementations[{index}]: image must be an immutable "
                    "registry/path@sha256:<64 lowercase hex> reference"
                )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(".github/workflows/models-ci-config.json"),
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path(".github/workflows/models-ci-config-schema.json"),
    )
    args = parser.parse_args()

    import jsonschema

    config = json.loads(args.config.read_text())
    schema = json.loads(args.schema.read_text())
    jsonschema.validate(instance=config, schema=schema)
    errors = validate_implementation_identities(config)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("models-ci-config.json schema and implementation identities are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
