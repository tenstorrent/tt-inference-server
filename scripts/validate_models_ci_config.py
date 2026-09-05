#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Validate Models CI schema and implementation-aware identities."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple


def iter_implementations(model_entry: dict) -> Iterable[dict]:
    """Yield either the flat model entry or its implementation entries."""
    yield from model_entry.get("implementations", [model_entry])


def validate_implementation_identities(config: dict) -> List[str]:
    """Return errors for ambiguous or duplicate per-model engine identities.

    A single row for an inference engine may omit ``impl`` and select that
    engine's default implementation. When an engine is repeated for a model,
    every row must provide ``impl`` and each ``(engine, impl)`` pair must be
    unique.
    """
    errors: List[str] = []
    for model, model_entry in config.get("models", {}).items():
        implementations = list(iter_implementations(model_entry))
        engine_counts = Counter(
            str(item.get("inference_engine", "")).casefold() for item in implementations
        )
        identities: Set[Tuple[str, Optional[str]]] = set()

        for index, item in enumerate(implementations):
            engine = str(item.get("inference_engine", "")).casefold()
            impl = item.get("impl")
            path = f"models.{model}.implementations[{index}]"

            if engine_counts[engine] > 1 and impl is None:
                errors.append(
                    f"{path}: impl is required because inference_engine="
                    f"{item.get('inference_engine')!r} appears more than once"
                )

            identity = (engine, impl)
            if identity in identities:
                errors.append(
                    f"{path}: duplicate Models CI identity "
                    f"(inference_engine={item.get('inference_engine')!r}, "
                    f"impl={impl!r})"
                )
            identities.add(identity)

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
    try:
        jsonschema.validate(instance=config, schema=schema)
    except jsonschema.ValidationError as error:
        print(f"ERROR: schema validation failed: {error.message}")
        print("Path: " + " -> ".join(str(part) for part in error.absolute_path))
        return 1

    errors = validate_implementation_identities(config)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("models-ci-config.json schema and implementation identities are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
