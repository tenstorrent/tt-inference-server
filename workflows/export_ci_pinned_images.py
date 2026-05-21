# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Export a manifest of ModelSpec entries whose docker_image must NOT be
replaced with a freshly-built image at tt-shield CI time
(`ci_pinned_docker_image=True`).

Output JSON, structure:

    {
      "schema_version": "0.1.0",
      "ci_pinned_images": {
        "<model_name>": {"<INFERENCE_ENGINE_NAME>": true, ...},
        ...
      }
    }

`INFERENCE_ENGINE_NAME` is the InferenceEngine enum **name** (uppercase: VLLM,
MEDIA, FORGE) — same convention `InferenceEngine.from_string` normalizes to,
so callers can match against `impl["inference_engine"].upper()`.

Consumed by tt-shield's `.github/scripts/generate_model_ci_workflows/
generate_ci_matrix.py` via the `--pinned-images-manifest` CLI flag.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from workflows.model_spec import MODEL_SPECS, MODEL_SPECS_SCHEMA_VERSION
from workflows.workflow_types import InferenceEngine


def build_pinned_images_manifest() -> dict:
    """Build the pinned-images manifest from in-memory ModelSpecs."""
    pinned: dict = defaultdict(dict)
    for spec in MODEL_SPECS.values():
        if not spec.ci_pinned_docker_image:
            continue
        engine_name = InferenceEngine.from_string(spec.inference_engine).name
        pinned[spec.model_name][engine_name] = True

    return {
        "schema_version": MODEL_SPECS_SCHEMA_VERSION,
        "ci_pinned_images": dict(pinned),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to write manifest JSON. Default: stdout.",
    )
    args = parser.parse_args()

    manifest = build_pinned_images_manifest()
    payload = json.dumps(manifest, indent=2, sort_keys=True)

    if args.output:
        args.output.write_text(payload + "\n")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
