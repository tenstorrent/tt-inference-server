#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""
Promote dev model specs to prod for every model-device combination marked
``release`` in models-ci-config.json.

For each (model, inference_engine, device) under ``ci.release`` in the CI config,
the matching template in workflows/model_specs/dev/ is copied (whole, with inline
comments preserved) into the same-named file in workflows/model_specs/prod/,
upserting by (impl, inference_engine, weights) identity.
"""

import sys
from collections import namedtuple
from pathlib import Path

from ruamel.yaml import YAML

# Add repo root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from workflows.workflow_types import DeviceTypes, InferenceEngine  # noqa: E402

REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_CI_CONFIG = REPO_ROOT / ".github" / "workflows" / "models-ci-config.json"
DEFAULT_DEV_DIR = REPO_ROOT / "workflows" / "model_specs" / "dev"
DEFAULT_PROD_DIR = REPO_ROOT / "workflows" / "model_specs" / "prod"

ReleaseCombo = namedtuple("ReleaseCombo", ["model_name", "engine", "device"])


def _yaml() -> YAML:
    """A round-trip YAML configured to preserve comments and avoid line wrapping."""
    y = YAML()
    y.preserve_quotes = True
    y.width = 4096
    return y


def model_name_from_weight(weight: str) -> str:
    """Extract the model name (basename) from a HuggingFace repo path."""
    return Path(weight).name


def iter_implementations(model_entry: dict):
    """Yield each implementation dict for a CI-config model entry.

    Handles both the flat shape ({inference_engine, ci}) and the
    implementations:[...] array shape.
    """
    if "implementations" in model_entry:
        yield from model_entry["implementations"]
    else:
        yield model_entry


def collect_release_combos(ci_config: dict) -> set:
    """Return the set of ReleaseCombo(model_name, engine, device) marked release."""
    combos = set()
    for model_name, entry in ci_config.get("models", {}).items():
        for impl in iter_implementations(entry):
            release = impl.get("ci", {}).get("release")
            if not release:
                continue
            engine = InferenceEngine.from_string(impl["inference_engine"])
            for device in release.get("devices", []):
                combos.add(
                    ReleaseCombo(model_name, engine, DeviceTypes.from_string(device))
                )
    return combos


def template_engine(template: dict) -> InferenceEngine:
    return InferenceEngine.from_string(template["inference_engine"])


def template_devices(template: dict) -> set:
    return {
        DeviceTypes.from_string(d["device"])
        for d in template.get("device_model_specs", [])
    }


def template_model_names(template: dict) -> set:
    return {model_name_from_weight(w) for w in template.get("weights", [])}


def template_matches(template: dict, combo: ReleaseCombo) -> bool:
    """True if the template provides the given release combo."""
    return (
        combo.model_name in template_model_names(template)
        and combo.engine == template_engine(template)
        and combo.device in template_devices(template)
    )


def template_identity(template: dict):
    """Upsert identity for a template: (impl, engine, frozenset(weights))."""
    return (
        template["impl"],
        template_engine(template),
        frozenset(template.get("weights", [])),
    )


def find_matches(dev_dir: Path, combos: set):
    """Scan dev catalog files for templates matching any release combo.

    Returns (matches_by_file, unmatched):
      - matches_by_file: dict filename -> list of matched template objects,
        in file order, de-duplicated by identity.
      - unmatched: set of combos that matched no dev template.
    """
    yaml = _yaml()
    matched_combos = set()
    matches_by_file = {}
    for dev_file in sorted(dev_dir.glob("*.yaml")):
        doc = yaml.load(dev_file.read_text())
        templates = (doc or {}).get("templates") or []
        picked = []
        picked_ids = set()
        for template in templates:
            hits = [c for c in combos if template_matches(template, c)]
            if not hits:
                continue
            matched_combos.update(hits)
            identity = template_identity(template)
            if identity not in picked_ids:
                picked.append(template)
                picked_ids.add(identity)
        if picked:
            matches_by_file[dev_file.name] = picked
    unmatched = combos - matched_combos
    return matches_by_file, unmatched
