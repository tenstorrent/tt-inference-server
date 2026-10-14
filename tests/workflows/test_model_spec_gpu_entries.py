# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""At most one GPU spec per model in each model-spec catalog.

The loader builds model ids from (impl, model, device), so two GPU entries for
one model (for example a reviewed template plus a generated copy from
scripts/gpu_reference_colab) can collapse into one id with no error, and the
spec that wins depends on file order."""

from collections import defaultdict
from pathlib import Path

import pytest
import yaml

CATALOGS = sorted(
    (Path(__file__).resolve().parents[2] / "workflows" / "model_specs").glob("*/*.yaml")
)


def gpu_spec_counts(text: str):
    counts = defaultdict(int)
    for tpl in (yaml.safe_load(text) or {}).get("templates") or []:
        gpu = sum(
            1 for d in tpl.get("device_model_specs") or [] if d.get("device") == "GPU"
        )
        for model in tpl.get("weights") or []:
            counts[model] += gpu
    return counts


@pytest.mark.parametrize("catalog", CATALOGS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_one_gpu_spec_per_model(catalog):
    duplicated = {
        m: n for m, n in gpu_spec_counts(catalog.read_text()).items() if n > 1
    }
    assert not duplicated, f"models with more than one GPU spec: {duplicated}"


def test_the_check_catches_a_duplicate():
    text = (
        "templates:\n"
        "- weights: [org/m]\n  device_model_specs: [{device: GPU}]\n"
        "- weights: [org/m, org/n]\n  device_model_specs: [{device: GPU}, {device: P300X2}]\n"
    )
    assert dict(gpu_spec_counts(text)) == {"org/m": 2, "org/n": 1}
