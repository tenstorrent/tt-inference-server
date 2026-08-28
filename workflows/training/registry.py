# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Maps a (model, device) training run to its checked-in loss expectation.

Explicit rather than derived: the expectation file name encodes the dataset and
device, which cannot be reconstructed from the model name alone. Add a row here
when onboarding a new training test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

# Goldens live under reference_config/training/ for symmetry with the perf
# (reference_config/benchmarking/) and accuracy (reference_config/evals/)
# targets, not next to this module. registry.py is at
# <repo>/workflows/training/registry.py, so parents[2] is the repo root.
EXPECTED_CONFIG_DIR = (
    Path(__file__).resolve().parents[2] / "reference_config" / "training"
)

# Keyed on model_spec.model_name (the weights basename, e.g. "Llama-3.1-8B"),
# which is what _build_training_cmd forwards — NOT the full HF repo path.
_EXPECTED_BY_MODEL_DEVICE: Dict[Tuple[str, str], str] = {
    ("Llama-3.1-8B", "p150"): "llama_3_1_8b_sst2_p150.yaml",
}


def expected_config_path(model_name: str, device: str) -> Path:
    """Resolve the expectation YAML for a (model, device), or raise.

    Raises loudly with the known keys so a missing mapping is obvious in CI
    rather than silently skipping the loss gate.
    """
    key = (model_name, str(device).lower())
    filename = _EXPECTED_BY_MODEL_DEVICE.get(key)
    if filename is None:
        known = ", ".join(f"{m} on {d}" for (m, d) in _EXPECTED_BY_MODEL_DEVICE)
        raise KeyError(
            f"No training loss expectation registered for {model_name} on {device}. "
            f"Known: {known}. Add a row to workflows/training/registry.py and a "
            f"YAML under {EXPECTED_CONFIG_DIR}."
        )
    return EXPECTED_CONFIG_DIR / filename
