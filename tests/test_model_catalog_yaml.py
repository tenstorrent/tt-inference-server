# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import json
from pathlib import Path

import pytest

from workflows.model_spec import MODEL_SPECS, export_model_specs_json

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "model_specs_golden.json"


def _current_export(tmp_path: Path) -> dict:
    out = tmp_path / "current.json"
    export_model_specs_json(MODEL_SPECS, out)
    return json.loads(out.read_text())


def test_model_specs_match_golden(tmp_path):
    """MODEL_SPECS output must be byte-identical to the pre-migration snapshot."""
    assert GOLDEN_PATH.exists(), (
        "Golden snapshot missing. Regenerate with: python -c \"...\" "
        "(see plan task 1 step 1)."
    )
    golden = json.loads(GOLDEN_PATH.read_text())
    current = _current_export(tmp_path)
    # Compare only the catalog payload; schema_version / release_version
    # can drift independently of catalog content.
    assert current["model_specs"] == golden["model_specs"]
