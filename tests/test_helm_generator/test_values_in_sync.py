# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import difflib

import pytest

from workflows.helm_generator.cli import DEFAULT_VALUES_PATH, filter_specs, generate
from workflows.model_spec import IMAGE_PINNED_MODEL_SPECS


def test_committed_values_yaml_in_sync_with_model_spec(tmp_path):
    work = tmp_path / "values.yaml"
    work.write_text(DEFAULT_VALUES_PATH.read_text())

    specs = filter_specs(IMAGE_PINNED_MODEL_SPECS, include_multihost=False)
    generate(values_path=work, specs=specs)

    committed = DEFAULT_VALUES_PATH.read_text()
    regenerated = work.read_text()
    if regenerated == committed:
        return

    diff = "".join(
        difflib.unified_diff(
            committed.splitlines(keepends=True),
            regenerated.splitlines(keepends=True),
            fromfile="charts/tt-inference-server/values.yaml (committed)",
            tofile="charts/tt-inference-server/values.yaml (regenerated)",
        )
    )
    pytest.fail(
        "charts/tt-inference-server/values.yaml drifted from "
        "workflows/model_spec.py.\n"
        "\n"
        f"FIX: python3 -m workflows.helm_generator\n"
        "      git add charts/tt-inference-server/values.yaml\n"
        "\n"
        f"{diff}"
    )
