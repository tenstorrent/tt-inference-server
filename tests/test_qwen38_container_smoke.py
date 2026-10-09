# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import json

import pytest

from scripts.release.run_qwen38_container_smoke import (
    evaluation_command,
    predecessor_ready,
)


def test_live_controller_cannot_be_overridden_by_completed_receipt():
    properties = dict(
        MainPID="123",
        ActiveState="active",
        LoadState="loaded",
        Result="success",
        InvocationID="original",
    )
    assert not predecessor_ready(
        properties, dict(state="completed", cleanup_completed=True), "original"
    )


@pytest.mark.parametrize(
    "change", [dict(InvocationID="replacement"), dict(Result="timeout")]
)
def test_changed_or_failed_predecessor_rejected(change):
    properties = dict(
        MainPID="0",
        ActiveState="failed",
        LoadState="loaded",
        Result="success",
        InvocationID="original",
    )
    with pytest.raises(ValueError):
        predecessor_ready(
            {**properties, **change},
            dict(state="completed", cleanup_completed=True),
            "original",
        )


def test_collected_unit_requires_durable_clean_receipt():
    properties = dict(
        MainPID="0", ActiveState="inactive", LoadState="not-found", InvocationID=""
    )
    assert predecessor_ready(
        properties, dict(state="completed", cleanup_completed=True), "original"
    )
    with pytest.raises(ValueError):
        predecessor_ready(
            properties, dict(state="failed", cleanup_completed=True), "original"
        )


@pytest.mark.parametrize(
    "command", ["echo unsafe", [], ["python", "run.py"], ["/bin/python", None]]
)
def test_evaluation_command_requires_explicit_argv(tmp_path, command):
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(command))
    with pytest.raises(ValueError):
        evaluation_command(path)


def test_evaluation_keeps_arguments_literal(tmp_path):
    path = tmp_path / "evaluation.json"
    command = ["/bin/python", "a file.py", "$(literal)"]
    path.write_text(json.dumps(command))
    assert evaluation_command(path) == command
