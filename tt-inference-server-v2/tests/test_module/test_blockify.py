# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for the ``_test_common.blockify`` helpers used by media runners."""

from __future__ import annotations

import re
from types import SimpleNamespace

from test_module._test_common import block_id, block_targets, sweep_envelope


def _ctx(model: str = "tt-sdxl-1.0", device: str = "n300") -> SimpleNamespace:
    return SimpleNamespace(
        model_spec=SimpleNamespace(model_name=model),
        device=SimpleNamespace(name=device),
    )


def test_block_targets_only_carries_per_block_fields():
    targets = block_targets(_ctx(), task_type="image")
    assert targets == {"task_type": "image"}
    # model/device/timestamp deliberately absent — they live in the
    # sweep envelope (schema metadata), not on every block.
    assert "model" not in targets
    assert "device" not in targets
    assert "timestamp" not in targets


def test_block_targets_extra_kwargs_get_merged():
    targets = block_targets(_ctx(), task_type="image", task_name="sdxl-prompts")
    assert targets["task_name"] == "sdxl-prompts"
    assert targets["task_type"] == "image"


def test_sweep_envelope_carries_model_device_timestamp():
    env = sweep_envelope(_ctx())
    assert env["model_name"] == "tt-sdxl-1.0"
    assert env["device"] == "n300"
    assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", env["generated_at"])


def test_block_id_slugifies_model_and_device():
    assert block_id(_ctx("meta-llama/Llama-3.1-8B", "n300")) == (
        "meta-llama__Llama-3.1-8B_n300"
    )


def test_block_id_empty_when_both_missing():
    ctx = SimpleNamespace(
        model_spec=SimpleNamespace(model_name=""),
        device=SimpleNamespace(name=""),
    )
    assert block_id(ctx) == ""
