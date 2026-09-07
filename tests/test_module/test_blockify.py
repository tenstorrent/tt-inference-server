# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for the ``_test_common.blockify`` helpers used by media runners."""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Optional

from test_module._test_common import block_id, report_model_fields, sweep_envelope


def _ctx(
    model: str = "tt-sdxl-1.0",
    device: str = "n300",
    hf_model_repo: Optional[str] = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        model_spec=SimpleNamespace(
            model_name=model,
            hf_model_repo=hf_model_repo if hf_model_repo is not None else model,
        ),
        device=SimpleNamespace(name=device),
    )


def test_sweep_envelope_carries_model_device_timestamp():
    env = sweep_envelope(_ctx())
    assert env["model_name"] == "tt-sdxl-1.0"
    assert env["model_repo"] == "tt-sdxl-1.0"
    assert env["device"] == "n300"
    assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", env["generated_at"])


def test_sweep_envelope_keeps_bare_name_and_full_repo():
    env = sweep_envelope(
        _ctx(model="whisper-large-v3", hf_model_repo="openai/whisper-large-v3")
    )
    assert env["model_name"] == "whisper-large-v3"
    assert env["model_repo"] == "openai/whisper-large-v3"


def test_report_model_fields_split():
    fields = report_model_fields(
        SimpleNamespace(
            model_name="whisper-large-v3",
            hf_model_repo="openai/whisper-large-v3",
        )
    )
    assert fields == {
        "model_name": "whisper-large-v3",
        "model_repo": "openai/whisper-large-v3",
    }


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
