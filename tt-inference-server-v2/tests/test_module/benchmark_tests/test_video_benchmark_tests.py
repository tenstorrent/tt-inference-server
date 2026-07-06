# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for video benchmark dispatch guards."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from test_module._test_common import SkipTest
from test_module.benchmark_tests import video_benchmark_tests as mod


def _ctx(model_name):
    return SimpleNamespace(
        model_spec=SimpleNamespace(model_name=model_name),
        device=SimpleNamespace(name="t3k"),
    )


def test_unsupported_video_model_raises_skip():
    # I2V has no inference-step profile: previously a KeyError (crash -> ERROR),
    # now a visible, non-blocking SkipTest that run_media_task maps to SKIP.
    with pytest.raises(SkipTest) as exc:
        mod.run_video_benchmark(_ctx("Wan2.2-I2V-A14B-Diffusers"))
    assert "not implemented" in str(exc.value)
    assert "Wan2.2-I2V-A14B-Diffusers" in str(exc.value)
