# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Validate performance-target lookup for image models.

Ports the image (SDXL) coverage from v1's
``tests/workflows/test_utils_report.py::test_get_performance_targets`` after
image models moved to v2. Asserts ``get_performance_targets`` reads the image
perf targets from v2's ``model_performance_reference.json``.
"""

from __future__ import annotations

import pytest

from test_module._test_common.target_check import (
    PerformanceTargets,
    get_performance_targets,
)


@pytest.mark.parametrize(
    "model_name,device_str,expected",
    [
        (
            "stable-diffusion-xl-base-1.0",
            "n150",
            PerformanceTargets(
                ttft_ms=12500,
                tput_user=0.08,
                max_concurrency=1,
                num_eval_runs=100,
                task_type="image",
            ),
        ),
        (
            "FLUX.1-dev",
            "t3k",
            PerformanceTargets(
                ttft_ms=11000,
                tput_user=0.091,
                max_concurrency=1,
                task_type="image",
            ),
        ),
    ],
)
def test_get_performance_targets_image_models(model_name, device_str, expected):
    actual = get_performance_targets(model_name=model_name, device_str=device_str)
    assert actual == expected, (
        f"Performance targets mismatch for {model_name} on {device_str}:\n"
        f"Expected: {expected}\nActual: {actual}"
    )


def test_get_performance_targets_unknown_model_returns_empty():
    """Unknown model/device yields all-None defaults (callers treat as NA)."""
    assert (
        get_performance_targets(model_name="nonexistent-model", device_str="n150")
        == PerformanceTargets()
    )
