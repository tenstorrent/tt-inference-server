#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC


import pytest

from workflows.utils_report import PerformanceTargets, get_performance_targets


@pytest.fixture
def mock_cnn_performance_targets():
    """Mock performance targets for CNN/image models"""
    return {
        "ttft_ms": 12500,
        "ttft_streaming_ms": None,
        "tput_user": 0.08,
        "tput": None,
        "rtr": None,
        "tolerance": 0.05,
        "max_concurrency": 1,
        "num_eval_runs": None,
        "task_type": "cnn",
    }


@pytest.fixture
def mock_audio_performance_targets():
    """Mock performance targets for audio models"""
    return {
        "ttft_ms": 400,
        "ttft_streaming_ms": 842.3,
        "tput_user": 112.62,
        "tput": None,
        "rtr": 15.61,
        "tolerance": 0.05,
        "max_concurrency": 1,
        "num_eval_runs": 2,
        "task_type": "audio",
    }


@pytest.mark.parametrize(
    "model_name,device_str,model_type,expected_targets_dict",
    [
        (
            "stable-diffusion-xl-base-1.0",
            "n150",
            "IMAGE",
            {
                "ttft_ms": 12500,
                "ttft_streaming_ms": None,
                "tput_user": 0.08,
                "tput": None,
                "rtr": None,
                "tolerance": 0.05,
                "max_concurrency": 1,
                "num_eval_runs": None,
                "task_type": "image",
            },
        ),
        (
            "distil-large-v3",
            "n150",
            "AUDIO",
            {
                "ttft_ms": 400,
                "ttft_streaming_ms": 842.3,
                "tput_user": 112.62,
                "tput": None,
                "rtr": 15.61,
                "tolerance": 0.05,
                "max_concurrency": 1,
                "num_eval_runs": 2,
                "task_type": "audio",
            },
        ),
    ],
)
def test_get_performance_targets(
    model_name, device_str, model_type, expected_targets_dict
):
    """Test that get_performance_targets returns correct values for different models"""
    # Arrange
    expected_targets = PerformanceTargets(**expected_targets_dict)

    # Act
    actual_targets = get_performance_targets(
        model_name=model_name, device_str=device_str, model_type=model_type
    )

    # Assert
    assert actual_targets == expected_targets, (
        f"Performance targets mismatch for {model_name}:\n"
        f"Expected: {expected_targets}\n"
        f"Actual: {actual_targets}"
    )
