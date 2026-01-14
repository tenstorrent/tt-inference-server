# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from unittest.mock import MagicMock, patch

import pytest
from tt_model_runners.runner_fabric import get_device_runner


@patch("tt_model_runners.runner_fabric.settings")
@patch("tt_model_runners.base_device_runner.get_settings")
@pytest.mark.parametrize(
    "runner_name_param, expected_class_name",
    [
        ("vllm_forge", "VLLMForgeRunner"),
    ],
)
def test_runner_creation_unique(
    mock_get_settings,
    mock_runner_fabric_settings,
    runner_name_param,
    expected_class_name,
):
    """Test that each runner type creates the correct class instance."""
    # Mock the settings object for runner_fabric
    mock_runner_fabric_settings.model_runner = runner_name_param

    # Mock the settings object returned by get_settings in base_device_runner
    mock_settings = MagicMock()
    mock_settings.model_runner = runner_name_param
    mock_settings.device_mesh_shape = (1, 8)
    mock_get_settings.return_value = mock_settings

    # Call the function under test
    runner = get_device_runner("test_worker")

    # Assert the runner is of the expected type
    assert expected_class_name in type(runner).__name__


@patch("tt_model_runners.runner_fabric.settings")
@patch("tt_model_runners.base_device_runner.get_settings")
def test_invalid_runner(mock_get_settings, mock_runner_fabric_settings):
    """Test that invalid runner name raises ValueError."""
    # Mock the settings object for runner_fabric
    mock_runner_fabric_settings.model_runner = "invalid"

    # Mock the settings object returned by get_settings in base_device_runner
    mock_settings = MagicMock()
    mock_settings.model_runner = "invalid"
    mock_get_settings.return_value = mock_settings

    with pytest.raises(ValueError, match="Unknown model runner: invalid"):
        get_device_runner("test_worker")
