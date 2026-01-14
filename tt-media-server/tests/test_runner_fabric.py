# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from unittest.mock import PropertyMock, patch

import pytest
from tt_model_runners.runner_fabric import get_device_runner


def test_runner_creation(runner_name, expected_class_name):
    """Test that each runner type creates the correct class instance."""
    # Patch the settings module where it's imported in runner_fabric
    with patch("tt_model_runners.runner_fabric.settings") as mock_settings:
        # Use PropertyMock to make model_runner return the test value
        type(mock_settings).model_runner = PropertyMock(return_value=runner_name)
        runner = get_device_runner("test_worker")
        assert expected_class_name in type(runner).__name__


def test_invalid_runner():
    """Test that invalid runner name raises ValueError."""
    with patch("tt_model_runners.runner_fabric.settings") as mock_settings:
        type(mock_settings).model_runner = PropertyMock(return_value="invalid")
        with pytest.raises(ValueError, match="Unknown model runner: invalid"):
            get_device_runner("test_worker")
