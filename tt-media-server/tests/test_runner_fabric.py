# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import pytest
from config.settings import settings
from tt_model_runners.runner_fabric import get_device_runner


def test_runner_creation(monkeypatch, runner_name, expected_class_name):
    """Test that each runner type creates the correct class instance."""
    monkeypatch.setattr(settings, "model_runner", runner_name)
    runner = get_device_runner("test_worker")
    assert expected_class_name in type(runner).__name__


def test_invalid_runner(monkeypatch):
    monkeypatch.setattr(settings, "model_runner", "invalid")
    with pytest.raises(ValueError):
        get_device_runner("test_worker")
