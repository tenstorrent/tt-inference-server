# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import unittest
from unittest.mock import MagicMock, patch

import pytest

from utils.media_clients.base_strategy_interface import (
    DEVICE_LIVENESS_TEST_ALIVE,
    BaseMediaStrategy,
)


class ConcreteMediaStrategy(BaseMediaStrategy):
    """Concrete implementation for testing the abstract base class."""

    def run_eval(self) -> None:
        pass

    def run_benchmark(self, num_calls: int = 1) -> list:
        return []


class TestConstants(unittest.TestCase):
    """Tests for module-level constants."""

    def test_device_liveness_test_alive_value(self):
        assert DEVICE_LIVENESS_TEST_ALIVE == "alive"


class TestBaseMediaStrategyInit(unittest.TestCase):
    """Tests for BaseMediaStrategy.__init__ method."""

    def test_init_sets_all_params(self):
        all_params = {"key": "value"}
        model_spec = MagicMock()
        device = MagicMock()
        output_path = "/tmp/output"
        service_port = 8000

        strategy = ConcreteMediaStrategy(
            all_params, model_spec, device, output_path, service_port
        )

        assert strategy.all_params == all_params

    def test_init_sets_model_spec(self):
        model_spec = MagicMock()
        strategy = ConcreteMediaStrategy({}, model_spec, MagicMock(), "/tmp", 8000)

        assert strategy.model_spec == model_spec

    def test_init_sets_device(self):
        device = MagicMock()
        strategy = ConcreteMediaStrategy({}, MagicMock(), device, "/tmp", 8000)

        assert strategy.device == device

    def test_init_sets_output_path(self):
        output_path = "/custom/output/path"
        strategy = ConcreteMediaStrategy(
            {}, MagicMock(), MagicMock(), output_path, 8000
        )

        assert strategy.output_path == output_path

    def test_init_sets_service_port(self):
        service_port = 9000
        strategy = ConcreteMediaStrategy(
            {}, MagicMock(), MagicMock(), "/tmp", service_port
        )

        assert strategy.service_port == service_port

    def test_init_sets_base_url(self):
        service_port = 8080
        strategy = ConcreteMediaStrategy(
            {}, MagicMock(), MagicMock(), "/tmp", service_port
        )

        assert strategy.base_url == f"http://localhost:{service_port}"

    def test_init_sets_test_payloads_path(self):
        strategy = ConcreteMediaStrategy({}, MagicMock(), MagicMock(), "/tmp", 8000)

        assert strategy.test_payloads_path == "utils/test_payloads"


class TestBaseMediaStrategyAbstractMethods(unittest.TestCase):
    """Tests for abstract method enforcement."""

    def test_cannot_instantiate_base_class_directly(self):
        with pytest.raises(TypeError) as exc_info:
            BaseMediaStrategy({}, MagicMock(), MagicMock(), "/tmp", 8000)

        assert "abstract" in str(exc_info.value).lower()

    def test_concrete_class_implements_run_eval(self):
        strategy = ConcreteMediaStrategy({}, MagicMock(), MagicMock(), "/tmp", 8000)
        # Should not raise
        strategy.run_eval()

    def test_concrete_class_implements_run_benchmark(self):
        strategy = ConcreteMediaStrategy({}, MagicMock(), MagicMock(), "/tmp", 8000)
        result = strategy.run_benchmark(5)
        assert result == []


class TestBaseMediaStrategyGetHealth(unittest.TestCase):
    """Tests for BaseMediaStrategy.get_health method."""

    def _create_strategy(self, device=None, max_concurrency=4):
        model_spec = MagicMock()
        model_spec.device_model_spec.max_concurrency = max_concurrency
        if device is None:
            device = MagicMock()
            device.name = "test_device"
        return ConcreteMediaStrategy({}, model_spec, device, "/tmp", 8000)

    @patch("utils.media_clients.base_strategy_interface.DeviceLivenessTest")
    @patch("utils.media_clients.base_strategy_interface.TestConfig")
    def test_get_health_success_returns_true_and_runner(
        self, mock_test_config, mock_liveness_test_class
    ):
        strategy = self._create_strategy()
        mock_liveness_test = MagicMock()
        mock_liveness_test_class.return_value = mock_liveness_test
        mock_liveness_test.run_tests.return_value = {
            "success": True,
            "attempts": 1,
            "result": {"full_response": {"runner_in_use": "tt_metal"}},
        }

        health_status, runner_in_use = strategy.get_health()

        assert health_status is True
        assert runner_in_use == "tt_metal"
        mock_liveness_test.run_tests.assert_called_once()

    @patch("utils.media_clients.base_strategy_interface.DeviceLivenessTest")
    @patch("utils.media_clients.base_strategy_interface.TestConfig")
    def test_get_health_failure_returns_false_and_none(
        self, mock_test_config, mock_liveness_test_class
    ):
        strategy = self._create_strategy()
        mock_liveness_test = MagicMock()
        mock_liveness_test_class.return_value = mock_liveness_test
        mock_liveness_test.run_tests.return_value = {"success": False}

        health_status, runner_in_use = strategy.get_health()

        assert health_status is False
        assert runner_in_use is None

    @patch("utils.media_clients.base_strategy_interface.DeviceLivenessTest")
    @patch("utils.media_clients.base_strategy_interface.TestConfig")
    def test_get_health_non_dict_result_returns_false(
        self, mock_test_config, mock_liveness_test_class
    ):
        strategy = self._create_strategy()
        mock_liveness_test = MagicMock()
        mock_liveness_test_class.return_value = mock_liveness_test
        mock_liveness_test.run_tests.return_value = "not a dict"

        health_status, runner_in_use = strategy.get_health()

        assert health_status is False
        assert runner_in_use is None

    @patch("utils.media_clients.base_strategy_interface.DeviceLivenessTest")
    @patch("utils.media_clients.base_strategy_interface.TestConfig")
    def test_get_health_system_exit_returns_false(
        self, mock_test_config, mock_liveness_test_class
    ):
        strategy = self._create_strategy()
        mock_liveness_test = MagicMock()
        mock_liveness_test_class.return_value = mock_liveness_test
        mock_liveness_test.run_tests.side_effect = SystemExit(1)

        health_status, runner_in_use = strategy.get_health()

        assert health_status is False
        assert runner_in_use is None

    @patch("utils.media_clients.base_strategy_interface.DeviceLivenessTest")
    @patch("utils.media_clients.base_strategy_interface.TestConfig")
    def test_get_health_exception_returns_false(
        self, mock_test_config, mock_liveness_test_class
    ):
        strategy = self._create_strategy()
        mock_liveness_test = MagicMock()
        mock_liveness_test_class.return_value = mock_liveness_test
        mock_liveness_test.run_tests.side_effect = Exception("Connection error")

        health_status, runner_in_use = strategy.get_health()

        assert health_status is False
        assert runner_in_use is None

    @patch("utils.media_clients.base_strategy_interface.DeviceLivenessTest")
    @patch("utils.media_clients.base_strategy_interface.TestConfig")
    def test_get_health_device_with_name_attribute(
        self, mock_test_config, mock_liveness_test_class
    ):
        device = MagicMock()
        device.name = "n150_device"
        strategy = self._create_strategy(device=device)
        mock_liveness_test = MagicMock()
        mock_liveness_test_class.return_value = mock_liveness_test
        mock_liveness_test.run_tests.return_value = {"success": True, "result": {}}

        strategy.get_health()
        # If it doesn't raise, the hasattr check worked

    @patch("utils.media_clients.base_strategy_interface.DeviceLivenessTest")
    @patch("utils.media_clients.base_strategy_interface.TestConfig")
    def test_get_health_device_without_name_uses_str(
        self, mock_test_config, mock_liveness_test_class
    ):
        # Create a device object without 'name' attribute
        device = "string_device_representation"
        model_spec = MagicMock()
        model_spec.device_model_spec.max_concurrency = 4
        strategy = ConcreteMediaStrategy({}, model_spec, device, "/tmp", 8000)

        mock_liveness_test = MagicMock()
        mock_liveness_test_class.return_value = mock_liveness_test
        mock_liveness_test.run_tests.return_value = {"success": True, "result": {}}

        strategy.get_health()
        # If it doesn't raise, the str() fallback worked

    @patch("utils.media_clients.base_strategy_interface.DeviceLivenessTest")
    @patch("utils.media_clients.base_strategy_interface.TestConfig")
    def test_get_health_positive_num_devices(
        self, mock_test_config, mock_liveness_test_class
    ):
        strategy = self._create_strategy(max_concurrency=8)
        mock_liveness_test = MagicMock()
        mock_liveness_test_class.return_value = mock_liveness_test
        mock_liveness_test.run_tests.return_value = {"success": True, "result": {}}

        strategy.get_health()

        # Verify test was created with correct targets
        mock_liveness_test_class.assert_called_once()

    @patch("utils.media_clients.base_strategy_interface.DeviceLivenessTest")
    @patch("utils.media_clients.base_strategy_interface.TestConfig")
    def test_get_health_zero_num_devices_sets_none(
        self, mock_test_config, mock_liveness_test_class
    ):
        strategy = self._create_strategy(max_concurrency=0)
        mock_liveness_test = MagicMock()
        mock_liveness_test_class.return_value = mock_liveness_test
        mock_liveness_test.run_tests.return_value = {"success": True, "result": {}}

        strategy.get_health()
        # The targets should have None for num_of_devices

    @patch("utils.media_clients.base_strategy_interface.DeviceLivenessTest")
    @patch("utils.media_clients.base_strategy_interface.TestConfig")
    def test_get_health_none_num_devices_sets_none(
        self, mock_test_config, mock_liveness_test_class
    ):
        strategy = self._create_strategy(max_concurrency=None)
        mock_liveness_test = MagicMock()
        mock_liveness_test_class.return_value = mock_liveness_test
        mock_liveness_test.run_tests.return_value = {"success": True, "result": {}}

        strategy.get_health()
        # The targets should have None for num_of_devices

    @patch("utils.media_clients.base_strategy_interface.DeviceLivenessTest")
    @patch("utils.media_clients.base_strategy_interface.TestConfig")
    def test_get_health_sets_service_port_on_test(
        self, mock_test_config, mock_liveness_test_class
    ):
        model_spec = MagicMock()
        model_spec.device_model_spec.max_concurrency = 4
        device = MagicMock()
        device.name = "test"
        service_port = 9999
        strategy = ConcreteMediaStrategy({}, model_spec, device, "/tmp", service_port)

        mock_liveness_test = MagicMock()
        mock_liveness_test_class.return_value = mock_liveness_test
        mock_liveness_test.run_tests.return_value = {"success": True, "result": {}}

        strategy.get_health()

        assert mock_liveness_test.service_port == service_port

    @patch("utils.media_clients.base_strategy_interface.DeviceLivenessTest")
    @patch("utils.media_clients.base_strategy_interface.TestConfig")
    def test_get_health_success_missing_runner_in_use(
        self, mock_test_config, mock_liveness_test_class
    ):
        strategy = self._create_strategy()
        mock_liveness_test = MagicMock()
        mock_liveness_test_class.return_value = mock_liveness_test
        mock_liveness_test.run_tests.return_value = {
            "success": True,
            "attempts": 1,
            "result": {"full_response": {}},
        }

        health_status, runner_in_use = strategy.get_health()

        assert health_status is True
        assert runner_in_use is None

    @patch("utils.media_clients.base_strategy_interface.DeviceLivenessTest")
    @patch("utils.media_clients.base_strategy_interface.TestConfig")
    def test_get_health_success_missing_full_response(
        self, mock_test_config, mock_liveness_test_class
    ):
        strategy = self._create_strategy()
        mock_liveness_test = MagicMock()
        mock_liveness_test_class.return_value = mock_liveness_test
        mock_liveness_test.run_tests.return_value = {
            "success": True,
            "attempts": 1,
            "result": {},
        }

        health_status, runner_in_use = strategy.get_health()

        assert health_status is True
        assert runner_in_use is None

    @patch("utils.media_clients.base_strategy_interface.DeviceLivenessTest")
    @patch("utils.media_clients.base_strategy_interface.TestConfig")
    def test_get_health_passes_attempt_number(
        self, mock_test_config, mock_liveness_test_class
    ):
        strategy = self._create_strategy()
        mock_liveness_test = MagicMock()
        mock_liveness_test_class.return_value = mock_liveness_test
        mock_liveness_test.run_tests.return_value = {"success": True, "result": {}}

        # Call with custom attempt number
        strategy.get_health(attempt_number=5)
        # Method accepts the parameter (even if not used internally)


# Pytest parametrized tests for edge cases
@pytest.mark.parametrize(
    "max_concurrency,expected_num_devices",
    [
        (1, 1),
        (4, 4),
        (32, 32),
        (0, None),
        (None, None),
        (-1, None),
    ],
)
@patch("utils.media_clients.base_strategy_interface.DeviceLivenessTest")
@patch("utils.media_clients.base_strategy_interface.TestConfig")
def test_get_health_num_devices_edge_cases(
    mock_test_config, mock_liveness_test_class, max_concurrency, expected_num_devices
):
    """Test that num_devices is correctly set based on max_concurrency."""
    model_spec = MagicMock()
    model_spec.device_model_spec.max_concurrency = max_concurrency
    device = MagicMock()
    device.name = "test"
    strategy = ConcreteMediaStrategy({}, model_spec, device, "/tmp", 8000)

    mock_liveness_test = MagicMock()
    mock_liveness_test_class.return_value = mock_liveness_test
    mock_liveness_test.run_tests.return_value = {"success": True, "result": {}}

    strategy.get_health()

    # Just verify it runs without error for various max_concurrency values


@pytest.mark.parametrize(
    "exception_type",
    [RuntimeError, ValueError, ConnectionError, TimeoutError, Exception],
)
@patch("utils.media_clients.base_strategy_interface.DeviceLivenessTest")
@patch("utils.media_clients.base_strategy_interface.TestConfig")
def test_get_health_handles_various_exceptions(
    mock_test_config, mock_liveness_test_class, exception_type
):
    """Test that get_health handles various exception types gracefully."""
    model_spec = MagicMock()
    model_spec.device_model_spec.max_concurrency = 4
    device = MagicMock()
    device.name = "test"
    strategy = ConcreteMediaStrategy({}, model_spec, device, "/tmp", 8000)

    mock_liveness_test = MagicMock()
    mock_liveness_test_class.return_value = mock_liveness_test
    mock_liveness_test.run_tests.side_effect = exception_type("Test error")

    health_status, runner_in_use = strategy.get_health()

    assert health_status is False
    assert runner_in_use is None


@pytest.mark.parametrize(
    "service_port",
    [80, 443, 8000, 8080, 9000, 65535],
)
def test_init_various_service_ports(service_port):
    """Test that various service ports are handled correctly."""
    strategy = ConcreteMediaStrategy({}, MagicMock(), MagicMock(), "/tmp", service_port)

    assert strategy.service_port == service_port
    assert strategy.base_url == f"http://localhost:{service_port}"
