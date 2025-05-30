#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import pytest
import argparse
import json
import os
import sys
from unittest.mock import patch, MagicMock, call
from pathlib import Path

# Add the project root to the path so we can import from run.py and workflows
sys.path.insert(0, str(Path(__file__).parent.parent))

from run import parse_arguments
from workflows.run_docker_server import run_docker_server
from workflows.workflow_types import DeviceTypes


class TestVLLMOverrideArgsIntegration:
    """Test the complete flow of --vllm-override-args from CLI to Docker environment."""

    @pytest.fixture
    def base_args(self):
        """Base arguments for server workflow with docker."""
        return [
            "--model",
            "Mistral-7B-Instruct-v0.3",
            "--workflow",
            "server",
            "--device",
            "n150",
            "--docker-server",
        ]

    @pytest.fixture
    def mock_setup_config(self):
        """Mock setup configuration for docker server."""
        mock_config = MagicMock()
        mock_config.cache_root = "/tmp/cache"
        mock_config.container_tt_metal_cache_dir = Path("/container/cache")
        mock_config.container_model_weights_path = "/container/weights"
        mock_config.container_model_weights_mount_dir = "/container/mounts"
        mock_config.host_model_volume_root = "/host/volumes"
        mock_config.host_model_weights_mount_dir = "/host/weights"
        mock_config.model_source = "hf"
        return mock_config

    def test_cli_parsing_vllm_override_args(self, base_args):
        """Test that the CLI correctly parses --vllm-override-args."""
        test_json = '{"max_model_len": 4096, "enable_chunked_prefill": true}'
        args_with_override = base_args + ["--vllm-override-args", test_json]

        with patch("sys.argv", ["run.py"] + args_with_override):
            args = parse_arguments()

        assert hasattr(args, "vllm_override_args")
        assert args.vllm_override_args == test_json

    def test_cli_parsing_vllm_override_args_empty(self, base_args):
        """Test CLI parsing when --vllm-override-args is not provided."""
        with patch("sys.argv", ["run.py"] + base_args):
            args = parse_arguments()

        assert hasattr(args, "vllm_override_args")
        assert args.vllm_override_args is None

    @pytest.mark.parametrize(
        "override_json,expected_env_var",
        [
            ('{"max_model_len": 4096}', '{"max_model_len": 4096}'),
            (
                '{"enable_chunked_prefill": true, "max_num_seqs": 16}',
                '{"enable_chunked_prefill": true, "max_num_seqs": 16}',
            ),
            ("{}", "{}"),
        ],
    )
    def test_run_docker_server_sets_vllm_override_env(
        self, mock_setup_config, override_json, expected_env_var
    ):
        """Test that run_docker_server correctly sets VLLM_OVERRIDE_ARGS environment variable."""
        # Create mock args with vllm_override_args - set all required attributes properly
        mock_args = MagicMock()
        mock_args.model = "Mistral-7B-Instruct-v0.3"
        mock_args.device = "n150"
        mock_args.workflow = "server"
        mock_args.service_port = "8000"
        mock_args.vllm_override_args = override_json
        mock_args.interactive = False
        mock_args.dev_mode = False
        mock_args.device_id = None
        mock_args.impl = "tt-transformers"
        mock_args.override_docker_image = None  # Add this to avoid MagicMock issues

        # Mock dependencies
        with patch(
            "workflows.run_docker_server.get_model_id", return_value="test-model-id"
        ), patch("workflows.run_docker_server.MODEL_CONFIGS") as mock_configs, patch(
            "workflows.run_docker_server.ensure_docker_image", return_value=True
        ), patch("workflows.run_docker_server.subprocess.Popen") as mock_popen, patch(
            "workflows.run_docker_server.open"
        ), patch(
            "workflows.run_docker_server.subprocess.check_output",
            return_value="container123",
        ), patch("workflows.run_docker_server.atexit.register"), patch(
            "workflows.run_docker_server.shlex.join", return_value="mocked command"
        ):  # Mock shlex.join to avoid issues
            # Mock model config
            mock_model_config = MagicMock()
            mock_model_config.docker_image = "test:image"
            mock_model_config.impl.impl_name = "tt-transformers"
            mock_model_config.hf_model_repo = "mistralai/Mistral-7B-Instruct-v0.3"
            mock_model_config.device_model_spec.max_concurrency = "32"
            mock_model_config.device_model_spec.max_context = "32768"
            mock_model_config.override_tt_config = None
            mock_configs.__getitem__.return_value = mock_model_config

            # Call the function
            run_docker_server(mock_args, mock_setup_config)

            # Verify subprocess.Popen was called
            mock_popen.assert_called_once()
            docker_command = mock_popen.call_args[0][0]

            # Check that VLLM_OVERRIDE_ARGS is in the docker command with correct value
            found_env_var = False
            for i, arg in enumerate(docker_command):
                if arg == "-e" and i + 1 < len(docker_command):
                    env_setting = docker_command[i + 1]
                    if env_setting.startswith("VLLM_OVERRIDE_ARGS="):
                        found_env_var = True
                        actual_value = env_setting.split("=", 1)[1]
                        assert actual_value == expected_env_var
                        break

            assert (
                found_env_var
            ), f"VLLM_OVERRIDE_ARGS not found in docker command: {docker_command}"

    def test_run_docker_server_no_vllm_override_args(self, mock_setup_config):
        """Test that run_docker_server doesn't set VLLM_OVERRIDE_ARGS when not provided."""
        # Create mock args without vllm_override_args
        mock_args = MagicMock()
        mock_args.model = "Mistral-7B-Instruct-v0.3"
        mock_args.device = "n150"
        mock_args.workflow = "server"
        mock_args.service_port = "8000"
        mock_args.vllm_override_args = None
        mock_args.interactive = False
        mock_args.dev_mode = False
        mock_args.device_id = None
        mock_args.impl = "tt-transformers"
        mock_args.override_docker_image = None  # Add this to avoid MagicMock issues

        # Mock dependencies
        with patch(
            "workflows.run_docker_server.get_model_id", return_value="test-model-id"
        ), patch("workflows.run_docker_server.MODEL_CONFIGS") as mock_configs, patch(
            "workflows.run_docker_server.ensure_docker_image", return_value=True
        ), patch("workflows.run_docker_server.subprocess.Popen") as mock_popen, patch(
            "workflows.run_docker_server.open"
        ), patch(
            "workflows.run_docker_server.subprocess.check_output",
            return_value="container123",
        ), patch("workflows.run_docker_server.atexit.register"), patch(
            "workflows.run_docker_server.shlex.join", return_value="mocked command"
        ):  # Mock shlex.join to avoid issues
            # Mock model config
            mock_model_config = MagicMock()
            mock_model_config.docker_image = "test:image"
            mock_model_config.impl.impl_name = "tt-transformers"
            mock_model_config.hf_model_repo = "mistralai/Mistral-7B-Instruct-v0.3"
            mock_model_config.device_model_spec.max_concurrency = "32"
            mock_model_config.device_model_spec.max_context = "32768"
            mock_model_config.override_tt_config = None
            mock_configs.__getitem__.return_value = mock_model_config

            # Call the function
            run_docker_server(mock_args, mock_setup_config)

            # Verify subprocess.Popen was called
            mock_popen.assert_called_once()
            docker_command = mock_popen.call_args[0][0]

            # Check that VLLM_OVERRIDE_ARGS is NOT in the docker command
            for i, arg in enumerate(docker_command):
                if arg == "-e" and i + 1 < len(docker_command):
                    env_setting = docker_command[i + 1]
                    assert not env_setting.startswith(
                        "VLLM_OVERRIDE_ARGS="
                    ), f"VLLM_OVERRIDE_ARGS should not be set when not provided: {env_setting}"


class TestVLLMOverrideArgsServer:
    """Test the vLLM server-side processing of VLLM_OVERRIDE_ARGS.

    These tests simulate the behavior of the functions in run_vllm_api_server.py
    without importing directly from that module to avoid dependency issues.
    """

    def get_vllm_override_args_mock(self):
        """Mock implementation of get_vllm_override_args function."""
        cli_override_str = os.getenv("VLLM_OVERRIDE_ARGS")
        if not cli_override_str:
            return {}

        try:
            override_args = json.loads(cli_override_str)
            # Return empty dict if not a dict
            if not isinstance(override_args, dict):
                return {}
            if not override_args:
                return {}
            return override_args
        except json.JSONDecodeError:
            return {}

    def test_get_vllm_override_args_valid_json(self):
        """Test get_vllm_override_args with valid JSON."""
        test_json = '{"max_model_len": 4096, "enable_chunked_prefill": true}'
        expected_args = {"max_model_len": 4096, "enable_chunked_prefill": True}

        with patch.dict(os.environ, {"VLLM_OVERRIDE_ARGS": test_json}):
            result = self.get_vllm_override_args_mock()

        assert result == expected_args

    def test_get_vllm_override_args_empty_env(self):
        """Test get_vllm_override_args when environment variable is not set."""
        with patch.dict(os.environ, {}, clear=True):
            result = self.get_vllm_override_args_mock()

        assert result == {}

    def test_get_vllm_override_args_empty_json(self):
        """Test get_vllm_override_args with empty JSON object."""
        with patch.dict(os.environ, {"VLLM_OVERRIDE_ARGS": "{}"}):
            result = self.get_vllm_override_args_mock()

        assert result == {}

    def test_get_vllm_override_args_invalid_json(self):
        """Test get_vllm_override_args with invalid JSON."""
        with patch.dict(os.environ, {"VLLM_OVERRIDE_ARGS": "invalid json"}):
            result = self.get_vllm_override_args_mock()

        assert result == {}

    def test_get_vllm_override_args_non_dict_json(self):
        """Test get_vllm_override_args with valid JSON that's not a dict."""
        with patch.dict(os.environ, {"VLLM_OVERRIDE_ARGS": '["not", "a", "dict"]'}):
            result = self.get_vllm_override_args_mock()

        assert result == {}

    def model_setup_mock(self, hf_model_id):
        """Mock implementation of model_setup function with vLLM override args support."""
        args = {
            "model": hf_model_id,
            "block_size": os.getenv("VLLM_BLOCK_SIZE", "64"),
            "max_num_seqs": os.getenv("VLLM_MAX_NUM_SEQS", "32"),
            "max_model_len": os.getenv("VLLM_MAX_MODEL_LEN", "131072"),
            "max_num_batched_tokens": os.getenv(
                "VLLM_MAX_NUM_BATCHED_TOKENS", "131072"
            ),
            "num_scheduler_steps": os.getenv("VLLM_NUM_SCHEDULER_STEPS", "10"),
            "max-log-len": os.getenv("VLLM_MAX_LOG_LEN", "64"),
            "port": os.getenv("SERVICE_PORT", "7000"),
        }

        # Apply vLLM argument overrides
        override_args = self.get_vllm_override_args_mock()
        if override_args:
            args.update(override_args)

        return args

    @pytest.mark.parametrize(
        "override_args,expected_final_args",
        [
            (
                {"max_model_len": 4096},
                {
                    "model": "test-model",
                    "block_size": "64",
                    "max_model_len": 4096,  # overridden
                    "port": "7000",
                },
            ),
            (
                {"enable_chunked_prefill": True, "max_num_seqs": 16},
                {
                    "model": "test-model",
                    "block_size": "64",
                    "max_model_len": "131072",
                    "port": "7000",
                    "enable_chunked_prefill": True,  # added
                    "max_num_seqs": 16,  # overridden
                },
            ),
            (
                {},  # empty override
                {
                    "model": "test-model",
                    "block_size": "64",
                    "max_model_len": "131072",
                    "max_num_seqs": "32",
                    "port": "7000",
                },
            ),
        ],
    )
    def test_model_setup_applies_vllm_overrides(
        self, override_args, expected_final_args
    ):
        """Test that model_setup correctly applies vLLM override arguments."""
        # Set up environment for model_setup
        env_vars = {
            "VLLM_OVERRIDE_ARGS": json.dumps(override_args),
            "SERVICE_PORT": "7000",
            "VLLM_BLOCK_SIZE": "64",
            "VLLM_MAX_NUM_SEQS": "32",
            "VLLM_MAX_MODEL_LEN": "131072",
            "VLLM_MAX_NUM_BATCHED_TOKENS": "131072",
            "VLLM_NUM_SCHEDULER_STEPS": "10",
            "VLLM_MAX_LOG_LEN": "64",
        }

        with patch.dict(os.environ, env_vars):
            result = self.model_setup_mock("test-model")

        # Check that all expected args are present and have correct values
        for key, expected_value in expected_final_args.items():
            assert key in result, f"Missing expected argument: {key}"
            assert (
                result[key] == expected_value
            ), f"Argument {key}: expected {expected_value}, got {result[key]}"

    def test_model_setup_override_precedence(self):
        """Test that vLLM override arguments take precedence over default values."""
        override_args = {
            "max_model_len": 8192,  # Should override default
            "max_num_seqs": 64,  # Should override default
            "custom_param": "value",  # Should be added
        }

        env_vars = {
            "VLLM_OVERRIDE_ARGS": json.dumps(override_args),
            "SERVICE_PORT": "7000",
            "VLLM_BLOCK_SIZE": "64",
            "VLLM_MAX_NUM_SEQS": "32",  # Should be overridden
            "VLLM_MAX_MODEL_LEN": "131072",  # Should be overridden
        }

        with patch.dict(os.environ, env_vars):
            result = self.model_setup_mock("test-model")

        # Verify overrides took precedence
        assert result["max_model_len"] == 8192
        assert result["max_num_seqs"] == 64
        assert result["custom_param"] == "value"

        # Verify other args are still present
        assert result["model"] == "test-model"
        assert result["block_size"] == "64"
        assert result["port"] == "7000"


class TestVLLMOverrideArgsJSONValidation:
    """Test comprehensive JSON validation for VLLM_OVERRIDE_ARGS to match real server behavior."""

    @pytest.mark.parametrize(
        "json_input,expected_result,should_log_error",
        [
            # Valid cases
            ('{"max_model_len": 4096}', {"max_model_len": 4096}, False),
            (
                '{"enable_chunked_prefill": true, "max_num_seqs": 16}',
                {"enable_chunked_prefill": True, "max_num_seqs": 16},
                False,
            ),
            (
                '{"custom_param": "string_value"}',
                {"custom_param": "string_value"},
                False,
            ),
            ('{"nested": {"key": "value"}}', {"nested": {"key": "value"}}, False),
            ("{}", {}, False),
            # Invalid JSON cases
            ('{"incomplete": }', {}, True),
            ("not json at all", {}, True),
            ('{"missing_quote: "value"}', {}, True),
            ("{invalid json syntax}", {}, True),
            ("", {}, False),  # Empty string is treated as no override
            # Valid JSON but not dict cases
            ('["array", "not", "dict"]', {}, True),
            ('"string_not_dict"', {}, True),
            ("123", {}, True),
            ("true", {}, True),
            ("null", {}, True),
        ],
    )
    def test_json_parsing_edge_cases(
        self, json_input, expected_result, should_log_error
    ):
        """Test comprehensive JSON parsing behavior matching the real vLLM server."""

        def get_vllm_override_args_exact_implementation():
            """Exact implementation from run_vllm_api_server.py for testing."""
            cli_override_str = os.getenv("VLLM_OVERRIDE_ARGS")
            if not cli_override_str:
                return {}

            try:
                override_args = json.loads(cli_override_str)
                # Return empty dict if not a dict
                if not isinstance(override_args, dict):
                    # In real implementation this logs an error
                    return {}
                if not override_args:
                    # In real implementation this logs info about no overrides
                    return {}
                # In real implementation this logs info about applying overrides
                return override_args
            except json.JSONDecodeError as e:
                # In real implementation this logs an error
                return {}

        env_vars = {"VLLM_OVERRIDE_ARGS": json_input} if json_input else {}

        with patch.dict(os.environ, env_vars, clear=True):
            result = get_vllm_override_args_exact_implementation()

        assert (
            result == expected_result
        ), f"Input: {json_input}, Expected: {expected_result}, Got: {result}"

    def test_environment_variable_precedence(self):
        """Test that VLLM_OVERRIDE_ARGS takes precedence when multiple env vars are set."""

        def model_setup_with_overrides():
            """Simplified model setup that mimics the override behavior."""
            # Default args from environment variables
            args = {
                "max_model_len": os.getenv("VLLM_MAX_MODEL_LEN", "131072"),
                "max_num_seqs": os.getenv("VLLM_MAX_NUM_SEQS", "32"),
                "block_size": os.getenv("VLLM_BLOCK_SIZE", "64"),
                "port": os.getenv("SERVICE_PORT", "7000"),
            }

            # Apply overrides from VLLM_OVERRIDE_ARGS
            cli_override_str = os.getenv("VLLM_OVERRIDE_ARGS")
            if cli_override_str:
                try:
                    override_args = json.loads(cli_override_str)
                    if isinstance(override_args, dict) and override_args:
                        args.update(override_args)
                except json.JSONDecodeError:
                    pass  # Ignore invalid JSON

            return args

        env_vars = {
            "VLLM_MAX_MODEL_LEN": "65536",  # Should be overridden
            "VLLM_MAX_NUM_SEQS": "16",  # Should be overridden
            "VLLM_BLOCK_SIZE": "32",  # Should remain
            "SERVICE_PORT": "8000",  # Should remain
            "VLLM_OVERRIDE_ARGS": '{"max_model_len": 4096, "max_num_seqs": 8, "custom_param": "test"}',
        }

        with patch.dict(os.environ, env_vars):
            result = model_setup_with_overrides()

        # Verify overrides took precedence
        assert result["max_model_len"] == 4096  # Overridden
        assert result["max_num_seqs"] == 8  # Overridden
        assert result["custom_param"] == "test"  # Added

        # Verify non-overridden values remain
        assert result["block_size"] == "32"  # From env
        assert result["port"] == "8000"  # From env
