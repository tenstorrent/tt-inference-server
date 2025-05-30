#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import pytest
import argparse
import os
import sys
import subprocess
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add the project root to the path so we can import from run.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from run import (
    parse_arguments,
    infer_args,
    validate_runtime_args,
    handle_secrets,
    get_current_commit_sha,
    validate_local_setup,
    main,
)
from workflows.run_docker_server import run_docker_server
from utils.vllm_run_utils import get_vllm_override_args, get_override_tt_config


@pytest.fixture
def base_args():
    """Base valid arguments for testing."""
    return [
        "--model",
        "Mistral-7B-Instruct-v0.3",
        "--workflow",
        "benchmarks",
        "--device",
        "n150",
    ]


@pytest.fixture
def mock_args():
    """Create a mock args object with default values."""
    return argparse.Namespace(
        model="Mistral-7B-Instruct-v0.3",
        workflow="benchmarks",
        device="n150",
        impl="tt-transformers",
        docker_server=False,
        local_server=False,
        interactive=False,
        workflow_args=None,
        override_docker_image=None,
        service_port="8000",
        disable_trace_capture=False,
        dev_mode=False,
        override_tt_config=None,
        vllm_override_args=None,
        device_id=None,
    )


@pytest.fixture
def mock_setup_config():
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


class TestArgumentParsing:
    """Compact tests for argument parsing and validation."""

    def test_required_args_success(self, base_args):
        """Test successful parsing of required arguments."""
        with patch("sys.argv", ["run.py"] + base_args):
            args = parse_arguments()
        assert args.model == "Mistral-7B-Instruct-v0.3"
        assert args.workflow == "benchmarks"
        assert args.device == "n150"

    @pytest.mark.parametrize(
        "missing_arg,remaining_args",
        [
            ("--model", ["--workflow", "benchmarks", "--device", "n150"]),
            ("--workflow", ["--model", "Mistral-7B-Instruct-v0.3", "--device", "n150"]),
            (
                "--device",
                ["--model", "Mistral-7B-Instruct-v0.3", "--workflow", "benchmarks"],
            ),
        ],
    )
    def test_missing_required_args(self, missing_arg, remaining_args, capsys):
        """Test that missing required arguments show proper error messages."""
        with patch("sys.argv", ["run.py"] + remaining_args):
            with pytest.raises(SystemExit):
                parse_arguments()
        captured = capsys.readouterr()
        assert f"the following arguments are required: {missing_arg}" in captured.err

    @pytest.mark.parametrize(
        "invalid_arg,invalid_value",
        [
            ("--model", "invalid-model"),
            ("--workflow", "invalid-workflow"),
            ("--device", "invalid-device"),
        ],
    )
    def test_invalid_choices(self, base_args, invalid_arg, invalid_value):
        """Test that invalid choices raise SystemExit."""
        args = base_args.copy()
        idx = args.index(invalid_arg) + 1
        args[idx] = invalid_value
        with patch("sys.argv", ["run.py"] + args):
            with patch("sys.stderr") as mock_stderr:
                with pytest.raises(SystemExit) as exc_info:
                    parse_arguments()

                # Verify it's an error exit (code 2 for argparse errors)
                assert exc_info.value.code == 2

                # Verify error message was written to stderr
                stderr_calls = [str(call) for call in mock_stderr.write.call_args_list]
                stderr_output = "".join(stderr_calls)
                assert (
                    "invalid choice" in stderr_output.lower()
                    or "error" in stderr_output.lower()
                )

    @pytest.mark.parametrize(
        "override_arg,test_value",
        [
            ("--override-tt-config", '{"data_parallel": 16}'),
            (
                "--vllm-override-args",
                '{"max_model_len": 4096, "enable_chunked_prefill": true}',
            ),
        ],
    )
    def test_override_args_parsing(self, base_args, override_arg, test_value):
        """Test parsing of override arguments."""
        args_with_override = base_args + [override_arg, test_value]

        with patch("sys.argv", ["run.py"] + args_with_override):
            args = parse_arguments()

        if override_arg == "--override-tt-config":
            assert args.override_tt_config == test_value
        else:
            assert args.vllm_override_args == test_value

    def test_optional_args_and_defaults(self, base_args):
        """Test optional arguments and default values."""
        # Test with all optional args
        full_args = base_args + [
            "--impl",
            "tt-transformers",
            "--local-server",
            "--docker-server",
            "--interactive",
            "--workflow-args",
            "param=value",
            "--service-port",
            "9000",
            "--disable-trace-capture",
            "--dev-mode",
            "--override-docker-image",
            "custom:latest",
            "--device-id",
            "1",
            "--override-tt-config",
            '{"data_parallel": 16}',
            "--vllm-override-args",
            '{"max_model_len": 4096}',
        ]
        with patch("sys.argv", ["run.py"] + full_args):
            args = parse_arguments()

        assert args.impl == "tt-transformers"
        assert args.local_server is True
        assert args.docker_server is True
        assert args.interactive is True
        assert args.workflow_args == "param=value"
        assert args.service_port == "9000"
        assert args.disable_trace_capture is True
        assert args.dev_mode is True
        assert args.override_docker_image == "custom:latest"
        assert args.device_id == "1"
        assert args.override_tt_config == '{"data_parallel": 16}'
        assert args.vllm_override_args == '{"max_model_len": 4096}'

        # Test defaults
        with patch("sys.argv", ["run.py"] + base_args):
            args = parse_arguments()

        assert args.local_server is False
        assert args.docker_server is False
        assert args.interactive is False
        assert args.workflow_args is None
        assert args.service_port == "8000"
        assert args.disable_trace_capture is False
        assert args.dev_mode is False
        assert args.override_docker_image is None
        assert args.override_tt_config is None
        assert args.vllm_override_args is None


class TestArgsInference:
    """Tests for argument inference and validation."""

    def test_infer_impl_success(self, mock_args):
        """Test successful impl inference."""
        mock_args.impl = None
        with patch("run.logger"):
            infer_args(mock_args)
        assert mock_args.impl == "tt-transformers"

    def test_infer_impl_already_set(self, mock_args):
        """Test that existing impl is preserved."""
        mock_args.impl = "existing-impl"
        with patch("run.logger"):
            infer_args(mock_args)
        assert mock_args.impl == "existing-impl"

    def test_infer_impl_no_default(self, mock_args):
        """Test error when no default impl available."""
        mock_args.model = "NonExistentModel"
        mock_args.impl = None
        with pytest.raises(ValueError, match="does not have a default impl"):
            infer_args(mock_args)


class TestRuntimeValidation:
    """Tests for runtime argument validation."""

    @pytest.mark.parametrize(
        "workflow,should_pass",
        [
            ("benchmarks", True),
            ("evals", True),  # Mistral-7B-Instruct-v0.3 is in EVAL_CONFIGS
            ("reports", True),
            ("release", True),  # Mistral-7B-Instruct-v0.3 is in both configs
            ("tests", False),  # Not implemented
        ],
    )
    def test_workflow_validation(self, mock_args, workflow, should_pass):
        """Test validation for different workflows."""
        mock_args.workflow = workflow
        if should_pass:
            validate_runtime_args(mock_args)
        else:
            with pytest.raises(NotImplementedError):
                validate_runtime_args(mock_args)

    def test_server_workflow_validation(self, mock_args):
        """Test server workflow specific validation."""
        mock_args.workflow = "server"

        # Should fail without docker or local server
        with pytest.raises(ValueError, match="requires --docker-server"):
            validate_runtime_args(mock_args)

        # Should pass with docker server
        mock_args.docker_server = True
        validate_runtime_args(mock_args)

        # Should fail with local server (not implemented)
        mock_args.docker_server = False
        mock_args.local_server = True
        with pytest.raises(
            NotImplementedError, match="not implemented for --local-server"
        ):
            validate_runtime_args(mock_args)

    def test_conflicting_server_options(self, mock_args):
        """Test that both docker and local server raises error."""
        mock_args.docker_server = True
        mock_args.local_server = True
        with pytest.raises(
            AssertionError, match="Cannot run --docker-server and --local-server"
        ):
            validate_runtime_args(mock_args)


class TestOverrideArgsIntegration:
    """Test override arguments integration from CLI to Docker to server."""

    @pytest.mark.parametrize(
        "override_type,cli_arg,env_var,test_value",
        [
            (
                "tt_config",
                "--override-tt-config",
                "OVERRIDE_TT_CONFIG",
                '{"data_parallel": 16}',
            ),
            (
                "vllm_args",
                "--vllm-override-args",
                "VLLM_OVERRIDE_ARGS",
                '{"max_model_len": 4096}',
            ),
        ],
    )
    def test_docker_server_sets_override_env_vars(
        self, mock_setup_config, override_type, cli_arg, env_var, test_value
    ):
        """Test that run_docker_server correctly sets override environment variables."""
        mock_args = MagicMock()
        mock_args.model = "Mistral-7B-Instruct-v0.3"
        mock_args.device = "n150"
        mock_args.workflow = "server"
        mock_args.service_port = "8000"
        mock_args.interactive = False
        mock_args.dev_mode = False
        mock_args.device_id = None
        mock_args.impl = "tt-transformers"
        mock_args.override_docker_image = None

        # Set the specific override arg being tested
        if override_type == "tt_config":
            mock_args.override_tt_config = test_value
            mock_args.vllm_override_args = None
        else:
            mock_args.vllm_override_args = test_value
            mock_args.override_tt_config = None

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
        ):
            # Mock model config
            mock_model_config = MagicMock()
            mock_model_config.docker_image = "test:image"
            mock_model_config.impl.impl_name = "tt-transformers"
            mock_model_config.hf_model_repo = "mistralai/Mistral-7B-Instruct-v0.3"
            mock_model_config.device_model_spec.max_concurrency = "32"
            mock_model_config.device_model_spec.max_context = "32768"
            mock_model_config.device_model_spec.override_tt_config = None
            mock_configs.__getitem__.return_value = mock_model_config

            # Call the function
            run_docker_server(mock_args, mock_setup_config)

            # Verify subprocess.Popen was called
            mock_popen.assert_called_once()
            docker_command = mock_popen.call_args[0][0]

            # Check that the environment variable is in the docker command
            found_env_var = False
            for i, arg in enumerate(docker_command):
                if arg == "-e" and i + 1 < len(docker_command):
                    env_setting = docker_command[i + 1]
                    if env_setting.startswith(f"{env_var}="):
                        found_env_var = True
                        actual_value = env_setting.split("=", 1)[1]
                        assert actual_value == test_value
                        break

            assert (
                found_env_var
            ), f"{env_var} not found in docker command: {docker_command}"

    def test_docker_server_no_override_args(self, mock_setup_config):
        """Test that run_docker_server doesn't set override env vars when not provided."""
        mock_args = MagicMock()
        mock_args.model = "Mistral-7B-Instruct-v0.3"
        mock_args.device = "n150"
        mock_args.workflow = "server"
        mock_args.service_port = "8000"
        mock_args.override_tt_config = None
        mock_args.vllm_override_args = None
        mock_args.interactive = False
        mock_args.dev_mode = False
        mock_args.device_id = None
        mock_args.impl = "tt-transformers"
        mock_args.override_docker_image = None

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
        ):
            # Mock model config
            mock_model_config = MagicMock()
            mock_model_config.docker_image = "test:image"
            mock_model_config.impl.impl_name = "tt-transformers"
            mock_model_config.hf_model_repo = "mistralai/Mistral-7B-Instruct-v0.3"
            mock_model_config.device_model_spec.max_concurrency = "32"
            mock_model_config.device_model_spec.max_context = "32768"
            mock_model_config.device_model_spec.override_tt_config = None
            mock_configs.__getitem__.return_value = mock_model_config

            # Call the function
            run_docker_server(mock_args, mock_setup_config)

            # Verify subprocess.Popen was called
            mock_popen.assert_called_once()
            docker_command = mock_popen.call_args[0][0]

            # Check that override env vars are NOT in the docker command
            override_env_vars = ["OVERRIDE_TT_CONFIG=", "VLLM_OVERRIDE_ARGS="]
            for i, arg in enumerate(docker_command):
                if arg == "-e" and i + 1 < len(docker_command):
                    env_setting = docker_command[i + 1]
                    for env_var in override_env_vars:
                        assert not env_setting.startswith(
                            env_var
                        ), f"{env_var} should not be set when not provided: {env_setting}"


class TestOverrideArgsServerProcessing:
    """Test server-side processing of override arguments."""

    @pytest.mark.parametrize(
        "env_var,processor_func,test_input,expected_output",
        [
            (
                "VLLM_OVERRIDE_ARGS",
                get_vllm_override_args,
                '{"max_model_len": 4096}',
                {"max_model_len": 4096},
            ),
            (
                "OVERRIDE_TT_CONFIG",
                get_override_tt_config,
                '{"data_parallel": 16}',
                '{"data_parallel": 16}',
            ),
            ("VLLM_OVERRIDE_ARGS", get_vllm_override_args, "{}", {}),
            ("OVERRIDE_TT_CONFIG", get_override_tt_config, "{}", None),
            ("VLLM_OVERRIDE_ARGS", get_vllm_override_args, "invalid json", {}),
            (
                "OVERRIDE_TT_CONFIG",
                get_override_tt_config,
                "invalid json",
                {},
            ),  # Returns {} for invalid JSON, not None
        ],
    )
    def test_override_args_processing(
        self, env_var, processor_func, test_input, expected_output
    ):
        """Test processing of override arguments from environment variables."""
        with patch.dict(
            os.environ, {env_var: test_input} if test_input else {}, clear=True
        ):
            result = processor_func()

        assert result == expected_output

    def test_override_args_precedence_in_model_setup(self):
        """Test that override arguments take precedence in model setup."""

        # Mock the model_setup functionality
        def mock_model_setup(hf_model_id):
            args = {
                "model": hf_model_id,
                "max_model_len": os.getenv("VLLM_MAX_MODEL_LEN", "131072"),
                "max_num_seqs": os.getenv("VLLM_MAX_NUM_SEQS", "32"),
                "port": os.getenv("SERVICE_PORT", "7000"),
                "override_tt_config": get_override_tt_config(),
            }

            # Apply vLLM argument overrides
            override_args = get_vllm_override_args()
            if override_args:
                args.update(override_args)

            return args

        env_vars = {
            "VLLM_OVERRIDE_ARGS": '{"max_model_len": 8192, "custom_param": "test"}',
            "OVERRIDE_TT_CONFIG": '{"data_parallel": 32}',
            "VLLM_MAX_MODEL_LEN": "131072",  # Should be overridden
            "VLLM_MAX_NUM_SEQS": "32",
            "SERVICE_PORT": "7000",
        }

        with patch.dict(os.environ, env_vars):
            result = mock_model_setup("test-model")

        # Verify overrides took precedence
        assert result["max_model_len"] == 8192  # Overridden
        assert result["custom_param"] == "test"  # Added
        assert result["override_tt_config"] == '{"data_parallel": 32}'

        # Verify other args are still present
        assert result["model"] == "test-model"
        assert result["max_num_seqs"] == "32"
        assert result["port"] == "7000"


class TestSecretsHandling:
    """Tests for secrets handling functionality."""

    @pytest.mark.parametrize(
        "workflow,docker_server,interactive,jwt_required,hf_required",
        [
            ("benchmarks", False, False, False, False),  # Client-side
            ("evals", False, False, False, False),  # Client-side
            ("server", True, False, True, True),  # Server with docker
            ("server", False, False, False, True),  # Server without docker
            ("server", True, True, False, False),  # Interactive mode
            ("release", False, False, False, True),  # Non-client workflow
            ("reports", False, False, False, True),  # Non-client workflow
        ],
    )
    def test_secrets_requirements(
        self, mock_args, workflow, docker_server, interactive, jwt_required, hf_required
    ):
        """Test secret requirements for different configurations."""
        mock_args.workflow = workflow
        mock_args.docker_server = docker_server
        mock_args.interactive = interactive

        env_vars = {}
        if jwt_required:
            env_vars["JWT_SECRET"] = "test-jwt"
        if hf_required:
            env_vars["HF_TOKEN"] = "test-hf"

        with patch("run.load_dotenv", return_value=True):
            with patch.dict(os.environ, env_vars, clear=True):
                if jwt_required or hf_required:
                    if not env_vars:
                        with pytest.raises(
                            AssertionError, match="is not set in .env file"
                        ):
                            handle_secrets(mock_args)
                    else:
                        handle_secrets(mock_args)
                else:
                    handle_secrets(mock_args)

    @patch("run.load_dotenv")
    @patch("run.write_dotenv")
    @patch("getpass.getpass")
    def test_secrets_prompting(
        self, mock_getpass, mock_write_dotenv, mock_load_dotenv, mock_args
    ):
        """Test prompting for missing secrets."""
        mock_args.workflow = "server"
        mock_args.docker_server = True
        mock_args.interactive = False

        mock_load_dotenv.side_effect = [False, True]
        mock_getpass.side_effect = ["test-jwt", "test-hf"]

        with patch.dict(os.environ, {}, clear=True):
            handle_secrets(mock_args)

        assert mock_getpass.call_count == 2
        mock_write_dotenv.assert_called_once()


class TestUtilityFunctions:
    """Tests for utility functions."""

    @patch("subprocess.check_output")
    def test_get_commit_sha(self, mock_check_output):
        """Test git commit SHA retrieval."""
        mock_check_output.return_value = b"abc123def456\n"
        result = get_current_commit_sha()
        assert result == "abc123def456"

        # Test error handling
        mock_check_output.side_effect = subprocess.CalledProcessError(1, "git")
        with pytest.raises(subprocess.CalledProcessError):
            get_current_commit_sha()

    @patch("run.ensure_readwriteable_dir")
    @patch("run.get_default_workflow_root_log_dir")
    def test_validate_local_setup(self, mock_get_log_dir, mock_ensure_dir):
        """Test local setup validation."""
        mock_log_dir = Path("/tmp/test_logs")
        mock_get_log_dir.return_value = mock_log_dir

        validate_local_setup("test-model")

        mock_get_log_dir.assert_called_once()
        mock_ensure_dir.assert_called_once_with(mock_log_dir)


class TestMainInitializationStates:
    """Comprehensive tests for main function initialization states."""

    def test_main_workflow_failure_handling(self, mock_args):
        """Test main function handles workflow failures."""
        mock_args.workflow = "benchmarks"

        mock_logger = MagicMock()

        mock_model_config = MagicMock()
        mock_model_config.tt_metal_commit = "test-commit"
        mock_model_config.vllm_commit = "test-vllm-commit"

        with patch("run.parse_arguments", return_value=mock_args), patch(
            "run.infer_args"
        ), patch("run.validate_runtime_args"), patch("run.handle_secrets"), patch(
            "run.validate_local_setup"
        ), patch("run.get_model_id", return_value="test-model-id"), patch(
            "run.get_current_commit_sha", return_value="abc123"
        ), patch("run.get_run_id", return_value="test-run-id"), patch(
            "run.get_default_workflow_root_log_dir", return_value=Path("/tmp/logs")
        ), patch("run.setup_run_logger"), patch(
            "run.run_workflows", return_value=[1, 0]
        ), patch("run.logger", mock_logger):
            with patch.dict("run.MODEL_CONFIGS", {"test-model-id": mock_model_config}):
                with patch("datetime.datetime") as mock_datetime:
                    mock_datetime.now.return_value.strftime.return_value = (
                        "2024-01-01_12-00-00"
                    )
                    with patch("pathlib.Path.read_text", return_value="1.0.0\n"):
                        main()

        mock_logger.error.assert_called()

    @pytest.mark.parametrize(
        "workflow,docker_server,local_server,expects_run_workflows,expects_server_setup,should_raise",
        [
            ("benchmarks", False, False, True, False, None),
            ("evals", False, False, True, False, None),
            ("reports", False, False, True, False, None),
            ("release", False, False, True, False, None),
            ("server", True, False, False, True, None),  # Server workflow with docker
            (
                "server",
                False,
                True,
                False,
                False,
                NotImplementedError,
            ),  # Local server not implemented
        ],
    )
    def test_main_workflow_execution_paths(
        self,
        workflow,
        docker_server,
        local_server,
        expects_run_workflows,
        expects_server_setup,
        should_raise,
        mock_args,
    ):
        """Test different workflow execution paths and server configurations."""
        mock_args.workflow = workflow
        mock_args.docker_server = docker_server
        mock_args.local_server = local_server

        mock_run_workflows = MagicMock(return_value=[0])
        mock_setup_host = MagicMock(return_value={"test": "config"})
        mock_run_docker_server = MagicMock()

        # Create a mock model config for the test
        mock_model_config = MagicMock()
        mock_model_config.tt_metal_commit = "test-commit"
        mock_model_config.vllm_commit = "test-vllm-commit"

        with patch("run.parse_arguments", return_value=mock_args), patch(
            "run.infer_args"
        ), patch("run.validate_runtime_args"), patch("run.handle_secrets"), patch(
            "run.validate_local_setup"
        ), patch("run.get_model_id", return_value="test-model-id"), patch(
            "run.get_current_commit_sha", return_value="abc123"
        ), patch("run.get_run_id", return_value="test-run-id"), patch(
            "run.get_default_workflow_root_log_dir", return_value=Path("/tmp/logs")
        ), patch("run.setup_run_logger"), patch(
            "run.run_workflows", mock_run_workflows
        ), patch("run.setup_host", mock_setup_host), patch(
            "run.run_docker_server", mock_run_docker_server
        ), patch("run.logger"):
            with patch.dict("run.MODEL_CONFIGS", {"test-model-id": mock_model_config}):
                with patch("datetime.datetime") as mock_datetime:
                    mock_datetime.now.return_value.strftime.return_value = (
                        "2024-01-01_12-00-00"
                    )
                    with patch("pathlib.Path.read_text", return_value="1.0.0\n"):
                        with patch.dict(
                            os.environ, {"JWT_SECRET": "test", "HF_TOKEN": "test"}
                        ):
                            if should_raise:
                                with pytest.raises(should_raise, match="TODO"):
                                    main()
                            else:
                                main()

        # Verify expectations
        if expects_run_workflows and not should_raise:
            mock_run_workflows.assert_called_once()
        else:
            mock_run_workflows.assert_not_called()

        if expects_server_setup and not should_raise:
            mock_setup_host.assert_called_once()
            mock_run_docker_server.assert_called_once()
        else:
            mock_setup_host.assert_not_called()
            mock_run_docker_server.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__])
