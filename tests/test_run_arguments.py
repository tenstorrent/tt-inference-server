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

# Add the project root to the path so we can import run.py
sys.path.insert(0, str(Path(__file__).parent.parent))

import run


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
    )


class TestArgumentParsing:
    """Compact tests for argument parsing and validation."""

    def test_required_args_success(self, base_args):
        """Test successful parsing of required arguments."""
        with patch("sys.argv", ["run.py"] + base_args):
            args = run.parse_arguments()
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
    def test_missing_required_args(self, missing_arg, remaining_args):
        """Test that missing required arguments raise SystemExit."""
        with patch("sys.argv", ["run.py"] + remaining_args):
            with pytest.raises(SystemExit):
                run.parse_arguments()

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
            with pytest.raises(SystemExit):
                run.parse_arguments()

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
        ]
        with patch("sys.argv", ["run.py"] + full_args):
            args = run.parse_arguments()

        assert args.impl == "tt-transformers"
        assert args.local_server is True
        assert args.docker_server is True
        assert args.interactive is True
        assert args.workflow_args == "param=value"
        assert args.service_port == "9000"
        assert args.disable_trace_capture is True
        assert args.dev_mode is True
        assert args.override_docker_image == "custom:latest"

        # Test defaults
        with patch("sys.argv", ["run.py"] + base_args):
            args = run.parse_arguments()

        assert args.local_server is False
        assert args.docker_server is False
        assert args.interactive is False
        assert args.workflow_args is None
        assert args.service_port == "8000"
        assert args.disable_trace_capture is False
        assert args.dev_mode is False
        assert args.override_docker_image is None

    def test_service_port_env_var(self, base_args):
        """Test SERVICE_PORT environment variable."""
        with patch.dict(os.environ, {"SERVICE_PORT": "7777"}):
            with patch("sys.argv", ["run.py"] + base_args):
                args = run.parse_arguments()
        assert args.service_port == "7777"


class TestArgsInference:
    """Tests for argument inference and validation."""

    def test_infer_impl_success(self, mock_args):
        """Test successful impl inference."""
        mock_args.impl = None
        with patch("run.logger"):
            run.infer_args(mock_args)
        assert mock_args.impl == "tt-transformers"

    def test_infer_impl_already_set(self, mock_args):
        """Test that existing impl is preserved."""
        mock_args.impl = "existing-impl"
        with patch("run.logger"):
            run.infer_args(mock_args)
        assert mock_args.impl == "existing-impl"

    def test_infer_impl_no_default(self, mock_args):
        """Test error when no default impl available."""
        mock_args.model = "NonExistentModel"
        mock_args.impl = None
        with pytest.raises(ValueError, match="does not have a default impl"):
            run.infer_args(mock_args)


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
            run.validate_runtime_args(mock_args)
        else:
            with pytest.raises(NotImplementedError):
                run.validate_runtime_args(mock_args)

    def test_server_workflow_validation(self, mock_args):
        """Test server workflow specific validation."""
        mock_args.workflow = "server"

        # Should fail without docker or local server
        with pytest.raises(ValueError, match="requires --docker-server"):
            run.validate_runtime_args(mock_args)

        # Should pass with docker server
        mock_args.docker_server = True
        run.validate_runtime_args(mock_args)

        # Should fail with local server (not implemented)
        mock_args.docker_server = False
        mock_args.local_server = True
        with pytest.raises(
            NotImplementedError, match="not implemented for --local-server"
        ):
            run.validate_runtime_args(mock_args)

    def test_conflicting_server_options(self, mock_args):
        """Test that both docker and local server raises error."""
        mock_args.docker_server = True
        mock_args.local_server = True
        with pytest.raises(
            AssertionError, match="Cannot run --docker-server and --local-server"
        ):
            run.validate_runtime_args(mock_args)

    def test_gpu_server_not_implemented(self, mock_args):
        """Test GPU with server raises NotImplementedError."""
        mock_args.device = "gpu"
        mock_args.docker_server = True
        with pytest.raises(
            NotImplementedError, match="GPU support for running inference server"
        ):
            run.validate_runtime_args(mock_args)

    def test_device_none_not_implemented(self, mock_args):
        """Test None device raises NotImplementedError."""
        mock_args.device = None
        with pytest.raises(
            NotImplementedError, match="Device detection not implemented"
        ):
            run.validate_runtime_args(mock_args)


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
                        with pytest.raises(AssertionError):
                            run.handle_secrets(mock_args)
                    else:
                        run.handle_secrets(mock_args)
                else:
                    run.handle_secrets(mock_args)

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
            run.handle_secrets(mock_args)

        assert mock_getpass.call_count == 2
        mock_write_dotenv.assert_called_once()


class TestUtilityFunctions:
    """Tests for utility functions."""

    @patch("subprocess.check_output")
    def test_get_commit_sha(self, mock_check_output):
        """Test git commit SHA retrieval."""
        mock_check_output.return_value = b"abc123def456\n"
        result = run.get_current_commit_sha()
        assert result == "abc123def456"

        # Test error handling
        mock_check_output.side_effect = subprocess.CalledProcessError(1, "git")
        with pytest.raises(subprocess.CalledProcessError):
            run.get_current_commit_sha()

    @patch("run.ensure_readwriteable_dir")
    @patch("run.get_default_workflow_root_log_dir")
    def test_validate_local_setup(self, mock_get_log_dir, mock_ensure_dir):
        """Test local setup validation."""
        mock_log_dir = Path("/tmp/test_logs")
        mock_get_log_dir.return_value = mock_log_dir

        run.validate_local_setup("test-model")

        mock_get_log_dir.assert_called_once()
        mock_ensure_dir.assert_called_once_with(mock_log_dir)


class TestMainInitializationStates:
    """Comprehensive tests for main function initialization states."""

    def test_main_client_workflow_success(self, mock_args):
        """Test successful client-side workflow execution."""
        mock_args.workflow = "benchmarks"

        mock_run_workflows = MagicMock(return_value=[0])
        mock_parse_arguments = MagicMock(return_value=mock_args)

        with patch.multiple(
            "run",
            parse_arguments=mock_parse_arguments,
            infer_args=MagicMock(),
            validate_runtime_args=MagicMock(),
            handle_secrets=MagicMock(),
            validate_local_setup=MagicMock(),
            get_model_id=MagicMock(return_value="test-model-id"),
            get_current_commit_sha=MagicMock(return_value="abc123"),
            get_run_id=MagicMock(return_value="test-run-id"),
            get_default_workflow_root_log_dir=MagicMock(return_value=Path("/tmp/logs")),
            setup_run_logger=MagicMock(),
            run_workflows=mock_run_workflows,
            logger=MagicMock(),
        ):
            with patch("datetime.datetime") as mock_datetime:
                mock_datetime.now.return_value.strftime.return_value = (
                    "2024-01-01_12-00-00"
                )
                with patch("pathlib.Path.read_text", return_value="1.0.0\n"):
                    run.main()

        # Verify key functions were called
        mock_run_workflows.assert_called_once()
        mock_parse_arguments.assert_called_once()

    def test_main_server_workflow_docker(self, mock_args):
        """Test server workflow with docker initialization."""
        mock_args.workflow = "server"
        mock_args.docker_server = True

        mock_setup_host = MagicMock(return_value={"test": "config"})
        mock_run_docker_server = MagicMock()

        with patch.multiple(
            "run",
            parse_arguments=MagicMock(return_value=mock_args),
            infer_args=MagicMock(),
            validate_runtime_args=MagicMock(),
            handle_secrets=MagicMock(),
            validate_local_setup=MagicMock(),
            get_model_id=MagicMock(return_value="test-model-id"),
            get_current_commit_sha=MagicMock(return_value="abc123"),
            get_run_id=MagicMock(return_value="test-run-id"),
            get_default_workflow_root_log_dir=MagicMock(return_value=Path("/tmp/logs")),
            setup_run_logger=MagicMock(),
            setup_host=mock_setup_host,
            run_docker_server=mock_run_docker_server,
            logger=MagicMock(),
        ):
            with patch("datetime.datetime") as mock_datetime:
                mock_datetime.now.return_value.strftime.return_value = (
                    "2024-01-01_12-00-00"
                )
                with patch("pathlib.Path.read_text", return_value="1.0.0\n"):
                    with patch.dict(
                        os.environ, {"JWT_SECRET": "test", "HF_TOKEN": "test"}
                    ):
                        run.main()

        mock_setup_host.assert_called_once()
        mock_run_docker_server.assert_called_once()

    def test_main_server_workflow_local_not_implemented(self, mock_args):
        """Test that local server raises NotImplementedError."""
        mock_args.workflow = "server"
        mock_args.local_server = True

        with patch.multiple(
            "run",
            parse_arguments=MagicMock(return_value=mock_args),
            infer_args=MagicMock(),
            validate_runtime_args=MagicMock(),
            handle_secrets=MagicMock(),
            validate_local_setup=MagicMock(),
            get_model_id=MagicMock(return_value="test-model-id"),
            get_current_commit_sha=MagicMock(return_value="abc123"),
            get_run_id=MagicMock(return_value="test-run-id"),
            get_default_workflow_root_log_dir=MagicMock(return_value=Path("/tmp/logs")),
            setup_run_logger=MagicMock(),
            logger=MagicMock(),
        ):
            with patch("datetime.datetime") as mock_datetime:
                mock_datetime.now.return_value.strftime.return_value = (
                    "2024-01-01_12-00-00"
                )
                with patch("pathlib.Path.read_text", return_value="1.0.0\n"):
                    with pytest.raises(NotImplementedError, match="TODO"):
                        run.main()

    def test_main_workflow_failure_handling(self, mock_args):
        """Test main function handles workflow failures."""
        mock_args.workflow = "benchmarks"

        mock_logger = MagicMock()

        with patch.multiple(
            "run",
            parse_arguments=MagicMock(return_value=mock_args),
            infer_args=MagicMock(),
            validate_runtime_args=MagicMock(),
            handle_secrets=MagicMock(),
            validate_local_setup=MagicMock(),
            get_model_id=MagicMock(return_value="test-model-id"),
            get_current_commit_sha=MagicMock(return_value="abc123"),
            get_run_id=MagicMock(return_value="test-run-id"),
            get_default_workflow_root_log_dir=MagicMock(return_value=Path("/tmp/logs")),
            setup_run_logger=MagicMock(),
            run_workflows=MagicMock(return_value=[1, 0]),
            logger=mock_logger,
        ):
            with patch("datetime.datetime") as mock_datetime:
                mock_datetime.now.return_value.strftime.return_value = (
                    "2024-01-01_12-00-00"
                )
                with patch("pathlib.Path.read_text", return_value="1.0.0\n"):
                    run.main()

        mock_logger.error.assert_called()

    @pytest.mark.parametrize(
        "workflow,expects_run_workflows",
        [
            ("benchmarks", True),
            ("evals", True),
            ("reports", True),
            ("release", True),
            ("server", False),  # Server workflow skips run_workflows
        ],
    )
    def test_main_workflow_execution_paths(
        self, workflow, expects_run_workflows, mock_args
    ):
        """Test different workflow execution paths."""
        mock_args.workflow = workflow
        if workflow == "server":
            mock_args.docker_server = True

        mock_run_workflows = MagicMock(return_value=[0])

        with patch.multiple(
            "run",
            parse_arguments=MagicMock(return_value=mock_args),
            infer_args=MagicMock(),
            validate_runtime_args=MagicMock(),
            handle_secrets=MagicMock(),
            validate_local_setup=MagicMock(),
            get_model_id=MagicMock(return_value="test-model-id"),
            get_current_commit_sha=MagicMock(return_value="abc123"),
            get_run_id=MagicMock(return_value="test-run-id"),
            get_default_workflow_root_log_dir=MagicMock(return_value=Path("/tmp/logs")),
            setup_run_logger=MagicMock(),
            run_workflows=mock_run_workflows,
            setup_host=MagicMock(return_value={"test": "config"}),
            run_docker_server=MagicMock(),
            logger=MagicMock(),
        ):
            with patch("datetime.datetime") as mock_datetime:
                mock_datetime.now.return_value.strftime.return_value = (
                    "2024-01-01_12-00-00"
                )
                with patch("pathlib.Path.read_text", return_value="1.0.0\n"):
                    with patch.dict(
                        os.environ, {"JWT_SECRET": "test", "HF_TOKEN": "test"}
                    ):
                        run.main()

        if expects_run_workflows:
            mock_run_workflows.assert_called_once()
        else:
            mock_run_workflows.assert_not_called()

    def test_main_environment_variable_handling(self, mock_args):
        """Test main function handles environment variables correctly."""
        mock_args.workflow = "server"
        mock_args.docker_server = True
        mock_args.override_docker_image = "custom:latest"

        mock_setup_host = MagicMock(return_value={"test": "config"})
        mock_logger = MagicMock()

        with patch.multiple(
            "run",
            parse_arguments=MagicMock(return_value=mock_args),
            infer_args=MagicMock(),
            validate_runtime_args=MagicMock(),
            handle_secrets=MagicMock(),
            validate_local_setup=MagicMock(),
            get_model_id=MagicMock(return_value="test-model-id"),
            get_current_commit_sha=MagicMock(return_value="abc123"),
            get_run_id=MagicMock(return_value="test-run-id"),
            get_default_workflow_root_log_dir=MagicMock(return_value=Path("/tmp/logs")),
            setup_run_logger=MagicMock(),
            setup_host=mock_setup_host,
            run_docker_server=MagicMock(),
            logger=mock_logger,
        ):
            with patch("datetime.datetime") as mock_datetime:
                mock_datetime.now.return_value.strftime.return_value = (
                    "2024-01-01_12-00-00"
                )
                with patch("pathlib.Path.read_text", return_value="1.0.0\n"):
                    with patch.dict(
                        os.environ,
                        {
                            "JWT_SECRET": "test",
                            "HF_TOKEN": "test",
                            "AUTOMATIC_HOST_SETUP": "true",
                        },
                    ):
                        run.main()

        # Verify environment variables are passed correctly
        mock_setup_host.assert_called_once_with(
            model_id="test-model-id",
            jwt_secret="test",
            hf_token="test",
            automatic_setup="true",
        )

        # Verify docker image override is logged
        mock_logger.info.assert_any_call("docker_image:     custom:latest")


if __name__ == "__main__":
    pytest.main([__file__])
