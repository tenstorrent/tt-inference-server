#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import pytest
import argparse
import os
import sys
import subprocess
from unittest.mock import patch, MagicMock, call
from pathlib import Path

# Add the project root to the path so we can import run.py
sys.path.insert(0, str(Path(__file__).parent.parent))

import run


class TestRunArgumentParsing:
    """Test suite for run.py argument parsing and validation."""

    def test_parse_arguments_required_args_success(self):
        """Test that required arguments are parsed correctly."""
        # Get valid choices from the actual configurations
        valid_model = "Llama-3.2-1B"  # A model that exists in MODEL_CONFIGS
        valid_workflow = "benchmarks"
        valid_device = "n150"

        test_args = [
            "--model",
            valid_model,
            "--workflow",
            valid_workflow,
            "--device",
            valid_device,
        ]

        with patch("sys.argv", ["run.py"] + test_args):
            args = run.parse_arguments()

        assert args.model == valid_model
        assert args.workflow == valid_workflow
        assert args.device == valid_device

    def test_parse_arguments_missing_required_args(self):
        """Test that missing required arguments raise SystemExit."""
        # Test missing model
        with patch(
            "sys.argv", ["run.py", "--workflow", "benchmarks", "--device", "n150"]
        ):
            with pytest.raises(SystemExit):
                run.parse_arguments()

        # Test missing workflow
        with patch(
            "sys.argv", ["run.py", "--model", "Llama-3.2-1B", "--device", "n150"]
        ):
            with pytest.raises(SystemExit):
                run.parse_arguments()

        # Test missing device
        with patch(
            "sys.argv",
            ["run.py", "--model", "Llama-3.2-1B", "--workflow", "benchmarks"],
        ):
            with pytest.raises(SystemExit):
                run.parse_arguments()

    def test_parse_arguments_invalid_choices(self):
        """Test that invalid choices for arguments raise SystemExit."""
        # Test invalid model
        with patch(
            "sys.argv",
            [
                "run.py",
                "--model",
                "invalid-model",
                "--workflow",
                "benchmarks",
                "--device",
                "n150",
            ],
        ):
            with pytest.raises(SystemExit):
                run.parse_arguments()

        # Test invalid workflow
        with patch(
            "sys.argv",
            [
                "run.py",
                "--model",
                "Llama-3.2-1B",
                "--workflow",
                "invalid-workflow",
                "--device",
                "n150",
            ],
        ):
            with pytest.raises(SystemExit):
                run.parse_arguments()

        # Test invalid device
        with patch(
            "sys.argv",
            [
                "run.py",
                "--model",
                "Llama-3.2-1B",
                "--workflow",
                "benchmarks",
                "--device",
                "invalid-device",
            ],
        ):
            with pytest.raises(SystemExit):
                run.parse_arguments()

    def test_parse_arguments_optional_args(self):
        """Test that optional arguments are parsed correctly."""
        test_args = [
            "--model",
            "Llama-3.2-1B",
            "--workflow",
            "benchmarks",
            "--device",
            "n150",
            "--impl",
            "tt-transformers",
            "--local-server",
            "--docker-server",
            "--interactive",
            "--workflow-args",
            "param1=value1 param2=value2",
            "--service-port",
            "9000",
            "--disable-trace-capture",
            "--dev-mode",
            "--override-docker-image",
            "custom-image:latest",
        ]

        with patch("sys.argv", ["run.py"] + test_args):
            args = run.parse_arguments()

        assert args.impl == "tt-transformers"
        assert args.local_server is True
        assert args.docker_server is True
        assert args.interactive is True
        assert args.workflow_args == "param1=value1 param2=value2"
        assert args.service_port == "9000"
        assert args.disable_trace_capture is True
        assert args.dev_mode is True
        assert args.override_docker_image == "custom-image:latest"

    def test_parse_arguments_default_values(self):
        """Test that default values are set correctly."""
        test_args = [
            "--model",
            "Llama-3.2-1B",
            "--workflow",
            "benchmarks",
            "--device",
            "n150",
        ]

        with patch("sys.argv", ["run.py"] + test_args):
            args = run.parse_arguments()

        assert args.local_server is False
        assert args.docker_server is False
        assert args.interactive is False
        assert args.workflow_args is None
        assert args.service_port == "8000"  # Default from env or hardcoded
        assert args.disable_trace_capture is False
        assert args.dev_mode is False
        assert args.override_docker_image is None

    def test_service_port_environment_variable(self):
        """Test that SERVICE_PORT environment variable is used as default."""
        test_args = [
            "--model",
            "Llama-3.2-1B",
            "--workflow",
            "benchmarks",
            "--device",
            "n150",
        ]

        with patch.dict(os.environ, {"SERVICE_PORT": "7777"}):
            with patch("sys.argv", ["run.py"] + test_args):
                args = run.parse_arguments()

        assert args.service_port == "7777"


class TestInferArgs:
    """Test suite for the infer_args function."""

    def test_infer_args_success(self):
        """Test successful impl inference."""
        # Create a mock args object
        args = argparse.Namespace()
        args.model = "Llama-3.2-1B"
        args.device = "n150"
        args.impl = None

        with patch("run.logger") as mock_logger:
            run.infer_args(args)

        # Should infer tt-transformers as the impl
        assert args.impl == "tt-transformers"
        mock_logger.info.assert_called()

    def test_infer_args_already_set(self):
        """Test that existing impl is not overridden."""
        args = argparse.Namespace()
        args.model = "Llama-3.2-1B"
        args.device = "n150"
        args.impl = "existing-impl"

        with patch("run.logger") as mock_logger:
            run.infer_args(args)

        # Should keep the existing impl
        assert args.impl == "existing-impl"
        mock_logger.info.assert_called_with(
            "Using impl:=existing-impl for model:=Llama-3.2-1B"
        )

    def test_infer_args_no_default_impl(self):
        """Test error when no default impl can be inferred."""
        args = argparse.Namespace()
        args.model = "NonExistentModel"
        args.device = "n150"
        args.impl = None

        with pytest.raises(ValueError, match="does not have a default impl"):
            run.infer_args(args)


class TestValidateRuntimeArgs:
    """Test suite for the validate_runtime_args function."""

    def test_validate_runtime_args_benchmarks_success(self):
        """Test successful validation for benchmarks workflow."""
        args = argparse.Namespace()
        args.model = "Llama-3.2-1B"
        args.device = "n150"
        args.impl = "tt-transformers"
        args.workflow = "benchmarks"
        args.docker_server = False
        args.local_server = False

        # Should not raise any exception
        run.validate_runtime_args(args)

    def test_validate_runtime_args_evals_success(self):
        """Test successful validation for evals workflow."""
        args = argparse.Namespace()
        args.model = "Mistral-7B-Instruct-v0.3"  # Model that exists in EVAL_CONFIGS
        args.device = "n150"
        args.impl = "tt-transformers"
        args.workflow = "evals"
        args.docker_server = False
        args.local_server = False

        # Should not raise any exception
        run.validate_runtime_args(args)

    def test_validate_runtime_args_server_requires_docker_or_local(self):
        """Test that server workflow requires --docker-server or --local-server."""
        args = argparse.Namespace()
        args.model = "Llama-3.2-1B"
        args.device = "n150"
        args.impl = "tt-transformers"
        args.workflow = "server"
        args.docker_server = False
        args.local_server = False

        with pytest.raises(ValueError, match="requires --docker-server argument"):
            run.validate_runtime_args(args)

    def test_validate_runtime_args_server_with_docker_success(self):
        """Test successful validation for server workflow with docker."""
        args = argparse.Namespace()
        args.model = "Llama-3.2-1B"
        args.device = "n150"
        args.impl = "tt-transformers"
        args.workflow = "server"
        args.docker_server = True
        args.local_server = False

        # Should not raise any exception
        run.validate_runtime_args(args)

    def test_validate_runtime_args_server_local_not_implemented(self):
        """Test that server workflow with local server raises NotImplementedError."""
        args = argparse.Namespace()
        args.model = "Llama-3.2-1B"
        args.device = "n150"
        args.impl = "tt-transformers"
        args.workflow = "server"
        args.docker_server = False
        args.local_server = True

        with pytest.raises(
            NotImplementedError, match="not implemented for --local-server"
        ):
            run.validate_runtime_args(args)

    def test_validate_runtime_args_tests_not_implemented(self):
        """Test that tests workflow raises NotImplementedError."""
        args = argparse.Namespace()
        args.model = "Llama-3.2-1B"
        args.device = "n150"
        args.impl = "tt-transformers"
        args.workflow = "tests"
        args.docker_server = False
        args.local_server = False

        with pytest.raises(NotImplementedError, match="not implemented yet"):
            run.validate_runtime_args(args)

    def test_validate_runtime_args_both_docker_and_local_error(self):
        """Test that both --docker-server and --local-server raises assertion error."""
        args = argparse.Namespace()
        args.model = "Llama-3.2-1B"
        args.device = "n150"
        args.impl = "tt-transformers"
        args.workflow = "benchmarks"
        args.docker_server = True
        args.local_server = True

        with pytest.raises(
            AssertionError, match="Cannot run --docker-server and --local-server"
        ):
            run.validate_runtime_args(args)

    def test_validate_runtime_args_gpu_with_server_not_implemented(self):
        """Test that GPU device with server raises NotImplementedError."""
        args = argparse.Namespace()
        args.model = "Llama-3.2-1B"
        args.device = "gpu"
        args.impl = "tt-transformers"
        args.workflow = "benchmarks"
        args.docker_server = True
        args.local_server = False

        with pytest.raises(
            NotImplementedError,
            match="GPU support for running inference server not implemented",
        ):
            run.validate_runtime_args(args)

    def test_validate_runtime_args_unsupported_device_for_model(self):
        """Test that unsupported device for model raises KeyError when model_id doesn't exist."""
        args = argparse.Namespace()
        args.model = "QwQ-32B"  # This model only supports T3K and GPU
        args.device = "n150"  # Not supported for this model
        args.impl = "tt-transformers"
        args.workflow = "benchmarks"
        args.docker_server = False
        args.local_server = False

        # The function tries to get MODEL_CONFIGS[model_id] but the model_id doesn't exist
        # for unsupported device combinations, so it raises KeyError
        with pytest.raises(KeyError):
            run.validate_runtime_args(args)

    def test_validate_runtime_args_reports_workflow_success(self):
        """Test that reports workflow passes validation."""
        args = argparse.Namespace()
        args.model = "Llama-3.2-1B"
        args.device = "n150"
        args.impl = "tt-transformers"
        args.workflow = "reports"
        args.docker_server = False
        args.local_server = False

        # Should not raise any exception
        run.validate_runtime_args(args)

    def test_validate_runtime_args_release_workflow_success(self):
        """Test successful validation for release workflow."""
        args = argparse.Namespace()
        args.model = "Mistral-7B-Instruct-v0.3"  # Model that exists in both EVAL_CONFIGS and BENCHMARK_CONFIGS
        args.device = "n150"
        args.impl = "tt-transformers"
        args.workflow = "release"
        args.docker_server = False
        args.local_server = False

        # Should not raise any exception
        run.validate_runtime_args(args)

    def test_validate_runtime_args_benchmarks_with_override(self):
        """Test benchmarks workflow with OVERRIDE_BENCHMARKS environment variable."""
        args = argparse.Namespace()
        args.model = "Llama-3.2-1B"
        args.device = "n150"
        args.impl = "tt-transformers"
        args.workflow = "benchmarks"
        args.docker_server = False
        args.local_server = False

        with patch.dict(os.environ, {"OVERRIDE_BENCHMARKS": "true"}):
            with patch("run.logger") as mock_logger:
                run.validate_runtime_args(args)
                mock_logger.warning.assert_called_with(
                    "OVERRIDE_BENCHMARKS is active, using override benchmarks"
                )

    def test_validate_runtime_args_device_none_not_implemented(self):
        """Test that None device raises NotImplementedError."""
        args = argparse.Namespace()
        args.model = "Llama-3.2-1B"
        args.device = None
        args.impl = "tt-transformers"
        args.workflow = "benchmarks"
        args.docker_server = False
        args.local_server = False

        # The function should raise NotImplementedError before trying to get model_id
        # when device is None, so we don't need to mock get_model_id
        with pytest.raises(
            NotImplementedError, match="Device detection not implemented yet"
        ):
            run.validate_runtime_args(args)


class TestHandleSecrets:
    """Test suite for the handle_secrets function."""

    @patch("run.load_dotenv")
    @patch("run.logger")
    def test_handle_secrets_existing_dotenv(self, mock_logger, mock_load_dotenv):
        """Test that existing .env file is used."""
        mock_load_dotenv.return_value = True

        args = argparse.Namespace()
        args.workflow = "server"
        args.docker_server = True
        args.interactive = False

        with patch.dict(
            os.environ, {"JWT_SECRET": "test-secret", "HF_TOKEN": "test-token"}
        ):
            run.handle_secrets(args)

        mock_logger.info.assert_called_with("Using secrets from .env file.")

    @patch("run.load_dotenv")
    @patch("run.write_dotenv")
    @patch("getpass.getpass")
    def test_handle_secrets_prompt_for_secrets(
        self, mock_getpass, mock_write_dotenv, mock_load_dotenv
    ):
        """Test that user is prompted for secrets when .env doesn't exist."""
        mock_load_dotenv.side_effect = [
            False,
            True,
        ]  # First call returns False, second returns True
        mock_getpass.side_effect = ["test-secret", "test-token"]

        args = argparse.Namespace()
        args.workflow = "server"
        args.docker_server = True
        args.interactive = False

        with patch.dict(os.environ, {}, clear=True):
            run.handle_secrets(args)

        mock_write_dotenv.assert_called_once()
        assert mock_getpass.call_count == 2

    def test_handle_secrets_client_side_workflow_no_secrets_required(self):
        """Test that client-side workflows (benchmarks, evals) don't require secrets."""
        args = argparse.Namespace()
        args.workflow = "benchmarks"  # Client-side workflow
        args.docker_server = False
        args.interactive = False

        # Should not raise any exception even without secrets
        run.handle_secrets(args)

    def test_handle_secrets_interactive_mode_no_secrets_required(self):
        """Test that interactive mode doesn't require secrets for any workflow."""
        # Test server workflow in interactive mode
        args = argparse.Namespace()
        args.workflow = "server"
        args.docker_server = True
        args.interactive = True

        # Should not raise any exception even without secrets
        run.handle_secrets(args)

        # Test release workflow in interactive mode
        args.workflow = "release"
        run.handle_secrets(args)

    @patch("run.load_dotenv")
    @patch("run.write_dotenv")
    @patch("getpass.getpass")
    def test_handle_secrets_server_docker_workflow_requires_both_secrets(
        self, mock_getpass, mock_write_dotenv, mock_load_dotenv
    ):
        """Test that server workflow with docker requires both JWT_SECRET and HF_TOKEN when not interactive."""
        mock_load_dotenv.side_effect = [False, True]
        mock_getpass.side_effect = ["test-jwt-secret", "test-hf-token"]

        args = argparse.Namespace()
        args.workflow = "server"
        args.docker_server = True
        args.interactive = False

        with patch.dict(os.environ, {}, clear=True):
            run.handle_secrets(args)

        # Should prompt for both secrets
        expected_calls = [
            call("Enter your JWT_SECRET: "),
            call("Enter your HF_TOKEN: "),
        ]
        mock_getpass.assert_has_calls(expected_calls)
        mock_write_dotenv.assert_called_once_with(
            {"JWT_SECRET": "test-jwt-secret", "HF_TOKEN": "test-hf-token"}
        )

    @patch("run.load_dotenv")
    @patch("run.write_dotenv")
    def test_handle_secrets_server_no_docker_only_hf_token_required(
        self, mock_write_dotenv, mock_load_dotenv
    ):
        """Test that server workflow without docker only requires HF_TOKEN when not interactive."""
        mock_load_dotenv.side_effect = [False, True]

        args = argparse.Namespace()
        args.workflow = "server"
        args.docker_server = False
        args.interactive = False

        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "getpass.getpass", side_effect=["test-hf-token"]
            ) as mock_getpass:
                run.handle_secrets(args)

        # Should only prompt for HF_TOKEN, not JWT_SECRET
        mock_getpass.assert_called_once_with("Enter your HF_TOKEN: ")
        mock_write_dotenv.assert_called_once_with({"HF_TOKEN": "test-hf-token"})

    @patch("run.load_dotenv")
    @patch("run.write_dotenv")
    def test_handle_secrets_hf_token_only_workflows(
        self, mock_write_dotenv, mock_load_dotenv
    ):
        """Test that release/reports/tests workflows only require HF_TOKEN when not interactive."""
        mock_load_dotenv.side_effect = [False, True]

        # Test release workflow
        args = argparse.Namespace()
        args.workflow = "release"
        args.docker_server = False
        args.interactive = False

        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "getpass.getpass", side_effect=["test-hf-token"]
            ) as mock_getpass:
                run.handle_secrets(args)

        # Should only prompt for HF_TOKEN, not JWT_SECRET
        mock_getpass.assert_called_once_with("Enter your HF_TOKEN: ")
        mock_write_dotenv.assert_called_once_with({"HF_TOKEN": "test-hf-token"})

    @patch("run.load_dotenv")
    @patch("run.write_dotenv")
    @patch("getpass.getpass")
    def test_handle_secrets_partial_env_vars_from_environment(
        self, mock_getpass, mock_write_dotenv, mock_load_dotenv
    ):
        """Test when some secrets are in environment and others need to be prompted."""
        mock_load_dotenv.side_effect = [False, True]
        mock_getpass.side_effect = [
            "prompted-hf-token"
        ]  # Only HF_TOKEN will be prompted

        args = argparse.Namespace()
        args.workflow = "server"
        args.docker_server = True
        args.interactive = False

        # JWT_SECRET is already in environment, HF_TOKEN is not
        with patch.dict(os.environ, {"JWT_SECRET": "existing-jwt-secret"}, clear=True):
            run.handle_secrets(args)

        # Should only prompt for HF_TOKEN
        mock_getpass.assert_called_once_with("Enter your HF_TOKEN: ")
        mock_write_dotenv.assert_called_once_with(
            {"JWT_SECRET": "existing-jwt-secret", "HF_TOKEN": "prompted-hf-token"}
        )

    @patch("run.load_dotenv")
    def test_handle_secrets_existing_dotenv_missing_required_vars(
        self, mock_load_dotenv
    ):
        """Test that assertion error is raised when .env exists but required vars are missing."""
        mock_load_dotenv.return_value = True  # .env file exists

        args = argparse.Namespace()
        args.workflow = "server"
        args.docker_server = True
        args.interactive = False

        # .env file exists but JWT_SECRET is missing
        with patch.dict(os.environ, {"HF_TOKEN": "test-token"}, clear=True):
            with pytest.raises(
                AssertionError,
                match="Required environment variable JWT_SECRET is not set",
            ):
                run.handle_secrets(args)

    @patch("run.load_dotenv")
    @patch("run.write_dotenv")
    def test_handle_secrets_write_dotenv_failure(
        self, mock_write_dotenv, mock_load_dotenv
    ):
        """Test assertion error when write_dotenv succeeds but subsequent load_dotenv fails."""
        mock_load_dotenv.side_effect = [
            False,
            False,
        ]  # First call False, second call also False

        args = argparse.Namespace()
        args.workflow = "server"
        args.docker_server = True
        args.interactive = False

        with patch("getpass.getpass", side_effect=["test-secret", "test-token"]):
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(
                    AssertionError,
                    match="load_dotenv\\(\\) failed after write_dotenv\\(env_vars\\)",
                ):
                    run.handle_secrets(args)

    @patch("run.load_dotenv")
    @patch("run.write_dotenv")
    @patch("getpass.getpass")
    def test_handle_secrets_empty_secret_input_assertion_error(
        self, mock_getpass, mock_write_dotenv, mock_load_dotenv
    ):
        """Test assertion error when user provides empty secret."""
        mock_load_dotenv.side_effect = [False, True]
        mock_getpass.side_effect = ["", "test-token"]  # Empty JWT_SECRET

        args = argparse.Namespace()
        args.workflow = "server"
        args.docker_server = True
        args.interactive = False

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(AssertionError):
                run.handle_secrets(args)


class TestGetCurrentCommitSha:
    """Test suite for the get_current_commit_sha function."""

    @patch("subprocess.check_output")
    def test_get_current_commit_sha_success(self, mock_check_output):
        """Test successful git commit SHA retrieval."""
        mock_check_output.return_value = b"abc123def456\n"

        result = run.get_current_commit_sha()

        assert result == "abc123def456"
        mock_check_output.assert_called_once()
        # Verify git command structure
        call_args = mock_check_output.call_args[0][0]
        assert call_args[0] == "git"
        assert "-C" in call_args
        assert "rev-parse" in call_args
        assert "HEAD" in call_args

    @patch("subprocess.check_output")
    def test_get_current_commit_sha_subprocess_error(self, mock_check_output):
        """Test that subprocess errors are propagated."""
        mock_check_output.side_effect = subprocess.CalledProcessError(1, "git")

        with pytest.raises(subprocess.CalledProcessError):
            run.get_current_commit_sha()

    @patch("subprocess.check_output")
    def test_get_current_commit_sha_strips_whitespace(self, mock_check_output):
        """Test that whitespace is properly stripped from git output."""
        mock_check_output.return_value = b"  abc123def456  \n\t  "

        result = run.get_current_commit_sha()

        assert result == "abc123def456"


class TestValidateLocalSetup:
    """Test suite for the validate_local_setup function."""

    @patch("run.ensure_readwriteable_dir")
    @patch("run.get_default_workflow_root_log_dir")
    def test_validate_local_setup_success(self, mock_get_log_dir, mock_ensure_dir):
        """Test successful local setup validation."""
        mock_log_dir = Path("/tmp/test_logs")
        mock_get_log_dir.return_value = mock_log_dir

        run.validate_local_setup("test-model")

        mock_get_log_dir.assert_called_once()
        mock_ensure_dir.assert_called_once_with(mock_log_dir)

    @patch("run.ensure_readwriteable_dir")
    @patch("run.get_default_workflow_root_log_dir")
    def test_validate_local_setup_directory_error(
        self, mock_get_log_dir, mock_ensure_dir
    ):
        """Test that directory creation errors are propagated."""
        mock_log_dir = Path("/tmp/test_logs")
        mock_get_log_dir.return_value = mock_log_dir
        mock_ensure_dir.side_effect = PermissionError("Cannot create directory")

        with pytest.raises(PermissionError):
            run.validate_local_setup("test-model")


class TestMainFunction:
    """Test suite for the main function."""

    def create_mock_args(self, **kwargs):
        """Helper to create mock args with default values."""
        defaults = {
            "model": "Llama-3.2-1B",
            "workflow": "benchmarks",
            "device": "n150",
            "impl": "tt-transformers",
            "docker_server": False,
            "local_server": False,
            "interactive": False,
            "workflow_args": None,
            "override_docker_image": None,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    @patch("run.run_workflows")
    @patch("run.run_docker_server")
    @patch("run.setup_host")
    @patch("run.setup_run_logger")
    @patch("run.get_current_commit_sha")
    @patch("run.validate_local_setup")
    @patch("run.handle_secrets")
    @patch("run.validate_runtime_args")
    @patch("run.infer_args")
    @patch("run.parse_arguments")
    @patch("run.get_model_id")
    @patch("run.get_run_id")
    @patch("run.get_default_workflow_root_log_dir")
    @patch("datetime.datetime")
    @patch("builtins.open", new_callable=MagicMock)
    def test_main_benchmarks_workflow_success(
        self,
        mock_open,
        mock_datetime,
        mock_get_log_dir,
        mock_get_run_id,
        mock_get_model_id,
        mock_parse_args,
        mock_infer_args,
        mock_validate_runtime,
        mock_handle_secrets,
        mock_validate_local,
        mock_get_commit_sha,
        mock_setup_logger,
        mock_setup_host,
        mock_run_docker_server,
        mock_run_workflows,
    ):
        """Test successful main function execution for benchmarks workflow."""
        # Setup mocks
        args = self.create_mock_args(workflow="benchmarks")
        mock_parse_args.return_value = args
        mock_get_model_id.return_value = "test-model-id"
        mock_get_commit_sha.return_value = "abc123"
        mock_datetime.now.return_value.strftime.return_value = "2024-01-01_12-00-00"
        mock_get_run_id.return_value = "test-run-id"
        mock_get_log_dir.return_value = Path("/tmp/logs")
        mock_run_workflows.return_value = [0]  # Success return code

        # Mock Path.read_text for VERSION file
        with patch("pathlib.Path.read_text", return_value="1.0.0\n"):
            with patch("run.logger") as mock_logger:
                run.main()

        # Verify function calls
        mock_parse_args.assert_called_once()
        mock_infer_args.assert_called_once_with(args)
        mock_validate_runtime.assert_called_once_with(args)
        mock_handle_secrets.assert_called_once_with(args)
        mock_validate_local.assert_called_once_with(model_name=args.model)
        mock_get_commit_sha.assert_called_once()
        mock_setup_logger.assert_called_once()
        mock_run_workflows.assert_called_once_with(args)

        # Verify docker server was not called for benchmarks
        mock_run_docker_server.assert_not_called()
        mock_setup_host.assert_not_called()

        # Verify success logging
        mock_logger.info.assert_any_call("✅ Completed run.py successfully.")

    @patch("run.run_workflows")
    @patch("run.run_docker_server")
    @patch("run.setup_host")
    @patch("run.setup_run_logger")
    @patch("run.get_current_commit_sha")
    @patch("run.validate_local_setup")
    @patch("run.handle_secrets")
    @patch("run.validate_runtime_args")
    @patch("run.infer_args")
    @patch("run.parse_arguments")
    @patch("run.get_model_id")
    @patch("run.get_run_id")
    @patch("run.get_default_workflow_root_log_dir")
    @patch("datetime.datetime")
    @patch.dict(os.environ, {"JWT_SECRET": "test-secret", "HF_TOKEN": "test-token"})
    def test_main_server_workflow_with_docker(
        self,
        mock_datetime,
        mock_get_log_dir,
        mock_get_run_id,
        mock_get_model_id,
        mock_parse_args,
        mock_infer_args,
        mock_validate_runtime,
        mock_handle_secrets,
        mock_validate_local,
        mock_get_commit_sha,
        mock_setup_logger,
        mock_setup_host,
        mock_run_docker_server,
        mock_run_workflows,
    ):
        """Test main function execution for server workflow with docker."""
        # Setup mocks
        args = self.create_mock_args(workflow="server", docker_server=True)
        mock_parse_args.return_value = args
        mock_get_model_id.return_value = "test-model-id"
        mock_get_commit_sha.return_value = "abc123"
        mock_datetime.now.return_value.strftime.return_value = "2024-01-01_12-00-00"
        mock_get_run_id.return_value = "test-run-id"
        mock_get_log_dir.return_value = Path("/tmp/logs")
        mock_setup_host.return_value = {"test": "config"}

        # Mock Path.read_text for VERSION file
        with patch("pathlib.Path.read_text", return_value="1.0.0\n"):
            with patch("run.logger") as mock_logger:
                run.main()

        # Verify docker server setup was called
        mock_setup_host.assert_called_once_with(
            model_id="test-model-id",
            jwt_secret="test-secret",
            hf_token="test-token",
            automatic_setup=None,
        )
        mock_run_docker_server.assert_called_once_with(args, {"test": "config"})

        # Verify workflows was not called for server workflow
        mock_run_workflows.assert_not_called()

        # Verify server completion logging
        mock_logger.info.assert_any_call(
            "Completed server workflow, skipping run_workflows()."
        )

    @patch("run.run_workflows")
    @patch("run.run_docker_server")
    @patch("run.setup_host")
    @patch("run.setup_run_logger")
    @patch("run.get_current_commit_sha")
    @patch("run.validate_local_setup")
    @patch("run.handle_secrets")
    @patch("run.validate_runtime_args")
    @patch("run.infer_args")
    @patch("run.parse_arguments")
    @patch("run.get_model_id")
    @patch("run.get_run_id")
    @patch("run.get_default_workflow_root_log_dir")
    @patch("datetime.datetime")
    def test_main_local_server_not_implemented(
        self,
        mock_datetime,
        mock_get_log_dir,
        mock_get_run_id,
        mock_get_model_id,
        mock_parse_args,
        mock_infer_args,
        mock_validate_runtime,
        mock_handle_secrets,
        mock_validate_local,
        mock_get_commit_sha,
        mock_setup_logger,
        mock_setup_host,
        mock_run_docker_server,
        mock_run_workflows,
    ):
        """Test that local server raises NotImplementedError."""
        # Setup mocks
        args = self.create_mock_args(workflow="server", local_server=True)
        mock_parse_args.return_value = args
        mock_get_model_id.return_value = "test-model-id"
        mock_get_commit_sha.return_value = "abc123"
        mock_datetime.now.return_value.strftime.return_value = "2024-01-01_12-00-00"
        mock_get_run_id.return_value = "test-run-id"
        mock_get_log_dir.return_value = Path("/tmp/logs")

        # Mock Path.read_text for VERSION file
        with patch("pathlib.Path.read_text", return_value="1.0.0\n"):
            with patch("run.logger"):
                with pytest.raises(NotImplementedError, match="TODO"):
                    run.main()

    @patch("run.run_workflows")
    @patch("run.run_docker_server")
    @patch("run.setup_host")
    @patch("run.setup_run_logger")
    @patch("run.get_current_commit_sha")
    @patch("run.validate_local_setup")
    @patch("run.handle_secrets")
    @patch("run.validate_runtime_args")
    @patch("run.infer_args")
    @patch("run.parse_arguments")
    @patch("run.get_model_id")
    @patch("run.get_run_id")
    @patch("run.get_default_workflow_root_log_dir")
    @patch("datetime.datetime")
    def test_main_workflow_failure(
        self,
        mock_datetime,
        mock_get_log_dir,
        mock_get_run_id,
        mock_get_model_id,
        mock_parse_args,
        mock_infer_args,
        mock_validate_runtime,
        mock_handle_secrets,
        mock_validate_local,
        mock_get_commit_sha,
        mock_setup_logger,
        mock_setup_host,
        mock_run_docker_server,
        mock_run_workflows,
    ):
        """Test main function when workflows fail."""
        # Setup mocks
        args = self.create_mock_args(workflow="benchmarks")
        mock_parse_args.return_value = args
        mock_get_model_id.return_value = "test-model-id"
        mock_get_commit_sha.return_value = "abc123"
        mock_datetime.now.return_value.strftime.return_value = "2024-01-01_12-00-00"
        mock_get_run_id.return_value = "test-run-id"
        mock_get_log_dir.return_value = Path("/tmp/logs")
        mock_run_workflows.return_value = [1, 0]  # Mixed success/failure return codes

        # Mock Path.read_text for VERSION file
        with patch("pathlib.Path.read_text", return_value="1.0.0\n"):
            with patch("run.logger") as mock_logger:
                run.main()

        # Verify failure logging
        mock_logger.error.assert_any_call(
            "⛔ run.py failed with return codes: [1, 0]. See logs above for details."
        )

    @patch("run.run_workflows")
    @patch("run.run_docker_server")
    @patch("run.setup_host")
    @patch("run.setup_run_logger")
    @patch("run.get_current_commit_sha")
    @patch("run.validate_local_setup")
    @patch("run.handle_secrets")
    @patch("run.validate_runtime_args")
    @patch("run.infer_args")
    @patch("run.parse_arguments")
    @patch("run.get_model_id")
    @patch("run.get_run_id")
    @patch("run.get_default_workflow_root_log_dir")
    @patch("datetime.datetime")
    def test_main_with_override_docker_image(
        self,
        mock_datetime,
        mock_get_log_dir,
        mock_get_run_id,
        mock_get_model_id,
        mock_parse_args,
        mock_infer_args,
        mock_validate_runtime,
        mock_handle_secrets,
        mock_validate_local,
        mock_get_commit_sha,
        mock_setup_logger,
        mock_setup_host,
        mock_run_docker_server,
        mock_run_workflows,
    ):
        """Test main function with override docker image logging."""
        # Setup mocks
        args = self.create_mock_args(
            workflow="benchmarks", override_docker_image="custom:latest"
        )
        mock_parse_args.return_value = args
        mock_get_model_id.return_value = "test-model-id"
        mock_get_commit_sha.return_value = "abc123"
        mock_datetime.now.return_value.strftime.return_value = "2024-01-01_12-00-00"
        mock_get_run_id.return_value = "test-run-id"
        mock_get_log_dir.return_value = Path("/tmp/logs")
        mock_run_workflows.return_value = [0]

        # Mock Path.read_text for VERSION file
        with patch("pathlib.Path.read_text", return_value="1.0.0\n"):
            with patch("run.logger") as mock_logger:
                run.main()

        # Verify docker image override is logged
        mock_logger.info.assert_any_call("docker_image:     custom:latest")

    @patch("run.run_workflows")
    @patch("run.run_docker_server")
    @patch("run.setup_host")
    @patch("run.setup_run_logger")
    @patch("run.get_current_commit_sha")
    @patch("run.validate_local_setup")
    @patch("run.handle_secrets")
    @patch("run.validate_runtime_args")
    @patch("run.infer_args")
    @patch("run.parse_arguments")
    @patch("run.get_model_id")
    @patch("run.get_run_id")
    @patch("run.get_default_workflow_root_log_dir")
    @patch("datetime.datetime")
    @patch.dict(os.environ, {"AUTOMATIC_HOST_SETUP": "true"})
    def test_main_with_automatic_host_setup(
        self,
        mock_datetime,
        mock_get_log_dir,
        mock_get_run_id,
        mock_get_model_id,
        mock_parse_args,
        mock_infer_args,
        mock_validate_runtime,
        mock_handle_secrets,
        mock_validate_local,
        mock_get_commit_sha,
        mock_setup_logger,
        mock_setup_host,
        mock_run_docker_server,
        mock_run_workflows,
    ):
        """Test main function with automatic host setup environment variable."""
        # Setup mocks
        args = self.create_mock_args(workflow="server", docker_server=True)
        mock_parse_args.return_value = args
        mock_get_model_id.return_value = "test-model-id"
        mock_get_commit_sha.return_value = "abc123"
        mock_datetime.now.return_value.strftime.return_value = "2024-01-01_12-00-00"
        mock_get_run_id.return_value = "test-run-id"
        mock_get_log_dir.return_value = Path("/tmp/logs")
        mock_setup_host.return_value = {"test": "config"}

        # Mock Path.read_text for VERSION file
        with patch("pathlib.Path.read_text", return_value="1.0.0\n"):
            with patch("run.logger"):
                with patch.dict(os.environ, {"JWT_SECRET": "test", "HF_TOKEN": "test"}):
                    run.main()

        # Verify setup_host was called with automatic_setup
        mock_setup_host.assert_called_once_with(
            model_id="test-model-id",
            jwt_secret="test",
            hf_token="test",
            automatic_setup="true",
        )


if __name__ == "__main__":
    pytest.main([__file__])
