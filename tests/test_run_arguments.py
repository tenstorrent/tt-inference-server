#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import pytest
import argparse
import os
import sys
from unittest.mock import patch, MagicMock, call
from pathlib import Path

# Add the project root to the path so we can import run.py
sys.path.insert(0, str(Path(__file__).parent.parent))

import run
from workflows.workflow_types import WorkflowType, DeviceTypes
from workflows.model_config import MODEL_CONFIGS


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
        """Test that client-side workflows don't require secrets."""
        args = argparse.Namespace()
        args.workflow = "benchmarks"  # Client-side workflow
        args.docker_server = False
        args.interactive = False

        # Should not raise any exception even without secrets
        run.handle_secrets(args)

    def test_handle_secrets_interactive_mode_no_secrets_required(self):
        """Test that interactive mode doesn't require secrets."""
        args = argparse.Namespace()
        args.workflow = "server"
        args.docker_server = True
        args.interactive = True

        # Should not raise any exception even without secrets
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
    def test_handle_secrets_release_workflow_only_hf_token_required(
        self, mock_write_dotenv, mock_load_dotenv
    ):
        """Test that release workflow only requires HF_TOKEN when not interactive."""
        mock_load_dotenv.side_effect = [False, True]

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
    def test_handle_secrets_benchmarks_workflow_no_secrets_required(
        self, mock_write_dotenv, mock_load_dotenv
    ):
        """Test that benchmarks workflow (client-side) doesn't require secrets."""
        mock_load_dotenv.side_effect = [
            False,
            True,
        ]  # First call False, second call True

        args = argparse.Namespace()
        args.workflow = "benchmarks"
        args.docker_server = False
        args.interactive = False

        # Should not raise any exception or prompt for secrets
        with patch.dict(os.environ, {}, clear=True):
            run.handle_secrets(args)

        # Should call load_dotenv twice and write_dotenv once with empty dict
        assert mock_load_dotenv.call_count == 2
        mock_write_dotenv.assert_called_once_with({})

    @patch("run.load_dotenv")
    @patch("run.write_dotenv")
    def test_handle_secrets_evals_workflow_no_secrets_required(
        self, mock_write_dotenv, mock_load_dotenv
    ):
        """Test that evals workflow (client-side) doesn't require secrets."""
        mock_load_dotenv.side_effect = [
            False,
            True,
        ]  # First call False, second call True

        args = argparse.Namespace()
        args.workflow = "evals"
        args.docker_server = False
        args.interactive = False

        # Should not raise any exception or prompt for secrets
        with patch.dict(os.environ, {}, clear=True):
            run.handle_secrets(args)

        # Should call load_dotenv twice and write_dotenv once with empty dict
        assert mock_load_dotenv.call_count == 2
        mock_write_dotenv.assert_called_once_with({})

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
    def test_handle_secrets_existing_dotenv_all_vars_present(self, mock_load_dotenv):
        """Test successful execution when .env exists and all required vars are present."""
        mock_load_dotenv.return_value = True

        args = argparse.Namespace()
        args.workflow = "server"
        args.docker_server = True
        args.interactive = False

        with patch("run.logger") as mock_logger:
            with patch.dict(
                os.environ, {"JWT_SECRET": "test-secret", "HF_TOKEN": "test-token"}
            ):
                run.handle_secrets(args)

        mock_logger.info.assert_called_with("Using secrets from .env file.")

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

    def test_handle_secrets_interactive_server_workflow_no_requirements(self):
        """Test that interactive server workflow doesn't require any secrets."""
        args = argparse.Namespace()
        args.workflow = "server"
        args.docker_server = True
        args.interactive = True

        # Should not raise any exception even without any environment variables
        with patch.dict(os.environ, {}, clear=True):
            run.handle_secrets(args)

    def test_handle_secrets_interactive_release_workflow_no_requirements(self):
        """Test that interactive release workflow doesn't require any secrets."""
        args = argparse.Namespace()
        args.workflow = "release"
        args.docker_server = False
        args.interactive = True

        # Should not raise any exception even without any environment variables
        with patch.dict(os.environ, {}, clear=True):
            run.handle_secrets(args)

    @patch("run.load_dotenv")
    @patch("run.write_dotenv")
    def test_handle_secrets_reports_workflow_only_hf_token_required(
        self, mock_write_dotenv, mock_load_dotenv
    ):
        """Test that reports workflow only requires HF_TOKEN when not interactive."""
        mock_load_dotenv.side_effect = [False, True]

        args = argparse.Namespace()
        args.workflow = "reports"
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
    def test_handle_secrets_tests_workflow_only_hf_token_required(
        self, mock_write_dotenv, mock_load_dotenv
    ):
        """Test that tests workflow only requires HF_TOKEN when not interactive."""
        mock_load_dotenv.side_effect = [False, True]

        args = argparse.Namespace()
        args.workflow = "tests"
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


class TestWorkflowValidation:
    """Test suite for workflow-specific validation."""

    def test_all_workflow_types_are_valid_choices(self):
        """Test that all WorkflowType enum values are valid argument choices."""
        # Get valid workflows from parse_arguments
        valid_workflows = {w.name.lower() for w in WorkflowType}

        # Test each workflow type
        expected_workflows = {
            "benchmarks",
            "evals",
            "tests",
            "reports",
            "server",
            "release",
        }
        assert valid_workflows == expected_workflows

    def test_all_device_types_are_valid_choices(self):
        """Test that all DeviceTypes enum values are valid argument choices."""
        # Get valid devices from parse_arguments
        valid_devices = {device.name.lower() for device in DeviceTypes}

        # Test that common device types are included
        expected_devices = {
            "cpu",
            "e150",
            "n150",
            "p100",
            "p150",
            "n300",
            "t3k",
            "galaxy",
            "gpu",
        }
        assert valid_devices == expected_devices

    def test_model_choices_from_config(self):
        """Test that model choices come from MODEL_CONFIGS."""
        valid_models = {config.model_name for _, config in MODEL_CONFIGS.items()}

        # Should have some common models
        assert len(valid_models) > 0
        # Check for some expected models (these should exist based on the config we saw)
        expected_models = {"Llama-3.2-1B", "Llama-3.2-3B", "Mistral-7B-Instruct-v0.3"}
        assert expected_models.issubset(valid_models)


class TestIntegrationScenarios:
    """Integration tests for common usage scenarios."""

    @patch("run.handle_secrets")
    @patch("run.validate_local_setup")
    @patch("run.get_current_commit_sha")
    @patch("run.setup_run_logger")
    @patch("run.run_workflows")
    def test_benchmarks_workflow_integration(
        self,
        mock_run_workflows,
        mock_setup_logger,
        mock_get_sha,
        mock_validate_setup,
        mock_handle_secrets,
    ):
        """Test a complete benchmarks workflow scenario."""
        mock_get_sha.return_value = "abc123"
        mock_run_workflows.return_value = [0]  # Success

        test_args = [
            "--model",
            "Llama-3.2-1B",
            "--workflow",
            "benchmarks",
            "--device",
            "n150",
        ]

        with patch("sys.argv", ["run.py"] + test_args):
            with patch("run.get_default_workflow_root_log_dir") as mock_log_dir:
                mock_log_dir.return_value = Path("/tmp/test_logs")
                # Should not raise any exception
                run.main()

    @patch("run.handle_secrets")
    @patch("run.validate_local_setup")
    @patch("run.get_current_commit_sha")
    @patch("run.setup_run_logger")
    @patch("run.run_docker_server")
    @patch("run.setup_host")
    def test_server_workflow_integration(
        self,
        mock_setup_host,
        mock_run_docker,
        mock_setup_logger,
        mock_get_sha,
        mock_validate_setup,
        mock_handle_secrets,
    ):
        """Test a complete server workflow scenario."""
        mock_get_sha.return_value = "abc123"
        mock_setup_host.return_value = MagicMock()

        test_args = [
            "--model",
            "Llama-3.2-1B",
            "--workflow",
            "server",
            "--device",
            "n150",
            "--docker-server",
        ]

        with patch("sys.argv", ["run.py"] + test_args):
            with patch("run.get_default_workflow_root_log_dir") as mock_log_dir:
                with patch.dict(os.environ, {"JWT_SECRET": "test", "HF_TOKEN": "test"}):
                    mock_log_dir.return_value = Path("/tmp/test_logs")
                    # Should not raise any exception
                    run.main()


if __name__ == "__main__":
    pytest.main([__file__])
