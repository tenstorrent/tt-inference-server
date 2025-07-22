#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import tempfile
import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from argparse import Namespace

from run import main
from workflows.setup_host import HostSetupManager
from workflows.run_workflows import WorkflowSetup, run_single_workflow, run_workflows
from workflows.model_spec import MODEL_SPECS
from workflows.workflow_types import WorkflowType
from workflows.workflow_config import WORKFLOW_CONFIGS
from workflows.utils import (
    get_model_id,
    ensure_readwriteable_dir,
)


class TestWorkflowUtils:
    """Test workflow utility functions without heavy mocking."""

    def test_get_model_id_construction(self):
        """Test model ID construction logic."""
        # Test valid case
        model_id = get_model_id("tt-transformers", "Llama-3.1-8B-Instruct", "n150")
        expected = "id_tt-transformers_Llama-3.1-8B-Instruct_n150"
        assert model_id == expected

        # Test validation - None device should raise AssertionError
        with pytest.raises(AssertionError, match="Device must be a string"):
            get_model_id("tt-transformers", "Llama-3.1-8B-Instruct", None)

        # Test validation - empty string device should raise AssertionError
        with pytest.raises(
            AssertionError, match="Device cannot be empty or whitespace-only"
        ):
            get_model_id("tt-transformers", "Llama-3.1-8B-Instruct", "")

        # Test validation - whitespace-only device should raise AssertionError
        with pytest.raises(
            AssertionError, match="Device cannot be empty or whitespace-only"
        ):
            get_model_id("tt-transformers", "Llama-3.1-8B-Instruct", "   ")

        # Test validation - None impl_name should raise AssertionError
        with pytest.raises(AssertionError, match="Impl name must be a string"):
            get_model_id(None, "Llama-3.1-8B-Instruct", "n150")

        # Test validation - empty model_name should raise AssertionError
        with pytest.raises(
            AssertionError, match="Model name cannot be empty or whitespace-only"
        ):
            get_model_id("tt-transformers", "", "n150")

    def test_ensure_readwriteable_dir_creation(self):
        """Test directory creation and permission checking."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "new_dir" / "nested"

            # Test directory creation
            result = ensure_readwriteable_dir(test_dir, raise_on_fail=False)
            assert result is True
            assert test_dir.exists()
            assert test_dir.is_dir()

    def test_ensure_readwriteable_dir_existing_file_error(self):
        """Test error when path exists but is not a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "existing_file"
            test_file.write_text("content")

            # Should return False when raise_on_fail=False
            result = ensure_readwriteable_dir(test_file, raise_on_fail=False)
            assert result is False

            # Should raise when raise_on_fail=True
            with pytest.raises(ValueError, match="exists but is not a directory"):
                ensure_readwriteable_dir(test_file, raise_on_fail=True)


class TestWorkflowConfigurationValidation:
    """Test workflow configuration validation logic."""

    def test_workflow_config_consistency(self):
        """Test that workflow configurations are consistent without mocking."""
        # Verify all required workflow types have configurations
        required_workflows = [
            WorkflowType.BENCHMARKS,
            WorkflowType.EVALS,
            WorkflowType.SERVER,
            WorkflowType.REPORTS,
        ]

        for workflow_type in required_workflows:
            assert workflow_type in WORKFLOW_CONFIGS
            config = WORKFLOW_CONFIGS[workflow_type]

            # Verify configuration is valid
            assert config.workflow_type == workflow_type
            assert config.run_script_path is not None
            assert isinstance(config.run_script_path, Path)
            assert config.name is not None

    def test_model_config_data_integrity(self):
        """Test model configuration data integrity without mocking."""
        # Test that model configurations are properly structured
        for model_id, config in MODEL_SPECS.items():
            assert model_id.startswith("id_")
            assert config.model_name is not None
            assert config.impl is not None
            assert config.device_type is not None
            assert config.hf_model_repo is not None
            assert config.model_id == model_id

    def test_invalid_workflow_type_handling(self):
        """Test handling of invalid workflow types."""
        args = Namespace(
            model="Llama-3.1-8B-Instruct",
            impl="tt-transformers",
            device="n150",
            workflow="invalid_workflow",
            run_id="test",
        )

        # Should raise ValueError for invalid workflow type
        with pytest.raises(ValueError, match="Invalid TaskType"):
            WorkflowSetup(args)

    def test_invalid_model_config_handling(self):
        """Test handling of invalid model configurations."""
        args = Namespace(
            model="InvalidModel",
            impl="tt-transformers",
            device="n150",
            workflow="benchmarks",
            run_id="test",
        )

        # Should raise KeyError for invalid model ID
        with pytest.raises(KeyError):
            WorkflowSetup(args)


class TestWorkflowSetupConfiguration:
    """Test WorkflowSetup configuration logic with minimal mocking."""

    @pytest.fixture
    def sample_args(self):
        """Create sample args for testing."""
        return Namespace(
            model="Llama-3.1-8B-Instruct",
            impl="tt-transformers",
            device="n150",
            workflow="benchmarks",
            service_port="8000",
            disable_trace_capture=False,
            run_id="test_run_123",
        )

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "venvs").mkdir()
            (workspace / "logs").mkdir()
            yield workspace

    def test_workflow_setup_initialization(self, sample_args, temp_workspace):
        """Test WorkflowSetup object initialization."""
        with patch(
            "workflows.workflow_venvs.default_venv_path", temp_workspace / "venvs"
        ):
            workflow_setup = WorkflowSetup(sample_args)

            # Verify basic attributes are set correctly
            assert workflow_setup.args == sample_args
            assert (
                workflow_setup.workflow_config
                == WORKFLOW_CONFIGS[WorkflowType.BENCHMARKS]
            )
            assert (
                workflow_setup.model_id
                == "id_tt-transformers_Llama-3.1-8B-Instruct_n150"
            )
            assert workflow_setup.model_config == MODEL_SPECS[workflow_setup.model_id]

    def test_workflow_setup_different_workflow_types(self, sample_args, temp_workspace):
        """Test WorkflowSetup with different workflow types."""
        with patch(
            "workflows.workflow_venvs.default_venv_path", temp_workspace / "venvs"
        ):
            # Test benchmarks workflow
            sample_args.workflow = "benchmarks"
            setup = WorkflowSetup(sample_args)
            assert setup.workflow_config.workflow_type == WorkflowType.BENCHMARKS

            # Test evals workflow
            sample_args.workflow = "evals"
            setup = WorkflowSetup(sample_args)
            assert setup.workflow_config.workflow_type == WorkflowType.EVALS

    def test_get_output_path_creation(self, sample_args, temp_workspace):
        """Test output path creation without mocking directory operations."""
        with patch(
            "workflows.workflow_venvs.default_venv_path", temp_workspace / "venvs"
        ), patch(
            "workflows.workflow_config.get_default_workflow_root_log_dir",
            return_value=temp_workspace / "logs",
        ):
            workflow_setup = WorkflowSetup(sample_args)
            output_path = workflow_setup.get_output_path()

            # Verify path was created and is accessible
            assert output_path.exists()
            assert output_path.is_dir()
            assert output_path.name == "benchmarks_output"

    def test_python_version_check(self, sample_args, temp_workspace):
        """Test Python version checking logic."""
        with patch(
            "workflows.workflow_venvs.default_venv_path", temp_workspace / "venvs"
        ):
            workflow_setup = WorkflowSetup(sample_args)

            # Test with current Python version (should pass)
            workflow_setup.boostrap_uv()  # Should not raise

            # Test with insufficient Python version
            with patch("sys.version_info", (3, 5, 0)):
                workflow_setup = WorkflowSetup(sample_args)
                with pytest.raises(
                    ValueError, match="Python 3.6 or higher is required"
                ):
                    workflow_setup.boostrap_uv()

    def test_command_construction_logic(self, sample_args, temp_workspace):
        """Test command construction logic without executing commands."""
        with patch(
            "workflows.workflow_venvs.default_venv_path", temp_workspace / "venvs"
        ), patch(
            "workflows.workflow_config.get_default_workflow_root_log_dir",
            return_value=temp_workspace / "logs",
        ), patch(
            "workflows.run_workflows.run_command", return_value=0
        ) as mock_run_command:
            workflow_setup = WorkflowSetup(sample_args)

            # Mock the venv config to have a known python path
            mock_python_path = "/fake/venv/bin/python"
            mock_venv_config = MagicMock()
            mock_venv_config.venv_python = mock_python_path
            workflow_setup.workflow_venv_config = mock_venv_config

            return_code = workflow_setup.run_workflow_script(sample_args)

            # Verify command was constructed correctly
            assert mock_run_command.called
            called_cmd = mock_run_command.call_args[0][0]

            # Check key components of the command
            assert mock_python_path in called_cmd
            assert "--model" in called_cmd
            assert "Llama-3.1-8B-Instruct" in called_cmd
            assert "--impl" in called_cmd
            assert "tt-transformers" in called_cmd
            assert "--device" in called_cmd
            assert "n150" in called_cmd
            assert "--run-id" in called_cmd
            assert "test_run_123" in called_cmd
            assert return_code == 0


class TestWorkflowVenvValidation:
    """Test virtual environment configuration validation for different workflow types."""

    def test_server_workflow_venv_handling(self):
        """Test that server workflow properly handles None venv configuration."""
        args = Namespace(
            model="Llama-3.1-8B-Instruct",
            impl="tt-transformers",
            device="n150",
            workflow="server",
            service_port="8000",
            run_id="test",
        )

        # Server workflows have workflow_run_script_venv_type=None and should not be used with WorkflowSetup
        # The assertion in WorkflowSetup.__init__ should prevent this
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "workflows.run_workflows.default_venv_path", Path(temp_dir) / "venvs"
            ):
                # Should raise AssertionError because server workflows don't have venv configs
                with pytest.raises(AssertionError):
                    WorkflowSetup(args)

    def test_workflows_that_require_venv(self):
        """Test that workflows requiring venv configs have them properly defined."""
        workflows_requiring_venv = [
            WorkflowType.BENCHMARKS,
            WorkflowType.EVALS,
            WorkflowType.REPORTS,
        ]

        for workflow_type in workflows_requiring_venv:
            config = WORKFLOW_CONFIGS[workflow_type]
            assert (
                config.workflow_run_script_venv_type is not None
            ), f"{workflow_type.name} workflow must have a venv configuration"

    def test_server_workflow_is_special_case(self):
        """Document that server workflow intentionally has no venv config."""
        server_config = WORKFLOW_CONFIGS[WorkflowType.SERVER]

        # Document the current behavior: server workflow has None venv type
        assert server_config.workflow_run_script_venv_type is None

        # This test documents that server workflow is a special case
        # The code should be updated to handle this gracefully


class TestWorkflowExecution:
    """Test workflow execution with minimal but strategic mocking."""

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "venvs").mkdir()
            (workspace / "logs").mkdir()
            (workspace / "persistent_volume").mkdir()
            yield workspace

    def test_release_workflow_sequence(self):
        """Test release workflow sequence logic."""
        args = Namespace(
            model="Llama-3.1-8B-Instruct",
            impl="tt-transformers",
            device="n150",
            workflow="release",
            run_id="test",
            disable_trace_capture=False,
        )

        # Track workflow calls in order
        workflow_calls = []

        def mock_run_single(args_copy):
            # Capture workflow name at time of call
            workflow_calls.append(args_copy.workflow)
            return 0

        # Mock run_single_workflow to return success codes
        with patch(
            "workflows.run_workflows.run_single_workflow", side_effect=mock_run_single
        ) as mock_run_single:
            return_codes = run_workflows(args)

            # Verify all expected workflows were called
            assert len(return_codes) == 3  # benchmarks, evals, reports
            assert all(code == 0 for code in return_codes)
            assert mock_run_single.call_count == 3

            # The order should be BENCHMARKS, EVALS, REPORTS
            expected_order = ["BENCHMARKS", "EVALS", "REPORTS"]
            assert (
                workflow_calls == expected_order
            ), f"Expected {expected_order}, got {workflow_calls}"

            # Check trace capture logic by examining args modifications
            # Note: The args object is modified in place, so we rely on the implementation details
            # First workflow should start without trace capture disabled
            # Subsequent workflows should have trace capture disabled

    @patch("workflows.run_workflows.run_command")
    def test_workflow_execution_flow(self, mock_run_command, temp_workspace):
        """Test the complete workflow execution flow with minimal mocking."""
        # Mock only the external command execution
        mock_run_command.return_value = 0

        args = Namespace(
            model="Llama-3.1-8B-Instruct",
            impl="tt-transformers",
            device="n150",
            workflow="benchmarks",
            service_port="8000",
            disable_trace_capture=False,
            run_id="test_run_123",
        )

        # Ensure logs directory exists
        logs_dir = temp_workspace / "logs"
        logs_dir.mkdir(exist_ok=True)

        with patch(
            "workflows.workflow_venvs.default_venv_path", temp_workspace / "venvs"
        ), patch(
            "workflows.run_workflows.get_default_workflow_root_log_dir",
            return_value=logs_dir,
        ):
            # Run the workflow
            return_code = run_single_workflow(args)

            # Verify successful execution
            assert return_code == 0

            # Verify that directories were created
            output_dir = logs_dir / "benchmarks_output"
            assert output_dir.exists()

            # Verify the command execution sequence
            assert (
                mock_run_command.call_count >= 1
            )  # At least one command should be run

    def test_error_propagation(self, temp_workspace):
        """Test that errors are properly propagated through the workflow system."""
        args = Namespace(
            model="Llama-3.1-8B-Instruct",
            impl="tt-transformers",
            device="n150",
            workflow="benchmarks",
            run_id="test",
        )

        with patch(
            "workflows.workflow_venvs.default_venv_path", temp_workspace / "venvs"
        ), patch(
            "workflows.run_workflows.run_command", return_value=1
        ):  # Simulate command failure
            return_code = run_single_workflow(args)

            # Verify error is propagated
            assert return_code == 1


class TestHostSetupIntegration:
    """Test host setup with strategic mocking for external dependencies."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def mock_env_vars(self, temp_dir):
        """Set up mock environment variables."""
        env_vars = {
            "HF_TOKEN": "hf_test_token_123456",
            "JWT_SECRET": "test_jwt_secret_123",
            "PERSISTENT_VOLUME_ROOT": str(temp_dir / "persistent_volume"),
            "HF_HOME": str(temp_dir / "hf_home"),
            "SERVICE_PORT": "8000",
        }
        with patch.dict(os.environ, env_vars):
            yield env_vars

    @pytest.fixture
    def mock_system_calls(self):
        """Mock system calls that interact with external services."""
        with patch("subprocess.run") as mock_subprocess, patch(
            "subprocess.check_output"
        ) as mock_check_output, patch("shutil.disk_usage") as mock_disk_usage, patch(
            "workflows.setup_host.http_request"
        ) as mock_http_request:
            # Mock successful subprocess calls
            mock_subprocess.return_value.returncode = 0
            mock_check_output.return_value = b"abc123def456\n"  # Mock git commit SHA

            # Mock sufficient disk space (100GB free)
            mock_disk_usage.return_value = (
                1000 * 1024**3,
                500 * 1024**3,
                100 * 1024**3,
            )

            # Mock HuggingFace API responses
            def mock_hf_api(url, method="GET", headers=None):
                if "whoami-v2" in url:
                    return b'{"name": "test_user"}', 200, {}
                elif "api/models" in url:
                    return (
                        json.dumps(
                            {"siblings": [{"rfilename": "config.json"}]}
                        ).encode(),
                        200,
                        {},
                    )
                elif "resolve/main" in url:
                    return b"", 200, {}
                return b"", 404, {}

            mock_http_request.side_effect = mock_hf_api

            yield {
                "subprocess": mock_subprocess,
                "check_output": mock_check_output,
                "disk_usage": mock_disk_usage,
                "http_request": mock_http_request,
            }

    @pytest.fixture
    def mock_ram_check(self):
        """Mock RAM check to return sufficient memory."""
        mock_meminfo = "MemAvailable:    52428800 kB\n"  # 50GB in KB
        with patch("builtins.open", mock_open(read_data=mock_meminfo)) as mock_file:
            yield mock_file

    def test_setup_host_huggingface_source(
        self, temp_dir, mock_env_vars, mock_system_calls, mock_ram_check
    ):
        """Test host setup with HuggingFace model source."""
        model_id = "id_tt-transformers_Llama-3.1-8B-Instruct_n150"
        model_config = MODEL_SPECS[model_id]

        # Create setup manager
        manager = HostSetupManager(
            model_config=model_config,
            automatic=True,
            jwt_secret="test_jwt_secret",
            hf_token="hf_test_token_123456",
        )

        # Mock the setup flow properly
        with patch.object(manager, "check_setup", return_value=False), patch.object(
            manager, "check_model_weights_dir", return_value=True
        ), patch.object(manager, "setup_weights_huggingface") as mock_setup_weights:
            # Run setup
            manager.run_setup()

            # Verify that HF environment was set up
            assert str(manager.setup_config.host_hf_home) == str(temp_dir / "hf_home")
            assert mock_setup_weights.called

        # Verify that setup completed successfully
        assert manager.setup_config.model_source == "huggingface"
        assert manager.setup_config.persistent_volume_root.exists()

    def test_error_handling_insufficient_resources(
        self, temp_dir, mock_env_vars, mock_system_calls
    ):
        """Test error handling when system resources are insufficient."""
        # Mock insufficient disk space
        mock_system_calls["disk_usage"].return_value = (
            1000 * 1024**3,
            995 * 1024**3,
            5 * 1024**3,
        )  # Only 5GB free

        model_id = "id_tt-transformers_Llama-3.1-8B-Instruct_n150"
        model_config = MODEL_SPECS[model_id]

        manager = HostSetupManager(
            model_config=model_config,
            automatic=True,
            jwt_secret="test_jwt_secret",
            hf_token="hf_test_token_123456",
        )

        # Should raise assertion error due to insufficient disk space
        with pytest.raises(AssertionError, match="Insufficient disk space"):
            manager.setup_model_environment()

    def test_hf_token_validation_failure(
        self, temp_dir, mock_env_vars, mock_system_calls, mock_ram_check
    ):
        """Test handling of invalid HuggingFace token."""
        # Mock failed HF API response
        mock_system_calls["http_request"].side_effect = lambda url, **kwargs: (
            b"Unauthorized",
            401,
            {},
        )

        model_id = "id_tt-transformers_Llama-3.1-8B-Instruct_n150"
        model_config = MODEL_SPECS[model_id]

        manager = HostSetupManager(
            model_config=model_config,
            automatic=True,
            jwt_secret="test_jwt_secret",
            hf_token="invalid_token",
        )

        # Should raise assertion error due to invalid token
        with pytest.raises(AssertionError, match="HF_TOKEN validation failed"):
            manager.setup_model_environment()


class TestMainWorkflowIntegration:
    """Test main run.py integration with minimal mocking."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def mock_env_vars(self, temp_dir):
        """Set up mock environment variables."""
        env_vars = {
            "HF_TOKEN": "hf_test_token_123456",
            "JWT_SECRET": "test_jwt_secret_123",
            "PERSISTENT_VOLUME_ROOT": str(temp_dir / "persistent_volume"),
            "HF_HOME": str(temp_dir / "hf_home"),
            "SERVICE_PORT": "8000",
            "AUTOMATIC_HOST_SETUP": "1",
        }
        with patch.dict(os.environ, env_vars):
            yield env_vars

    @pytest.fixture
    def mock_version_file(self):
        """Mock VERSION file read."""
        with patch("pathlib.Path.read_text", return_value="1.0.0-test"):
            yield

    def test_main_workflow_benchmarks_no_docker(
        self, temp_dir, mock_env_vars, mock_version_file
    ):
        """Test main run.py workflow for benchmarks without docker server."""
        test_args = [
            "run.py",
            "--model",
            "Llama-3.1-8B-Instruct",
            "--device",
            "n150",
            "--workflow",
            "benchmarks",
        ]

        with patch("sys.argv", test_args), patch(
            "run.run_workflows", return_value=[0]
        ) as mock_run_workflows, patch(
            "workflows.run_workflows.run_single_workflow"
        ) as mock_run_single, patch(
            "workflows.utils.get_default_workflow_root_log_dir", return_value=temp_dir
        ), patch("workflows.log_setup.setup_run_logger"):
            mock_run_workflows.return_value = [0]
            mock_run_single.return_value = 0

            # Run main
            result = main()

            # Verify workflow ran without setup_host
            assert mock_run_workflows.called
            assert result == 0

    def test_error_handling_invalid_model(self, mock_env_vars):
        """Test error handling for invalid model configuration."""
        test_args = [
            "run.py",
            "--model",
            "InvalidModel",
            "--device",
            "n150",
            "--workflow",
            "benchmarks",
        ]

        with patch("sys.argv", test_args):
            with pytest.raises(SystemExit):  # argparse should exit on invalid choice
                main()


if __name__ == "__main__":
    pytest.main([__file__])
