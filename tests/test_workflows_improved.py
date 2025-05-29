#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import sys
import tempfile
import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from argparse import Namespace

# Add the project root to Python path
project_root = Path(__file__).resolve().parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

from workflows.setup_host import SetupConfig
from workflows.run_workflows import WorkflowSetup, run_single_workflow, run_workflows
from workflows.model_config import MODEL_CONFIGS
from workflows.workflow_types import WorkflowType, DeviceTypes
from workflows.workflow_config import WORKFLOW_CONFIGS
from workflows.utils import get_model_id, ensure_readwriteable_dir, get_repo_root_path


class TestWorkflowUtils:
    """Test workflow utility functions without heavy mocking."""

    def test_get_model_id_construction(self):
        """Test model ID construction logic."""
        model_id = get_model_id("tt-transformers", "Llama-3.1-8B-Instruct", "n150")
        expected = "id_tt-transformers_Llama-3.1-8B-Instruct_n150"
        assert model_id == expected

        # Test with no device - device is only added if it exists
        model_id = get_model_id("tt-transformers", "Llama-3.1-8B-Instruct", None)
        expected = "id_tt-transformers_Llama-3.1-8B-Instruct"
        assert model_id == expected

        # Test with empty string device
        model_id = get_model_id("tt-transformers", "Llama-3.1-8B-Instruct", "")
        expected = "id_tt-transformers_Llama-3.1-8B-Instruct"
        assert model_id == expected

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


class TestWorkflowSetupConfiguration:
    """Test WorkflowSetup configuration logic without external dependencies."""

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

    def test_workflow_setup_initialization(self, sample_args):
        """Test WorkflowSetup object initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "workflows.workflow_venvs.default_venv_path", Path(temp_dir) / "venvs"
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
                assert (
                    workflow_setup.model_config
                    == MODEL_CONFIGS[workflow_setup.model_id]
                )

    def test_workflow_setup_different_workflow_types(self, sample_args):
        """Test WorkflowSetup with different workflow types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "workflows.workflow_venvs.default_venv_path", Path(temp_dir) / "venvs"
            ):
                # Test benchmarks workflow
                sample_args.workflow = "benchmarks"
                setup = WorkflowSetup(sample_args)
                assert setup.workflow_config.workflow_type == WorkflowType.BENCHMARKS

                # Test evals workflow
                sample_args.workflow = "evals"
                setup = WorkflowSetup(sample_args)
                assert setup.workflow_config.workflow_type == WorkflowType.EVALS

                # Test server workflow
                sample_args.workflow = "server"
                setup = WorkflowSetup(sample_args)
                assert setup.workflow_config.workflow_type == WorkflowType.SERVER

    def test_get_output_path_creation(self, sample_args):
        """Test output path creation without mocking directory operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "workflows.workflow_venvs.default_venv_path", Path(temp_dir) / "venvs"
            ), patch(
                "workflows.workflow_config.get_default_workflow_root_log_dir",
                return_value=Path(temp_dir),
            ):
                workflow_setup = WorkflowSetup(sample_args)
                output_path = workflow_setup.get_output_path()

                # Verify path was created and is accessible
                assert output_path.exists()
                assert output_path.is_dir()
                assert output_path.name == "benchmarks_output"

    def test_command_construction_logic(self, sample_args):
        """Test command construction logic without executing commands."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "workflows.workflow_venvs.default_venv_path", Path(temp_dir) / "venvs"
            ), patch(
                "workflows.workflow_config.get_default_workflow_root_log_dir",
                return_value=Path(temp_dir),
            ), patch(
                "workflows.run_workflows.run_command", return_value=0
            ) as mock_run_command:
                workflow_setup = WorkflowSetup(sample_args)

                # Mock the venv config to have a known python path
                mock_python_path = "/fake/venv/bin/python"
                workflow_setup.workflow_venv_config.venv_python = mock_python_path

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


class TestWorkflowConfigurationValidation:
    """Test workflow configuration validation logic."""

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
        with pytest.raises(ValueError, match="No enum member with name"):
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

        # Mock run_single_workflow to return success codes
        with patch(
            "workflows.run_workflows.run_single_workflow", return_value=0
        ) as mock_run_single:
            return_codes = run_workflows(args)

            # Verify all expected workflows were called
            assert len(return_codes) == 3  # benchmarks, evals, reports
            assert all(code == 0 for code in return_codes)
            assert mock_run_single.call_count == 3

            # Verify the workflow sequence and trace capture logic
            call_args = [call[0][0] for call in mock_run_single.call_args_list]

            # First call should be benchmarks with trace capture enabled
            assert call_args[0].workflow == "benchmarks"
            assert not call_args[0].disable_trace_capture

            # Subsequent calls should have trace capture disabled
            assert call_args[1].workflow == "evals"
            assert call_args[1].disable_trace_capture

            assert call_args[2].workflow == "reports"
            assert call_args[2].disable_trace_capture


class TestWorkflowIntegrationRealistic:
    """More realistic integration tests that minimize mocking."""

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)

            # Create minimal required directory structure
            (workspace / "venvs").mkdir()
            (workspace / "logs").mkdir()
            (workspace / "persistent_volume").mkdir()

            yield workspace

    def test_python_version_check(self, temp_workspace):
        """Test Python version checking logic."""
        args = Namespace(
            model="Llama-3.1-8B-Instruct",
            impl="tt-transformers",
            device="n150",
            workflow="benchmarks",
            run_id="test",
        )

        with patch(
            "workflows.workflow_venvs.default_venv_path", temp_workspace / "venvs"
        ):
            workflow_setup = WorkflowSetup(args)

            # Test with current Python version (should pass)
            workflow_setup.boostrap_uv()  # Should not raise

            # Test with insufficient Python version
            with patch("sys.version_info", (3, 5, 0)):
                workflow_setup = WorkflowSetup(args)
                with pytest.raises(
                    ValueError, match="Python 3.6 or higher is required"
                ):
                    workflow_setup.boostrap_uv()

    def test_directory_structure_creation(self, temp_workspace):
        """Test that required directories are created properly."""
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
            "workflows.workflow_config.get_default_workflow_root_log_dir",
            return_value=temp_workspace / "logs",
        ):
            workflow_setup = WorkflowSetup(args)
            output_path = workflow_setup.get_output_path()

            # Verify directory structure was created
            assert output_path.exists()
            assert output_path.is_dir()
            assert output_path.parent == temp_workspace / "logs"

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

        with patch(
            "workflows.workflow_venvs.default_venv_path", temp_workspace / "venvs"
        ), patch(
            "workflows.workflow_config.get_default_workflow_root_log_dir",
            return_value=temp_workspace / "logs",
        ):
            # Run the workflow
            return_code = run_single_workflow(args)

            # Verify successful execution
            assert return_code == 0

            # Verify that directories were created
            output_dir = temp_workspace / "logs" / "benchmarks_output"
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


class TestMinimalMockingPatterns:
    """Examples of testing patterns that minimize mocking."""

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
        for model_id, config in MODEL_CONFIGS.items():
            assert model_id.startswith("id_")
            assert config.model_name is not None
            assert config.impl is not None
            assert config.device_type is not None
            assert config.hf_model_repo is not None
            assert config.model_id == model_id

    def test_workflow_venv_path_generation(self):
        """Test virtual environment path generation logic."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Patch the default_venv_path from the run_workflows module import
            with patch("workflows.run_workflows.default_venv_path", Path(temp_dir)):
                args = Namespace(
                    model="Llama-3.1-8B-Instruct",
                    impl="tt-transformers",
                    device="n150",
                    workflow="benchmarks",
                    run_id="test",
                )

                workflow_setup = WorkflowSetup(args)

                # Verify venv path is constructed correctly
                expected_base = Path(temp_dir) / ".venv_setup_workflow"
                assert workflow_setup.workflow_setup_venv == expected_base


class TestRealBugDiscovery:
    """Demonstrate how improved tests catch real bugs that mocked tests miss."""

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

        # This test catches the real bug: server workflow has workflow_run_script_venv_type=None
        # but WorkflowSetup.__init__ doesn't handle this case
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "workflows.run_workflows.default_venv_path", Path(temp_dir) / "venvs"
            ):
                # This should either:
                # 1. Handle None venv type gracefully, or
                # 2. Raise a clear error about missing venv config
                try:
                    workflow_setup = WorkflowSetup(args)
                    # If no exception, the code was fixed to handle None
                    assert hasattr(workflow_setup, "workflow_venv_config")
                except KeyError as e:
                    # Expected behavior: KeyError for None venv type
                    assert "None" in str(e)
                    pytest.fail(
                        "Server workflow needs proper venv configuration or None handling"
                    )
                except Exception as e:
                    pytest.fail(f"Unexpected error type: {type(e).__name__}: {e}")

    def test_workflow_types_that_require_venv(self):
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


if __name__ == "__main__":
    pytest.main([__file__])
