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
from unittest.mock import patch, MagicMock, call, mock_open
from argparse import Namespace

# Add the project root to Python path
project_root = Path(__file__).resolve().parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

import run
from workflows.setup_host import HostSetupManager, SetupConfig
from workflows.run_workflows import WorkflowSetup, run_single_workflow
from workflows.model_config import MODEL_CONFIGS
from workflows.workflow_types import WorkflowType, DeviceTypes


class TestWorkflowIntegration:
    """Integration tests for the main workflow calls as if made by CLI."""

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
    def mock_hf_model_files(self, temp_dir):
        """Create mock HuggingFace model files structure."""
        # Create a mock HF cache structure
        hf_home = temp_dir / "hf_home"
        model_cache = (
            hf_home
            / "hub"
            / "models--meta-llama--Llama-3.1-8B-Instruct"
            / "snapshots"
            / "abc123"
        )
        model_cache.mkdir(parents=True)

        # Create mock model files
        (model_cache / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (model_cache / "tokenizer.json").write_text(json.dumps({"vocab": {}}))
        (model_cache / "model-00001-of-00004.safetensors").write_text("fake_weights")
        (model_cache / "model-00002-of-00004.safetensors").write_text("fake_weights")

        return model_cache

    @pytest.fixture
    def mock_system_calls(self):
        """Mock system calls that interact with external services."""
        with patch("subprocess.run") as mock_subprocess, patch(
            "subprocess.check_output"
        ) as mock_check_output, patch("shutil.disk_usage") as mock_disk_usage, patch(
            "workflows.setup_host.http_request"
        ) as mock_http_request, patch(
            "workflows.run_workflows.run_command"
        ) as mock_run_command, patch(
            "workflows.run_docker_server.run_docker_server"
        ) as mock_docker_server:
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

            # Mock successful workflow commands
            mock_run_command.return_value = 0

            yield {
                "subprocess": mock_subprocess,
                "check_output": mock_check_output,
                "disk_usage": mock_disk_usage,
                "http_request": mock_http_request,
                "run_command": mock_run_command,
                "docker_server": mock_docker_server,
            }

    @pytest.fixture
    def mock_ram_check(self):
        """Mock RAM check to return sufficient memory."""
        mock_meminfo = "MemAvailable:    52428800 kB\n"  # 50GB in KB
        with patch("builtins.open", mock_open(read_data=mock_meminfo)) as mock_file:
            yield mock_file

    @pytest.fixture
    def mock_version_file(self):
        """Mock VERSION file read."""
        with patch("pathlib.Path.read_text", return_value="1.0.0-test"):
            yield

    def test_setup_host_huggingface_source(
        self,
        temp_dir,
        mock_env_vars,
        mock_hf_model_files,
        mock_system_calls,
        mock_ram_check,
    ):
        """Test host setup with HuggingFace model source."""
        # Get a valid model config (use the correct model ID format)
        model_id = "id_tt-transformers_Llama-3.1-8B-Instruct_n150"
        model_config = MODEL_CONFIGS[model_id]

        # Create setup manager
        manager = HostSetupManager(
            model_config=model_config,
            automatic=True,
            jwt_secret="test_jwt_secret",
            hf_token="hf_test_token_123456",
        )

        # Mock the setup flow properly:
        # 1. First check_setup() should return False to trigger setup
        # 2. After setup, check_model_weights_dir should return True
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

    def test_setup_host_local_source(
        self, temp_dir, mock_env_vars, mock_system_calls, mock_ram_check
    ):
        """Test host setup with local model source."""
        # Create local model directory with required files
        local_model_dir = temp_dir / "local_model"
        local_model_dir.mkdir()
        (local_model_dir / "config.json").write_text(
            json.dumps({"model_type": "llama"})
        )
        (local_model_dir / "tokenizer.json").write_text(json.dumps({"vocab": {}}))
        (local_model_dir / "model-00001-of-00001.safetensors").write_text(
            "fake_weights"
        )

        # Set up environment for local source
        env_vars = {
            **mock_env_vars,  # Include all the existing env vars
            "MODEL_SOURCE": "local",
            "MODEL_WEIGHTS_DIR": str(local_model_dir),
        }
        with patch.dict(os.environ, env_vars, clear=True):
            model_id = "id_tt-transformers_Llama-3.1-8B-Instruct_n150"
            model_config = MODEL_CONFIGS[model_id]

            # Patch SetupConfig to use local source
            with patch("workflows.setup_host.SetupConfig") as mock_setup_config_class:
                # Create a real SetupConfig but override the model_source
                real_setup_config = SetupConfig(model_config=model_config)
                real_setup_config.model_source = "local"
                real_setup_config.update_host_model_weights_mount_dir(local_model_dir)
                mock_setup_config_class.return_value = real_setup_config

                manager = HostSetupManager(
                    model_config=model_config,
                    automatic=True,
                    jwt_secret="test_jwt_secret",
                    hf_token="hf_test_token_123456",
                )

                # Verify the model source was set correctly
                assert manager.setup_config.model_source == "local"

                # Mock the setup flow for local source
                with patch.object(
                    manager, "check_setup", return_value=False
                ), patch.object(
                    manager, "check_model_weights_dir", return_value=True
                ), patch.object(manager, "setup_weights_local") as mock_setup_weights:
                    # Run setup
                    manager.run_setup()

                    # Verify that local setup was called
                    assert mock_setup_weights.called

                # Verify setup
                assert manager.setup_config.model_source == "local"
                assert (
                    manager.setup_config.host_model_weights_mount_dir == local_model_dir
                )

    def test_workflow_setup_benchmarks(
        self, temp_dir, mock_env_vars, mock_system_calls
    ):
        """Test workflow setup for benchmarks."""
        args = Namespace(
            model="Llama-3.1-8B-Instruct",
            impl="tt-transformers",
            device="n150",
            workflow="benchmarks",
            service_port="8000",
            disable_trace_capture=False,
            run_id="test_run_123",
        )

        # Mock the entire workflow setup process
        with patch(
            "workflows.run_workflows.WorkflowSetup.boostrap_uv"
        ) as mock_bootstrap, patch(
            "workflows.run_workflows.WorkflowSetup.create_required_venvs"
        ) as mock_create_venvs, patch(
            "workflows.run_workflows.WorkflowSetup.get_output_path",
            return_value=temp_dir / "output",
        ), patch(
            "workflows.run_workflows.run_command", return_value=0
        ) as mock_run_command:
            # Create workflow setup
            workflow_setup = WorkflowSetup(args)

            # Run workflow setup
            workflow_setup.boostrap_uv()
            workflow_setup.setup_workflow()

            # Test running workflow script
            return_code = workflow_setup.run_workflow_script(args)
            assert return_code == 0

            # Verify mocks were called
            assert mock_bootstrap.called
            assert mock_create_venvs.called

    @pytest.mark.parametrize(
        "workflow_type,device",
        [
            ("benchmarks", "n150"),
            ("benchmarks", "t3k"),
        ],
    )
    def test_run_single_workflow_variations(
        self, workflow_type, device, temp_dir, mock_env_vars, mock_system_calls
    ):
        """Test running single workflows with different configurations."""
        args = Namespace(
            model="Llama-3.1-8B-Instruct",
            impl="tt-transformers",
            device=device,
            workflow=workflow_type,
            service_port="8000",
            disable_trace_capture=False,
            run_id="test_run_123",
        )

        with patch(
            "workflows.run_workflows.get_default_workflow_root_log_dir",
            return_value=temp_dir,
        ), patch(
            "workflows.run_workflows.default_venv_path", temp_dir / "venvs"
        ), patch(
            "workflows.run_workflows.WorkflowSetup.boostrap_uv"
        ) as mock_bootstrap, patch(
            "workflows.run_workflows.WorkflowSetup.create_required_venvs"
        ) as mock_create_venvs, patch(
            "workflows.run_workflows.run_command", return_value=0
        ):
            return_code = run_single_workflow(args)
            assert return_code == 0
            assert mock_bootstrap.called
            assert mock_create_venvs.called

    def test_main_workflow_benchmarks_docker(
        self,
        temp_dir,
        mock_env_vars,
        mock_hf_model_files,
        mock_system_calls,
        mock_ram_check,
        mock_version_file,
    ):
        """Test main run.py workflow for benchmarks with docker server."""
        test_args = [
            "run.py",
            "--model",
            "Llama-3.1-8B-Instruct",
            "--device",
            "n150",
            "--workflow",
            "benchmarks",
            "--docker-server",
            "--dev-mode",
        ]

        with patch("sys.argv", test_args), patch(
            "run.setup_host"
        ) as mock_setup_host, patch(
            "run.run_docker_server"
        ) as mock_run_docker_server, patch(
            "run.run_workflows"
        ) as mock_run_workflows, patch(
            "workflows.run_workflows.run_single_workflow"
        ) as mock_run_single, patch.object(
            run, "get_default_workflow_root_log_dir", return_value=temp_dir
        ), patch.object(run, "setup_run_logger"), patch(
            "workflows.workflow_venvs.default_venv_path", temp_dir / "venvs"
        ), patch.dict(os.environ, {"AUTOMATIC_HOST_SETUP": "1"}):
            # Mock setup_host return
            mock_setup_config = SetupConfig(
                model_config=MODEL_CONFIGS[
                    "id_tt-transformers_Llama-3.1-8B-Instruct_n150"
                ]
            )
            mock_setup_host.return_value = mock_setup_config
            mock_run_workflows.return_value = [0]
            mock_run_single.return_value = 0

            # Run main
            result = run.main()

            # Verify calls were made
            assert mock_setup_host.called
            assert mock_run_docker_server.called
            assert mock_run_workflows.called
            assert result == 0

    def test_main_workflow_benchmarks_no_docker(
        self, temp_dir, mock_env_vars, mock_system_calls, mock_version_file
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
            "run.run_workflows"
        ) as mock_run_workflows, patch(
            "workflows.run_workflows.run_single_workflow"
        ) as mock_run_single, patch.object(
            run, "get_default_workflow_root_log_dir", return_value=temp_dir
        ), patch.object(run, "setup_run_logger"):
            mock_run_workflows.return_value = [0]
            mock_run_single.return_value = 0

            # Run main
            result = run.main()

            # Verify workflow ran without setup_host
            assert mock_run_workflows.called
            assert result == 0

    def test_main_workflow_server_mode(
        self,
        temp_dir,
        mock_env_vars,
        mock_hf_model_files,
        mock_system_calls,
        mock_ram_check,
        mock_version_file,
    ):
        """Test main run.py workflow for server mode."""
        test_args = [
            "run.py",
            "--model",
            "Llama-3.1-8B-Instruct",
            "--device",
            "n150",
            "--workflow",
            "server",
            "--docker-server",
        ]

        with patch("sys.argv", test_args), patch(
            "run.setup_host"
        ) as mock_setup_host, patch(
            "run.run_docker_server"
        ) as mock_run_docker_server, patch(
            "run.run_workflows"
        ) as mock_run_workflows, patch(
            "workflows.run_workflows.run_single_workflow"
        ) as mock_run_single, patch.object(
            run, "get_default_workflow_root_log_dir", return_value=temp_dir
        ), patch.object(run, "setup_run_logger"), patch(
            "workflows.workflow_venvs.default_venv_path", temp_dir / "venvs"
        ), patch.dict(os.environ, {"AUTOMATIC_HOST_SETUP": "1"}):
            # Mock setup_host return
            mock_setup_config = SetupConfig(
                model_config=MODEL_CONFIGS[
                    "id_tt-transformers_Llama-3.1-8B-Instruct_n150"
                ]
            )
            mock_setup_host.return_value = mock_setup_config
            mock_run_workflows.return_value = [0]
            mock_run_single.return_value = 0

            # Run main
            result = run.main()

            # Verify setup_host was called but not run_workflows (server mode skips it)
            assert mock_setup_host.called
            assert mock_run_docker_server.called
            assert not mock_run_workflows.called  # Server workflow skips run_workflows
            assert result == 0

    def test_main_workflow_release_mode(
        self, temp_dir, mock_env_vars, mock_system_calls, mock_version_file
    ):
        """Test main run.py workflow for release mode (runs multiple workflows)."""
        test_args = [
            "run.py",
            "--model",
            "Llama-3.1-8B-Instruct",
            "--device",
            "n150",
            "--workflow",
            "release",
        ]

        with patch("sys.argv", test_args), patch(
            "run.run_workflows"
        ) as mock_run_workflows, patch(
            "workflows.run_workflows.run_single_workflow"
        ) as mock_run_single, patch.object(
            run, "get_default_workflow_root_log_dir", return_value=temp_dir
        ), patch.object(run, "setup_run_logger"):
            # Mock successful return codes for all workflows in release
            mock_run_workflows.return_value = [0, 0, 0]  # benchmarks, evals, reports
            mock_run_single.return_value = 0

            # Run main
            result = run.main()

            # Verify workflow ran
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
                run.main()

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
        model_config = MODEL_CONFIGS[model_id]

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
        model_config = MODEL_CONFIGS[model_id]

        manager = HostSetupManager(
            model_config=model_config,
            automatic=True,
            jwt_secret="test_jwt_secret",
            hf_token="invalid_token",
        )

        # Should raise assertion error due to invalid token
        with pytest.raises(AssertionError, match="HF_TOKEN validation failed"):
            manager.setup_model_environment()


if __name__ == "__main__":
    pytest.main([__file__])
