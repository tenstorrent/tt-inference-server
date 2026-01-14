#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import pytest
import argparse
import os
import sys
import subprocess
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add the project root to the path so we can import from run.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from run import (
    parse_arguments,
    validate_runtime_args,
    handle_secrets,
    get_current_commit_sha,
    validate_local_setup,
)
from workflows.model_spec import get_runtime_model_spec
from workflows.run_docker_server import run_docker_server


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
        reset_venvs=False,
        model_spec_json=None,
        tt_metal_python_venv_dir=None,
    )


@pytest.fixture
def mock_model_spec():
    """Create a mock model_spec object with default values."""
    mock_spec = MagicMock()
    mock_spec.model_id = "id_tt-transformers_Mistral-7B-Instruct-v0.3_n150"
    mock_spec.model_name = "Mistral-7B-Instruct-v0.3"
    mock_spec.tt_metal_commit = "test-commit"
    mock_spec.vllm_commit = "test-vllm-commit"
    mock_spec.to_json.return_value = "/tmp/test-model-spec.json"

    # Create mock cli_args
    mock_cli_args = MagicMock()
    mock_cli_args.workflow = "benchmarks"
    mock_cli_args.docker_server = False
    mock_cli_args.local_server = False
    mock_cli_args.interactive = False
    mock_cli_args.device = "n150"
    mock_cli_args.model = "Mistral-7B-Instruct-v0.3"
    mock_spec.cli_args = mock_cli_args

    return mock_spec


@pytest.fixture
def mock_setup_config():
    """Mock setup configuration for docker server."""
    mock_config = MagicMock()
    mock_config.cache_root = Path("/tmp/cache")
    mock_config.container_model_spec_dir = Path("/home/container_app_user/model_spec")
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
        "device_ids,expected",
        [("0", [0]), ("0,1", [0, 1]), ("0,1,2,3", [0, 1, 2, 3]), ("0,3", [0, 3])],
    )
    def test_parse_device_ids_valid(self, base_args, device_ids, expected):
        """Test valid device-id values."""
        full_args = base_args.copy()
        full_args += ["--device-id", device_ids]
        with patch("sys.argv", ["run.py"] + full_args):
            args = parse_arguments()
        assert args.device_id == expected

    @pytest.mark.parametrize(
        "device_ids",
        [
            "0 1",  # space instead of comma
            "-1",  # negative value
            "1,-2,3",  # mixed negative
            "abc",  # non-integer
            "1,,2",  # empty entry
            "",  # empty string
        ],
    )
    def test_parse_device_ids_invalid(self, base_args, device_ids):
        """Test invalid device-id values raise the correct error."""
        full_args = base_args.copy()
        full_args += ["--device-id", device_ids]
        with patch("sys.argv", ["run.py"] + full_args):
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
        assert args.device_id == [1]
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
        """Test successful impl inference via get_runtime_model_spec."""
        mock_args.impl = None

        # Mock get_model_id and MODEL_SPECS in the correct module
        mock_model_spec = MagicMock()
        mock_model_spec.apply_runtime_args = MagicMock()

        with patch(
            "workflows.model_spec.get_model_id", return_value="test-model-id"
        ), patch.dict(
            "workflows.model_spec.MODEL_SPECS", {"test-model-id": mock_model_spec}
        ):
            result = get_runtime_model_spec(mock_args)

            # Verify that impl was inferred
            assert mock_args.impl == "tt-transformers"
            assert result == mock_model_spec

    def test_infer_impl_already_set(self, mock_args):
        """Test that existing impl is preserved."""
        mock_args.impl = "existing-impl"

        # Mock get_model_id and MODEL_SPECS in the correct module
        mock_model_spec = MagicMock()
        mock_model_spec.apply_runtime_args = MagicMock()

        with patch(
            "workflows.model_spec.get_model_id", return_value="test-model-id"
        ), patch.dict(
            "workflows.model_spec.MODEL_SPECS", {"test-model-id": mock_model_spec}
        ):
            result = get_runtime_model_spec(mock_args)

            # Verify that existing impl was preserved
            assert mock_args.impl == "existing-impl"
            assert result == mock_model_spec

    def test_infer_impl_no_default(self, mock_args):
        """Test error when no default impl available."""
        mock_args.model = "NonExistentModel"
        mock_args.impl = None

        with pytest.raises(ValueError, match="does not have a default impl"):
            get_runtime_model_spec(mock_args)


class TestRuntimeValidation:
    """Tests for runtime argument validation."""

    @pytest.mark.parametrize(
        "workflow,should_pass",
        [
            ("benchmarks", True),
            ("evals", True),  # Mistral-7B-Instruct-v0.3 is in EVAL_CONFIGS
            ("reports", True),
            ("release", True),  # Mistral-7B-Instruct-v0.3 is in both configs
            ("stress_tests", True),
        ],
    )
    def test_workflow_validation(self, mock_model_spec, workflow, should_pass):
        """Test validation for different workflows."""
        mock_model_spec.cli_args.workflow = workflow
        with patch.dict("run.MODEL_SPECS", {mock_model_spec.model_id: mock_model_spec}):
            if should_pass:
                validate_runtime_args(mock_model_spec)
            else:
                with pytest.raises(AssertionError):
                    validate_runtime_args(mock_model_spec)

    def test_server_workflow_validation(self, mock_model_spec):
        """Test server workflow specific validation."""
        mock_model_spec.cli_args.workflow = "server"

        with patch.dict("run.MODEL_SPECS", {mock_model_spec.model_id: mock_model_spec}):
            # Should fail without docker or local server
            with pytest.raises(ValueError, match="requires --docker-server"):
                validate_runtime_args(mock_model_spec)

            # Should pass with docker server
            mock_model_spec.cli_args.docker_server = True
            validate_runtime_args(mock_model_spec)

            # Should fail with local server (not implemented)
            mock_model_spec.cli_args.docker_server = False
            mock_model_spec.cli_args.local_server = True
            with pytest.raises(
                NotImplementedError, match="not implemented for --local-server"
            ):
                validate_runtime_args(mock_model_spec)

    def test_conflicting_server_options(self, mock_model_spec):
        """Test that both docker and local server raises error."""
        mock_model_spec.cli_args.docker_server = True
        mock_model_spec.cli_args.local_server = True
        with patch.dict("run.MODEL_SPECS", {mock_model_spec.model_id: mock_model_spec}):
            with pytest.raises(
                AssertionError, match="Cannot run --docker-server and --local-server"
            ):
                validate_runtime_args(mock_model_spec)


class TestOverrideArgsIntegration:
    """Test override arguments integration with model_spec apply_runtime_args."""

    @pytest.mark.parametrize(
        "override_type,cli_arg_name,test_value",
        [
            ("tt_config", "override_tt_config", '{"data_parallel": 16}'),
            ("vllm_args", "vllm_override_args", '{"max_model_len": 4096}'),
        ],
    )
    def test_get_runtime_model_spec_applies_overrides(
        self, override_type, cli_arg_name, test_value
    ):
        """Test that get_runtime_model_spec correctly applies override arguments."""
        # Create args with override values using argparse.Namespace instead of MagicMock
        # to avoid issues with mutable defaults in dataclass creation
        mock_args = argparse.Namespace()
        mock_args.impl = "tt-transformers"
        mock_args.model = "Mistral-7B-Instruct-v0.3"
        mock_args.device = "n150"
        mock_args.dev_mode = False
        mock_args.override_docker_image = None

        # Set the specific override arg being tested
        setattr(mock_args, cli_arg_name, test_value)
        # Set the other one to None
        other_arg = (
            "vllm_override_args"
            if cli_arg_name == "override_tt_config"
            else "override_tt_config"
        )
        setattr(mock_args, other_arg, None)

        # Mock get_model_id and MODEL_SPECS in the correct module
        mock_model_spec = MagicMock()
        mock_model_spec.apply_runtime_args = MagicMock()

        with patch(
            "workflows.model_spec.get_model_id", return_value="test-model-id"
        ), patch.dict(
            "workflows.model_spec.MODEL_SPECS", {"test-model-id": mock_model_spec}
        ):
            result = get_runtime_model_spec(mock_args)

            # Verify that apply_runtime_args was called with the args
            mock_model_spec.apply_runtime_args.assert_called_once_with(mock_args)
            assert result == mock_model_spec

    def test_docker_server_mounts_model_spec_json(self, mock_setup_config):
        """Test that run_docker_server mounts the model_spec JSON file into the container."""
        from pathlib import Path

        # Create mock model_spec
        mock_model_spec = MagicMock()
        mock_model_spec.model_id = "test-model-id"
        mock_model_spec.device_type = "n150"
        mock_model_spec.docker_image = "test:image"
        mock_model_spec.impl.impl_name = "tt-transformers"
        mock_model_spec.hf_model_repo = "mistralai/Mistral-7B-Instruct-v0.3"
        mock_model_spec.subdevice_type = None

        # Create cli_args
        mock_cli_args = MagicMock()
        mock_cli_args.model = "Mistral-7B-Instruct-v0.3"
        mock_cli_args.device = "n150"
        mock_cli_args.workflow = "server"
        mock_cli_args.service_port = "8000"
        mock_cli_args.interactive = False
        mock_cli_args.dev_mode = False
        mock_cli_args.device_id = None
        mock_cli_args.impl = "tt-transformers"
        mock_cli_args.override_docker_image = None
        mock_model_spec.cli_args = mock_cli_args

        # Mock dependencies
        with patch(
            "workflows.run_docker_server.ensure_docker_image", return_value=True
        ), patch("workflows.run_docker_server.subprocess.Popen") as mock_popen, patch(
            "workflows.run_docker_server.open"
        ), patch(
            "workflows.run_docker_server.subprocess.check_output",
            return_value="container123",
        ), patch("workflows.run_docker_server.atexit.register"), patch(
            "workflows.run_docker_server.shlex.join", return_value="mocked command"
        ), patch(
            "workflows.run_docker_server.get_repo_root_path", return_value=Path("/tmp")
        ), patch(
            "workflows.run_docker_server.get_default_workflow_root_log_dir",
            return_value=Path("/tmp/logs"),
        ), patch("workflows.run_docker_server.ensure_readwriteable_dir"), patch(
            "workflows.run_docker_server.DeviceTypes"
        ), patch("workflows.run_docker_server.short_uuid", return_value="test123"):
            # Call the function with model_spec, setup_config, and json_fpath
            json_fpath = Path("/tmp/test-model-spec.json")
            run_docker_server(mock_model_spec, mock_setup_config, json_fpath)

            # Verify subprocess.Popen was called
            mock_popen.assert_called_once()
            docker_command = mock_popen.call_args[0][0]

            # Check that the JSON file is mounted and TT_MODEL_SPEC_JSON_PATH is set
            json_mount_found = False
            env_var_found = False

            for i, arg in enumerate(docker_command):
                # Check for JSON file mount
                if arg == "--mount" and i + 1 < len(docker_command):
                    mount_spec = docker_command[i + 1]
                    if (
                        "test-model-spec.json" in mount_spec
                        and "readonly" in mount_spec
                    ):
                        json_mount_found = True

                # Check for TT_MODEL_SPEC_JSON_PATH environment variable
                if arg == "-e" and i + 1 < len(docker_command):
                    env_setting = docker_command[i + 1]
                    if env_setting.startswith("TT_MODEL_SPEC_JSON_PATH="):
                        env_var_found = True
                        assert "test-model-spec.json" in env_setting

            assert json_mount_found, (
                f"JSON file mount not found in docker command: {docker_command}"
            )
            assert env_var_found, (
                f"TT_MODEL_SPEC_JSON_PATH not found in docker command: {docker_command}"
            )


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
        self,
        mock_model_spec,
        workflow,
        docker_server,
        interactive,
        jwt_required,
        hf_required,
    ):
        """Test secret requirements for different configurations."""
        mock_model_spec.cli_args.workflow = workflow
        mock_model_spec.cli_args.docker_server = docker_server
        mock_model_spec.cli_args.interactive = interactive

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
                            handle_secrets(mock_model_spec)
                    else:
                        handle_secrets(mock_model_spec)
                else:
                    handle_secrets(mock_model_spec)

    @patch("run.load_dotenv")
    @patch("run.write_dotenv")
    @patch("getpass.getpass")
    def test_secrets_prompting(
        self, mock_getpass, mock_write_dotenv, mock_load_dotenv, mock_model_spec
    ):
        """Test prompting for missing secrets."""
        mock_model_spec.cli_args.workflow = "server"
        mock_model_spec.cli_args.docker_server = True
        mock_model_spec.cli_args.interactive = False

        mock_load_dotenv.side_effect = [False, True]
        mock_getpass.side_effect = ["test-jwt", "test-hf"]

        with patch.dict(os.environ, {}, clear=True):
            handle_secrets(mock_model_spec)

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
    def test_validate_local_setup(
        self,
        mock_get_log_dir,
        mock_ensure_dir,
        mock_model_spec,
    ):
        """Test local setup validation."""
        mock_log_dir = Path("/tmp/test_logs")
        mock_get_log_dir.return_value = mock_log_dir

        # Create a temporary directory for the model spec JSON
        with patch.dict(
            "run.MODEL_SPECS", {mock_model_spec.model_id: mock_model_spec}
        ), tempfile.TemporaryDirectory() as tempdir:
            # dump the ModelSpec to a tempdir
            model_spec_path = mock_model_spec.to_json(run_id="temp", output_dir=tempdir)
            validate_local_setup(mock_model_spec, model_spec_path)

        mock_get_log_dir.assert_called_once()
        mock_ensure_dir.assert_called_once_with(mock_log_dir)


if __name__ == "__main__":
    pytest.main([__file__])
