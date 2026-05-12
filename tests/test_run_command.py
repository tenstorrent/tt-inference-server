#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""
Tests for run_command error handling and check parameter behavior.

These tests validate:
1. run_command with check=False (default) returns the return code without raising
2. run_command with check=True raises RuntimeError on non-zero return codes
3. All critical commands that use check=True properly raise errors on failure
4. setup_*() functions return setup_succeeded boolean based on command success

Critical commands using check=True:
- workflows/bootstrap_uv.py: venv creation, ensurepip, pip install uv, uv version
- workflows/workflow_venvs.py: VenvConfig.setup() venv creation, docker version check
- run.py: system software validation
"""

import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from workflows.runtime_config import RuntimeConfig
from workflows.utils import run_command


@pytest.fixture
def test_logger():
    """Create a test logger."""
    logger = logging.getLogger("test_run_command")
    logger.setLevel(logging.DEBUG)
    return logger


# =============================================================================
# Core run_command behavior tests
# =============================================================================


class TestRunCommandCheckBehavior:
    """Test run_command check parameter behavior."""

    def test_run_command_check_false_returns_nonzero_code(self, test_logger):
        """Test that check=False (default) returns non-zero code without raising."""
        return_code = run_command("false", logger=test_logger, check=False)
        assert return_code != 0

    def test_run_command_check_false_is_default(self, test_logger):
        """Test that check=False is the default behavior."""
        # Run a failing command without specifying check
        return_code = run_command("false", logger=test_logger)
        assert return_code != 0  # Should return code, not raise

    def test_run_command_check_true_raises_on_failure(self, test_logger):
        """Test that check=True raises RuntimeError on non-zero return code."""
        with pytest.raises(RuntimeError, match="command failed with return code"):
            run_command("false", logger=test_logger, check=True)

    def test_run_command_check_true_success(self, test_logger):
        """Test that check=True does not raise on success."""
        return_code = run_command("true", logger=test_logger, check=True)
        assert return_code == 0

    def test_run_command_check_false_success(self, test_logger):
        """Test that check=False returns 0 on success."""
        return_code = run_command("true", logger=test_logger, check=False)
        assert return_code == 0

    def test_run_command_with_log_file_check_false(self, test_logger):
        """Test check=False behavior when using log file output."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            log_file_path = f.name

        return_code = run_command(
            "false", logger=test_logger, log_file_path=log_file_path, check=False
        )
        assert return_code != 0

        Path(log_file_path).unlink(missing_ok=True)

    def test_run_command_with_log_file_check_true(self, test_logger):
        """Test check=True raises when using log file output."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            log_file_path = f.name

        with pytest.raises(subprocess.CalledProcessError):
            run_command(
                "false", logger=test_logger, log_file_path=log_file_path, check=True
            )

        Path(log_file_path).unlink(missing_ok=True)


class TestRunCommandReturnCode:
    """Test run_command return code handling."""

    def test_returns_exact_exit_code(self, test_logger):
        """Test that the exact exit code is returned."""
        return_code = run_command(
            ["bash", "-c", "exit 42"], logger=test_logger, check=False
        )
        assert return_code == 42

    def test_returns_zero_on_success(self, test_logger):
        """Test that 0 is returned on success."""
        return_code = run_command("echo hello", logger=test_logger, check=False)
        assert return_code == 0

    def test_logs_error_on_failure_check_false(self, test_logger, caplog):
        """Test that errors are logged when check=False and command fails."""
        with caplog.at_level(logging.ERROR):
            run_command("false", logger=test_logger, check=False)
        assert "command failed with return code" in caplog.text


# =============================================================================
# Bootstrap UV critical commands - check=True
# =============================================================================


class TestBootstrapUvCheckTrue:
    """Test bootstrap_uv uses check=True for all critical commands."""

    def test_bootstrap_uv_all_commands_use_check_true(self):
        """Verify all bootstrap_uv commands use check=True."""
        from workflows.bootstrap_uv import bootstrap_uv

        with patch("workflows.bootstrap_uv.UV_VENV_PATH") as mock_venv_path, patch(
            "workflows.bootstrap_uv.run_command"
        ) as mock_run_command:
            # Simulate venv needs creation
            mock_venv_path.exists.return_value = False
            mock_pip_exec = MagicMock()
            mock_pip_exec.exists.return_value = False
            mock_venv_path.__truediv__ = MagicMock(return_value=mock_pip_exec)

            try:
                bootstrap_uv()
            except Exception:
                pass  # We only care about the run_command calls

            # Verify all run_command calls use check=True
            for call_item in mock_run_command.call_args_list:
                assert call_item[1].get("check") is True, (
                    f"Expected check=True in call: {call_item}"
                )

    def test_bootstrap_uv_venv_creation_raises_on_failure(self):
        """Test that venv creation failure raises RuntimeError."""
        from workflows.bootstrap_uv import bootstrap_uv

        with patch("workflows.bootstrap_uv.UV_VENV_PATH") as mock_venv_path, patch(
            "workflows.bootstrap_uv.run_command"
        ) as mock_run_command:
            mock_venv_path.exists.return_value = False
            mock_pip_exec = MagicMock()
            mock_pip_exec.exists.return_value = False
            mock_venv_path.__truediv__ = MagicMock(return_value=mock_pip_exec)

            # Simulate venv creation failure
            mock_run_command.side_effect = RuntimeError("command failed")

            with pytest.raises(RuntimeError, match="command failed"):
                bootstrap_uv()

    def test_bootstrap_uv_pip_install_raises_on_failure(self):
        """Test that pip install uv failure raises RuntimeError."""
        from workflows.bootstrap_uv import bootstrap_uv

        with patch("workflows.bootstrap_uv.UV_VENV_PATH") as mock_venv_path, patch(
            "workflows.bootstrap_uv.run_command"
        ) as mock_run_command:
            mock_venv_path.exists.return_value = False
            mock_pip_exec = MagicMock()
            mock_pip_exec.exists.return_value = True  # pip exists after venv creation
            mock_venv_path.__truediv__ = MagicMock(return_value=mock_pip_exec)

            # First call (venv creation) succeeds, second (pip install) fails
            mock_run_command.side_effect = [
                0,  # venv creation
                RuntimeError("pip install failed"),  # pip install uv
            ]

            with pytest.raises(RuntimeError, match="pip install failed"):
                bootstrap_uv()


# =============================================================================
# VenvConfig.setup() critical commands - check=True
# =============================================================================


class TestVenvConfigSetupCheckTrue:
    """Test VenvConfig.setup() venv creation uses check=True."""

    @pytest.fixture
    def mock_model_spec(self):
        """Create a mock model spec."""
        mock_spec = MagicMock()
        mock_spec.model_name = "test-model"
        return mock_spec

    def test_venv_creation_uses_check_true(self, mock_model_spec):
        """Test VenvConfig.setup() calls run_command with check=True for venv creation."""
        from workflows.workflow_types import WorkflowVenvType
        from workflows.workflow_venvs import VenvConfig

        def passing_setup(venv_config, model_spec):
            return True

        with tempfile.TemporaryDirectory() as temp_dir:
            venv_path = Path(temp_dir) / "test_venv"

            with patch(
                "workflows.workflow_venvs.run_command", return_value=0
            ) as mock_run:
                config = VenvConfig(
                    venv_type=WorkflowVenvType.EVALS_RUN_SCRIPT,
                    setup_function=passing_setup,
                    venv_path=venv_path,
                )

                config.setup(mock_model_spec)

                # Verify run_command was called with check=True
                mock_run.assert_called()
                venv_call = mock_run.call_args_list[0]
                assert "venv" in venv_call[0][0]
                assert venv_call[1].get("check") is True

    def test_venv_creation_raises_on_failure(self, mock_model_spec):
        """Test VenvConfig.setup() raises RuntimeError when venv creation fails."""
        from workflows.workflow_types import WorkflowVenvType
        from workflows.workflow_venvs import VenvConfig

        def passing_setup(venv_config, model_spec):
            return True

        with tempfile.TemporaryDirectory() as temp_dir:
            venv_path = Path(temp_dir) / "test_venv"

            with patch(
                "workflows.workflow_venvs.run_command",
                side_effect=RuntimeError("venv creation failed"),
            ):
                config = VenvConfig(
                    venv_type=WorkflowVenvType.EVALS_RUN_SCRIPT,
                    setup_function=passing_setup,
                    venv_path=venv_path,
                )

                with pytest.raises(RuntimeError, match="venv creation failed"):
                    config.setup(mock_model_spec)

    def test_setup_function_failure_raises(self, mock_model_spec):
        """Test VenvConfig.setup() raises RuntimeError when setup_function returns False."""
        from workflows.workflow_types import WorkflowVenvType
        from workflows.workflow_venvs import VenvConfig

        def failing_setup(venv_config, model_spec):
            return False

        with patch("workflows.workflow_venvs.run_command", return_value=0):
            config = VenvConfig(
                venv_type=WorkflowVenvType.EVALS_RUN_SCRIPT,
                setup_function=failing_setup,
            )

            with pytest.raises(RuntimeError, match="Failed to setup venv"):
                config.setup(mock_model_spec)


# =============================================================================
# BENCHMARKS_GENAI_PERF docker check hook - check=True
# =============================================================================


class TestGenaiPerfDockerCheckTrue:
    """Test the BENCHMARKS_GENAI_PERF setup hook uses check=True for docker --version.

    The genai-perf venv is Docker-based — no pip installs. Its `setup_function`
    hook (`check_docker_available`) verifies Docker is available on the host
    via `docker --version` and must raise on failure.
    """

    @pytest.fixture
    def mock_venv_config(self):
        """Create a mock venv config."""
        mock_config = MagicMock()
        mock_config.venv_python = "/fake/venv/bin/python"
        mock_config.venv_path = MagicMock()
        mock_config.venv_path.exists.return_value = True
        return mock_config

    @pytest.fixture
    def mock_model_spec(self):
        """Create a mock model spec."""
        return MagicMock()

    def test_docker_version_uses_check_true(self, mock_venv_config, mock_model_spec):
        """Test docker --version command uses check=True."""
        from workflows.workflow_venvs import check_docker_available

        with patch("workflows.workflow_venvs.run_command") as mock_run:
            check_docker_available(mock_venv_config, mock_model_spec)

            # Find the docker --version call
            docker_call = None
            for call_item in mock_run.call_args_list:
                if "docker --version" in str(call_item):
                    docker_call = call_item
                    break

            assert docker_call is not None, "docker --version call not found"
            assert docker_call[1].get("check") is True

    def test_docker_version_raises_on_failure(self, mock_venv_config, mock_model_spec):
        """Test docker --version failure raises RuntimeError."""
        from workflows.workflow_venvs import check_docker_available

        with patch(
            "workflows.workflow_venvs.run_command",
            side_effect=RuntimeError("docker not found"),
        ):
            with pytest.raises(RuntimeError, match="docker not found"):
                check_docker_available(mock_venv_config, mock_model_spec)

    def test_genai_perf_config_registers_docker_hook(self):
        """Registry sanity: BENCHMARKS_GENAI_PERF declares the docker hook and no pip deps."""
        from workflows.workflow_types import WorkflowVenvType
        from workflows.workflow_venvs import VENV_CONFIGS, check_docker_available

        cfg = VENV_CONFIGS[WorkflowVenvType.BENCHMARKS_GENAI_PERF]
        assert cfg.requirements_file is None
        assert cfg.setup_function is check_docker_available
        assert "artifacts" in cfg.extra_dirs


# =============================================================================
# run.py system software validation - check=False with manual error handling
# =============================================================================


class TestSystemSoftwareValidationCheckFalse:
    """Test validate_local_setup uses check=False and raises ValueError on failure."""

    def test_system_validation_uses_check_false(self):
        """Test system software validation command uses check=False (default)."""
        from workflows.validate_setup import validate_local_setup

        mock_model_spec = MagicMock()
        mock_runtime_config = RuntimeConfig(
            model="test",
            workflow="server",
            device="n150",
            skip_system_sw_validation=False,
        )

        with patch(
            "workflows.validate_setup.get_default_workflow_root_log_dir"
        ) as mock_log_dir, patch(
            "workflows.validate_setup.ensure_readwriteable_dir"
        ), patch("workflows.validate_setup.VENV_CONFIGS") as mock_venv_configs, patch(
            "workflows.validate_setup.run_command", return_value=0
        ) as mock_run, patch(
            "workflows.validate_setup.get_repo_root_path",
            return_value=Path("/fake/repo"),
        ):
            mock_log_dir.return_value = Path("/fake/logs")
            mock_venv_config = MagicMock()
            mock_venv_config.venv_python = "/fake/python"
            mock_venv_config.setup.return_value = True
            mock_venv_configs.__getitem__.return_value = mock_venv_config

            validate_local_setup(
                mock_model_spec, mock_runtime_config, "/fake/json/path"
            )

            # Verify run_command was called with check=False (default)
            mock_run.assert_called_once()
            assert mock_run.call_args[1].get("check", False) is False

    def test_system_validation_raises_valueerror_on_failure(self):
        """Test system software validation failure raises ValueError."""
        from workflows.validate_setup import validate_local_setup

        mock_model_spec = MagicMock()
        mock_runtime_config = RuntimeConfig(
            model="test",
            workflow="server",
            device="n150",
            skip_system_sw_validation=False,
        )

        with patch(
            "workflows.validate_setup.get_default_workflow_root_log_dir"
        ) as mock_log_dir, patch(
            "workflows.validate_setup.ensure_readwriteable_dir"
        ), patch("workflows.validate_setup.VENV_CONFIGS") as mock_venv_configs, patch(
            "workflows.validate_setup.run_command", return_value=1
        ), patch(
            "workflows.validate_setup.get_repo_root_path",
            return_value=Path("/fake/repo"),
        ):
            mock_log_dir.return_value = Path("/fake/logs")
            mock_venv_config = MagicMock()
            mock_venv_config.venv_python = "/fake/python"
            mock_venv_config.setup.return_value = True
            mock_venv_configs.__getitem__.return_value = mock_venv_config

            with pytest.raises(
                ValueError, match="validating system software dependencies failed"
            ):
                validate_local_setup(
                    mock_model_spec, mock_runtime_config, "/fake/json/path"
                )


# =============================================================================
# install_requirements() helper return value
# =============================================================================


class TestInstallRequirementsReturnBool:
    """Test install_requirements() returns True/False based on the pip install result.

    Pure-pip workflow venvs declare a `requirements_file` and rely on
    install_requirements() to drive the install. The helper wraps a single
    `uv pip install -r ...` invocation, so its return value is determined by
    a single `run_command` exit code.
    """

    @pytest.fixture
    def mock_venv_config(self):
        """Create a mock venv config that points install_requirements at any file."""
        mock_config = MagicMock()
        mock_config.venv_python = "/fake/venv/bin/python"
        return mock_config

    @pytest.mark.parametrize(
        "requirements_file",
        [
            "evals-common.txt",
            "evals-vision.txt",
            "evals-audio.txt",
            "evals-embedding.txt",
            "evals-video.txt",
            "evals-run-script.txt",
            "stress-tests-run-script.txt",
            "benchmarks-run-script.txt",
            "benchmarks-vllm.txt",
            "benchmarks-aiperf.txt",
            "reports-run-script.txt",
            "hf-setup.txt",
            "system-software-validation.txt",
            "tt-smi.txt",
            "tt-topology.txt",
            "tests-run-script.txt",
        ],
    )
    def test_install_requirements_returns_true_on_success(
        self, mock_venv_config, requirements_file
    ):
        """install_requirements() returns True when uv pip install succeeds (rc=0)."""
        from workflows.workflow_venvs import install_requirements

        with patch("workflows.workflow_venvs.run_command", return_value=0):
            assert install_requirements(mock_venv_config, requirements_file) is True

    def test_install_requirements_returns_false_on_failure(self, mock_venv_config):
        """install_requirements() returns False when uv pip install fails (rc!=0)."""
        from workflows.workflow_venvs import install_requirements

        with patch("workflows.workflow_venvs.run_command", return_value=1):
            assert install_requirements(mock_venv_config, "evals-common.txt") is False

    def test_install_requirements_raises_for_missing_file(self, mock_venv_config):
        """install_requirements() raises FileNotFoundError if the file doesn't exist."""
        from workflows.workflow_venvs import install_requirements

        with pytest.raises(FileNotFoundError, match="Requirements file not found"):
            install_requirements(mock_venv_config, "this-file-does-not-exist.txt")


class TestVenvConfigRegistry:
    """Sanity checks on the declarative VENV_CONFIGS registry.

    Catches accidental deletion or mis-pointing of a per-venv requirements file.
    """

    def test_every_referenced_requirements_file_exists(self):
        """Every `requirements_file` referenced in VENV_CONFIGS must exist on disk."""
        from workflows.workflow_venvs import REQUIREMENTS_DIR, VENV_CONFIGS

        missing = []
        for venv_type, cfg in VENV_CONFIGS.items():
            if cfg.requirements_file is None:
                continue
            path = REQUIREMENTS_DIR / cfg.requirements_file
            if not path.is_file():
                missing.append(f"{venv_type.name} -> {path}")
        assert missing == [], f"Missing requirements files: {missing}"

    def test_every_pure_pip_venv_has_no_setup_function(self):
        """Pure-pip venvs (no extra_dirs, no hook) must declare a requirements_file."""
        from workflows.workflow_venvs import VENV_CONFIGS

        for venv_type, cfg in VENV_CONFIGS.items():
            if cfg.setup_function is None and not cfg.extra_dirs:
                assert cfg.requirements_file is not None, (
                    f"Venv {venv_type.name} has no requirements_file, no extra_dirs, "
                    f"and no setup_function — it would do nothing on setup()."
                )


# =============================================================================
# Workflow execution with check=False (default)
# =============================================================================


class TestWorkflowExecutionCheckFalse:
    """Test workflow execution commands use check=False to allow continuation."""

    def test_run_workflow_script_does_not_raise_on_failure(self):
        """Test run_workflow_script doesn't raise when workflow fails."""
        from workflows.run_workflows import WorkflowSetup

        mock_model_spec = MagicMock()
        mock_model_spec.model_name = "test-model"
        mock_model_spec.model_id = "test-model-id"
        mock_runtime_config = RuntimeConfig(
            model="test-model",
            workflow="tests",
            device="n150",
        )

        with patch("workflows.run_workflows.WorkflowType") as mock_wf_type, patch(
            "workflows.run_workflows.WORKFLOW_CONFIGS"
        ) as mock_configs, patch(
            "workflows.run_workflows.VENV_CONFIGS"
        ) as mock_venvs, patch(
            "workflows.run_workflows.run_command", return_value=1
        ) as mock_run, patch(
            "workflows.run_workflows.get_default_workflow_root_log_dir"
        ), patch("workflows.run_workflows.ensure_readwriteable_dir"):
            mock_wf_type.from_string.return_value = MagicMock()
            mock_config = MagicMock()
            mock_config.workflow_run_script_venv_type = MagicMock()
            mock_config.name = "tests"
            mock_config.run_script_path = "/fake/script.py"
            mock_configs.__getitem__.return_value = mock_config
            mock_venv_config = MagicMock()
            mock_venv_config.venv_python = "/fake/python"
            mock_venvs.__getitem__.return_value = mock_venv_config

            setup = WorkflowSetup(mock_model_spec, mock_runtime_config, "/fake/json")

            # This should NOT raise - just return non-zero code
            return_code = setup.run_workflow_script()

            assert return_code == 1
            # Verify check=False (default) was used
            mock_run.assert_called_once()
            assert mock_run.call_args[1].get("check", False) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
