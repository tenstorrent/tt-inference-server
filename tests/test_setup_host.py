# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Integration tests for setup_host volume/weight combinations.

Tests the full run.py code path: setup_host() -> SetupConfig -> generate_docker_run_command().
Each test verifies SetupConfig fields, setup_host() completion, and docker command output.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from workflows.runtime_config import RuntimeConfig
from workflows.validate_setup import validate_runtime_args
from workflows.model_spec import (
    DeviceModelSpec,
    ImplSpec,
    ModelSpec,
)
from workflows.run_docker_server import generate_docker_run_command
from workflows.setup_host import HostSetupManager, SetupConfig, setup_host
from workflows.workflow_types import (
    DeviceTypes,
    InferenceEngine,
    ModelSource,
)

# A tiny public HF model (~500KB, no auth required) for testing
TINY_HF_REPO = "hf-internal-testing/tiny-random-gpt2"
TINY_MODEL_NAME = "tiny-random-gpt2"


@pytest.fixture
def tiny_impl():
    """Minimal ImplSpec for testing."""
    return ImplSpec(
        impl_id="tt-transformers",
        impl_name="tt-transformers",
        repo_url="https://github.com/test/repo",
        code_path="models/test",
    )


@pytest.fixture
def tiny_device_model_spec():
    """Minimal DeviceModelSpec for testing."""
    return DeviceModelSpec(
        device=DeviceTypes.N150,
        max_concurrency=16,
        max_context=4096,
        default_impl=True,
    )


@pytest.fixture
def tiny_model_spec(tiny_impl, tiny_device_model_spec):
    """A real ModelSpec using a tiny public HF model."""
    return ModelSpec(
        device_type=DeviceTypes.N150,
        impl=tiny_impl,
        hf_model_repo=TINY_HF_REPO,
        model_id=f"id_tt-transformers_{TINY_MODEL_NAME}_n150",
        model_name=TINY_MODEL_NAME,
        tt_metal_commit="v1.0.0",
        vllm_commit="abc123",
        inference_engine=InferenceEngine.VLLM.value,
        device_model_spec=tiny_device_model_spec,
        docker_image="test-image:latest",
        min_disk_gb=1,
        min_ram_gb=1,
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_env_vars(temp_dir):
    """Set up mock environment variables for setup_host."""
    env_vars = {
        "HF_TOKEN": "hf_test_token_123456",
        "JWT_SECRET": "test_jwt_secret_123",
        "HF_HOME": str(temp_dir / "hf_home"),
        "SERVICE_PORT": "8000",
    }
    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture
def mock_system_calls():
    """Mock external system calls: subprocess, disk, HF API."""
    with patch("subprocess.run") as mock_subprocess, patch(
        "subprocess.check_output"
    ) as mock_check_output, patch("shutil.disk_usage") as mock_disk_usage, patch(
        "workflows.setup_host.http_request"
    ) as mock_http_request:
        mock_subprocess.return_value.returncode = 0
        mock_check_output.return_value = b"abc123def456\n"

        # 100GB free
        mock_disk_usage.return_value = (
            1000 * 1024**3,
            500 * 1024**3,
            100 * 1024**3,
        )

        def mock_hf_api(url, method="GET", headers=None):
            if "whoami-v2" in url:
                return b'{"name": "test_user"}', 200, {}
            elif "api/models" in url:
                return (
                    json.dumps({"siblings": [{"rfilename": "config.json"}]}).encode(),
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
def mock_ram_check():
    """Mock RAM check to return sufficient memory (50GB)."""
    mock_meminfo = "MemAvailable:    52428800 kB\n"
    with patch("builtins.open", mock_open(read_data=mock_meminfo)):
        yield


@pytest.fixture
def mock_cli_args(tiny_model_spec):
    """Create minimal RuntimeConfig that generate_docker_run_command reads."""
    return RuntimeConfig(
        model=TINY_MODEL_NAME,
        device="n150",
        workflow="server",
        service_port="8000",
        interactive=False,
        dev_mode=False,
        device_id=None,
        image_user="1000",
        docker_server=True,
        local_server=False,
        no_auth=False,
        host_volume=None,
        host_hf_cache=None,
        host_weights_dir=None,
    )


def _join_docker_cmd(docker_command):
    """Join docker command list into a string for easier searching."""
    return " ".join(str(c) for c in docker_command)


def _find_env_var(docker_command, key):
    """Find an environment variable value in a docker command list."""
    for i, arg in enumerate(docker_command):
        if arg == "-e" and i + 1 < len(docker_command):
            val = str(docker_command[i + 1])
            if val.startswith(f"{key}="):
                return val[len(f"{key}=") :]
    return None


class TestSetupHostCombinations:
    """Test SetupConfig field population for each volume/weight mode."""

    def test_default_docker_volume_mode(self, tiny_model_spec):
        """Mode 1: No host mounts, container handles download via docker volume."""
        config = SetupConfig(model_spec=tiny_model_spec)

        assert config.host_model_volume_root is None
        assert config.host_model_weights_mount_dir is None
        assert config.container_readonly_model_weights_dir is None
        assert config.container_model_weights_path == (
            config.cache_root / "weights" / TINY_MODEL_NAME
        )
        assert config.docker_volume_name.startswith("volume_id_")

    def test_host_volume_mode(self, tiny_model_spec, temp_dir):
        """Mode 2: Host volume bind mount for entire cache_root."""
        host_volume = str(temp_dir / "persistent_volume")
        config = SetupConfig(
            model_spec=tiny_model_spec,
            host_volume=host_volume,
        )

        assert config.host_model_volume_root is not None
        assert config.persistent_volume_root.resolve() == Path(host_volume).resolve()
        assert config.host_tt_metal_cache_dir is not None
        # Weights still default to cache_root path (downloaded into host volume)
        assert config.container_model_weights_path == (
            config.cache_root / "weights" / TINY_MODEL_NAME
        )
        # No readonly weights mount (weights are in the bind-mounted cache_root)
        assert config.host_model_weights_mount_dir is None

    def test_host_hf_cache_mode(self, tiny_model_spec, temp_dir):
        """Mode 3: Host HF cache for readonly weights mount."""
        hf_cache = str(temp_dir / "hf_home")
        config = SetupConfig(
            model_spec=tiny_model_spec,
            host_hf_cache=hf_cache,
        )

        assert Path(config.host_hf_cache).resolve() == Path(hf_cache).resolve()
        assert config.container_readonly_model_weights_dir is not None
        assert config.container_model_weights_mount_dir is not None
        assert config.container_readonly_model_weights_dir == (
            Path("/home/container_app_user/readonly_weights_mount")
        )
        # host_model_weights_mount_dir depends on HF cache snapshot existing;
        # without an actual download it's None at config time
        assert config.host_model_volume_root is None

    def test_host_weights_dir_mode(self, tiny_model_spec, temp_dir):
        """Mode 4: Pre-downloaded weights directory bind mount."""
        weights_dir = temp_dir / "my_weights"
        weights_dir.mkdir()
        # Create a dummy weight file so it looks like a real weights dir
        (weights_dir / "config.json").write_text("{}")

        config = SetupConfig(
            model_spec=tiny_model_spec,
            host_weights_dir=str(weights_dir),
        )

        assert config.host_model_weights_mount_dir.resolve() == weights_dir.resolve()
        assert config.container_readonly_model_weights_dir == (
            Path("/home/container_app_user/readonly_weights_mount")
        )
        expected_container_mount = (
            config.container_readonly_model_weights_dir / weights_dir.name
        )
        assert config.container_model_weights_mount_dir == expected_container_mount
        assert config.container_model_weights_path == expected_container_mount
        # No host volume set
        assert config.host_model_volume_root is None

    def test_local_server_defaults_to_repo_persistent_volume(self, tiny_model_spec):
        """Local server always gets a host-backed cache root."""
        config = SetupConfig(model_spec=tiny_model_spec, local_server=True)

        assert config.host_volume is not None
        assert config.persistent_volume_root.name == "persistent_volume"
        assert config.host_model_volume_root is not None
        assert config.host_tt_metal_cache_dir is not None

    def test_local_server_hf_cache_still_uses_host_volume_for_cache(
        self, tiny_model_spec, temp_dir
    ):
        """Local server keeps host volume storage for cache/logs when using HF cache weights."""
        hf_cache = temp_dir / "hf_home"
        snapshot_dir = (
            hf_cache
            / "hub"
            / f"models--{TINY_HF_REPO.replace('/', '--')}"
            / "snapshots"
            / "abc123"
        )
        snapshot_dir.mkdir(parents=True)
        config = SetupConfig(
            model_spec=tiny_model_spec,
            host_hf_cache=str(hf_cache),
            local_server=True,
        )

        assert config.host_model_volume_root is not None
        assert config.host_hf_cache == str(hf_cache.resolve())
        assert (
            config.host_model_weights_snapshot_dir.resolve() == snapshot_dir.resolve()
        )

    def test_local_model_source_mode(self, tiny_model_spec, temp_dir):
        """Mode 5: MODEL_SOURCE=local with MODEL_WEIGHTS_DIR env var.

        Note: model_source default is evaluated at class definition time via os.getenv,
        so we must pass it explicitly here. In production, MODEL_SOURCE is set before
        the process starts.
        """
        weights_dir = temp_dir / "local_weights"
        weights_dir.mkdir()
        (weights_dir / "config.json").write_text("{}")

        with patch.dict(os.environ, {"MODEL_WEIGHTS_DIR": str(weights_dir)}):
            config = SetupConfig(
                model_spec=tiny_model_spec,
                model_source=ModelSource.LOCAL.value,
            )

        assert config.model_source == ModelSource.LOCAL.value
        assert config.container_readonly_model_weights_dir is not None
        assert config.host_model_weights_mount_dir.resolve() == weights_dir.resolve()
        assert config.container_model_weights_path == (
            config.container_readonly_model_weights_dir / weights_dir.name
        )

    def test_default_mode_check_setup_returns_true(self, tiny_model_spec):
        """Default mode: check_setup() returns True (container handles download)."""
        manager = HostSetupManager(
            model_spec=tiny_model_spec,
            automatic=True,
            jwt_secret="test_jwt",
            hf_token="hf_test_token",
        )
        assert manager.check_setup() is True

    def test_host_volume_mode_check_setup_returns_false(
        self, tiny_model_spec, temp_dir
    ):
        """Host volume mode: check_setup() returns False (needs host download)."""
        host_volume = str(temp_dir / "persistent_volume")
        manager = HostSetupManager(
            model_spec=tiny_model_spec,
            automatic=True,
            jwt_secret="test_jwt",
            hf_token="hf_test_token",
            host_volume=host_volume,
        )
        assert manager.check_setup() is False

    def test_host_weights_dir_check_setup_with_valid_weights(
        self, tiny_model_spec, temp_dir
    ):
        """Host weights dir: check_setup() returns True when HF-format weights exist."""
        weights_dir = temp_dir / "my_weights"
        weights_dir.mkdir()
        # Create minimal HF-format weight files
        (weights_dir / "config.json").write_text("{}")
        (weights_dir / "tokenizer.json").write_text("{}")
        (weights_dir / "model.safetensors").write_bytes(b"\x00" * 16)

        manager = HostSetupManager(
            model_spec=tiny_model_spec,
            automatic=True,
            jwt_secret="test_jwt",
            hf_token="hf_test_token",
            host_weights_dir=str(weights_dir),
        )
        assert manager.check_setup() is True

    def test_host_weights_dir_check_setup_with_empty_dir(
        self, tiny_model_spec, temp_dir
    ):
        """Host weights dir: check_setup() returns False when directory is empty."""
        weights_dir = temp_dir / "empty_weights"
        weights_dir.mkdir()

        manager = HostSetupManager(
            model_spec=tiny_model_spec,
            automatic=True,
            jwt_secret="test_jwt",
            hf_token="hf_test_token",
            host_weights_dir=str(weights_dir),
        )
        assert manager.check_setup() is False


class TestSetupHostRunSetup:
    """Test setup_host() / run_setup() for each mode using the same code path as run.py."""

    def test_default_mode_skips_host_download(
        self, tiny_model_spec, mock_env_vars, mock_system_calls, mock_ram_check
    ):
        """Default mode: run_setup() returns early because check_setup() is True."""
        setup_config = setup_host(
            model_spec=tiny_model_spec,
            jwt_secret="test_jwt_secret_123",
            hf_token="hf_test_token_123456",
            automatic_setup=True,
        )

        assert setup_config.host_model_volume_root is None
        assert setup_config.host_model_weights_mount_dir is None

    def test_host_volume_mode_calls_setup_weights(
        self,
        tiny_model_spec,
        temp_dir,
        mock_env_vars,
        mock_system_calls,
        mock_ram_check,
    ):
        """Host volume mode: run_setup() calls setup_weights_huggingface()."""
        host_volume = str(temp_dir / "persistent_volume")

        with patch("workflows.setup_host.VENV_CONFIGS") as mock_venv_configs:
            mock_venv = MagicMock()
            mock_venv.venv_path = temp_dir / "fake_venv"
            mock_venv.venv_path.mkdir(parents=True, exist_ok=True)
            hf_bin = mock_venv.venv_path / "bin"
            hf_bin.mkdir(parents=True, exist_ok=True)
            (hf_bin / "hf").write_text("#!/bin/bash")
            mock_venv_configs.__getitem__ = MagicMock(return_value=mock_venv)

            setup_config = setup_host(
                model_spec=tiny_model_spec,
                jwt_secret="test_jwt_secret_123",
                hf_token="hf_test_token_123456",
                automatic_setup=True,
                host_volume=host_volume,
            )

        assert setup_config.host_model_volume_root is not None
        assert setup_config.host_model_volume_root.exists()

    def test_host_weights_dir_mode_skips_download(
        self,
        tiny_model_spec,
        temp_dir,
        mock_env_vars,
        mock_system_calls,
        mock_ram_check,
    ):
        """Host weights dir mode: run_setup() skips download when weights are valid."""
        weights_dir = temp_dir / "pre_downloaded"
        weights_dir.mkdir()
        (weights_dir / "config.json").write_text("{}")
        (weights_dir / "tokenizer.json").write_text("{}")
        (weights_dir / "model.safetensors").write_bytes(b"\x00" * 16)

        setup_config = setup_host(
            model_spec=tiny_model_spec,
            jwt_secret="test_jwt_secret_123",
            hf_token="hf_test_token_123456",
            automatic_setup=True,
            host_weights_dir=str(weights_dir),
        )

        assert Path(setup_config.host_weights_dir).resolve() == weights_dir.resolve()
        assert (
            setup_config.host_model_weights_mount_dir.resolve() == weights_dir.resolve()
        )
        assert setup_config.container_model_weights_path is not None

    def test_host_weights_dir_mode_runs_setup_when_incomplete(
        self,
        tiny_model_spec,
        temp_dir,
        mock_env_vars,
        mock_system_calls,
        mock_ram_check,
    ):
        """Host weights dir mode: still runs setup_model_environment when weights are incomplete."""
        weights_dir = temp_dir / "incomplete_weights"
        weights_dir.mkdir()
        # Only create config.json, not enough for check_setup() to return True

        setup_config = setup_host(
            model_spec=tiny_model_spec,
            jwt_secret="test_jwt_secret_123",
            hf_token="hf_test_token_123456",
            automatic_setup=True,
            host_weights_dir=str(weights_dir),
        )

        # Should still complete without error; setup_weights_huggingface returns
        # early for host_weights_dir
        assert Path(setup_config.host_weights_dir).resolve() == weights_dir.resolve()

    def test_local_server_skips_image_user_permission_fixes(
        self, tiny_model_spec, temp_dir
    ):
        """Local server uses the invoking host user, not Docker UID fixes."""
        host_volume = str(temp_dir / "persistent_volume")

        with patch(
            "workflows.setup_host._try_fix_path_permissions_for_uid"
        ) as fix_permissions_mock:
            setup_config = setup_host(
                model_spec=tiny_model_spec,
                jwt_secret="test_jwt_secret_123",
                hf_token="hf_test_token_123456",
                automatic_setup=True,
                host_volume=host_volume,
                image_user="1000",
                local_server=True,
            )

        assert setup_config.host_model_volume_root is not None
        assert setup_config.host_model_volume_root.exists()
        fix_permissions_mock.assert_not_called()

    def test_local_source_mode(
        self, tiny_model_spec, temp_dir, mock_system_calls, mock_ram_check
    ):
        """Local source mode: uses MODEL_WEIGHTS_DIR env var.

        model_source default is evaluated at import time, so we create
        HostSetupManager directly and set model_source on its config.
        In production, MODEL_SOURCE is set before the process starts.
        """
        weights_dir = temp_dir / "local_model"
        weights_dir.mkdir()
        (weights_dir / "config.json").write_text("{}")
        (weights_dir / "tokenizer.json").write_text("{}")
        (weights_dir / "model.safetensors").write_bytes(b"\x00" * 16)

        env_vars = {
            "HF_TOKEN": "hf_test_token_123456",
            "JWT_SECRET": "test_jwt_secret_123",
            "MODEL_WEIGHTS_DIR": str(weights_dir),
        }
        with patch.dict(os.environ, env_vars):
            manager = HostSetupManager(
                model_spec=tiny_model_spec,
                automatic=True,
                jwt_secret="test_jwt_secret_123",
                hf_token="hf_test_token_123456",
            )
            # Set model_source explicitly (in production this is set before import)
            manager.setup_config.model_source = ModelSource.LOCAL.value
            manager.setup_config._infer_data()
            manager.run_setup()

        assert manager.setup_config.model_source == ModelSource.LOCAL.value
        assert (
            manager.setup_config.host_model_weights_mount_dir.resolve()
            == weights_dir.resolve()
        )


class TestSetupHostDockerCommand:
    """Test that generate_docker_run_command() produces correct mounts and env vars."""

    def _make_json_fpath(self, temp_dir):
        """Create a dummy model spec JSON file."""
        json_fpath = temp_dir / "model_spec.json"
        json_fpath.write_text("{}")
        return json_fpath

    def _generate_cmd(self, tiny_model_spec, runtime_config, config, json_fpath):
        """Helper to call generate_docker_run_command with standard mocks."""
        with patch(
            "workflows.run_docker_server.get_repo_root_path",
            return_value=Path("/tmp"),
        ), patch(
            "workflows.run_docker_server.DeviceTypes",
        ), patch("workflows.run_docker_server.short_uuid", return_value="test123"):
            return generate_docker_run_command(
                tiny_model_spec, runtime_config, config, json_fpath
            )

    def test_default_mode_docker_command(
        self, tiny_model_spec, mock_cli_args, temp_dir
    ):
        """Default mode: docker volume mount, no MODEL_WEIGHTS_DIR env var."""
        config = SetupConfig(model_spec=tiny_model_spec)
        json_fpath = self._make_json_fpath(temp_dir)

        docker_command, _ = self._generate_cmd(
            tiny_model_spec, mock_cli_args, config, json_fpath
        )

        assert "--volume" in docker_command
        assert _find_env_var(docker_command, "MODEL_WEIGHTS_DIR") is None
        assert _find_env_var(docker_command, "TT_CACHE_PATH") is None

    def test_host_volume_mode_docker_command(
        self, tiny_model_spec, mock_cli_args, temp_dir
    ):
        """Host volume mode: bind mount of cache_root, TT_CACHE_PATH set."""
        host_volume = str(temp_dir / "persistent_volume")
        config = SetupConfig(
            model_spec=tiny_model_spec,
            host_volume=host_volume,
        )
        json_fpath = self._make_json_fpath(temp_dir)

        docker_command, _ = self._generate_cmd(
            tiny_model_spec, mock_cli_args, config, json_fpath
        )

        cmd_str = _join_docker_cmd(docker_command)
        assert f"type=bind,src={config.host_model_volume_root}" in cmd_str
        assert _find_env_var(docker_command, "TT_CACHE_PATH") is not None
        # No separate readonly weights mount (weights in the cache_root volume)
        assert _find_env_var(docker_command, "MODEL_WEIGHTS_DIR") is None

    def test_host_hf_cache_mode_docker_command(
        self, tiny_model_spec, mock_cli_args, temp_dir
    ):
        """Host HF cache mode: readonly weights mount + MODEL_WEIGHTS_DIR when snapshot exists."""
        hf_cache = str(temp_dir / "hf_home")

        # Simulate an HF cache snapshot directory structure
        snapshot_hash = "abcdef1234567890"
        snapshot_dir = (
            temp_dir
            / "hf_home"
            / "hub"
            / f"models--{TINY_HF_REPO.replace('/', '--')}"
            / "snapshots"
            / snapshot_hash
        )
        snapshot_dir.mkdir(parents=True)
        (snapshot_dir / "config.json").write_text("{}")

        config = SetupConfig(
            model_spec=tiny_model_spec,
            host_hf_cache=hf_cache,
        )

        json_fpath = self._make_json_fpath(temp_dir)

        docker_command, _ = self._generate_cmd(
            tiny_model_spec, mock_cli_args, config, json_fpath
        )

        cmd_str = _join_docker_cmd(docker_command)
        # Readonly weights mount present
        assert "readonly" in cmd_str
        assert str(config.host_model_weights_mount_dir) in cmd_str
        # MODEL_WEIGHTS_DIR set because host weights are mounted
        assert _find_env_var(docker_command, "MODEL_WEIGHTS_DIR") is not None

    def test_host_weights_dir_mode_docker_command(
        self, tiny_model_spec, mock_cli_args, temp_dir
    ):
        """Host weights dir mode: readonly weights mount + MODEL_WEIGHTS_DIR."""
        weights_dir = temp_dir / "my_weights"
        weights_dir.mkdir()
        (weights_dir / "config.json").write_text("{}")

        config = SetupConfig(
            model_spec=tiny_model_spec,
            host_weights_dir=str(weights_dir),
        )
        json_fpath = self._make_json_fpath(temp_dir)

        docker_command, _ = self._generate_cmd(
            tiny_model_spec, mock_cli_args, config, json_fpath
        )

        cmd_str = _join_docker_cmd(docker_command)
        # Readonly weights mount present
        assert f"type=bind,src={weights_dir.resolve()}" in cmd_str
        assert "readonly" in cmd_str
        # MODEL_WEIGHTS_DIR set to container mount path
        model_weights_dir = _find_env_var(docker_command, "MODEL_WEIGHTS_DIR")
        assert model_weights_dir is not None
        assert "readonly_weights_mount" in model_weights_dir
        # Docker named volume used for cache_root (not host volume)
        assert "--volume" in docker_command


class TestSetupHostValidation:
    """Test validation: mutual exclusivity of weight source options."""

    def _make_mock_model_spec_and_config(self, **cli_overrides):
        """Create a mock model_spec and RuntimeConfig for validate_runtime_args."""
        mock_spec = MagicMock()
        mock_spec.model_id = "id_tt-transformers_Llama-3.1-8B-Instruct_n150"
        mock_spec.model_name = "Llama-3.1-8B-Instruct"

        cli_defaults = {
            "workflow": "benchmarks",
            "device": "n150",
            "model": "Llama-3.1-8B-Instruct",
            "docker_server": False,
            "local_server": False,
            "interactive": False,
            "no_auth": False,
            "host_volume": None,
            "host_hf_cache": None,
            "host_weights_dir": None,
        }
        cli_defaults.update(cli_overrides)

        runtime_config = RuntimeConfig(**cli_defaults)
        return mock_spec, runtime_config

    def test_mutual_exclusivity_host_volume_and_hf_cache(self):
        """Setting both --host-volume and --host-hf-cache should raise."""
        mock_spec, runtime_config = self._make_mock_model_spec_and_config(
            host_volume="/tmp/vol",
            host_hf_cache="/tmp/hf",
        )
        with patch.dict(
            "workflows.validate_setup.MODEL_SPECS", {mock_spec.model_id: mock_spec}
        ):
            with pytest.raises(
                ValueError,
                match="Only one of --host-volume, --host-hf-cache, --host-weights-dir",
            ):
                validate_runtime_args(mock_spec, runtime_config)

    def test_mutual_exclusivity_host_volume_and_weights_dir(self):
        """Setting both --host-volume and --host-weights-dir should raise."""
        mock_spec, runtime_config = self._make_mock_model_spec_and_config(
            host_volume="/tmp/vol",
            host_weights_dir="/tmp/weights",
        )
        with patch.dict(
            "workflows.validate_setup.MODEL_SPECS", {mock_spec.model_id: mock_spec}
        ):
            with pytest.raises(
                ValueError,
                match="Only one of --host-volume, --host-hf-cache, --host-weights-dir",
            ):
                validate_runtime_args(mock_spec, runtime_config)

    def test_mutual_exclusivity_hf_cache_and_weights_dir(self):
        """Setting both --host-hf-cache and --host-weights-dir should raise."""
        mock_spec, runtime_config = self._make_mock_model_spec_and_config(
            host_hf_cache="/tmp/hf",
            host_weights_dir="/tmp/weights",
        )
        with patch.dict(
            "workflows.validate_setup.MODEL_SPECS", {mock_spec.model_id: mock_spec}
        ):
            with pytest.raises(
                ValueError,
                match="Only one of --host-volume, --host-hf-cache, --host-weights-dir",
            ):
                validate_runtime_args(mock_spec, runtime_config)

    def test_all_three_set_raises(self):
        """Setting all three weight source options should raise."""
        mock_spec, runtime_config = self._make_mock_model_spec_and_config(
            host_volume="/tmp/vol",
            host_hf_cache="/tmp/hf",
            host_weights_dir="/tmp/weights",
        )
        with patch.dict(
            "workflows.validate_setup.MODEL_SPECS", {mock_spec.model_id: mock_spec}
        ):
            with pytest.raises(
                ValueError,
                match="Only one of --host-volume, --host-hf-cache, --host-weights-dir",
            ):
                validate_runtime_args(mock_spec, runtime_config)

    def test_single_option_passes_validation(self):
        """Setting only --host-weights-dir should pass mutual exclusivity check."""
        mock_spec, runtime_config = self._make_mock_model_spec_and_config(
            host_weights_dir="/tmp/weights",
        )
        with patch.dict(
            "workflows.validate_setup.MODEL_SPECS", {mock_spec.model_id: mock_spec}
        ):
            validate_runtime_args(mock_spec, runtime_config)

    def test_no_options_passes_validation(self):
        """Setting none of the weight source options should pass."""
        mock_spec, runtime_config = self._make_mock_model_spec_and_config()
        with patch.dict(
            "workflows.validate_setup.MODEL_SPECS", {mock_spec.model_id: mock_spec}
        ):
            validate_runtime_args(mock_spec, runtime_config)

    def test_host_weights_dir_nonexistent_directory(self, tiny_model_spec):
        """check_setup() returns False when host_weights_dir does not exist."""
        manager = HostSetupManager(
            model_spec=tiny_model_spec,
            automatic=True,
            jwt_secret="test_jwt",
            hf_token="hf_test_token",
            host_weights_dir="/nonexistent/path/to/weights",
        )
        assert manager.check_setup() is False
