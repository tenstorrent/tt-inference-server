#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Tests for docker-interface era detection and routing.

Covers:
  * parse_version_tuple / parse_image_version (workflows.utils)
  * DockerInterface enum + _DOCKER_INTERFACE_ERAS table + get_docker_interface
    (workflows.docker_interface)
  * The pre-0.11 vs post-0.11 branches of generate_docker_run_command —
    verifying the right docker run shape (--ipc host vs --shm-size 32G,
    CLI args after <image>, MODEL_WEIGHTS_DIR vs MODEL_WEIGHTS_PATH, etc.)
    is emitted for each era.

Setup-host integration (volume / weights / hf-cache combinations) lives in
test_setup_host.py. ModelSpec.apply_overrides version re-parsing lives in
test_model_specification.py.
"""

import dataclasses
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from workflows.model_spec import DeviceModelSpec, ImplSpec, ModelSpec
from workflows.run_docker_server import generate_docker_run_command
from workflows.runtime_config import RuntimeConfig
from workflows.workflow_types import DeviceTypes, InferenceEngine


TINY_HF_REPO = "hf-internal-testing/tiny-random-gpt2"
TINY_MODEL_NAME = "tiny-random-gpt2"


@pytest.fixture
def tiny_impl():
    return ImplSpec(
        impl_id="tt-transformers",
        impl_name="tt-transformers",
        repo_url="https://github.com/test/repo",
        code_path="models/test",
    )


@pytest.fixture
def tiny_device_model_spec():
    return DeviceModelSpec(
        device=DeviceTypes.N150,
        max_concurrency=16,
        max_context=4096,
        default_impl=True,
    )


@pytest.fixture
def tiny_model_spec(tiny_impl, tiny_device_model_spec):
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
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture(autouse=True)
def _isolate_default_volume_root(monkeypatch, tmp_path):
    """Pin get_default_persistent_volume_root so SetupConfig's legacy
    auto-fallback path doesn't touch the real repo's persistent_volume/
    directory during tests."""
    monkeypatch.setattr(
        "workflows.setup_host.get_default_persistent_volume_root",
        lambda repo_root: tmp_path / "persistent_volume",
    )


@pytest.fixture
def runtime_config():
    """Minimal RuntimeConfig sufficient for generate_docker_run_command."""
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


def _find_env_var(docker_command, key):
    """Return the value of an `-e KEY=value` entry in the command, or None."""
    for i, arg in enumerate(docker_command):
        if arg == "-e" and i + 1 < len(docker_command):
            val = str(docker_command[i + 1])
            if val.startswith(f"{key}="):
                return val[len(f"{key}=") :]
    return None


def _generate(model_spec, runtime_config, json_fpath):
    """Call generate_docker_run_command with standard mocks."""
    from workflows.setup_host import SetupConfig

    config = SetupConfig(model_spec=model_spec)
    with patch(
        "workflows.run_docker_server.get_repo_root_path",
        return_value=Path("/tmp"),
    ), patch(
        "workflows.run_docker_server.DeviceTypes",
    ), patch("workflows.run_docker_server.short_uuid", return_value="test123"):
        return generate_docker_run_command(
            model_spec, runtime_config, config, json_fpath
        )


def _make_json_fpath(temp_dir):
    p = temp_dir / "model_spec.json"
    p.write_text("{}")
    return p


# ---------------------------------------------------------------------------
# Helper sanity — pure functions, no fixtures needed.
# ---------------------------------------------------------------------------


class TestVersionParsers:
    """parse_version_tuple and parse_image_version (workflows.utils)."""

    def test_parse_version_tuple(self):
        from workflows.utils import parse_version_tuple

        assert parse_version_tuple("0.10.0") == (0, 10, 0)
        assert parse_version_tuple("0.11.0") == (0, 11, 0)
        assert parse_version_tuple("0.9") == (0, 9, 0)
        assert parse_version_tuple("0.13.0-suffix") == (0, 13, 0)
        # Non-version / empty / non-string inputs return None (defensive).
        assert parse_version_tuple("dev") is None
        assert parse_version_tuple("") is None
        assert parse_version_tuple(None) is None  # type: ignore[arg-type]

    def test_parse_image_version(self):
        from workflows.utils import parse_image_version

        assert parse_image_version("ghcr.io/foo/bar:0.10.0-abc") == (0, 10, 0)
        assert parse_image_version("ghcr.io/foo/bar:0.11.0") == (0, 11, 0)
        assert parse_image_version("ghcr.io/foo/bar:0.9-abc") == (0, 9, 0)
        # Unparseable tags / no tag / no version-prefix.
        assert parse_image_version("ghcr.io/foo/bar:dev") is None
        assert parse_image_version("ghcr.io/foo/bar:latest") is None
        assert parse_image_version("ghcr.io/foo/bar") is None


class TestDockerInterfaceEnum:
    """DockerInterface enum + _DOCKER_INTERFACE_ERAS table + get_docker_interface."""

    def test_get_docker_interface_legacy_era(self):
        from workflows.docker_interface import DockerInterface, get_docker_interface

        # Anything < 0.11.0 routes to V1_LEGACY.
        assert get_docker_interface("0.10.0") is DockerInterface.V1_LEGACY
        assert get_docker_interface("0.10.1") is DockerInterface.V1_LEGACY
        assert get_docker_interface("0.10.9") is DockerInterface.V1_LEGACY
        assert get_docker_interface("0.0.1") is DockerInterface.V1_LEGACY

    def test_get_docker_interface_modern_era(self):
        from workflows.docker_interface import DockerInterface, get_docker_interface

        # 0.11.0 and above route to V2_MODERN.
        assert get_docker_interface("0.11.0") is DockerInterface.V2_MODERN
        assert get_docker_interface("0.13.0") is DockerInterface.V2_MODERN
        assert get_docker_interface("1.0.0") is DockerInterface.V2_MODERN

    def test_get_docker_interface_unparseable_defaults_to_modern(self):
        from workflows.docker_interface import DockerInterface, get_docker_interface

        # Unparseable / empty version strings fall back to the newest era so
        # behaviour on `:dev` / `:latest` / handwritten "dev" is preserved.
        assert get_docker_interface("") is DockerInterface.V2_MODERN
        assert get_docker_interface("dev") is DockerInterface.V2_MODERN
        assert get_docker_interface("latest") is DockerInterface.V2_MODERN

    def test_eras_table_ordered_newest_first(self):
        """Selection rule walks the table top-to-bottom and returns on the
        first match, so newest era must come first."""
        from workflows.docker_interface import _DOCKER_INTERFACE_ERAS

        cut_versions = [entry[0] for entry in _DOCKER_INTERFACE_ERAS]
        assert cut_versions == sorted(cut_versions, reverse=True), (
            "_DOCKER_INTERFACE_ERAS must be ordered newest-first."
        )

    def test_eras_table_covers_zero(self):
        """The lowest entry must be (0, 0, 0) so every parseable version
        finds a match — otherwise get_docker_interface would silently fall
        through to the safety-net newest-era return at the bottom."""
        from workflows.docker_interface import _DOCKER_INTERFACE_ERAS

        assert _DOCKER_INTERFACE_ERAS[-1][0] == (0, 0, 0), (
            "Last era must anchor at (0, 0, 0) to catch every version."
        )


# ---------------------------------------------------------------------------
# Integration — verify generate_docker_run_command emits the right shape per era.
# ---------------------------------------------------------------------------


class TestEraAwareDockerCommand:
    """generate_docker_run_command must produce a different command shape for
    V1_LEGACY (docker-entrypoint.sh + gosu; no CLI args) vs V2_MODERN (bash-c
    ENTRYPOINT; CLI args after <image>). Era is read from model_spec.version.
    """

    # -- V2_MODERN -----------------------------------------------------------

    def test_modern_uses_ipc_host_and_cli_args(
        self, tiny_model_spec, runtime_config, temp_dir
    ):
        ms = dataclasses.replace(
            tiny_model_spec, docker_image="img:0.12.0-abc", version="0.12.0"
        )
        cmd, _ = _generate(ms, runtime_config, _make_json_fpath(temp_dir))

        # --ipc host (not --shm-size).
        assert "--ipc" in cmd
        assert cmd[cmd.index("--ipc") + 1] == "host"
        assert "--shm-size" not in cmd

        # CLI args appear after <image>.
        post_image = cmd[cmd.index(ms.docker_image) + 1 :]
        assert "--model" in post_image
        assert "--tt-device" in post_image

        # Legacy MODEL_WEIGHTS_PATH and CACHE_ROOT are NOT injected — the
        # image has these baked in / derives them.
        assert _find_env_var(cmd, "MODEL_WEIGHTS_PATH") is None
        assert _find_env_var(cmd, "CACHE_ROOT") is None

        # run.py forwards json_fpath unconditionally now, but V2_MODERN +
        # dev_mode=False must NOT mount it — the image uses its own catalog.
        assert _find_env_var(cmd, "RUNTIME_MODEL_SPEC_JSON_PATH") is None
        assert not any("model_spec.json" in str(a) for a in cmd)

    def test_unparseable_version_defaults_to_modern(
        self, tiny_model_spec, runtime_config, temp_dir
    ):
        ms = dataclasses.replace(tiny_model_spec, docker_image="img:dev", version="dev")
        cmd, _ = _generate(ms, runtime_config, _make_json_fpath(temp_dir))

        assert "--ipc" in cmd
        assert "--shm-size" not in cmd
        post_image = cmd[cmd.index(ms.docker_image) + 1 :]
        assert "--model" in post_image

    # -- V1_LEGACY -----------------------------------------------------------

    def test_legacy_uses_shm_size_not_ipc_host(
        self, tiny_model_spec, runtime_config, temp_dir
    ):
        ms = dataclasses.replace(
            tiny_model_spec, docker_image="img:0.10.0-abc", version="0.10.0"
        )
        cmd, _ = _generate(ms, runtime_config, _make_json_fpath(temp_dir))

        assert "--shm-size" in cmd
        assert cmd[cmd.index("--shm-size") + 1] == "32G"
        assert "--ipc" not in cmd

    def test_legacy_wraps_cli_args_in_bash_c_cmd_override(
        self, tiny_model_spec, runtime_config, temp_dir
    ):
        # Pre-0.11 ENTRYPOINT is docker-entrypoint.sh + gosu, which exec's
        # CMD verbatim. The image's default CMD runs the script with no args
        # but the script asserts on --tt-device, so the image cannot start
        # without a CMD override. We replace CMD with the post-0.11
        # bash -c "...python run_vllm_api_server.py <args>" shape so the
        # entrypoint forwards a working command to gosu.
        ms = dataclasses.replace(
            tiny_model_spec, docker_image="img:0.10.0-abc", version="0.10.0"
        )
        cmd, _ = _generate(ms, runtime_config, _make_json_fpath(temp_dir))

        post_image = cmd[cmd.index(ms.docker_image) + 1 :]
        # CMD is three argv elements: bash -c <script>
        assert post_image[0] == "bash"
        assert post_image[1] == "-c"
        script = post_image[2]
        assert "python run_vllm_api_server.py" in script
        assert "source" in script and "$PYTHON_ENV_DIR" in script
        # Required args present in the wrapped script.
        assert "--model" in script
        assert "--tt-device" in script
        # And NOT loose at the end (would be forwarded as gosu args, crash).
        assert "--model" not in post_image[3:]
        assert "--tt-device" not in post_image[3:]

    def test_legacy_sets_required_env_vars(
        self, tiny_model_spec, runtime_config, temp_dir
    ):
        # Pre-0.11 script asserts these unconditionally; the container
        # crashes at import time if any is unset.
        ms = dataclasses.replace(
            tiny_model_spec, docker_image="img:0.10.0-abc", version="0.10.0"
        )
        cmd, _ = _generate(ms, runtime_config, _make_json_fpath(temp_dir))

        assert _find_env_var(cmd, "CACHE_ROOT") is not None
        assert _find_env_var(cmd, "TT_CACHE_PATH") is not None
        assert _find_env_var(cmd, "TT_MODEL_SPEC_JSON_PATH") is not None

    def test_legacy_mounts_spec_json_outside_dev_mode(
        self, tiny_model_spec, runtime_config, temp_dir
    ):
        """TT_MODEL_SPEC_JSON_PATH is required by the in-image script, so
        the spec JSON must mount any time it's provided — not just under
        --dev-mode like the modern path."""
        rc = dataclasses.replace(runtime_config, dev_mode=False)
        ms = dataclasses.replace(
            tiny_model_spec, docker_image="img:0.10.0-abc", version="0.10.0"
        )
        cmd, _ = _generate(ms, rc, _make_json_fpath(temp_dir))

        cmd_str = " ".join(str(c) for c in cmd)
        assert "model_spec.json" in cmd_str
        assert _find_env_var(cmd, "TT_MODEL_SPEC_JSON_PATH") is not None
        # And NOT the modern var name.
        assert _find_env_var(cmd, "RUNTIME_MODEL_SPEC_JSON_PATH") is None

    def test_legacy_uses_model_weights_path_env(
        self, tiny_model_spec, runtime_config, temp_dir
    ):
        """When weights ARE bind-mounted, pre-0.11 uses MODEL_WEIGHTS_PATH
        (the old name; renamed to MODEL_WEIGHTS_DIR in 0.11)."""
        from workflows.setup_host import SetupConfig

        weights_dir = temp_dir / "my_weights"
        weights_dir.mkdir()
        (weights_dir / "config.json").write_text("{}")

        ms = dataclasses.replace(
            tiny_model_spec, docker_image="img:0.10.0-abc", version="0.10.0"
        )
        config = SetupConfig(model_spec=ms, host_weights_dir=str(weights_dir))
        json_fpath = _make_json_fpath(temp_dir)

        with patch(
            "workflows.run_docker_server.get_repo_root_path",
            return_value=Path("/tmp"),
        ), patch(
            "workflows.run_docker_server.DeviceTypes",
        ), patch("workflows.run_docker_server.short_uuid", return_value="test123"):
            cmd, _ = generate_docker_run_command(ms, runtime_config, config, json_fpath)

        assert _find_env_var(cmd, "MODEL_WEIGHTS_PATH") is not None
        assert _find_env_var(cmd, "MODEL_WEIGHTS_DIR") is None


class TestLegacyRealisticFlagCombinations:
    """Pre-0.11 invariants (TT_MODEL_SPEC_JSON_PATH, MODEL_WEIGHTS_PATH) must
    hold under the flag combinations a user actually runs with — not only
    the contrived "host_weights_dir is set" path. Each test exercises one
    realistic combination of run.py / SetupConfig inputs.
    """

    def test_legacy_mounts_spec_json_without_dev_mode(
        self, tiny_model_spec, runtime_config, temp_dir
    ):
        """The legacy in-image script reads TT_MODEL_SPEC_JSON_PATH at import
        time and crashes if it's unset. Verify the env var and JSON bind-mount
        both appear for --docker-server without --dev-mode."""
        from workflows.setup_host import SetupConfig

        ms = dataclasses.replace(
            tiny_model_spec, docker_image="img:0.10.0-abc", version="0.10.0"
        )
        rc = dataclasses.replace(runtime_config, dev_mode=False, docker_server=True)
        config = SetupConfig(model_spec=ms)
        json_fpath = _make_json_fpath(temp_dir)

        with patch(
            "workflows.run_docker_server.get_repo_root_path",
            return_value=Path("/tmp"),
        ), patch(
            "workflows.run_docker_server.DeviceTypes",
        ), patch("workflows.run_docker_server.short_uuid", return_value="test123"):
            cmd, _ = generate_docker_run_command(ms, rc, config, json_fpath)

        assert _find_env_var(cmd, "TT_MODEL_SPEC_JSON_PATH") is not None
        # And the file is actually bind-mounted into the container.
        assert any("model_spec.json" in str(a) for a in cmd)

    def test_legacy_host_volume_sets_model_weights_path(
        self, tiny_model_spec, runtime_config, temp_dir
    ):
        """--host-volume does NOT populate host_model_weights_mount_dir, but
        pre-0.11 still requires MODEL_WEIGHTS_PATH. Verify it's derived from
        cache_root/weights/<model> in this case."""
        from workflows.setup_host import SetupConfig

        volume_root = temp_dir / "persistent_volumes"
        volume_root.mkdir()

        ms = dataclasses.replace(
            tiny_model_spec, docker_image="img:0.10.0-abc", version="0.10.0"
        )
        config = SetupConfig(model_spec=ms, host_volume=str(volume_root))
        json_fpath = _make_json_fpath(temp_dir)

        with patch(
            "workflows.run_docker_server.get_repo_root_path",
            return_value=Path("/tmp"),
        ), patch(
            "workflows.run_docker_server.DeviceTypes",
        ), patch("workflows.run_docker_server.short_uuid", return_value="test123"):
            cmd, _ = generate_docker_run_command(ms, runtime_config, config, json_fpath)

        weights_path = _find_env_var(cmd, "MODEL_WEIGHTS_PATH")
        assert weights_path is not None
        # Path is cache_root / weights / <model_name>; default cache_root is
        # /home/container_app_user/cache_root.
        assert weights_path.endswith(f"weights/{TINY_MODEL_NAME}")
        assert "cache_root" in weights_path
        # And NOT the modern var name.
        assert _find_env_var(cmd, "MODEL_WEIGHTS_DIR") is None

    def test_legacy_default_no_flags_auto_creates_host_volume(
        self, tiny_model_spec, monkeypatch, tmp_path
    ):
        """Pre-0.11 images can't download weights inside the container. When
        the user passes no --host-volume / --host-hf-cache / --host-weights-dir,
        SetupConfig must auto-fall-back to the default host volume so weights
        can be staged once and reused."""
        from workflows.setup_host import SetupConfig

        ms = dataclasses.replace(
            tiny_model_spec, docker_image="img:0.10.0-abc", version="0.10.0"
        )

        # Pin the default volume root so we can assert against it without
        # touching the real user home.
        fake_default = tmp_path / "persistent_volume"
        monkeypatch.setattr(
            "workflows.setup_host.get_default_persistent_volume_root",
            lambda repo_root: fake_default,
        )

        config = SetupConfig(model_spec=ms)

        # Legacy path: host_volume now points at the default location.
        assert config.host_volume is not None
        assert str(fake_default) in config.host_volume
        # host_model_volume_root must be populated as a result so the cache_root
        # bind mount lands on the host (not on an anonymous docker volume).
        assert config.host_model_volume_root is not None

        # And the resulting docker command must satisfy the pre-0.11 invariants
        # (MODEL_WEIGHTS_PATH + CACHE_ROOT required), proving the auto-fallback
        # actually unblocks the legacy entrypoint end-to-end.
        rc = RuntimeConfig(
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
        json_fpath = _make_json_fpath(tmp_path)
        with patch(
            "workflows.run_docker_server.get_repo_root_path",
            return_value=Path("/tmp"),
        ), patch(
            "workflows.run_docker_server.DeviceTypes",
        ), patch("workflows.run_docker_server.short_uuid", return_value="test123"):
            cmd, _ = generate_docker_run_command(ms, rc, config, json_fpath)

        assert _find_env_var(cmd, "MODEL_WEIGHTS_PATH") is not None
        assert _find_env_var(cmd, "CACHE_ROOT") is not None
        assert _find_env_var(cmd, "TT_MODEL_SPEC_JSON_PATH") is not None

    def test_modern_default_no_flags_keeps_docker_named_volume(self, tiny_model_spec):
        """Post-0.11 images can download weights themselves; the auto-fallback
        must not fire and the path stays on an anonymous docker named volume."""
        from workflows.setup_host import SetupConfig

        ms = dataclasses.replace(
            tiny_model_spec, docker_image="img:0.13.0-abc", version="0.13.0"
        )
        config = SetupConfig(model_spec=ms)

        assert config.host_volume is None
        assert config.host_model_volume_root is None
