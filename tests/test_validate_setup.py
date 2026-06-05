# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Tests for bind mount permission validation in workflows/validate_setup.py."""

import os
from argparse import Namespace
from unittest.mock import ANY, MagicMock, patch

import pytest

from workflows.run_local_server import vllm_tt_plugin_source_path
from workflows.runtime_config import RuntimeConfig
from workflows.utils import check_path_permissions_for_uid
from workflows.validate_setup import (
    _check_image_version_supported,
    _try_fix_path_permissions_for_uid,  # noqa: F401
    validate_bind_mount_permissions,
    validate_local_setup,
    validate_local_server_paths,
    validate_runtime_args,
)


class TestCheckPathPermissionsForUid:
    """Tests for check_path_permissions_for_uid helper."""

    def test_nonexistent_path(self, tmp_path):
        ok, reason = check_path_permissions_for_uid(tmp_path / "nonexistent", uid=1000)
        assert not ok
        assert "does not exist" in reason

    def test_owner_has_read(self, tmp_path):
        """Owner UID matches, read bit set."""
        d = tmp_path / "owned"
        d.mkdir()
        uid = os.getuid()
        ok, reason = check_path_permissions_for_uid(d, uid=uid)
        assert ok
        assert reason == ""

    def test_owner_lacks_read(self, tmp_path):
        """Owner UID matches but read bit is cleared."""
        d = tmp_path / "no_read"
        d.mkdir()
        os.chmod(d, 0o300)
        try:
            uid = os.getuid()
            ok, reason = check_path_permissions_for_uid(d, uid=uid)
            assert not ok
            assert "lacks read permission" in reason
            assert "owner" in reason
        finally:
            os.chmod(d, 0o700)

    def test_owner_has_write(self, tmp_path):
        d = tmp_path / "writable"
        d.mkdir()
        uid = os.getuid()
        ok, reason = check_path_permissions_for_uid(d, uid=uid, need_write=True)
        assert ok

    def test_owner_lacks_write(self, tmp_path):
        d = tmp_path / "no_write"
        d.mkdir()
        os.chmod(d, 0o500)
        try:
            uid = os.getuid()
            ok, reason = check_path_permissions_for_uid(d, uid=uid, need_write=True)
            assert not ok
            assert "lacks write permission" in reason
        finally:
            os.chmod(d, 0o700)

    def test_directory_lacks_execute(self, tmp_path):
        """Directory without execute bit blocks traversal."""
        d = tmp_path / "no_exec"
        d.mkdir()
        os.chmod(d, 0o600)
        try:
            uid = os.getuid()
            ok, reason = check_path_permissions_for_uid(d, uid=uid)
            assert not ok
            assert "traverse" in reason
        finally:
            os.chmod(d, 0o700)

    def test_other_uid_world_readable(self, tmp_path):
        """Non-owner, non-group UID can read if world-readable."""
        d = tmp_path / "world_read"
        d.mkdir()
        os.chmod(d, 0o755)
        # UID 0 is root; use a UID that is not the owner and not in the group.
        # We use a mock to force the "other" code path.
        fake_uid = 99999
        with patch("workflows.utils.get_groups_for_uid", return_value=set()):
            ok, reason = check_path_permissions_for_uid(d, uid=fake_uid)
        assert ok

    def test_other_uid_not_world_readable(self, tmp_path):
        """Non-owner, non-group UID cannot read without world-read bit."""
        d = tmp_path / "no_world_read"
        d.mkdir()
        os.chmod(d, 0o750)
        fake_uid = 99999
        with patch("workflows.utils.get_groups_for_uid", return_value=set()):
            ok, reason = check_path_permissions_for_uid(d, uid=fake_uid)
        assert not ok
        assert "lacks read permission" in reason
        assert "other" in reason

    def test_other_uid_world_writable(self, tmp_path):
        d = tmp_path / "world_write"
        d.mkdir()
        os.chmod(d, 0o757)
        fake_uid = 99999
        with patch("workflows.utils.get_groups_for_uid", return_value=set()):
            ok, reason = check_path_permissions_for_uid(
                d, uid=fake_uid, need_write=True
            )
        assert ok

    def test_other_uid_not_world_writable(self, tmp_path):
        d = tmp_path / "no_world_write"
        d.mkdir()
        os.chmod(d, 0o755)
        fake_uid = 99999
        with patch("workflows.utils.get_groups_for_uid", return_value=set()):
            ok, reason = check_path_permissions_for_uid(
                d, uid=fake_uid, need_write=True
            )
        assert not ok
        assert "lacks write permission" in reason

    def test_group_member_can_read(self, tmp_path):
        """UID in the file's group can read with group-read bit."""
        d = tmp_path / "group_read"
        d.mkdir()
        st = d.stat()
        os.chmod(d, 0o750)
        fake_uid = 99999
        with patch(
            "workflows.utils.get_groups_for_uid",
            return_value={st.st_gid},
        ):
            ok, reason = check_path_permissions_for_uid(d, uid=fake_uid)
        assert ok

    def test_file_permissions(self, tmp_path):
        """Regular file (not directory) does not require execute bit."""
        f = tmp_path / "readable_file.txt"
        f.write_text("data")
        os.chmod(f, 0o644)
        uid = os.getuid()
        ok, reason = check_path_permissions_for_uid(f, uid=uid)
        assert ok


class TestValidateBindMountPermissions:
    """Tests for validate_bind_mount_permissions."""

    def _make_args(self, **overrides):
        defaults = {
            "image_user": str(os.getuid()),
            "host_volume": None,
            "host_hf_cache": None,
            "host_weights_dir": None,
        }
        defaults.update(overrides)
        return Namespace(**defaults)

    def test_no_bind_mounts_passes(self):
        """No host paths set -- nothing to validate."""
        args = self._make_args()
        validate_bind_mount_permissions(args)

    def test_host_volume_writable_passes(self, tmp_path):
        d = tmp_path / "volume"
        d.mkdir()
        args = self._make_args(host_volume=str(d))
        validate_bind_mount_permissions(args)

    def test_host_volume_not_writable_auto_fixed(self, tmp_path):
        """Auto-fix adds write permission when current user owns the directory."""
        d = tmp_path / "ro_volume"
        d.mkdir()
        os.chmod(d, 0o500)
        try:
            args = self._make_args(host_volume=str(d))
            validate_bind_mount_permissions(args)
            assert os.access(d, os.W_OK)
        finally:
            os.chmod(d, 0o700)

    def test_host_volume_not_writable_raises_when_fix_fails(self, tmp_path):
        d = tmp_path / "ro_volume"
        d.mkdir()
        os.chmod(d, 0o500)
        try:
            args = self._make_args(host_volume=str(d))
            with patch(
                "workflows.validate_setup._try_fix_path_permissions_for_uid",
                return_value=False,
            ):
                with pytest.raises(ValueError, match="bind-mounted"):
                    validate_bind_mount_permissions(args)
        finally:
            os.chmod(d, 0o700)

    def test_host_hf_cache_readable_passes(self, tmp_path):
        d = tmp_path / "hf_cache"
        d.mkdir()
        args = self._make_args(host_hf_cache=str(d))
        validate_bind_mount_permissions(args)

    def test_host_hf_cache_not_readable_auto_fixed(self, tmp_path):
        """Auto-fix adds read+execute permission for read-only mounts."""
        d = tmp_path / "hf_cache_noperm"
        d.mkdir()
        os.chmod(d, 0o300)
        try:
            args = self._make_args(host_hf_cache=str(d))
            validate_bind_mount_permissions(args)
            assert os.access(d, os.R_OK)
        finally:
            os.chmod(d, 0o700)

    def test_host_hf_cache_not_readable_raises_when_fix_fails(self, tmp_path):
        d = tmp_path / "hf_cache_noperm"
        d.mkdir()
        os.chmod(d, 0o300)
        try:
            args = self._make_args(host_hf_cache=str(d))
            with patch(
                "workflows.validate_setup._try_fix_path_permissions_for_uid",
                return_value=False,
            ):
                with pytest.raises(ValueError, match="bind-mounted"):
                    validate_bind_mount_permissions(args)
        finally:
            os.chmod(d, 0o700)

    def test_host_weights_dir_readable_passes(self, tmp_path):
        d = tmp_path / "weights"
        d.mkdir()
        args = self._make_args(host_weights_dir=str(d))
        validate_bind_mount_permissions(args)

    def test_host_weights_dir_not_readable_raises_when_fix_fails(self, tmp_path):
        d = tmp_path / "weights_noperm"
        d.mkdir()
        os.chmod(d, 0o300)
        try:
            args = self._make_args(host_weights_dir=str(d))
            with patch(
                "workflows.validate_setup._try_fix_path_permissions_for_uid",
                return_value=False,
            ):
                with pytest.raises(ValueError, match="bind-mounted"):
                    validate_bind_mount_permissions(args)
        finally:
            os.chmod(d, 0o700)

    def test_nonexistent_host_volume_is_created(self, tmp_path):
        missing = tmp_path / "missing"
        args = self._make_args(host_volume=str(missing))
        validate_bind_mount_permissions(args)
        assert missing.is_dir()

    def test_nonexistent_nested_host_volume_is_created(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        args = self._make_args(host_volume=str(nested))
        validate_bind_mount_permissions(args)
        assert nested.is_dir()

    def test_other_uid_volume_auto_fixed(self, tmp_path):
        """Auto-fix adds other rwx bits when container UID is not owner/group."""
        d = tmp_path / "other_fix"
        d.mkdir()
        os.chmod(d, 0o700)
        fake_uid = 99999
        args = self._make_args(image_user=str(fake_uid), host_volume=str(d))
        with patch("workflows.utils.get_groups_for_uid", return_value=set()):
            validate_bind_mount_permissions(args)
        mode = os.stat(d).st_mode
        assert mode & 0o007 == 0o007

    def test_error_message_includes_fix_guidance(self, tmp_path):
        d = tmp_path / "noperm"
        d.mkdir()
        os.chmod(d, 0o500)
        try:
            args = self._make_args(host_volume=str(d))
            with patch(
                "workflows.validate_setup._try_fix_path_permissions_for_uid",
                return_value=False,
            ):
                with pytest.raises(ValueError) as exc_info:
                    validate_bind_mount_permissions(args)
            msg = str(exc_info.value)
            assert "chown" in msg
            assert "chmod" in msg
        finally:
            os.chmod(d, 0o700)


class TestLocalServerValidation:
    def _make_model_spec(self):
        model_spec = MagicMock()
        model_spec.model_id = "id_tt-transformers_Mistral-7B-Instruct-v0.3_n150"
        model_spec.model_name = "Mistral-7B-Instruct-v0.3"
        model_spec.inference_engine = "vLLM"
        return model_spec

    def _make_runtime_config(self):
        runtime_config = RuntimeConfig(
            model="Mistral-7B-Instruct-v0.3",
            workflow="server",
            device="n150",
            local_server=True,
        )
        runtime_config.runtime_model_spec = {
            "hf_weights_repo": "mistralai/Mistral-7B-Instruct-v0.3"
        }
        return runtime_config

    def test_runtime_args_require_tt_metal_home_for_local_server(self):
        model_spec = self._make_model_spec()
        runtime_config = self._make_runtime_config()

        with patch.dict(
            "workflows.validate_setup.MODEL_SPECS",
            {model_spec.model_id: model_spec},
        ):
            with pytest.raises(ValueError, match="TT_METAL_HOME"):
                validate_runtime_args(model_spec, runtime_config)

    def test_runtime_args_allow_tt_metal_home_from_env(self):
        model_spec = self._make_model_spec()
        runtime_config = self._make_runtime_config()
        runtime_config.tt_metal_home = "/env/tt-metal"

        with patch.dict(
            "workflows.validate_setup.MODEL_SPECS",
            {model_spec.model_id: model_spec},
        ):
            validate_runtime_args(model_spec, runtime_config)

    def test_validate_local_server_paths_passes(self, tmp_path):
        tt_metal_home = tmp_path / "tt-metal"
        python_bin_dir = tt_metal_home / "python_env" / "bin"
        build_lib_dir = tt_metal_home / "build" / "lib"
        vllm_dir = tt_metal_home / "vllm"
        python_bin_dir.mkdir(parents=True)
        build_lib_dir.mkdir(parents=True)
        vllm_dir.mkdir(parents=True)
        (python_bin_dir / "python").write_text("")

        args = Namespace(
            local_server=True,
            tt_metal_home=str(tt_metal_home),
            tt_metal_python_venv_dir=None,
            vllm_dir=None,
            host_hf_cache=None,
            host_weights_dir=None,
            runtime_model_spec={
                "hf_weights_repo": "mistralai/Mistral-7B-Instruct-v0.3"
            },
        )

        validate_local_server_paths(args)

    def test_validate_local_server_paths_requires_python(self, tmp_path):
        tt_metal_home = tmp_path / "tt-metal"
        (tt_metal_home / "vllm").mkdir(parents=True)
        (tt_metal_home / "build" / "lib").mkdir(parents=True)

        args = Namespace(
            local_server=True,
            tt_metal_home=str(tt_metal_home),
            tt_metal_python_venv_dir=None,
            vllm_dir=None,
            host_hf_cache=None,
            host_weights_dir=None,
            runtime_model_spec={
                "hf_weights_repo": "mistralai/Mistral-7B-Instruct-v0.3"
            },
        )

        with pytest.raises(ValueError, match="python venv interpreter"):
            validate_local_server_paths(args)

    def test_validate_local_server_paths_requires_cached_hf_snapshot(self, tmp_path):
        tt_metal_home = tmp_path / "tt-metal"
        python_bin_dir = tt_metal_home / "python_env" / "bin"
        build_lib_dir = tt_metal_home / "build" / "lib"
        vllm_dir = tt_metal_home / "vllm"
        python_bin_dir.mkdir(parents=True)
        build_lib_dir.mkdir(parents=True)
        vllm_dir.mkdir(parents=True)
        (python_bin_dir / "python").write_text("")

        hf_home = tmp_path / "hf_home"
        hf_home.mkdir()
        args = Namespace(
            local_server=True,
            tt_metal_home=str(tt_metal_home),
            tt_metal_python_venv_dir=None,
            vllm_dir=None,
            host_hf_cache=str(hf_home),
            host_weights_dir=None,
            runtime_model_spec={
                "hf_weights_repo": "mistralai/Mistral-7B-Instruct-v0.3"
            },
        )

        with pytest.raises(ValueError, match="cached snapshot"):
            validate_local_server_paths(args)

    def test_validate_local_server_paths_accepts_cached_hf_snapshot(self, tmp_path):
        tt_metal_home = tmp_path / "tt-metal"
        python_bin_dir = tt_metal_home / "python_env" / "bin"
        build_lib_dir = tt_metal_home / "build" / "lib"
        vllm_dir = tt_metal_home / "vllm"
        python_bin_dir.mkdir(parents=True)
        build_lib_dir.mkdir(parents=True)
        vllm_dir.mkdir(parents=True)
        (python_bin_dir / "python").write_text("")

        snapshot_dir = (
            tmp_path
            / "hf_home"
            / "hub"
            / "models--mistralai--Mistral-7B-Instruct-v0.3"
            / "snapshots"
            / "abc123"
        )
        snapshot_dir.mkdir(parents=True)

        args = Namespace(
            local_server=True,
            tt_metal_home=str(tt_metal_home),
            tt_metal_python_venv_dir=None,
            vllm_dir=None,
            host_hf_cache=str(tmp_path / "hf_home"),
            host_weights_dir=None,
            runtime_model_spec={
                "hf_weights_repo": "mistralai/Mistral-7B-Instruct-v0.3"
            },
        )

        validate_local_server_paths(args)

    def test_validate_local_server_paths_accepts_explicit_vllm_dir(self, tmp_path):
        tt_metal_home = tmp_path / "tt-metal"
        python_bin_dir = tt_metal_home / "python_env" / "bin"
        build_lib_dir = tt_metal_home / "build" / "lib"
        python_bin_dir.mkdir(parents=True)
        build_lib_dir.mkdir(parents=True)
        (python_bin_dir / "python").write_text("")

        explicit_vllm_dir = tmp_path / "custom-vllm"
        explicit_vllm_dir.mkdir()
        args = Namespace(
            local_server=True,
            tt_metal_home=str(tt_metal_home),
            tt_metal_python_venv_dir=None,
            vllm_dir=str(explicit_vllm_dir),
            host_hf_cache=None,
            host_weights_dir=None,
            runtime_model_spec={
                "hf_weights_repo": "mistralai/Mistral-7B-Instruct-v0.3"
            },
        )

        validate_local_server_paths(args)

    @patch("workflows.validate_setup.run_command", return_value=0)
    @patch("workflows.validate_setup.ensure_readwriteable_dir")
    @patch("workflows.validate_setup.get_default_workflow_root_log_dir")
    def test_validate_local_setup_checks_vllm_installation(
        self,
        mock_get_log_dir,
        mock_ensure_dir,
        mock_run_command,
        tmp_path,
    ):
        tt_metal_home = tmp_path / "tt-metal"
        python_bin_dir = tt_metal_home / "python_env" / "bin"
        python_bin_dir.mkdir(parents=True)
        venv_python = python_bin_dir / "python"
        venv_python.write_text("")

        model_spec = self._make_model_spec()
        runtime_config = self._make_runtime_config()
        runtime_config.tt_metal_home = str(tt_metal_home)
        runtime_config.skip_system_sw_validation = True

        mock_get_log_dir.return_value = tmp_path / "logs"

        # No vllm checkout, so plugin install + check are skipped and the only
        # run_command call is the existing `import vllm` probe.
        validate_local_setup(model_spec, runtime_config, tmp_path / "runtime.json")

        mock_ensure_dir.assert_called_once_with(tmp_path / "logs")
        mock_run_command.assert_any_call(
            [str(venv_python), "-c", "import vllm"], logger=ANY
        )
        assert mock_run_command.call_count == 1

    @patch("workflows.validate_setup.run_command", return_value=1)
    @patch("workflows.validate_setup.ensure_readwriteable_dir")
    @patch("workflows.validate_setup.get_default_workflow_root_log_dir")
    def test_validate_local_setup_raises_when_vllm_not_installed(
        self,
        mock_get_log_dir,
        mock_ensure_dir,
        mock_run_command,
        tmp_path,
    ):
        tt_metal_home = tmp_path / "tt-metal"
        python_bin_dir = tt_metal_home / "python_env" / "bin"
        python_bin_dir.mkdir(parents=True)
        (python_bin_dir / "python").write_text("")

        model_spec = self._make_model_spec()
        runtime_config = self._make_runtime_config()
        runtime_config.tt_metal_home = str(tt_metal_home)
        runtime_config.skip_system_sw_validation = True

        mock_get_log_dir.return_value = tmp_path / "logs"

        with pytest.raises(ValueError, match="vllm Python package"):
            validate_local_setup(model_spec, runtime_config, tmp_path / "runtime.json")

        mock_ensure_dir.assert_called_once_with(tmp_path / "logs")
        mock_run_command.assert_called_once()

    @patch("workflows.run_local_server.run_command", return_value=0)
    @patch("workflows.validate_setup.run_command", return_value=0)
    @patch("workflows.validate_setup.ensure_readwriteable_dir")
    @patch("workflows.validate_setup.get_default_workflow_root_log_dir")
    def test_validate_local_setup_installs_and_verifies_tt_plugin_when_present(
        self,
        mock_get_log_dir,
        mock_ensure_dir,
        mock_validate_run_command,
        mock_runlocal_run_command,
        tmp_path,
    ):
        """When the vLLM checkout ships plugins/vllm-tt-plugin, local-server
        validation must (a) editable-install the plugin via run_local_server's
        install helper and (b) probe that the `tt` entry-point is registered.
        """
        tt_metal_home = tmp_path / "tt-metal"
        python_bin_dir = tt_metal_home / "python_env" / "bin"
        python_bin_dir.mkdir(parents=True)
        venv_python = python_bin_dir / "python"
        venv_python.write_text("")

        # Stage the plugin source so the validator detects it.
        vllm_dir = tmp_path / "vllm"
        plugin_dir = vllm_tt_plugin_source_path(vllm_dir)
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "pyproject.toml").write_text(
            "[project]\nname = 'vllm-tt-plugin'\n"
        )

        model_spec = self._make_model_spec()
        runtime_config = self._make_runtime_config()
        runtime_config.tt_metal_home = str(tt_metal_home)
        runtime_config.vllm_dir = str(vllm_dir)
        runtime_config.skip_system_sw_validation = True

        mock_get_log_dir.return_value = tmp_path / "logs"

        validate_local_setup(model_spec, runtime_config, tmp_path / "runtime.json")

        # The plugin install happens through workflows.run_local_server.run_command
        # because install_vllm_tt_plugin_if_present is defined in that module.
        mock_runlocal_run_command.assert_called_once()
        plugin_install_cmd = mock_runlocal_run_command.call_args.args[0]
        assert "pip install" in plugin_install_cmd
        assert "--no-deps" in plugin_install_cmd
        assert "plugins/vllm-tt-plugin" in plugin_install_cmd
        assert mock_runlocal_run_command.call_args.kwargs["check"] is True

        # The validator side does two calls: `import vllm` and the entry-point check.
        validate_calls = mock_validate_run_command.call_args_list
        assert len(validate_calls) == 2
        assert validate_calls[0].args[0] == [str(venv_python), "-c", "import vllm"]
        entry_point_cmd = validate_calls[1].args[0]
        assert entry_point_cmd[0] == str(venv_python)
        assert entry_point_cmd[1] == "-c"
        script = entry_point_cmd[2]
        assert "import vllm_tt_plugin" in script
        assert "vllm.platform_plugins" in script
        assert "'tt' in eps" in script

    @patch("workflows.run_local_server.run_command", return_value=0)
    @patch("workflows.validate_setup.ensure_readwriteable_dir")
    @patch("workflows.validate_setup.get_default_workflow_root_log_dir")
    def test_validate_local_setup_raises_when_tt_plugin_entry_point_missing(
        self,
        mock_get_log_dir,
        mock_ensure_dir,
        mock_runlocal_run_command,
        tmp_path,
    ):
        """If the plugin install succeeds but the entry-point check fails
        (e.g. wheel staleness, missing pip --editable refresh), validation
        must raise an actionable error rather than letting `vllm serve`
        crash later.
        """
        tt_metal_home = tmp_path / "tt-metal"
        python_bin_dir = tt_metal_home / "python_env" / "bin"
        python_bin_dir.mkdir(parents=True)
        (python_bin_dir / "python").write_text("")

        vllm_dir = tmp_path / "vllm"
        plugin_dir = vllm_tt_plugin_source_path(vllm_dir)
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "pyproject.toml").write_text(
            "[project]\nname = 'vllm-tt-plugin'\n"
        )

        model_spec = self._make_model_spec()
        runtime_config = self._make_runtime_config()
        runtime_config.tt_metal_home = str(tt_metal_home)
        runtime_config.vllm_dir = str(vllm_dir)
        runtime_config.skip_system_sw_validation = True

        mock_get_log_dir.return_value = tmp_path / "logs"

        # `import vllm` (call 1) succeeds with rc=0; the plugin entry-point
        # probe (call 2) fails with rc=1.
        with patch(
            "workflows.validate_setup.run_command", side_effect=[0, 1]
        ) as mock_validate_run_command:
            with pytest.raises(ValueError, match="vllm-tt-plugin"):
                validate_local_setup(
                    model_spec, runtime_config, tmp_path / "runtime.json"
                )

            assert mock_validate_run_command.call_count == 2

        mock_runlocal_run_command.assert_called_once()


class TestCheckImageVersionSupported:
    """run.py only emits the post-0.11 vLLM docker contract; pre-0.11 vLLM
    specs (or pre-0.11 override images) must be refused with a clear migration
    message rather than silently producing a broken docker run.

    Media-inference-server and forge images use a different Dockerfile that
    isn't affected by the 0.11.0 vLLM interface refactor, so the check must
    NOT fire for them.
    """

    def _spec(self, version, engine="vLLM"):
        s = MagicMock()
        s.version = version
        s.inference_engine = engine
        return s

    def test_post_0_11_vllm_versions_pass(self):
        # Boundary and above must not raise.
        _check_image_version_supported(self._spec("0.11.0"))
        _check_image_version_supported(self._spec("0.11.1"))
        _check_image_version_supported(self._spec("0.13.0"))
        _check_image_version_supported(self._spec("1.0.0"))

    def test_pre_0_11_vllm_versions_raise(self):
        for v in ("0.10.9", "0.10.1", "0.10.0", "0.9.0", "0.2.0"):
            with pytest.raises(RuntimeError, match="not supported"):
                _check_image_version_supported(self._spec(v))

    def test_error_names_exact_tag_to_checkout(self):
        # The matching tt-inference-server release tag is `v<spec.version>`.
        with pytest.raises(RuntimeError) as exc:
            _check_image_version_supported(self._spec("0.10.1"))
        msg = str(exc.value)
        assert "v0.10.1" in msg
        assert "git checkout v0.10.1" in msg

    def test_unparseable_versions_pass(self):
        # `dev`, `latest`, empty — let the runtime decide, matches main.
        _check_image_version_supported(self._spec("dev"))
        _check_image_version_supported(self._spec("latest"))
        _check_image_version_supported(self._spec(""))

    def test_media_engine_not_blocked_by_pre_0_11(self):
        # tt-media-inference-server images aren't affected by the vLLM
        # 0.11.0 interface change. Pre-0.11 media specs must still run.
        for v in ("0.2.0", "0.5.0", "0.9.0", "0.10.0", "0.10.1"):
            _check_image_version_supported(self._spec(v, engine="media"))

    def test_forge_engine_not_blocked_by_pre_0_11(self):
        # forge images are also outside the vLLM image-interface refactor.
        for v in ("0.2.0", "0.9.0", "0.10.1"):
            _check_image_version_supported(self._spec(v, engine="forge"))


class TestVersionParsers:
    """workflows.utils version helpers, used by _check_image_version_supported."""

    def test_parse_version_tuple(self):
        from workflows.utils import parse_version_tuple

        assert parse_version_tuple("0.10.0") == (0, 10, 0)
        assert parse_version_tuple("0.11.0") == (0, 11, 0)
        assert parse_version_tuple("0.9") == (0, 9, 0)
        assert parse_version_tuple("0.13.0-suffix") == (0, 13, 0)
        # Non-version / empty / non-string inputs return None.
        assert parse_version_tuple("dev") is None
        assert parse_version_tuple("") is None
        assert parse_version_tuple(None) is None  # type: ignore[arg-type]

    def test_parse_image_version(self):
        from workflows.utils import parse_image_version

        assert parse_image_version("ghcr.io/foo/bar:0.10.0-abc") == (0, 10, 0)
        assert parse_image_version("ghcr.io/foo/bar:0.11.0") == (0, 11, 0)
        assert parse_image_version("ghcr.io/foo/bar:0.9-abc") == (0, 9, 0)
        # Unparseable tags / no tag / no version-prefix / None.
        assert parse_image_version("ghcr.io/foo/bar:dev") is None
        assert parse_image_version("ghcr.io/foo/bar:latest") is None
        assert parse_image_version("ghcr.io/foo/bar") is None
        assert parse_image_version(None) is None  # type: ignore[arg-type]
