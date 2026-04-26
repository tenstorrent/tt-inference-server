# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Tests for bind mount permission validation in workflows/validate_setup.py."""

import os
from argparse import Namespace
from unittest.mock import ANY, MagicMock, patch

import pytest

from workflows.runtime_config import RuntimeConfig
from workflows.utils import check_path_permissions_for_uid
from workflows.validate_setup import (
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
                with pytest.raises(
                    ValueError, match="Bind mount permission check failed"
                ):
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
                with pytest.raises(
                    ValueError, match="Bind mount permission check failed"
                ):
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
                with pytest.raises(
                    ValueError, match="Bind mount permission check failed"
                ):
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
                with pytest.raises(ValueError, match="chmod/chown"):
                    validate_bind_mount_permissions(args)
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
            with pytest.raises(
                ValueError, match="requires --tt-metal-home or TT_METAL_HOME"
            ):
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

        with pytest.raises(ValueError, match="did not contain a cached snapshot"):
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

        validate_local_setup(model_spec, runtime_config, tmp_path / "runtime.json")

        mock_ensure_dir.assert_called_once_with(tmp_path / "logs")
        mock_run_command.assert_called_once_with(
            [str(venv_python), "-c", "import vllm"], logger=ANY
        )

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

        with pytest.raises(ValueError, match="requires the `vllm` Python package"):
            validate_local_setup(model_spec, runtime_config, tmp_path / "runtime.json")

        mock_ensure_dir.assert_called_once_with(tmp_path / "logs")
        mock_run_command.assert_called_once()
