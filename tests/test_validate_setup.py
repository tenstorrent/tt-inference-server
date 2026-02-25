# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Tests for bind mount permission validation in workflows/validate_setup.py."""

import os
from argparse import Namespace
from unittest.mock import patch

import pytest

from workflows.validate_setup import (
    _check_path_permissions_for_uid,
    validate_bind_mount_permissions,
)


class TestCheckPathPermissionsForUid:
    """Tests for _check_path_permissions_for_uid helper."""

    def test_nonexistent_path(self, tmp_path):
        ok, reason = _check_path_permissions_for_uid(tmp_path / "nonexistent", uid=1000)
        assert not ok
        assert "does not exist" in reason

    def test_owner_has_read(self, tmp_path):
        """Owner UID matches, read bit set."""
        d = tmp_path / "owned"
        d.mkdir()
        uid = os.getuid()
        ok, reason = _check_path_permissions_for_uid(d, uid=uid)
        assert ok
        assert reason == ""

    def test_owner_lacks_read(self, tmp_path):
        """Owner UID matches but read bit is cleared."""
        d = tmp_path / "no_read"
        d.mkdir()
        os.chmod(d, 0o300)
        try:
            uid = os.getuid()
            ok, reason = _check_path_permissions_for_uid(d, uid=uid)
            assert not ok
            assert "lacks read permission" in reason
            assert "owner" in reason
        finally:
            os.chmod(d, 0o700)

    def test_owner_has_write(self, tmp_path):
        d = tmp_path / "writable"
        d.mkdir()
        uid = os.getuid()
        ok, reason = _check_path_permissions_for_uid(d, uid=uid, need_write=True)
        assert ok

    def test_owner_lacks_write(self, tmp_path):
        d = tmp_path / "no_write"
        d.mkdir()
        os.chmod(d, 0o500)
        try:
            uid = os.getuid()
            ok, reason = _check_path_permissions_for_uid(d, uid=uid, need_write=True)
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
            ok, reason = _check_path_permissions_for_uid(d, uid=uid)
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
        with patch("workflows.validate_setup._get_groups_for_uid", return_value=set()):
            ok, reason = _check_path_permissions_for_uid(d, uid=fake_uid)
        assert ok

    def test_other_uid_not_world_readable(self, tmp_path):
        """Non-owner, non-group UID cannot read without world-read bit."""
        d = tmp_path / "no_world_read"
        d.mkdir()
        os.chmod(d, 0o750)
        fake_uid = 99999
        with patch("workflows.validate_setup._get_groups_for_uid", return_value=set()):
            ok, reason = _check_path_permissions_for_uid(d, uid=fake_uid)
        assert not ok
        assert "lacks read permission" in reason
        assert "other" in reason

    def test_other_uid_world_writable(self, tmp_path):
        d = tmp_path / "world_write"
        d.mkdir()
        os.chmod(d, 0o757)
        fake_uid = 99999
        with patch("workflows.validate_setup._get_groups_for_uid", return_value=set()):
            ok, reason = _check_path_permissions_for_uid(
                d, uid=fake_uid, need_write=True
            )
        assert ok

    def test_other_uid_not_world_writable(self, tmp_path):
        d = tmp_path / "no_world_write"
        d.mkdir()
        os.chmod(d, 0o755)
        fake_uid = 99999
        with patch("workflows.validate_setup._get_groups_for_uid", return_value=set()):
            ok, reason = _check_path_permissions_for_uid(
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
            "workflows.validate_setup._get_groups_for_uid",
            return_value={st.st_gid},
        ):
            ok, reason = _check_path_permissions_for_uid(d, uid=fake_uid)
        assert ok

    def test_file_permissions(self, tmp_path):
        """Regular file (not directory) does not require execute bit."""
        f = tmp_path / "readable_file.txt"
        f.write_text("data")
        os.chmod(f, 0o644)
        uid = os.getuid()
        ok, reason = _check_path_permissions_for_uid(f, uid=uid)
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

    def test_host_volume_not_writable_raises(self, tmp_path):
        d = tmp_path / "ro_volume"
        d.mkdir()
        os.chmod(d, 0o500)
        try:
            args = self._make_args(host_volume=str(d))
            with pytest.raises(ValueError, match="Bind mount permission check failed"):
                validate_bind_mount_permissions(args)
        finally:
            os.chmod(d, 0o700)

    def test_host_hf_cache_readable_passes(self, tmp_path):
        d = tmp_path / "hf_cache"
        d.mkdir()
        args = self._make_args(host_hf_cache=str(d))
        validate_bind_mount_permissions(args)

    def test_host_hf_cache_not_readable_raises(self, tmp_path):
        d = tmp_path / "hf_cache_noperm"
        d.mkdir()
        os.chmod(d, 0o300)
        try:
            args = self._make_args(host_hf_cache=str(d))
            with pytest.raises(ValueError, match="Bind mount permission check failed"):
                validate_bind_mount_permissions(args)
        finally:
            os.chmod(d, 0o700)

    def test_host_weights_dir_readable_passes(self, tmp_path):
        d = tmp_path / "weights"
        d.mkdir()
        args = self._make_args(host_weights_dir=str(d))
        validate_bind_mount_permissions(args)

    def test_host_weights_dir_not_readable_raises(self, tmp_path):
        d = tmp_path / "weights_noperm"
        d.mkdir()
        os.chmod(d, 0o300)
        try:
            args = self._make_args(host_weights_dir=str(d))
            with pytest.raises(ValueError, match="Bind mount permission check failed"):
                validate_bind_mount_permissions(args)
        finally:
            os.chmod(d, 0o700)

    def test_nonexistent_host_volume_raises(self, tmp_path):
        args = self._make_args(host_volume=str(tmp_path / "missing"))
        with pytest.raises(ValueError, match="does not exist"):
            validate_bind_mount_permissions(args)

    def test_error_message_includes_fix_guidance(self, tmp_path):
        d = tmp_path / "noperm"
        d.mkdir()
        os.chmod(d, 0o500)
        try:
            args = self._make_args(host_volume=str(d))
            with pytest.raises(ValueError, match="chmod/chown"):
                validate_bind_mount_permissions(args)
        finally:
            os.chmod(d, 0o700)
