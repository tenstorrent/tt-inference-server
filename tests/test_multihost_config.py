# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Tests for multi-host configuration generation."""

import os
import tempfile
from pathlib import Path

import pytest

from workflows.multihost_config import (
    CONTAINER_USER,
    ENV_PASSTHROUGH,
    MultiHostConfig,
    WORKER_SSH_PORT,
    build_mpi_args,
    build_override_tt_config,
    generate_rankfile,
    generate_ssh_config,
    get_rank_binding_path,
)
from workflows.workflow_types import DeviceTypes
from workflows.utils import (
    check_path_permissions_for_uid,
    get_groups_for_uid,
)
from workflows.run_multihost_validation import (
    _run_ssh_command,
    validate_multihost_bind_mount_permissions,
)


class TestGenerateSshConfig:
    def test_two_hosts(self):
        """SSH config uses real hostnames directly (not aliases)."""
        config = generate_ssh_config(["host-0", "host-1"])
        # Real hostnames as Host entries
        assert "Host host-0" in config
        assert "Host host-1" in config
        # No HostName directive needed (Host is the real hostname)
        assert "HostName" not in config
        assert f"Port {WORKER_SSH_PORT}" in config
        assert f"User {CONTAINER_USER}" in config
        assert "StrictHostKeyChecking no" in config
        assert "BatchMode yes" in config

    def test_four_hosts(self):
        config = generate_ssh_config(["host-0", "host-1", "host-2", "host-3"])
        assert "Host host-0" in config
        assert "Host host-1" in config
        assert "Host host-2" in config
        assert "Host host-3" in config

    def test_custom_parameters(self):
        config = generate_ssh_config(
            ["host-0", "host-1"],
            ssh_port=2222,
            ssh_user="testuser",
            identity_file="/custom/path/key",
        )
        assert "Port 2222" in config
        assert "User testuser" in config
        assert "IdentityFile /custom/path/key" in config


class TestGenerateRankfile:
    def test_two_hosts(self):
        """Rankfile uses real hostnames directly."""
        rankfile = generate_rankfile(["host-0", "host-1"])
        assert "rank 0=host-0 slot=0:*" in rankfile
        assert "rank 1=host-1 slot=0:*" in rankfile
        assert "# mpirun rankfile" in rankfile

    def test_four_hosts(self):
        hosts = ["h1", "h2", "h3", "h4"]
        rankfile = generate_rankfile(hosts)
        for i, host in enumerate(hosts):
            assert f"rank {i}={host} slot=0:*" in rankfile


class TestBuildMpiArgs:
    def test_two_hosts(self):
        """mpi_args includes --host with real hostnames and rankfile mapping."""
        args = build_mpi_args(["h1", "h2"], "/etc/mpirun/rankfile")
        # --host with real hostnames (not aliases)
        assert "--host h1,h2" in args
        assert "--map-by rankfile:file=/etc/mpirun/rankfile" in args
        assert "--bind-to none" in args


class TestGetRankBindingPath:
    def test_dual_galaxy(self):
        path = get_rank_binding_path(DeviceTypes.DUAL_GALAXY)
        assert "dual_galaxy_rank_bindings.yaml" in path
        assert "tt-metal" in path

    def test_quad_galaxy(self):
        path = get_rank_binding_path(DeviceTypes.QUAD_GALAXY)
        assert "quad_galaxy_rank_bindings.yaml" in path

    def test_unsupported_device_type(self):
        with pytest.raises(ValueError, match="Unsupported device type"):
            get_rank_binding_path(DeviceTypes.T3K)


class TestBuildOverrideTtConfig:
    def test_basic_config(self):
        config = build_override_tt_config(
            hosts=["host-0", "host-1"],
            mpi_interface="cnx1",
            config_pkl_dir="/mnt/shared/config_pkl",
            rankfile_path="/etc/mpirun/rankfile",
            device_type=DeviceTypes.DUAL_GALAXY,
        )

        assert "rank_binding" in config
        assert "dual_galaxy" in config["rank_binding"]
        assert "mpi_args" in config
        # mpi_args includes --host with real hostnames
        assert "--host host-0,host-1" in config["mpi_args"]
        assert "--map-by rankfile" in config["mpi_args"]
        assert config["extra_ttrun_args"] == "--tcp-interface cnx1"
        assert config["config_pkl_dir"] == "/mnt/shared/config_pkl"
        assert config["env_passthrough"] == ENV_PASSTHROUGH
        # Note: fabric_config, fabric_reliability_mode, trace_mode moved to DeviceModelSpec

    def test_custom_rank_binding(self):
        config = build_override_tt_config(
            hosts=["h1", "h2"],
            mpi_interface="eth0",
            config_pkl_dir="/tmp/config_pkl",
            rankfile_path="/etc/mpirun/rankfile",
            device_type=DeviceTypes.DUAL_GALAXY,
            rank_binding_path="/custom/rank_binding.yaml",
        )
        assert config["rank_binding"] == "/custom/rank_binding.yaml"


class TestMeshDeviceString:
    def test_dual_galaxy(self):
        assert DeviceTypes.DUAL_GALAXY.to_mesh_device_str() == "(8,8)"

    def test_quad_galaxy(self):
        assert DeviceTypes.QUAD_GALAXY.to_mesh_device_str() == "(8,16)"


class TestIsMultihost:
    def test_dual_galaxy_is_multihost(self):
        assert DeviceTypes.DUAL_GALAXY.is_multihost() is True

    def test_quad_galaxy_is_multihost(self):
        assert DeviceTypes.QUAD_GALAXY.is_multihost() is True

    def test_galaxy_is_not_multihost(self):
        assert DeviceTypes.GALAXY.is_multihost() is False

    def test_t3k_is_not_multihost(self):
        assert DeviceTypes.T3K.is_multihost() is False


class TestGetMultihostNumHosts:
    def test_dual_galaxy_num_hosts(self):
        assert DeviceTypes.DUAL_GALAXY.get_multihost_num_hosts() == 2

    def test_quad_galaxy_num_hosts(self):
        assert DeviceTypes.QUAD_GALAXY.get_multihost_num_hosts() == 4

    def test_non_multihost_device_raises(self):
        with pytest.raises(ValueError, match="not a multi-host device type"):
            DeviceTypes.GALAXY.get_multihost_num_hosts()


class TestRunSshCommand:
    def test_returns_tuple(self):
        """_run_ssh_command returns (success, output) tuple."""
        # Use localhost which should fail with BatchMode=yes if no key
        success, output = _run_ssh_command("localhost", ["echo", "test"], timeout=2)
        # Should return a tuple regardless of success/failure
        assert isinstance(success, bool)
        assert isinstance(output, str)

    def test_timeout_handling(self):
        """_run_ssh_command handles timeouts gracefully."""
        # Use a non-routable IP to trigger timeout
        success, output = _run_ssh_command("192.0.2.1", ["echo", "test"], timeout=1)
        assert success is False
        assert output  # Should have error message


class TestCheckPathPermissionsForUid:
    """Tests for the shared check_path_permissions_for_uid function."""

    def test_nonexistent_path_fails(self):
        """Non-existent path should return failure."""
        ok, reason = check_path_permissions_for_uid("/nonexistent/path", 1000)
        assert ok is False
        assert "does not exist" in reason

    def test_readable_path_passes(self):
        """Readable path should pass read check."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            os.chmod(path, 0o755)  # rwxr-xr-x
            ok, reason = check_path_permissions_for_uid(path, os.getuid())
            assert ok is True
            assert reason == ""

    def test_write_check_without_write_permission_fails(self):
        """Path without write permission should fail write check."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            st = path.stat()
            # Only test if we're not the owner (otherwise we have write permission)
            if st.st_uid != 1000:
                os.chmod(path, 0o755)  # rwxr-xr-x (no world write)
                ok, reason = check_path_permissions_for_uid(path, 1000, need_write=True)
                assert ok is False
                assert "lacks write permission" in reason

    def test_world_readable_passes_for_other_uid(self):
        """World-readable path should pass for any UID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            os.chmod(path, 0o755)  # rwxr-xr-x
            # Use a UID that's definitely not the owner
            other_uid = 99999
            ok, reason = check_path_permissions_for_uid(path, other_uid)
            assert ok is True


class TestGetGroupsForUid:
    """Tests for the shared get_groups_for_uid function."""

    def test_current_user_has_groups(self):
        """Current user should have at least one group (primary group)."""
        current_uid = os.getuid()
        groups = get_groups_for_uid(current_uid)
        assert len(groups) >= 1

    def test_nonexistent_uid_returns_empty(self):
        """Non-existent UID should return empty set."""
        # Use a UID that almost certainly doesn't exist
        groups = get_groups_for_uid(999999)
        assert groups == set()


class TestValidateMultihostBindMountPermissions:
    """Tests for the multihost bind mount permission validation."""

    def test_accessible_directories_pass(self):
        """Directories accessible by UID should pass validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shared_root = Path(tmpdir) / "shared"
            config_pkl = Path(tmpdir) / "config_pkl"
            shared_root.mkdir()
            config_pkl.mkdir()

            # Make world-writable for UID 1000
            os.chmod(shared_root, 0o777)
            os.chmod(config_pkl, 0o777)

            config = MultiHostConfig(
                hosts=["host1", "host2"],
                mpi_interface="eth0",
                shared_storage_root=str(shared_root),
                config_pkl_dir=str(config_pkl),
            )

            # Should not raise
            validate_multihost_bind_mount_permissions(config)

    def test_inaccessible_shared_root_fails(self):
        """Inaccessible shared root should fail validation with fix suggestions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shared_root = Path(tmpdir) / "shared"
            config_pkl = Path(tmpdir) / "config_pkl"
            shared_root.mkdir()
            config_pkl.mkdir()

            # Make config_pkl accessible but shared_root restricted
            os.chmod(config_pkl, 0o777)

            st = shared_root.stat()
            if st.st_uid != 1000:
                os.chmod(shared_root, 0o700)  # Only owner can access

                config = MultiHostConfig(
                    hosts=["host1", "host2"],
                    mpi_interface="eth0",
                    shared_storage_root=str(shared_root),
                    config_pkl_dir=str(config_pkl),
                )

                with pytest.raises(ValueError) as exc_info:
                    validate_multihost_bind_mount_permissions(config)

                error_msg = str(exc_info.value)
                assert "SHARED_STORAGE_ROOT" in error_msg
                assert "chown" in error_msg or "chmod" in error_msg

    def test_error_message_contains_all_failures(self):
        """Error message should contain all permission failures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shared_root = Path(tmpdir) / "shared"
            config_pkl = Path(tmpdir) / "config_pkl"
            shared_root.mkdir()
            config_pkl.mkdir()

            st = shared_root.stat()
            if st.st_uid != 1000:
                # Make both directories inaccessible
                os.chmod(shared_root, 0o700)
                os.chmod(config_pkl, 0o700)

                config = MultiHostConfig(
                    hosts=["host1", "host2"],
                    mpi_interface="eth0",
                    shared_storage_root=str(shared_root),
                    config_pkl_dir=str(config_pkl),
                )

                with pytest.raises(ValueError) as exc_info:
                    validate_multihost_bind_mount_permissions(config)

                error_msg = str(exc_info.value)
                # Both paths should be mentioned in the error
                assert "SHARED_STORAGE_ROOT" in error_msg
                assert "CONFIG_PKL_DIR" in error_msg


class TestAutoGeneratedConfigPklDir:
    """Tests for auto-generated config_pkl_dir functionality."""

    def test_generate_config_pkl_dir_format(self):
        """Generated path follows expected structure."""
        from workflows.multihost_orchestrator import _generate_config_pkl_path

        shared_root = "/mnt/shared"
        result = _generate_config_pkl_path(shared_root)

        assert result.startswith(shared_root)
        assert ".tt-inference-server" in result
        assert "session-" in result
        assert result.endswith("/config_pkl")

    def test_generate_config_pkl_dir_unique(self):
        """Each call generates a unique path."""
        from workflows.multihost_orchestrator import _generate_config_pkl_path

        shared_root = "/mnt/shared"
        paths = [_generate_config_pkl_path(shared_root) for _ in range(5)]

        # All paths should be unique
        assert len(set(paths)) == len(paths)

    def test_cleanup_config_pkl_dir_removes_directory(self):
        """Cleanup removes the config_pkl directory."""
        from workflows.multihost_orchestrator import _cleanup_config_pkl_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            config_pkl = Path(tmpdir) / "session-test" / "config_pkl"
            config_pkl.mkdir(parents=True)
            # Create a dummy file inside
            (config_pkl / "test.pkl").write_text("test")

            _cleanup_config_pkl_dir(str(config_pkl))

            assert not config_pkl.exists()
            # Session directory should also be removed (empty)
            assert not config_pkl.parent.exists()

    def test_cleanup_nonexistent_directory_no_error(self):
        """Cleanup handles non-existent directory gracefully."""
        from workflows.multihost_orchestrator import _cleanup_config_pkl_dir

        # Should not raise
        _cleanup_config_pkl_dir("/nonexistent/path/config_pkl")

    def test_create_with_permissions_sets_sticky_bit(self):
        """Created directory has sticky bit (1777) for UID 1000 write access."""
        import stat

        from workflows.multihost_orchestrator import (
            _create_config_pkl_dir_with_permissions,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_pkl = (
                Path(tmpdir) / ".tt-inference-server" / "session-test" / "config_pkl"
            )
            _create_config_pkl_dir_with_permissions(str(config_pkl))

            st = config_pkl.stat()
            # Check sticky bit and world-writable
            assert st.st_mode & stat.S_ISVTX, "Sticky bit should be set"
            assert st.st_mode & stat.S_IWOTH, "World-writable should be set"
            assert st.st_mode & stat.S_IROTH, "World-readable should be set"
            assert st.st_mode & stat.S_IXOTH, "World-executable should be set"

            # Also check session parent directory
            session_st = config_pkl.parent.stat()
            assert session_st.st_mode & stat.S_ISVTX, (
                "Session dir sticky bit should be set"
            )
