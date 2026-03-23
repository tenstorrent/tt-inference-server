#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from utils.cache_monitor import (
    CacheGenerationStatus,
    CacheMonitor,
    DockerVolumeInfo,
    inspect_docker_volume,
)


def _make_tensor_cache_model_spec(**overrides):
    defaults = {
        "impl": SimpleNamespace(impl_id="tt-transformers"),
        "model_name": "TestModel",
        "version": "1.0.0",
        "device_type": SimpleNamespace(name="N150"),
        "subdevice_type": None,
        "uses_tensor_model_cache": True,
        "device_model_spec": SimpleNamespace(tensor_cache_timeout=3600.0),
        "docker_image": "tt-inference-server:test",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_existing_tensor_cache_without_markers_is_not_treated_as_first_run(tmp_path):
    tensor_file = tmp_path / "tensor.bin"
    tensor_file.write_bytes(b"cached-tensor-data")

    monitor = CacheMonitor(cache_dir=tmp_path)

    status = monitor.get_cache_generation_status()

    assert status.is_generating is False
    assert status.is_first_run is False
    assert status.has_existing_cache is True
    assert status.is_stalled is False
    assert status.file_count == 1
    assert status.total_size_bytes == tensor_file.stat().st_size


def test_empty_cache_without_markers_is_observational(tmp_path, monkeypatch):
    fake_time = [1000.0]
    monkeypatch.setattr("utils.cache_monitor.time.time", lambda: fake_time[0])

    monitor = CacheMonitor(cache_dir=tmp_path)
    started_file, completed_file = monitor.get_cache_marker_files()

    initial_status = monitor.get_cache_generation_status()

    assert initial_status.is_first_run is True
    assert initial_status.is_generating is False
    assert initial_status.has_existing_cache is False
    assert initial_status.is_stalled is False
    assert initial_status.file_count == 0
    assert (
        monitor.get_effective_timeout(1200.0, initial_status)
        == monitor.DEFAULT_TENSOR_CACHE_TIMEOUT
    )
    assert started_file.exists() is False
    assert completed_file.exists() is False

    fake_time[0] += 600.0
    later_status = monitor.get_cache_generation_status()

    assert later_status.is_first_run is True
    assert later_status.is_generating is False
    assert later_status.has_existing_cache is False
    assert later_status.is_stalled is False


def test_tensor_cache_stall_detection_trips_after_no_growth_timeout(
    tmp_path, monkeypatch
):
    fake_time = [1000.0]
    monkeypatch.setattr("utils.cache_monitor.time.time", lambda: fake_time[0])
    monkeypatch.setattr(CacheMonitor, "TENSOR_CACHE_NO_CHANGE_TIMEOUT", 180)

    monitor = CacheMonitor(cache_dir=tmp_path)
    tensor_file = tmp_path / "cache.bin"
    tensor_file.write_bytes(b"1234")
    assert monitor.mark_cache_first_run_started() is True

    initial_status = monitor.get_cache_generation_status()
    assert initial_status.is_generating is True
    assert initial_status.is_stalled is False
    assert initial_status.file_count == 1

    fake_time[0] += 100.0
    mid_status = monitor.get_cache_generation_status()
    assert mid_status.is_stalled is False
    assert mid_status.no_progress_duration == pytest.approx(100.0)

    fake_time[0] += 81.0
    stalled_status = monitor.get_cache_generation_status()
    assert stalled_status.is_stalled is True
    assert stalled_status.no_progress_duration == pytest.approx(181.0)

    tensor_file.write_bytes(b"12345")
    recovered_status = monitor.get_cache_generation_status()
    assert recovered_status.is_stalled is False
    assert recovered_status.no_progress_duration == pytest.approx(0.0)
    assert recovered_status.total_size_bytes == 5


def test_effective_timeout_uses_tensor_cache_timeout_for_first_run_and_generation(
    tmp_path,
):
    model_spec = _make_tensor_cache_model_spec(
        impl=None,
        version=None,
        device_type=None,
    )
    monitor = CacheMonitor(model_spec=model_spec, cache_dir=tmp_path)

    first_run_status = CacheGenerationStatus(is_generating=False, is_first_run=True)
    generating_status = CacheGenerationStatus(is_generating=True)
    ready_status = CacheGenerationStatus(is_generating=False)

    assert monitor.get_tensor_cache_timeout() == 3600.0
    assert monitor.get_effective_timeout(1200.0, first_run_status) == 3600.0
    assert monitor.get_effective_timeout(1200.0, generating_status) == 3600.0
    assert monitor.get_effective_timeout(1200.0, ready_status) == 1200.0


@pytest.mark.parametrize("configured_timeout", [None, "invalid", 0, -1])
def test_invalid_tensor_cache_timeout_uses_default(tmp_path, configured_timeout):
    model_spec = _make_tensor_cache_model_spec(
        impl=None,
        version=None,
        device_type=None,
        device_model_spec=SimpleNamespace(tensor_cache_timeout=configured_timeout),
    )

    monitor = CacheMonitor(model_spec=model_spec, cache_dir=tmp_path)

    assert (
        monitor.get_tensor_cache_timeout() == CacheMonitor.DEFAULT_TENSOR_CACHE_TIMEOUT
    )


def test_explicit_cache_dir_takes_precedence_over_tt_cache_path(tmp_path, monkeypatch):
    explicit_cache_dir = tmp_path / "explicit-cache"
    env_cache_dir = tmp_path / "env-cache"
    monkeypatch.setenv("TT_CACHE_PATH", str(env_cache_dir))

    monitor = CacheMonitor(cache_dir=explicit_cache_dir)

    assert monitor.cache_dir == explicit_cache_dir


def test_detect_cache_directory_uses_tt_cache_path_env(tmp_path, monkeypatch):
    model_spec = _make_tensor_cache_model_spec(
        impl=None,
        version=None,
        device_type=None,
    )
    env_cache_dir = tmp_path / "in-container-cache"
    monkeypatch.setenv("TT_CACHE_PATH", str(env_cache_dir))

    monitor = CacheMonitor(model_spec=model_spec)

    assert monitor.cache_dir == env_cache_dir


def test_detect_cache_directory_uses_runtime_host_volume(tmp_path):
    model_spec = _make_tensor_cache_model_spec()
    runtime_config = SimpleNamespace(
        host_volume=str(tmp_path / "persistent_volume"),
        local_server=False,
        device="n150",
    )

    monitor = CacheMonitor(model_spec=model_spec, runtime_config=runtime_config)

    assert monitor.cache_dir == (
        tmp_path
        / "persistent_volume"
        / "volume_id_tt-transformers-TestModel-v1.0.0"
        / "tt_metal_cache"
        / "cache_TestModel"
        / "N150"
    )


def test_detect_cache_directory_uses_docker_volume_mountpoint(tmp_path, monkeypatch):
    model_spec = _make_tensor_cache_model_spec()

    monkeypatch.setattr(
        "utils.cache_monitor.inspect_docker_volume",
        lambda volume_name: DockerVolumeInfo(
            volume_name=volume_name,
            mountpoint=tmp_path,
            is_readable=True,
        ),
    )

    monitor = CacheMonitor(model_spec=model_spec)

    assert monitor.cache_dir == (
        tmp_path / "tt_metal_cache" / "cache_TestModel" / "N150"
    )


def test_inspect_docker_volume_treats_permission_error_as_unreadable(
    tmp_path, monkeypatch
):
    volume_name = "volume_id_tt-transformers-TestModel"
    original_exists = Path.exists

    def fake_run(command, capture_output, text, check, timeout):
        assert command[:4] == ["docker", "volume", "inspect", volume_name]
        return SimpleNamespace(
            returncode=0,
            stdout=f"{tmp_path}\n",
            stderr="",
        )

    def deny_exists(self):
        if self == tmp_path:
            raise PermissionError("permission denied")
        return original_exists(self)

    monkeypatch.setattr("utils.cache_monitor.subprocess.run", fake_run)
    monkeypatch.setattr(Path, "exists", deny_exists)

    volume_info = inspect_docker_volume(volume_name)

    assert volume_info == DockerVolumeInfo(
        volume_name=volume_name,
        mountpoint=tmp_path,
        is_readable=False,
    )


def test_unreadable_docker_volume_uses_cli_fallback(tmp_path, monkeypatch):
    model_spec = _make_tensor_cache_model_spec()

    monkeypatch.setattr(
        "utils.cache_monitor.inspect_docker_volume",
        lambda volume_name: DockerVolumeInfo(
            volume_name=volume_name,
            mountpoint=tmp_path,
            is_readable=False,
        ),
    )

    monitor = CacheMonitor(model_spec=model_spec)

    assert monitor.cache_target.uses_docker_cli() is True
    assert monitor.cache_target.cache_dir is None
    assert (
        monitor.cache_target.docker_volume_name == "volume_id_tt-transformers-TestModel"
    )
    assert monitor.cache_dir == Path("/cache/tt_metal_cache/cache_TestModel/N150")


def test_cache_monitor_uses_cli_fallback_when_docker_mountpoint_probe_is_denied(
    tmp_path, monkeypatch
):
    model_spec = _make_tensor_cache_model_spec()
    volume_name = "volume_id_tt-transformers-TestModel"
    original_exists = Path.exists

    def fake_run(command, capture_output, text, check, timeout):
        assert command[:4] == ["docker", "volume", "inspect", volume_name]
        return SimpleNamespace(
            returncode=0,
            stdout=f"{tmp_path}\n",
            stderr="",
        )

    def deny_exists(self):
        if self == tmp_path:
            raise PermissionError("permission denied")
        return original_exists(self)

    monkeypatch.setattr("utils.cache_monitor.subprocess.run", fake_run)
    monkeypatch.setattr(Path, "exists", deny_exists)

    monitor = CacheMonitor(model_spec=model_spec)

    assert monitor.cache_target.uses_docker_cli() is True
    assert monitor.cache_target.cache_dir is None
    assert monitor.cache_target.docker_volume_name == volume_name
    assert monitor.cache_dir == Path("/cache/tt_metal_cache/cache_TestModel/N150")


def test_docker_cli_snapshot_returns_expected_size_and_count(tmp_path, monkeypatch):
    model_spec = _make_tensor_cache_model_spec()

    monkeypatch.setattr(
        "utils.cache_monitor.inspect_docker_volume",
        lambda volume_name: DockerVolumeInfo(
            volume_name=volume_name,
            mountpoint=tmp_path,
            is_readable=False,
        ),
    )

    def fake_run(command, capture_output, text, check, timeout):
        assert command[:3] == ["docker", "run", "--rm"]
        assert "volume_id_tt-transformers-TestModel:/cache:ro" in command
        assert "tt-inference-server:test" in command
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"total_size_bytes": 1234, "file_count": 7}),
            stderr="",
        )

    monkeypatch.setattr("utils.cache_monitor.subprocess.run", fake_run)

    monitor = CacheMonitor(model_spec=model_spec)

    assert monitor._get_tensor_cache_snapshot() == (1234, 7)


def test_docker_cli_snapshot_failure_returns_empty_snapshot(tmp_path, monkeypatch):
    model_spec = _make_tensor_cache_model_spec()

    monkeypatch.setattr(
        "utils.cache_monitor.inspect_docker_volume",
        lambda volume_name: DockerVolumeInfo(
            volume_name=volume_name,
            mountpoint=tmp_path,
            is_readable=False,
        ),
    )
    monkeypatch.setattr(
        "utils.cache_monitor.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="helper image missing",
        ),
    )

    monitor = CacheMonitor(model_spec=model_spec)

    assert monitor._get_tensor_cache_snapshot() == (0, 0)


def test_docker_cli_snapshot_invalid_json_returns_empty_snapshot(tmp_path, monkeypatch):
    model_spec = _make_tensor_cache_model_spec()

    monkeypatch.setattr(
        "utils.cache_monitor.inspect_docker_volume",
        lambda volume_name: DockerVolumeInfo(
            volume_name=volume_name,
            mountpoint=tmp_path,
            is_readable=False,
        ),
    )
    monkeypatch.setattr(
        "utils.cache_monitor.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="not-json",
            stderr="",
        ),
    )

    monitor = CacheMonitor(model_spec=model_spec)

    assert monitor._get_tensor_cache_snapshot() == (0, 0)


def test_docker_cli_fallback_stall_detection_trips_after_no_growth_timeout(
    tmp_path, monkeypatch
):
    fake_time = [1000.0]
    monkeypatch.setattr("utils.cache_monitor.time.time", lambda: fake_time[0])
    monkeypatch.setattr(CacheMonitor, "TENSOR_CACHE_NO_CHANGE_TIMEOUT", 180)

    model_spec = _make_tensor_cache_model_spec()
    monkeypatch.setattr(
        "utils.cache_monitor.inspect_docker_volume",
        lambda volume_name: DockerVolumeInfo(
            volume_name=volume_name,
            mountpoint=tmp_path,
            is_readable=False,
        ),
    )

    monitor = CacheMonitor(model_spec=model_spec)
    snapshots = iter([(0, 0), (4, 1), (4, 1), (4, 1), (4, 1)])
    monkeypatch.setattr(monitor, "_get_docker_cache_snapshot", lambda: next(snapshots))

    initial_status = monitor.get_cache_generation_status()
    assert initial_status.is_first_run is True
    assert initial_status.is_generating is False
    assert initial_status.is_stalled is False
    assert initial_status.file_count == 0

    fake_time[0] += 1.0
    generating_status = monitor.get_cache_generation_status()
    assert generating_status.is_first_run is False
    assert generating_status.is_generating is True
    assert generating_status.file_count == 1
    assert generating_status.is_stalled is False

    fake_time[0] += 100.0
    mid_status = monitor.get_cache_generation_status()
    assert mid_status.is_generating is True
    assert mid_status.file_count == 1
    assert mid_status.is_stalled is False

    fake_time[0] += 81.0
    stalled_status = monitor.get_cache_generation_status()
    assert stalled_status.is_generating is True
    assert stalled_status.is_stalled is True
    assert stalled_status.no_progress_duration == pytest.approx(181.0)

    assert monitor.mark_cache_completed() is True
    completed_status = monitor.get_cache_generation_status()
    assert completed_status.is_generating is False


def test_completed_marker_suppresses_first_run(tmp_path):
    monitor = CacheMonitor(cache_dir=tmp_path)

    assert monitor.mark_cache_completed() is True

    status = monitor.get_cache_generation_status()

    assert status.is_first_run is False
    assert status.is_generating is False
    assert status.has_existing_cache is True
    assert status.is_stalled is False


def test_monitoring_is_disabled_when_tensor_cache_is_unused(tmp_path):
    model_spec = _make_tensor_cache_model_spec(uses_tensor_model_cache=False)

    monitor = CacheMonitor(model_spec=model_spec, cache_dir=tmp_path)
    status = monitor.get_cache_generation_status()

    assert monitor.cache_dir is None
    assert status.is_generating is False
    assert status.cache_dir is None
