#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from types import SimpleNamespace

import pytest

from utils.cache_monitor import CacheGenerationStatus, CacheMonitor


def test_existing_tensor_cache_without_markers_is_not_treated_as_first_run(tmp_path):
    tensor_file = tmp_path / "tensor.bin"
    tensor_file.write_bytes(b"cached-tensor-data")

    monitor = CacheMonitor(cache_dir=tmp_path)

    status = monitor.get_cache_generation_status()

    assert status.is_generating is False
    assert status.is_first_run is False
    assert status.is_stalled is False
    assert status.file_count == 1
    assert status.total_size_bytes == tensor_file.stat().st_size


def test_tensor_cache_stall_detection_trips_after_no_growth_timeout(
    tmp_path, monkeypatch
):
    fake_time = [1000.0]
    monkeypatch.setattr("utils.cache_monitor.time.time", lambda: fake_time[0])

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


def test_effective_timeout_uses_tensor_cache_timeout_only_during_generation(tmp_path):
    model_spec = SimpleNamespace(
        model_name="TestModel",
        uses_tensor_model_cache=True,
        device_model_spec=SimpleNamespace(tensor_cache_timeout=3600.0),
    )
    monitor = CacheMonitor(model_spec=model_spec, cache_dir=tmp_path)

    generating_status = CacheGenerationStatus(is_generating=True)
    ready_status = CacheGenerationStatus(is_generating=False)

    assert monitor.get_tensor_cache_timeout() == 3600.0
    assert monitor.get_effective_timeout(1200.0, generating_status) == 3600.0
    assert monitor.get_effective_timeout(1200.0, ready_status) == 1200.0
