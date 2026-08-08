#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Tests for collecting tt-triage hang reports out of the cache_root volume.

tt-metal writes a triage report into ``<cache_root>/logs`` when it detects a
device hang. ``cache_root`` is outside ``workflow_logs/``, the only tree CI
uploads, so these reports were invisible in CI until they are copied in.
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from workflows.run_docker_server import collect_tt_triage_logs

CONTAINER_CACHE_ROOT = Path("/home/container_app_user/cache_root")


@dataclass
class FakeSetupConfig:
    """Minimal stand-in for SetupConfig with only the fields the collector reads."""

    host_model_volume_root: Path = None
    cache_root: Path = CONTAINER_CACHE_ROOT
    docker_volume_name: str = "volume_id_qwen36_blackhole-Qwen3.6-27B"


def write_report(log_dir: Path, name: str, age_seconds: float = 0.0) -> Path:
    """Write a triage report, backdated by ``age_seconds``."""
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / name
    path.write_text("=== tt-triage ===\n")
    mtime = time.time() - age_seconds
    os.utime(path, (mtime, mtime))
    return path


@pytest.fixture
def bind_mounted_volume(tmp_path):
    """cache_root backed by a host bind mount (``--host-volume``)."""
    return tmp_path / "persistent_volume"


def test_collects_report_from_bind_mounted_cache_root(bind_mounted_volume, tmp_path):
    write_report(bind_mounted_volume / "logs", "tt-triage-20260807-212424.log")
    dest = tmp_path / "workflow_logs" / "tt_triage"

    collected = collect_tt_triage_logs(
        FakeSetupConfig(host_model_volume_root=bind_mounted_volume), dest
    )

    assert collected == 1
    assert [p.name for p in dest.iterdir()] == ["tt-triage-20260807-212424.log"]


def test_drops_reports_left_by_earlier_runs(bind_mounted_volume, tmp_path):
    """The volume is reused across runs, so stale reports must not be attributed here."""
    log_dir = bind_mounted_volume / "logs"
    write_report(log_dir, "tt-triage-20260807-212424.log")
    write_report(log_dir, "tt-triage-20260731-192602.log", age_seconds=86400)
    dest = tmp_path / "workflow_logs" / "tt_triage"

    collected = collect_tt_triage_logs(
        FakeSetupConfig(host_model_volume_root=bind_mounted_volume),
        dest,
        since_ts=time.time() - 60,
    )

    assert collected == 1
    assert [p.name for p in dest.iterdir()] == ["tt-triage-20260807-212424.log"]


def test_no_reports_when_nothing_hung(bind_mounted_volume, tmp_path):
    """No hang means tt-metal never ran triage, so there is no logs/ directory."""
    bind_mounted_volume.mkdir(parents=True)

    assert (
        collect_tt_triage_logs(
            FakeSetupConfig(host_model_volume_root=bind_mounted_volume),
            tmp_path / "workflow_logs" / "tt_triage",
        )
        == 0
    )


def test_named_volume_copies_from_container(tmp_path, monkeypatch):
    """Without a bind mount, cache_root is a named volume reachable via the container."""
    recorded = []

    def fake_run(cmd, **kwargs):
        recorded.append(cmd)

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr("workflows.run_docker_server.subprocess.run", fake_run)
    dest = tmp_path / "workflow_logs" / "tt_triage"

    collect_tt_triage_logs(
        FakeSetupConfig(), dest, container_name="tt-inference-server-b6acf2f9"
    )

    assert recorded == [
        [
            "docker",
            "cp",
            f"tt-inference-server-b6acf2f9:{CONTAINER_CACHE_ROOT}/logs/.",
            str(dest),
        ]
    ]


def test_collection_failure_never_raises(tmp_path, monkeypatch):
    """A collection problem must not mask the workload's own result."""

    def boom(*args, **kwargs):
        raise OSError("docker daemon unreachable")

    monkeypatch.setattr("workflows.run_docker_server.subprocess.run", boom)

    assert collect_tt_triage_logs(FakeSetupConfig(), tmp_path / "tt_triage") == 0
    assert collect_tt_triage_logs(None, tmp_path / "tt_triage") == 0
