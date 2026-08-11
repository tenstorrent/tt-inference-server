#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Tests for collecting tt-triage hang reports out of the cache_root volume.

tt-metal writes a triage report into ``<cache_root>/logs`` when it detects a
device hang. ``cache_root`` is outside ``workflow_logs/``, the only tree CI
uploads, so these reports were invisible in CI until they are copied in.
"""

import io
import os
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from workflows.run_docker_server import collect_tt_triage_logs

CONTAINER_CACHE_ROOT = Path("/home/container_app_user/cache_root")
DOCKER_IMAGE = "ghcr.io/tenstorrent/tt-shield/vllm-tt-metal-src-dev:0.19.0-abc123"


@dataclass
class FakeSetupConfig:
    """Minimal stand-in for SetupConfig with only the fields the collector reads."""

    host_model_volume_root: Path = None
    cache_root: Path = CONTAINER_CACHE_ROOT
    docker_volume_name: str = "volume_id_qwen36_blackhole-Qwen3.6-27B"


@dataclass
class FakeModelSpec:
    docker_image: str = DOCKER_IMAGE


def write_report(log_dir: Path, name: str, age_seconds: float = 0.0) -> Path:
    """Write a triage report, backdated by ``age_seconds``."""
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / name
    path.write_text("=== tt-triage ===\n")
    mtime = time.time() - age_seconds
    os.utime(path, (mtime, mtime))
    return path


def tar_bytes(names) -> bytes:
    """A tar stream like the helper container's `tar -cf -` would emit."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name in names:
            payload = b"=== tt-triage ===\n"
            info = tarfile.TarInfo(f"./{name}")
            info.size = len(payload)
            info.mtime = int(time.time())
            tar.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


@pytest.fixture
def bind_mounted_volume(tmp_path):
    """cache_root backed by a host bind mount (``--host-volume``)."""
    return tmp_path / "persistent_volume"


def test_collects_report_from_bind_mounted_cache_root(bind_mounted_volume, tmp_path):
    write_report(bind_mounted_volume / "logs", "tt-triage-20260808-195447.log")
    dest = tmp_path / "workflow_logs" / "tt_triage"

    collected = collect_tt_triage_logs(
        FakeSetupConfig(host_model_volume_root=bind_mounted_volume),
        FakeModelSpec(),
        dest,
    )

    assert collected == 1
    assert [p.name for p in dest.iterdir()] == ["tt-triage-20260808-195447.log"]


def test_drops_reports_left_by_earlier_runs(bind_mounted_volume, tmp_path):
    """The volume is reused across runs, so stale reports must not be attributed here."""
    log_dir = bind_mounted_volume / "logs"
    write_report(log_dir, "tt-triage-20260808-195447.log")
    write_report(log_dir, "tt-triage-20260731-192602.log", age_seconds=86400)
    dest = tmp_path / "workflow_logs" / "tt_triage"

    collected = collect_tt_triage_logs(
        FakeSetupConfig(host_model_volume_root=bind_mounted_volume),
        FakeModelSpec(),
        dest,
        since_ts=time.time() - 60,
    )

    assert collected == 1
    assert [p.name for p in dest.iterdir()] == ["tt-triage-20260808-195447.log"]


def test_no_reports_when_nothing_hung(bind_mounted_volume, tmp_path):
    """No hang means tt-metal never ran triage, so there is no logs/ directory."""
    bind_mounted_volume.mkdir(parents=True)

    assert (
        collect_tt_triage_logs(
            FakeSetupConfig(host_model_volume_root=bind_mounted_volume),
            FakeModelSpec(),
            tmp_path / "workflow_logs" / "tt_triage",
        )
        == 0
    )


def test_reads_named_volume_without_the_server_container(tmp_path, monkeypatch):
    """The server container runs with --rm, so a hang deletes it before collection.

    Regression test: reading the volume must not depend on that container still
    existing, otherwise reports are only ever collected when nothing hung.
    """
    recorded = []

    def fake_run(cmd, **kwargs):
        recorded.append(cmd)

        class Result:
            returncode = 0
            stdout = tar_bytes(["tt-triage-20260808-195447.log"])
            stderr = b""

        return Result()

    monkeypatch.setattr("workflows.run_docker_server.subprocess.run", fake_run)
    dest = tmp_path / "workflow_logs" / "tt_triage"

    collected = collect_tt_triage_logs(FakeSetupConfig(), FakeModelSpec(), dest)

    assert collected == 1
    assert [p.name for p in dest.iterdir()] == ["tt-triage-20260808-195447.log"]
    assert recorded == [
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "tar",
            "--volume",
            "volume_id_qwen36_blackhole-Qwen3.6-27B:/cache_root:ro",
            DOCKER_IMAGE,
            "-cf",
            "-",
            "-C",
            "/cache_root/logs",
            ".",
        ]
    ]


def test_named_volume_without_logs_directory(tmp_path, monkeypatch):
    """tar exits non-zero when logs/ is absent -- the no-hang case."""

    def fake_run(cmd, **kwargs):
        class Result:
            returncode = 2
            stdout = b""
            stderr = b"tar: /cache_root/logs: Cannot open: No such file or directory\n"

        return Result()

    monkeypatch.setattr("workflows.run_docker_server.subprocess.run", fake_run)

    assert (
        collect_tt_triage_logs(
            FakeSetupConfig(), FakeModelSpec(), tmp_path / "tt_triage"
        )
        == 0
    )


def test_collection_failure_never_raises(tmp_path, monkeypatch):
    """A collection problem must not mask the workload's own result."""

    def boom(*args, **kwargs):
        raise OSError("docker daemon unreachable")

    monkeypatch.setattr("workflows.run_docker_server.subprocess.run", boom)

    assert (
        collect_tt_triage_logs(
            FakeSetupConfig(), FakeModelSpec(), tmp_path / "tt_triage"
        )
        == 0
    )
    assert collect_tt_triage_logs(None, FakeModelSpec(), tmp_path / "tt_triage") == 0
