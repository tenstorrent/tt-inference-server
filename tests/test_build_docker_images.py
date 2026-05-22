# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import textwrap
from unittest.mock import patch, MagicMock

import pytest

from scripts.build_docker_images import (
    get_available_memory_gb,
    get_available_disk_gb,
    get_docker_root_dir,
    get_max_concurrent_builds,
    check_resources_for_new_build,
    log_resource_summary,
    MEMORY_PER_BUILD_GB,
    MEMORY_RESERVE_GB,
    DISK_PER_BUILD_GB,
    DISK_RESERVE_GB,
)


class TestGetAvailableMemoryGb:
    def test_reads_memavailable_from_proc(self, tmp_path):
        meminfo = textwrap.dedent("""\
            MemTotal:       65808552 kB
            MemFree:         3241232 kB
            MemAvailable:   52428800 kB
            Buffers:          123456 kB
        """)
        meminfo_file = tmp_path / "meminfo"
        meminfo_file.write_text(meminfo)

        real_open = open

        def mock_open(path, *a, **kw):
            if path == "/proc/meminfo":
                return real_open(meminfo_file, *a, **kw)
            return real_open(path, *a, **kw)

        with patch("builtins.open", side_effect=mock_open):
            result = get_available_memory_gb()

        expected = 52428800 / (1024 * 1024)
        assert abs(result - expected) < 0.01

    def test_falls_back_to_sysconf(self):
        with patch("builtins.open", side_effect=FileNotFoundError), patch(
            "os.sysconf"
        ) as mock_sysconf:
            page_size = 4096
            avail_pages = 4 * 1024 * 1024  # 16 GB worth of pages

            def sysconf_side_effect(name):
                if name == "SC_PAGE_SIZE":
                    return page_size
                if name == "SC_AVPHYS_PAGES":
                    return avail_pages
                raise ValueError(f"Unknown: {name}")

            mock_sysconf.side_effect = sysconf_side_effect
            result = get_available_memory_gb()
            expected = (page_size * avail_pages) / (1024**3)
            assert abs(result - expected) < 0.01

    def test_raises_when_nothing_works(self):
        with patch("builtins.open", side_effect=FileNotFoundError), patch(
            "os.sysconf", side_effect=OSError
        ):
            with pytest.raises(
                RuntimeError, match="Could not determine available memory"
            ):
                get_available_memory_gb()


class TestGetDockerRootDir:
    def test_returns_docker_info_output(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "/data/docker\n"

        with patch("subprocess.run", return_value=mock_result):
            assert get_docker_root_dir() == "/data/docker"

    def test_falls_back_on_failure(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            assert get_docker_root_dir() == "/var/lib/docker"

    def test_falls_back_on_timeout(self):
        import subprocess

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 10)):
            assert get_docker_root_dir() == "/var/lib/docker"


class TestGetAvailableDiskGb:
    def test_returns_free_disk_for_path(self):
        mock_usage = MagicMock()
        mock_usage.free = 500 * (1024**3)

        with patch("shutil.disk_usage", return_value=mock_usage):
            result = get_available_disk_gb("/some/path")
            assert abs(result - 500.0) < 0.01

    def test_falls_back_to_root_on_oserror(self):
        mock_usage = MagicMock()
        mock_usage.free = 200 * (1024**3)

        def disk_usage_side_effect(path):
            if path == "/nonexistent":
                raise OSError("No such path")
            return mock_usage

        with patch("shutil.disk_usage", side_effect=disk_usage_side_effect), patch(
            "scripts.build_docker_images.get_docker_root_dir",
            return_value="/nonexistent",
        ):
            result = get_available_disk_gb("/nonexistent")
            assert abs(result - 200.0) < 0.01


class TestGetMaxConcurrentBuilds:
    def _patch_resources(
        self, mem_gb, disk_gb, cpu_cores, docker_root="/var/lib/docker"
    ):
        return [
            patch(
                "scripts.build_docker_images.get_available_memory_gb",
                return_value=mem_gb,
            ),
            patch(
                "scripts.build_docker_images.get_available_disk_gb",
                return_value=disk_gb,
            ),
            patch(
                "scripts.build_docker_images.get_physical_cpu_count",
                return_value=cpu_cores,
            ),
            patch(
                "scripts.build_docker_images.get_docker_root_dir",
                return_value=docker_root,
            ),
        ]

    def test_memory_is_binding_constraint(self):
        # 32 GB mem - 16 reserve = 16 usable => 1, 500 GB disk => 11, 16 cores => 4 -> limit = 1
        patches = self._patch_resources(mem_gb=32, disk_gb=500, cpu_cores=16)
        for p in patches:
            p.start()
        try:
            limit, details = get_max_concurrent_builds()
            assert limit == 1
            assert details["binding_constraint"] == "memory"
            assert details["max_by_memory"] == 1
        finally:
            for p in patches:
                p.stop()

    def test_disk_is_binding_constraint(self):
        # 128 GB mem => 8, 70 GB disk => (70-20)/40 = 1, 32 cores => 8 -> limit = 1
        patches = self._patch_resources(mem_gb=128, disk_gb=70, cpu_cores=32)
        for p in patches:
            p.start()
        try:
            limit, details = get_max_concurrent_builds()
            assert limit == 1
            assert details["binding_constraint"] == "disk"
        finally:
            for p in patches:
                p.stop()

    def test_cpu_is_binding_constraint(self):
        # 256 GB mem => 16, 1000 GB disk => 24, 4 cores => 1 -> limit = 1
        patches = self._patch_resources(mem_gb=256, disk_gb=1000, cpu_cores=4)
        for p in patches:
            p.start()
        try:
            limit, details = get_max_concurrent_builds()
            assert limit == 1
            assert details["binding_constraint"] == "cpu"
        finally:
            for p in patches:
                p.stop()

    def test_max_workers_caps_resource_limit(self):
        # Resources allow 3 (64-16=48 GB mem / 16 = 3), but max_workers=2
        patches = self._patch_resources(mem_gb=64, disk_gb=500, cpu_cores=16)
        for p in patches:
            p.start()
        try:
            limit, details = get_max_concurrent_builds(max_workers=2)
            assert limit == 2
            assert details["max_workers_override"] == 2
        finally:
            for p in patches:
                p.stop()

    def test_max_workers_does_not_raise_above_resource_limit(self):
        # Resources allow 1 (32-16=16 GB / 16 = 1), max_workers=10 => still 1
        patches = self._patch_resources(mem_gb=32, disk_gb=500, cpu_cores=16)
        for p in patches:
            p.start()
        try:
            limit, _ = get_max_concurrent_builds(max_workers=10)
            assert limit == 1
        finally:
            for p in patches:
                p.stop()

    def test_minimum_is_one(self):
        # Very low resources, but should still allow at least 1
        patches = self._patch_resources(mem_gb=4, disk_gb=30, cpu_cores=2)
        for p in patches:
            p.start()
        try:
            limit, _ = get_max_concurrent_builds()
            assert limit == 1
        finally:
            for p in patches:
                p.stop()

    def test_custom_per_build_values(self):
        # 64 GB mem - 16 reserve = 48 usable, with 32 GB per build => 1
        patches = self._patch_resources(mem_gb=64, disk_gb=500, cpu_cores=16)
        for p in patches:
            p.start()
        try:
            limit, details = get_max_concurrent_builds(
                memory_per_build_gb=32, disk_per_build_gb=20
            )
            assert limit == 1
            assert details["memory_per_build_gb"] == 32
            assert details["disk_per_build_gb"] == 20
        finally:
            for p in patches:
                p.stop()


class TestCheckResourcesForNewBuild:
    def test_sufficient_resources(self):
        with patch(
            "scripts.build_docker_images.get_available_memory_gb", return_value=32.0
        ), patch(
            "scripts.build_docker_images.get_available_disk_gb", return_value=200.0
        ):
            ok, mem, disk = check_resources_for_new_build()
            assert ok is True
            assert mem == 32.0
            assert disk == 200.0

    def test_insufficient_memory(self):
        with patch(
            "scripts.build_docker_images.get_available_memory_gb", return_value=8.0
        ), patch(
            "scripts.build_docker_images.get_available_disk_gb", return_value=200.0
        ):
            ok, mem, disk = check_resources_for_new_build()
            assert ok is False

    def test_insufficient_disk(self):
        with patch(
            "scripts.build_docker_images.get_available_memory_gb", return_value=64.0
        ), patch(
            "scripts.build_docker_images.get_available_disk_gb", return_value=50.0
        ):
            # 50 - 20 reserve = 30, need 40 => not ok
            ok, mem, disk = check_resources_for_new_build()
            assert ok is False


class TestLogResourceSummary:
    def test_logs_without_error(self, caplog):
        details = {
            "available_memory_gb": 64.0,
            "available_disk_gb": 500.0,
            "docker_root": "/var/lib/docker",
            "physical_cpu_count": 16,
            "memory_per_build_gb": MEMORY_PER_BUILD_GB,
            "memory_reserve_gb": MEMORY_RESERVE_GB,
            "disk_per_build_gb": DISK_PER_BUILD_GB,
            "disk_reserve_gb": DISK_RESERVE_GB,
            "max_by_memory": 3,
            "max_by_disk": 12,
            "max_by_cpu": 4,
            "resource_limit": 3,
            "max_workers_override": None,
            "effective_limit": 3,
            "binding_constraint": "memory",
        }
        import logging

        with caplog.at_level(logging.INFO):
            log_resource_summary(details, total_builds=4)

        assert "BUILD RESOURCE SUMMARY" in caplog.text
        assert "Available memory: 64.0 GB" in caplog.text
        assert f"Memory reserve: {MEMORY_RESERVE_GB} GB" in caplog.text
        assert "Total builds queued: 4" in caplog.text

    def test_logs_throttling_warning(self, caplog):
        details = {
            "available_memory_gb": 32.0,
            "available_disk_gb": 500.0,
            "docker_root": "/var/lib/docker",
            "physical_cpu_count": 16,
            "memory_per_build_gb": MEMORY_PER_BUILD_GB,
            "memory_reserve_gb": MEMORY_RESERVE_GB,
            "disk_per_build_gb": DISK_PER_BUILD_GB,
            "disk_reserve_gb": DISK_RESERVE_GB,
            "max_by_memory": 1,
            "max_by_disk": 12,
            "max_by_cpu": 4,
            "resource_limit": 1,
            "max_workers_override": None,
            "effective_limit": 1,
            "binding_constraint": "memory",
        }
        import logging

        with caplog.at_level(logging.WARNING):
            log_resource_summary(details, total_builds=5)

        assert "Throttling active" in caplog.text

    def test_logs_critical_memory_error(self, caplog):
        details = {
            "available_memory_gb": 8.0,
            "available_disk_gb": 500.0,
            "docker_root": "/var/lib/docker",
            "physical_cpu_count": 16,
            "memory_per_build_gb": MEMORY_PER_BUILD_GB,
            "memory_reserve_gb": MEMORY_RESERVE_GB,
            "disk_per_build_gb": DISK_PER_BUILD_GB,
            "disk_reserve_gb": DISK_RESERVE_GB,
            "max_by_memory": 0,
            "max_by_disk": 12,
            "max_by_cpu": 4,
            "resource_limit": 1,
            "max_workers_override": None,
            "effective_limit": 1,
            "binding_constraint": "memory",
        }
        import logging

        with caplog.at_level(logging.ERROR):
            log_resource_summary(details, total_builds=1)

        assert "Available memory after reserve" in caplog.text
        assert "below the minimum per-build requirement" in caplog.text
