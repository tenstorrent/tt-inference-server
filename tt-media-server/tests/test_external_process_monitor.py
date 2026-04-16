# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import os
import tempfile
import time

import pytest

from utils.external_process_monitor import ExternalProcessMonitor


class MockSettings:
    def __init__(self, log_path, **overrides):
        self.external_process_log_path = log_path
        self.external_process_launch_command = overrides.get(
            "launch_command", "echo 'SHM bridge ready' && sleep 3600"
        )
        self.external_process_launch_cwd = overrides.get("launch_cwd", "")
        self.external_process_env_setup = overrides.get("env_setup", "")
        self.external_process_ready_pattern = overrides.get(
            "ready_pattern", "SHM bridge ready"
        )
        self.external_process_hang_patterns = overrides.get(
            "hang_patterns",
            "TIMEOUT: device timeout in fetch queue wait"
            ";TT_THROW.*Timed out while waiting for active ethernet core",
        )
        self.external_process_recovery_command = overrides.get("recovery_command", "")
        self.external_process_recovery_cwd = overrides.get("recovery_cwd", "")
        self.external_process_recovery_success_pattern = overrides.get(
            "recovery_success_pattern", "All Links Are Healthy"
        )
        self.external_process_check_interval_seconds = overrides.get(
            "check_interval", 0.1
        )
        self.external_process_recovery_cooldown_seconds = overrides.get(
            "cooldown_seconds", 0.5
        )
        self.external_process_max_recovery_attempts = overrides.get(
            "max_recovery_attempts", 3
        )


@pytest.fixture
def tmp_log_path():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        path = f.name
    yield path
    if os.path.exists(path):
        os.unlink(path)


class TestExternalProcessMonitor:
    def test_initial_state_disabled(self, tmp_log_path):
        settings = MockSettings(tmp_log_path)
        monitor = ExternalProcessMonitor(settings)
        status = monitor.get_status()
        assert status["state"] == "disabled"
        assert status["monitoring"] is False
        assert status["hang_detected_count"] == 0
        assert status["recovery_attempts"] == 0

    @pytest.mark.asyncio
    async def test_start_launches_process_and_detects_ready(self, tmp_log_path):
        settings = MockSettings(
            tmp_log_path,
            launch_command="echo 'starting up...' && echo 'SHM bridge ready' && sleep 3600",
            check_interval=0.1,
        )
        monitor = ExternalProcessMonitor(settings)
        await monitor.start()

        try:
            # Wait for ready pattern detection
            for _ in range(50):
                if monitor.get_status()["state"] == "healthy":
                    break
                await asyncio.sleep(0.1)

            status = monitor.get_status()
            assert status["state"] == "healthy"
            assert status["monitoring"] is True
            assert status["pid"] is not None
        finally:
            await monitor.stop()

    @pytest.mark.asyncio
    async def test_hang_detection_fires_callback(self, tmp_log_path):
        hang_events = []

        async def on_hang(pattern):
            hang_events.append(pattern)

        settings = MockSettings(
            tmp_log_path,
            launch_command="echo 'SHM bridge ready' && sleep 3600",
            check_interval=0.1,
        )
        monitor = ExternalProcessMonitor(settings, on_hang_detected=on_hang)
        await monitor.start()

        try:
            # Wait for healthy state
            for _ in range(50):
                if monitor.get_status()["state"] == "healthy":
                    break
                await asyncio.sleep(0.1)
            assert monitor.get_status()["state"] == "healthy"

            # Write hang pattern to log
            with open(tmp_log_path, "a") as f:
                f.write(
                    "TIMEOUT: device timeout in fetch queue wait, potential hang detected\n"
                )

            # Wait for hang detection
            for _ in range(50):
                if hang_events:
                    break
                await asyncio.sleep(0.1)

            assert len(hang_events) == 1
            assert "TIMEOUT" in hang_events[0]
            assert monitor.get_status()["hang_detected_count"] == 1
        finally:
            await monitor.stop()

    @pytest.mark.asyncio
    async def test_cooldown_prevents_rapid_recovery(self, tmp_log_path):
        hang_events = []

        async def on_hang(pattern):
            hang_events.append(pattern)

        settings = MockSettings(
            tmp_log_path,
            launch_command="echo 'SHM bridge ready' && sleep 3600",
            check_interval=0.1,
            cooldown_seconds=5.0,
        )
        monitor = ExternalProcessMonitor(settings, on_hang_detected=on_hang)
        await monitor.start()

        try:
            for _ in range(50):
                if monitor.get_status()["state"] == "healthy":
                    break
                await asyncio.sleep(0.1)

            # Write first hang pattern
            with open(tmp_log_path, "a") as f:
                f.write("TIMEOUT: device timeout in fetch queue wait\n")

            for _ in range(50):
                if hang_events:
                    break
                await asyncio.sleep(0.1)

            assert len(hang_events) == 1

            # Manually reset state to healthy to test cooldown
            monitor._state = "healthy"

            # Write second hang pattern immediately — should be ignored due to cooldown
            with open(tmp_log_path, "a") as f:
                f.write("TIMEOUT: device timeout in fetch queue wait again\n")

            await asyncio.sleep(0.5)
            assert len(hang_events) == 1  # Still 1 — cooldown active
            assert monitor.get_status()["in_cooldown"] is True
        finally:
            await monitor.stop()

    @pytest.mark.asyncio
    async def test_max_recovery_attempts_goes_fatal(self, tmp_log_path):
        settings = MockSettings(
            tmp_log_path,
            launch_command="echo 'SHM bridge ready' && sleep 3600",
            check_interval=0.1,
            cooldown_seconds=0.0,
            max_recovery_attempts=2,
        )
        monitor = ExternalProcessMonitor(settings)

        # Simulate reaching max recovery attempts
        monitor._recovery_attempts = 2
        monitor._state = "healthy"
        monitor._last_hang_time = None

        await monitor._handle_hang_detected("test_pattern")

        assert monitor.get_status()["state"] == "fatal"

    @pytest.mark.asyncio
    async def test_recovery_with_success(self, tmp_log_path):
        settings = MockSettings(
            tmp_log_path,
            launch_command="echo 'SHM bridge ready' && sleep 3600",
            recovery_command="echo 'All Links Are Healthy'",
            check_interval=0.1,
        )
        monitor = ExternalProcessMonitor(settings)
        await monitor.start()

        try:
            for _ in range(50):
                if monitor.get_status()["state"] == "healthy":
                    break
                await asyncio.sleep(0.1)

            success = await monitor.run_recovery()
            assert success is True
            assert monitor.get_status()["state"] == "starting"
            assert monitor.get_status()["recovery_attempts"] == 1
            assert monitor.get_status()["pid"] is not None
        finally:
            await monitor.stop()

    @pytest.mark.asyncio
    async def test_recovery_with_failure(self, tmp_log_path):
        settings = MockSettings(
            tmp_log_path,
            launch_command="echo 'SHM bridge ready' && sleep 3600",
            recovery_command="echo 'Links are broken'",
            check_interval=0.1,
        )
        monitor = ExternalProcessMonitor(settings)
        await monitor.start()

        try:
            for _ in range(50):
                if monitor.get_status()["state"] == "healthy":
                    break
                await asyncio.sleep(0.1)

            success = await monitor.run_recovery()
            assert success is False
            assert monitor.get_status()["state"] == "recovery_failed"
        finally:
            await monitor.stop()

    @pytest.mark.asyncio
    async def test_stop_terminates_process(self, tmp_log_path):
        settings = MockSettings(
            tmp_log_path,
            launch_command="sleep 3600",
            check_interval=0.1,
        )
        monitor = ExternalProcessMonitor(settings)
        await monitor.start()

        pid = monitor.get_status()["pid"]
        assert pid is not None

        await monitor.stop()

        status = monitor.get_status()
        assert status["state"] == "disabled"
        assert status["pid"] is None

    @pytest.mark.asyncio
    async def test_tt_throw_hang_pattern(self, tmp_log_path):
        hang_events = []

        async def on_hang(pattern):
            hang_events.append(pattern)

        settings = MockSettings(
            tmp_log_path,
            launch_command="echo 'SHM bridge ready' && sleep 3600",
            check_interval=0.1,
        )
        monitor = ExternalProcessMonitor(settings, on_hang_detected=on_hang)
        await monitor.start()

        try:
            for _ in range(50):
                if monitor.get_status()["state"] == "healthy":
                    break
                await asyncio.sleep(0.1)

            with open(tmp_log_path, "a") as f:
                f.write(
                    "TT_THROW @ some/file.cpp:123: Timed out while waiting for active ethernet core 0 to become active again\n"
                )

            for _ in range(50):
                if hang_events:
                    break
                await asyncio.sleep(0.1)

            assert len(hang_events) == 1
            assert "TT_THROW" in hang_events[0]
        finally:
            await monitor.stop()

    def test_get_status_shape(self, tmp_log_path):
        settings = MockSettings(tmp_log_path)
        monitor = ExternalProcessMonitor(settings)
        status = monitor.get_status()

        expected_keys = {
            "monitoring",
            "log_path",
            "state",
            "pid",
            "hang_detected_count",
            "recovery_attempts",
            "last_hang_time",
            "last_hang_pattern",
            "in_cooldown",
        }
        assert set(status.keys()) == expected_keys
