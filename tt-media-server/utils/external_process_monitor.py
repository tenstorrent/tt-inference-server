# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import os
import re
import signal
import subprocess
import time

from utils.logger import TTLogger


class ExternalProcessMonitor:
    def __init__(self, settings, on_hang_detected=None):
        self.settings = settings
        self.on_hang_detected = on_hang_detected
        self.logger = TTLogger()

        self.log_path = settings.external_process_log_path
        self.launch_command = settings.external_process_launch_command
        self.launch_cwd = settings.external_process_launch_cwd or None
        self.env_setup = settings.external_process_env_setup
        self.ready_pattern = re.compile(settings.external_process_ready_pattern)
        self.hang_patterns = [
            re.compile(p)
            for p in settings.external_process_hang_patterns.split(";")
            if p.strip()
        ]
        self.recovery_command = settings.external_process_recovery_command
        self.recovery_cwd = (
            settings.external_process_recovery_cwd
            or settings.external_process_launch_cwd
            or None
        )
        self.recovery_success_pattern = re.compile(
            settings.external_process_recovery_success_pattern
        )
        self.check_interval = settings.external_process_check_interval_seconds
        self.cooldown_seconds = settings.external_process_recovery_cooldown_seconds
        self.max_recovery_attempts = settings.external_process_max_recovery_attempts

        self._process = None
        self._log_file = None
        self._file_offset = 0
        self._monitor_task = None
        self._state = "disabled"
        self._hang_detected_count = 0
        self._recovery_attempts = 0
        self._last_hang_time = None
        self._last_hang_pattern = None

    async def start(self):
        self._state = "starting"
        self._file_offset = 0
        await asyncio.to_thread(self._launch_process)
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        self.logger.info(
            f"External process monitor started, PID={self._process.pid if self._process else None}"
        )

    async def stop(self):
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        await asyncio.to_thread(self._terminate_process)
        self._state = "disabled"
        self.logger.info("External process monitor stopped")

    def get_status(self) -> dict:
        now = time.time()
        in_cooldown = (
            self._last_hang_time is not None
            and (now - self._last_hang_time) < self.cooldown_seconds
        )
        return {
            "monitoring": self._state != "disabled",
            "log_path": self.log_path,
            "state": self._state,
            "pid": self._process.pid if self._process and self._process.poll() is None else None,
            "hang_detected_count": self._hang_detected_count,
            "recovery_attempts": self._recovery_attempts,
            "last_hang_time": self._last_hang_time,
            "last_hang_pattern": self._last_hang_pattern,
            "in_cooldown": in_cooldown,
        }

    def _build_shell_command(self, command):
        if self.env_setup:
            return f"{self.env_setup} && {command}"
        return command

    def _launch_process(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._log_file = open(self.log_path, "a+")
        shell_cmd = self._build_shell_command(self.launch_command)
        self.logger.info(f"Launching external process: {shell_cmd}")
        self._process = subprocess.Popen(
            ["bash", "-c", shell_cmd],
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            cwd=self.launch_cwd,
            preexec_fn=os.setsid,
        )
        # Start reading from end of file at launch time
        self._file_offset = os.path.getsize(self.log_path)
        self.logger.info(f"External process launched with PID {self._process.pid}")

    def _terminate_process(self):
        if self._process is None:
            return

        pid = self._process.pid
        try:
            if self._process.poll() is None:
                # Kill the entire process group
                os.killpg(os.getpgid(pid), signal.SIGTERM)
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.logger.warning(
                        f"External process {pid} did not exit after SIGTERM, sending SIGKILL"
                    )
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                    self._process.wait(timeout=5)
        except ProcessLookupError:
            pass
        except Exception as e:
            self.logger.error(f"Error terminating external process {pid}: {e}")
        finally:
            self._process = None
            if self._log_file:
                self._log_file.close()
                self._log_file = None

    def _read_new_log_content(self) -> str:
        try:
            file_size = os.path.getsize(self.log_path)
            # Handle log file truncation (e.g. after restart)
            if file_size < self._file_offset:
                self.logger.info("Log file truncated, resetting offset")
                self._file_offset = 0

            if file_size == self._file_offset:
                return ""

            with open(self.log_path, "r") as f:
                f.seek(self._file_offset)
                content = f.read()
                self._file_offset = f.tell()
                return content
        except FileNotFoundError:
            return ""
        except Exception as e:
            self.logger.error(f"Error reading log file: {e}")
            return ""

    async def _monitor_loop(self):
        try:
            while True:
                content = await asyncio.to_thread(self._read_new_log_content)

                if content:
                    if self._state == "starting":
                        if self.ready_pattern.search(content):
                            self._state = "healthy"
                            self.logger.info(
                                "External process ready pattern detected, state -> healthy"
                            )

                    if self._state == "healthy":
                        for pattern in self.hang_patterns:
                            match = pattern.search(content)
                            if match:
                                await self._handle_hang_detected(match.group())
                                break

                # Check if process exited unexpectedly
                if self._process and self._process.poll() is not None:
                    exit_code = self._process.returncode
                    if self._state in ("starting", "healthy"):
                        self.logger.error(
                            f"External process exited unexpectedly with code {exit_code}"
                        )
                        await self._handle_hang_detected(
                            f"process_exited_code_{exit_code}"
                        )

                await asyncio.sleep(self.check_interval)
        except asyncio.CancelledError:
            return

    async def _handle_hang_detected(self, pattern_matched: str):
        now = time.time()

        # Check cooldown
        if self._last_hang_time and (now - self._last_hang_time) < self.cooldown_seconds:
            self.logger.info(
                f"Hang pattern matched but in cooldown, ignoring: {pattern_matched}"
            )
            return

        self._hang_detected_count += 1
        self._last_hang_time = now
        self._last_hang_pattern = pattern_matched
        self._state = "hang_detected"

        self.logger.warning(
            f"Hang detected (count={self._hang_detected_count}): {pattern_matched}"
        )

        if self._recovery_attempts >= self.max_recovery_attempts:
            self._state = "fatal"
            self.logger.error(
                f"Max recovery attempts ({self.max_recovery_attempts}) reached, state -> fatal"
            )
            return

        if self.on_hang_detected:
            await self.on_hang_detected(pattern_matched)

    async def run_recovery(self) -> bool:
        self._state = "recovering"
        self._recovery_attempts += 1
        self.logger.info(
            f"Running recovery (attempt {self._recovery_attempts}/{self.max_recovery_attempts})"
        )

        # Step 1: Kill the subprocess
        await asyncio.to_thread(self._terminate_process)
        self.logger.info("External process terminated for recovery")

        # Step 2: Run recovery command
        if self.recovery_command:
            recovery_success = await self._run_recovery_command()
            if not recovery_success:
                self._state = "recovery_failed"
                self.logger.error("Recovery command failed, state -> recovery_failed")
                return False

        # Step 3: Restart the process
        self._file_offset = 0
        self._state = "starting"
        await asyncio.to_thread(self._launch_process)
        self.logger.info(
            f"External process re-launched with PID {self._process.pid}, waiting for ready"
        )
        return True

    async def _run_recovery_command(self) -> bool:
        shell_cmd = self._build_shell_command(self.recovery_command)
        self.logger.info(f"Running recovery command: {shell_cmd}")

        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["bash", "-c", shell_cmd],
                capture_output=True,
                text=True,
                cwd=self.recovery_cwd,
                timeout=300,
            )
            stdout = result.stdout
            self.logger.info(f"Recovery command stdout:\n{stdout}")
            if result.stderr:
                self.logger.warning(f"Recovery command stderr:\n{result.stderr}")

            if self.recovery_success_pattern.search(stdout):
                self.logger.info("Recovery success pattern found in output")
                return True
            else:
                self.logger.error(
                    f"Recovery success pattern not found. Expected: {self.recovery_success_pattern.pattern}"
                )
                return False
        except subprocess.TimeoutExpired:
            self.logger.error("Recovery command timed out after 300s")
            return False
        except Exception as e:
            self.logger.error(f"Recovery command failed: {e}")
            return False
