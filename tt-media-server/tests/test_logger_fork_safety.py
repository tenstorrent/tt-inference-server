# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Tests for fork-safe logging in utils/logger.py.

Verifies that the os.register_at_fork mechanism in logger.py prevents
deadlocks when child processes inherit logging handler locks held by a
thread that no longer exists after fork.

Scenario without fix:
    Parent bg-thread holds handler.lock  -->  os.fork()
    Child inherits lock (owner=bg-thread-id, count=1) but bg-thread is gone
    Child main thread calls logger.info() --> handler.acquire() --> hangs forever

Fix: os.register_at_fork(after_in_child=_reset_logging_locks_after_fork)
     replaces every handler lock with a fresh, released one before child runs.
"""

import importlib.util
import logging
import os
import signal
import subprocess
import sys
import threading

import pytest

FORK_TIMEOUT_SECONDS = 5
_LOGGER_MODULE_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, "utils", "logger.py"
)

requires_fork = pytest.mark.skipif(
    not hasattr(os, "fork"), reason="os.fork() not available"
)


def _load_real_logger_module():
    """Load utils/logger.py from disk, bypassing the conftest sys.modules mock.

    Each call executes the module top-level code, including the
    os.register_at_fork registration — which is the thing we need to verify.
    """
    spec = importlib.util.spec_from_file_location(
        "utils.logger._real", _LOGGER_MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hold_lock_until_released(lock, held_event, release_event):
    """Acquire *lock* on a background thread and hold until *release_event*."""
    lock.acquire()
    held_event.set()
    release_event.wait()
    lock.release()


class TestResetLoggingLocksFunction:
    """Unit tests for _reset_logging_locks_after_fork from the real module."""

    def test_resets_root_handler_lock(self):
        mod = _load_real_logger_module()
        handler = logging.StreamHandler()
        logging.root.addHandler(handler)
        try:
            handler.lock.acquire()
            mod._reset_logging_locks_after_fork()
            assert handler.lock.acquire(blocking=False)
            handler.lock.release()
        finally:
            logging.root.removeHandler(handler)

    def test_resets_named_logger_handler_lock(self):
        mod = _load_real_logger_module()
        logger = logging.getLogger("test_named_lock_reset")
        handler = logging.StreamHandler()
        logger.addHandler(handler)
        try:
            handler.lock.acquire()
            mod._reset_logging_locks_after_fork()
            assert handler.lock.acquire(blocking=False)
            handler.lock.release()
        finally:
            logger.removeHandler(handler)

    def test_skips_placeholder_loggers(self):
        """PlaceholderLoggers in the manager dict must not cause errors."""
        mod = _load_real_logger_module()
        logging.getLogger("placeholder.parent.child")
        mod._reset_logging_locks_after_fork()


class TestTTLoggerForkSafety:
    """End-to-end: real TTLogger + os.fork() with a handler lock held.

    Loading the real module registers the after_in_child callback via
    os.register_at_fork.  The child does NOT call the reset function
    manually — the OS fork machinery invokes it automatically.
    If someone removes the register_at_fork call, these tests fail.
    """

    @requires_fork
    def test_child_can_log_via_ttlogger_after_fork(self):
        """Create a real TTLogger, lock its handler from bg thread, fork, child logs."""
        mod = _load_real_logger_module()
        tt_logger = mod.TTLogger(name="test_fork_e2e")
        handler = tt_logger.logger.handlers[0]

        read_fd, write_fd = os.pipe()
        held = threading.Event()
        release = threading.Event()
        bg = threading.Thread(
            target=_hold_lock_until_released,
            args=(handler.lock, held, release),
            daemon=True,
        )
        bg.start()
        held.wait()

        try:
            pid = os.fork()
            if pid == 0:
                os.close(read_fd)
                signal.alarm(FORK_TIMEOUT_SECONDS)
                try:
                    tt_logger.info("child log after fork")
                    os.write(write_fd, b"OK")
                except Exception:
                    os.write(write_fd, b"FAIL")
                finally:
                    os.close(write_fd)
                    os._exit(0)

            os.close(write_fd)
            release.set()
            bg.join()
            _, status = os.waitpid(pid, 0)
            result = os.read(read_fd, 16)
            os.close(read_fd)
            assert result == b"OK", (
                f"Child failed to log via TTLogger — likely deadlocked "
                f"(got {result!r}, exit status {status}). "
                f"Verify os.register_at_fork is called in utils/logger.py"
            )
        finally:
            for h in tt_logger.logger.handlers[:]:
                tt_logger.logger.removeHandler(h)


class TestDeadlockWithoutFix:
    """Prove the deadlock is real when the fix is absent.

    Runs in a subprocess so no register_at_fork callback from logger.py
    is present — the child inherits the locked handler and hangs.

    Uses subprocess timeout (not signal.alarm) as the deadlock detector
    because SIGALRM may not interrupt C-level lock acquisition reliably
    across platforms.
    """

    @requires_fork
    def test_child_deadlocks_without_lock_reset(self):
        script = "\n".join(
            [
                "import logging, os, threading",
                "handler = logging.StreamHandler()",
                "logger = logging.getLogger('deadlock_proof')",
                "logger.propagate = False",
                "logger.setLevel(logging.DEBUG)",
                "logger.addHandler(handler)",
                "held = threading.Event()",
                "def hold():",
                "    handler.lock.acquire()",
                "    held.set()",
                "    threading.Event().wait()",
                "t = threading.Thread(target=hold, daemon=True)",
                "t.start()",
                "held.wait()",
                "pid = os.fork()",
                "if pid == 0:",
                "    logger.info('deadlock')",
                "    os._exit(0)",
                "_, status = os.waitpid(pid, 0)",
                "exit(0)",
            ]
        )
        try:
            result = subprocess.run(
                [sys.executable, "-c", script],
                timeout=3,
                capture_output=True,
            )
            pytest.fail(
                f"Expected child to deadlock but subprocess completed "
                f"(returncode={result.returncode})"
            )
        except subprocess.TimeoutExpired:
            pass
