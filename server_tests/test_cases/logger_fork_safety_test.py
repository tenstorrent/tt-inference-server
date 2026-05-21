# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import io
import logging
import os
import signal
import threading

from server_tests.base_test import BaseTest

logger = logging.getLogger(__name__)

FORK_TIMEOUT_SECONDS = 5


def _reset_logging_locks_after_fork():
    """Reset all logging handler locks — mirrors utils/logger.py."""
    for handler in logging.root.handlers:
        handler.createLock()
    for logger_ref in logging.Logger.manager.loggerDict.values():
        if isinstance(logger_ref, logging.Logger):
            for handler in logger_ref.handlers:
                handler.createLock()


def _hold_lock_until_released(lock, held_event, release_event):
    lock.acquire()
    held_event.set()
    release_event.wait()
    lock.release()


class LoggerForkSafetyTest(BaseTest):
    """Verify that forked child processes can log without deadlocking.

    Reproduces the production bug: a background thread holds a logging
    handler lock at fork time. The child inherits the lock in acquired
    state with no thread to release it, causing a deadlock on the first
    log call.

    Uses only stdlib logging — no colorama or utils.logger dependency.
    The fix in utils/logger.py registers an os.register_at_fork callback
    that replaces handler locks with fresh ones. This test validates that
    the mechanism works in the deployment environment.
    """

    async def _run_specific_test_async(self):
        if not hasattr(os, "fork"):
            logger.warning("os.fork() not available, skipping test")
            return {"success": True, "skipped": True, "reason": "fork not available"}

        logger.info("Testing logger fork safety with held lock...")

        test_logger = logging.getLogger("fork_safety_test")
        test_logger.propagate = False
        test_logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(io.StringIO())
        test_logger.addHandler(handler)

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

        logger.info("Handler lock held by background thread, forking...")

        try:
            pid = os.fork()
            if pid == 0:
                os.close(read_fd)
                signal.alarm(FORK_TIMEOUT_SECONDS)
                try:
                    _reset_logging_locks_after_fork()
                    test_logger.info("child logged after fork")
                    os.write(write_fd, b"OK")
                except Exception as exc:
                    os.write(write_fd, f"FAIL:{exc}".encode())
                finally:
                    os.close(write_fd)
                    os._exit(0)

            os.close(write_fd)
            release.set()
            bg.join()

            _, status = os.waitpid(pid, 0)
            result = os.read(read_fd, 1024).decode()
            os.close(read_fd)

            if result == "OK" and os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0:
                logger.info("Child logged without deadlock")
                return {"success": True, "child_result": result}

            killed = os.WIFSIGNALED(status)
            raise Exception(
                f"Child failed to log — likely deadlocked. "
                f"result={result!r}, killed_by_signal={killed}, "
                f"raw_status={status}"
            )
        finally:
            test_logger.removeHandler(handler)
