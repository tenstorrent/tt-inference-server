# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import importlib.util
import logging
import os
import signal
import threading
from pathlib import Path

from tests.server_tests.base_test import BaseTest

logger = logging.getLogger(__name__)


def _find_repo_root() -> Path:
    """Find repository root by locating .git directory."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent
    # Fallback to parents[3] for compatibility
    return Path(__file__).resolve().parents[3]


def _load_real_logger_module():
    """Load utils/logger.py from disk to trigger os.register_at_fork registration."""
    logger_path = _find_repo_root() / "tt-media-server" / "utils" / "logger.py"
    spec = importlib.util.spec_from_file_location(
        "utils.logger._real", str(logger_path)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LoggerForkSafetyTest(BaseTest):
    """Test fork-safety of logging in utils/logger.py.

    Validates that os.register_at_fork in utils/logger.py prevents deadlocks
    when child processes inherit logging handler locks in acquired state.

    This test simulates the production bug scenario:
    1. Create a TTLogger instance
    2. Hold its handler lock from a background thread (simulates logging activity)
    3. Fork while the lock is held
    4. Child process attempts to log (would deadlock without the fix)
    5. Verify child successfully logs and exits (fix is working)
    """

    async def _run_specific_test_async(self):
        if not hasattr(os, "fork"):
            logger.warning("os.fork() not available, skipping test")
            return {"success": True, "skipped": True, "reason": "fork not available"}

        logger.info("Testing logger fork safety with held lock...")

        # Load the real logger module (triggers os.register_at_fork)
        mod = _load_real_logger_module()
        tt_logger = mod.TTLogger(name="fork_safety_test")
        handler = tt_logger.logger.handlers[0]

        # Create pipe for child to report status
        read_fd, write_fd = os.pipe()

        # Background thread will hold the lock
        held_event = threading.Event()
        release_event = threading.Event()

        def hold_lock_until_released():
            handler.lock.acquire()
            held_event.set()
            release_event.wait()
            handler.lock.release()

        bg_thread = threading.Thread(target=hold_lock_until_released, daemon=True)
        bg_thread.start()
        held_event.wait()  # Wait until lock is definitely held

        logger.info("Lock is held by background thread, forking now...")

        try:
            pid = os.fork()
            if pid == 0:  # Child process
                os.close(read_fd)
                # Set alarm to kill us if we deadlock
                signal.alarm(5)
                try:
                    # This would deadlock without the os.register_at_fork fix
                    tt_logger.info("Child successfully logged after fork!")
                    os.write(write_fd, b"OK")
                except Exception as e:
                    os.write(write_fd, f"FAIL:{e}".encode())
                finally:
                    os.close(write_fd)
                    os._exit(0)
            else:  # Parent process
                os.close(write_fd)
                release_event.set()  # Release the lock in parent
                bg_thread.join()

                # Wait for child to finish
                _, status = os.waitpid(pid, 0)
                result = os.read(read_fd, 1024).decode()
                os.close(read_fd)

                if (
                    result == "OK"
                    and os.WIFEXITED(status)
                    and os.WEXITSTATUS(status) == 0
                ):
                    logger.info("✓ Child successfully logged without deadlock")
                    return {"success": True, "child_result": result}
                else:
                    raise Exception(
                        f"Child failed to log (likely deadlocked). "
                        f"Result: {result!r}, Exit status: {status}. "
                        f"Verify os.register_at_fork is called in utils/logger.py"
                    )
        finally:
            # Cleanup handlers
            for h in tt_logger.logger.handlers[:]:
                tt_logger.logger.removeHandler(h)
