# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import logging
import os
import signal
import sys
import threading

from tests.server_tests.base_test import BaseTest

logger = logging.getLogger(__name__)


def _get_ttlogger_class():
    """Get TTLogger class from already-imported utils.logger module.

    The server has already imported utils.logger, which registered the
    os.register_at_fork callback. We just need to get the TTLogger class.
    """
    # Import from the server's module path
    if "utils.logger" in sys.modules:
        return sys.modules["utils.logger"].TTLogger

    # Fallback: import it (will work if dependencies are installed)
    from utils.logger import TTLogger

    return TTLogger


class LoggerForkSafetyTest(BaseTest):
    """Test fork-safety of logging in utils/logger.py.

    Validates that os.register_at_fork in utils/logger.py prevents deadlocks
    when child processes inherit logging handler locks in acquired state.

    This test simulates the production bug scenario:
    1. Get TTLogger class from already-loaded module (server imports it at startup)
    2. Create a TTLogger instance
    3. Hold its handler lock from a background thread (simulates logging activity)
    4. Fork while the lock is held
    5. Child process attempts to log (would deadlock without the fix)
    6. Verify child successfully logs and exits (fix is working)

    Note: This test uses the server's already-imported logger module, so
    os.register_at_fork was already called at server startup.
    """

    async def _run_specific_test_async(self):
        if not hasattr(os, "fork"):
            logger.warning("os.fork() not available, skipping test")
            return {"success": True, "skipped": True, "reason": "fork not available"}

        logger.info("Testing logger fork safety with held lock...")

        # Get TTLogger class (os.register_at_fork already called when server started)
        TTLogger = _get_ttlogger_class()
        tt_logger = TTLogger(name="fork_safety_test")
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
