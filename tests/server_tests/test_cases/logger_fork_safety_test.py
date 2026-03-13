# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import logging
import subprocess
import sys
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


PYTEST_TEST_PATH = (
    _find_repo_root() / "tt-media-server" / "tests" / "test_logger_fork_safety.py"
)


class LoggerForkSafetyTest(BaseTest):
    """Run fork-safety logging tests via pytest subprocess.

    Validates that os.register_at_fork in utils/logger.py prevents deadlocks
    when child processes inherit logging handler locks in acquired state.
    """

    async def _run_specific_test_async(self):
        logger.info(f"Running pytest: {PYTEST_TEST_PATH}")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(PYTEST_TEST_PATH), "-v", "--tb=short"],
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        logger.info(f"pytest stdout:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"pytest stderr:\n{result.stderr}")

        if result.returncode != 0:
            raise Exception(
                f"Logger fork safety tests failed (exit code {result.returncode}):\n"
                f"{result.stdout}\n{result.stderr}"
            )

        logger.info("Logger fork safety tests passed")
        return {"success": True, "pytest_output": result.stdout}
