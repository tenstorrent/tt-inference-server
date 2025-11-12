# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base_test import BaseTest

# Set up logger for this module
logger = logging.getLogger(__name__)

# Constants
DEFAULT_TEST_LIMIT = 3
DEFAULT_TIMEOUT = 10

# Default paths to search for lmms-eval executable
DEFAULT_LMMS_EVAL_PATHS = [
    "lmms-eval",  # If it's in PATH
    "/usr/local/bin/lmms-eval",
]


class WhisperEvalTest(BaseTest):
    """
    Whisper Audio Model Test using installed lmms-eval package.

    This test class uses the lmms-eval subprocess approach for evaluation,
    similar to run_evals.py. It automatically discovers the lmms-eval executable
    from workflow virtual environments or common installation locations.
    """

    # Class-level configuration
    model_name: str = "whisper_tt"
    base_url: str = "http://127.0.0.1:8000"
    batch_size: int = 1
    test_limit: int = DEFAULT_TEST_LIMIT
    output_dir: str = "/tmp/whisper_eval_test_output"

    def __init__(self, config=None, targets=None, **kwargs):
        """Initialize and discover lmms-eval executable for performance."""
        super().__init__(config, targets)

        # Find lmms-eval executable during initialization (for performance)
        logger.info("Initializing WhisperEvalTest with lmms-eval subprocess approach")

        # Find lmms-eval executable (similar to run_evals.py)
        self.lmms_eval_exec = self._find_lmms_eval_executable()
        if not self.lmms_eval_exec:
            logger.warning("lmms-eval executable not found, test will fail")
        else:
            logger.info("Found lmms-eval executable: %s", self.lmms_eval_exec)

    async def _run_specific_test_async(self) -> Dict[str, Any]:
        """
        Run evaluation tests using lmms-eval subprocess (like run_evals.py).

        Returns:
            Dictionary containing test results and metadata.
        """
        start_time = time.time()

        logger.info("Running Whisper evaluation using lmms-eval subprocess...")

        if self.lmms_eval_exec:
            # Use lmms-eval subprocess (matching run_evals.py approach)
            results = await self._run_lmms_eval_subprocess(self.lmms_eval_exec)
        else:
            logger.error("lmms-eval executable not found, cannot run evaluation")
            results = {
                "status": "error",
                "method": "fallback",
                "error": "lmms-eval executable not found",
            }

        total_time = time.time() - start_time

        return self._build_final_results(results, total_time, self.lmms_eval_exec)

    def _build_final_results(
        self, results: Dict[str, Any], total_time: float, lmms_eval_exec: Optional[str]
    ) -> Dict[str, Any]:
        """
        Build the final results dictionary.

        Args:
            results: Raw results from evaluation
            total_time: Total execution time in seconds
            lmms_eval_exec: Path to lmms-eval executable (for mode determination)

        Returns:
            Formatted final results dictionary.
        """
        final_results = {
            "evaluation_results": results,
            "evaluation_mode": "lmms_eval_subprocess" if lmms_eval_exec else "fallback",
            "total_time_seconds": total_time,
            "test_type": "whisper_eval",
            "model": self.model_name,
            "base_url": self.base_url,
        }

        logger.info("✅ Whisper evaluation test completed successfully in %.2fs", total_time)
        return final_results

    def _find_lmms_eval_executable(self) -> Optional[str]:
        """
        Find the lmms-eval executable, similar to run_evals.py approach.

        Returns:
            Path to the lmms-eval executable, or None if not found.
        """
        # Try common locations where lmms-eval might be installed
        possible_paths = [
            *DEFAULT_LMMS_EVAL_PATHS,
            str(Path.home() / ".local/bin/lmms-eval"),
        ]

        # Add workflow venv path if available
        workflow_path = self._get_workflow_venv_path()
        if workflow_path:
            possible_paths.insert(0, str(workflow_path))

        return self._test_executable_paths(possible_paths)

    def _get_workflow_venv_path(self) -> Optional[Path]:
        """
        Get the lmms-eval path from workflow virtual environment.

        Returns:
            Path to lmms-eval in workflow venv, or None if not available.
        """
        try:
            # Add the actual repo root to sys.path to find workflows module
            repo_root = self._find_repo_root()
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))

            from workflows.workflow_types import WorkflowVenvType
            from workflows.workflow_venvs import VENV_CONFIGS

            if WorkflowVenvType.EVALS_AUDIO in VENV_CONFIGS:
                venv_config = VENV_CONFIGS[WorkflowVenvType.EVALS_AUDIO]
                return venv_config.venv_path / "bin" / "lmms-eval"

        except ImportError:
            # Silently continue if workflow modules aren't available
            pass
        except Exception:
            # Silently continue on other errors
            pass

        return None

    def _test_executable_paths(self, paths: List[str]) -> Optional[str]:
        """
        Test a list of possible executable paths.

        Args:
            paths: List of paths to test

        Returns:
            First valid executable path, or None if none found.
        """
        for path in paths:
            if self._is_valid_executable(path):
                return path
        return None

    def _is_valid_executable(self, path: str) -> bool:
        """
        Test if a path is a valid lmms-eval executable.

        Args:
            path: Path to test

        Returns:
            True if the path is a valid executable, False otherwise.
        """
        try:
            result = subprocess.run(
                [path, "--help"],
                capture_output=True,
                timeout=DEFAULT_TIMEOUT,
                check=False,
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            return False

    def _find_repo_root(self) -> Path:
        """
        Find the actual repository root directory by looking for workflows directory.

        This ensures we find the repo root regardless of how run.py sets up sys.path.

        Returns:
            Path to the repository root directory.
        """
        current = Path(__file__).resolve()
        while current.parent != current:  # Stop at filesystem root
            workflows_dir = current / "workflows"
            if workflows_dir.exists() and workflows_dir.is_dir():
                # Found the repo root with workflows directory
                return current
            current = current.parent

        # Fallback to using relative path - go up from tt-media-server/tests/server_tests/test_cases/
        # to reach the actual repo root (5 levels up)
        return Path(__file__).parent.parent.parent.parent.parent

    async def _run_lmms_eval_subprocess(self, lmms_eval_exec: str) -> Dict[str, Any]:
        """
        Run evaluation using lmms-eval subprocess (matching run_evals.py pattern).

        Args:
            lmms_eval_exec: Path to the lmms-eval executable

        Returns:
            Dictionary containing subprocess results and metadata.
        """
        try:
            logger.info("Running lmms-eval as subprocess...")

            # Set OPENAI_API_BASE for audio models (matching run_evals.py)
            import os
            os.environ["OPENAI_API_BASE"] = self.base_url

            # Create output directory for results
            output_dir = Path(self.output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)

            # Build and execute lmms-eval command
            cmd = self._build_lmms_eval_command(lmms_eval_exec)
            logger.info("Running command: %s", " ".join(cmd))

            return await self._execute_subprocess(cmd)

        except Exception as e:
            logger.exception("Error running lmms-eval subprocess: %s", e)
            return {
                "status": "error",
                "method": "lmms_eval_subprocess",
                "error": str(e),
                "fallback_used": False,
            }

    def _build_lmms_eval_command(self, lmms_eval_exec: str) -> List[str]:
        """
        Build the lmms-eval command arguments.

        Args:
            lmms_eval_exec: Path to the lmms-eval executable

        Returns:
            List of command arguments for lmms-eval.
        """
        return [
            str(lmms_eval_exec),
            "--model",
            self.model_name,
            "--model_args",
            f"model={self.model_name},base_url={self.base_url}",
            "--tasks",
            "whisper_cmu_arctic",
            "--batch_size",
            str(self.batch_size),
            "--output_path",
            str(self.output_dir),
            "--log_samples",
            "--limit",
            str(self.test_limit),
        ]

    async def _execute_subprocess(self, cmd: List[str]) -> Dict[str, Any]:
        """
        Execute the lmms-eval subprocess and handle results.

        Args:
            cmd: Command arguments to execute

        Returns:
            Dictionary containing execution results.
        """
        # Run the subprocess with timeout
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=1800  # 30 minutes
            )
        except asyncio.TimeoutError:
            logger.error("lmms-eval subprocess timed out after %d seconds", 1800)
            process.kill()
            await process.wait()
            return {
                "status": "error",
                "method": "lmms_eval_subprocess",
                "error": "Subprocess timed out after 1800 seconds",
                "return_code": -1,
            }

        success = process.returncode == 0
        if success:
            logger.info("lmms-eval subprocess completed successfully")
        else:
            logger.error("lmms-eval subprocess failed with return code: %d", process.returncode)

        result = {
            "status": "success" if success else "error",
            "method": "lmms_eval_subprocess",
            "command": " ".join(cmd),
            "stdout": stdout.decode() if stdout else "",
            "return_code": process.returncode,
        }

        if success:
            result["output_dir"] = str(self.output_dir)
        else:
            result.update(
                {
                    "stderr": stderr.decode() if stderr else "",
                    "fallback_used": False,
                }
            )

        return result
