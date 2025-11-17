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

    # Class-level configuration - use same settings as run_evals.py
    model_name: str = "whisper_tt"  # eval_class from eval_config.py
    hf_model_repo: str = "distil-whisper/distil-large-v3"  # Real model repo
    api_key: str = "your-secret-key"  # API key for authentication
    base_url: str = "http://127.0.0.1:8000"
    batch_size: int = 1
    test_limit: int = None  # Remove limit to match real run_evals.py behavior
    task_name: str = "librispeech_test_other"  # Same as run_evals.py
    num_concurrent: int = 1  # Added missing parameter

    # Note: output_dir is now computed dynamically in _construct_output_directory()
    # instead of being a class variable

    # Debug mode - set to True for fast testing with --limit 2
    debug_mode: bool = False  # Disable for real dataset download and evaluation

    def __init__(self, config=None, targets=None, **kwargs):
        """Initialize and discover lmms-eval executable for performance."""
        super().__init__(config, targets)

        # Set up output directory from config (with fallback)
        if config and config.get("output_path"):
            # Get base output path from config
            base_output_path = config.get("output_path")

            # Structure path as expected by run_reports.py:
            # {output_path}/eval_{model_id}/{hf_model_repo.replace('/', '__')}/
            # We need model_id, but it's not available here, so we'll construct it later
            # For now, create a base directory structure
            self.base_output_path = base_output_path
            logger.info(f"Using output path from config: {base_output_path}")
        else:
            # Fallback to hardcoded path
            self.base_output_path = "/tmp/whisper_eval_test_output"
            logger.info("Using fallback output path: /tmp/whisper_eval_test_output")

        # Find lmms-eval executable during initialization (for performance)
        logger.info("Initializing WhisperEvalTest with lmms-eval subprocess approach")

        # Find lmms-eval executable (similar to run_evals.py)
        self.lmms_eval_exec = self._find_lmms_eval_executable()
        if not self.lmms_eval_exec:
            logger.warning("lmms-eval executable not found, test will fail")
        else:
            logger.info("Found lmms-eval executable: %s", self.lmms_eval_exec)

    def _construct_output_directory(self):
        """
        Construct the output directory path that matches run_reports.py expectations.

        Note: lmms-eval automatically creates a subdirectory structure like:
        {output_path}/{hf_model_repo.replace('/', '__')}/results_{timestamp}.json

        So we only need to provide the base path: {base_output_path}/eval_{model_id}/
        lmms-eval will create the {hf_model_repo.replace('/', '__')} part.

        Expected final structure: {base_output_path}/eval_{model_id}/{hf_model_repo.replace('/', '__')}/*_results.json

        Returns:
            str: Base output directory path for lmms-eval
        """
        # Use model_id from config if available, otherwise construct it
        if self.config and self.config.get("model_id"):
            model_id = self.config.get("model_id")
            logger.info(f"Using model_id from config: {model_id}")
        else:
            # Construct model_id from model name and repo (fallback)
            # Format: model_name_repo_device -> whisper_distil_large_v3_n150
            repo_short = (
                self.hf_model_repo.split("/")[-1]
                if "/" in self.hf_model_repo
                else self.hf_model_repo
            )
            model_id = f"whisper_{repo_short}_n150"  # Assuming n150 device for now
            logger.info(f"Constructed fallback model_id: {model_id}")

        # Create base directory structure - lmms-eval will create the repo subdirectory
        output_dir = Path(self.base_output_path) / f"eval_{model_id}"

        # Create the expected final structure path for logging
        repo_dir_name = self.hf_model_repo.replace("/", "__")
        expected_final_path = output_dir / repo_dir_name

        logger.info(f"Constructed base output directory for lmms-eval: {output_dir}")
        logger.info(
            f"lmms-eval will create files in: {expected_final_path}/*_results.json"
        )
        logger.info(
            f"This matches run_reports.py pattern: eval_{model_id}/{repo_dir_name}/*_results.json"
        )

        return str(output_dir)

    @property
    def output_dir(self):
        """
        Get the output directory, constructing it dynamically based on configuration.

        Returns:
            str: Output directory path
        """
        return self._construct_output_directory()

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

        logger.info(
            "✅ Whisper evaluation test completed successfully in %.2fs", total_time
        )
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

            # Set API key for authentication
            os.environ["OPENAI_API_KEY"] = self.api_key
            logger.info("Set OPENAI_API_KEY for authentication")

            # Create output directory for results
            output_dir = Path(self.output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)

            # Build and execute lmms-eval command
            cmd = self._build_lmms_eval_command(lmms_eval_exec)
            logger.info("Running command: %s", " ".join(cmd))
            logger.info("Command comparison:")
            logger.info("  Model: %s", self.model_name)
            logger.info("  HF repo: %s", self.hf_model_repo)
            logger.info("  Task: %s", self.task_name)
            logger.info("  Limit: %s samples", self.test_limit)

            return await asyncio.get_running_loop().run_in_executor(
                None, self._execute_subprocess_sync, cmd
            )

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
        Build the lmms-eval command arguments matching run_evals.py exactly.

        Args:
            lmms_eval_exec: Path to the lmms-eval executable

        Returns:
            List of command arguments for lmms-eval.
        """
        cmd_args = [
            str(lmms_eval_exec),
            "--model",
            self.model_name,
            "--model_args",
            f"model={self.hf_model_repo},base_url={self.base_url},num_concurrent={self.num_concurrent}",
            "--tasks",
            self.task_name,
            "--batch_size",
            str(self.batch_size),
            "--output_path",
            str(self.output_dir),
            "--log_samples",
        ]

        # Only add --limit if test_limit is set (for debugging) or debug_mode is enabled
        if self.test_limit is not None:
            cmd_args.extend(["--limit", str(self.test_limit)])
        elif self.debug_mode:
            cmd_args.extend(["--limit", "2"])  # Fast debug mode

        return cmd_args

    def _execute_subprocess_sync(self, cmd: List[str]) -> Dict[str, Any]:
        """
        Execute the lmms-eval subprocess using the same method as run_evals.py.

        Args:
            cmd: Command arguments to execute

        Returns:
            Dictionary containing execution results.
        """
        logger.info("Starting lmms-eval subprocess with real-time logging...")

        # Import threading and subprocess for run_evals.py style execution
        import shlex
        import subprocess
        import threading

        def stream_subprocess_output(pipe, logger, level):
            """Same function as workflows.utils.stream_subprocess_output"""
            with pipe:
                for line in iter(pipe.readline, ""):
                    logger.log(level, line.strip(), extra={"raw": True})

        # Use the same subprocess approach as run_evals.py run_command
        logger.info(f"Running command: {shlex.join(cmd)}")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            text=True,
        )

        stdout_thread = threading.Thread(
            target=stream_subprocess_output,
            args=(
                process.stdout,
                logger,
                logging.INFO,
            ),  # Use INFO level for visibility
        )
        stderr_thread = threading.Thread(
            target=stream_subprocess_output,
            args=(process.stderr, logger, logging.ERROR),
        )

        stdout_thread.start()
        stderr_thread.start()

        stdout_thread.join()
        stderr_thread.join()

        process.wait()
        return_code = process.returncode

        success = return_code == 0
        if success:
            logger.info("✅ lmms-eval subprocess completed successfully")
        else:
            logger.error(
                "❌ lmms-eval subprocess failed with return code: %d", return_code
            )

        result = {
            "status": "success" if success else "error",
            "method": "lmms_eval_subprocess",
            "command": " ".join(cmd),
            "return_code": return_code,
        }

        if success:
            result["output_dir"] = str(self.output_dir)
        else:
            result.update(
                {
                    "fallback_used": False,
                }
            )

        return result
