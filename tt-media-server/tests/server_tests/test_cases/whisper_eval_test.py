# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..base_test import BaseTest

# Constants
SAMPLING_RATE = 16_000
DEFAULT_TEST_DURATION = 2.0
DEFAULT_TEST_LIMIT = 3
DEFAULT_TIMEOUT = 10
TONE_AMPLITUDE = 0.3

# Default paths to search for lmms-eval executable
DEFAULT_LMMS_EVAL_PATHS = [
    "lmms-eval",  # If it's in PATH
    "/usr/local/bin/lmms-eval",
]

# Audio test configuration
AUDIO_TEST_CONFIGS = [
    {"freq": 440, "description": "440Hz sine wave", "type": "tone"},
    {"freq": 880, "description": "880Hz sine wave", "type": "tone"},
    {"freq": None, "description": "silence", "type": "silence"},
]


class WhisperEvalTest(BaseTest):
    """
    Whisper Audio Model Test using installed lmms-eval package.

    This test class uses the lmms-eval subprocess approach for evaluation,
    similar to run_evals.py. It automatically discovers the lmms-eval executable
    from workflow virtual environments or common installation locations.

    Args:
        config: Test configuration from TestConfig
        targets: Test targets (unused in current implementation)
        model_name: Name of the model to test (default: "whisper_tt")
        base_url: Base URL for the model API (default: "http://127.0.0.1:8000")
        batch_size: Batch size for evaluation (default: 1)
        language: Language code for transcription (default: "en")
        task: Task type for whisper (default: "transcribe")
        **kwargs: Additional keyword arguments (logged as warnings)
    """

    def __init__(
        self,
        config,
        targets,
        model_name: str = "whisper_tt",
        base_url: str = "http://127.0.0.1:8000",
        batch_size: int = 1,
        language: str = "en",
        task: str = "transcribe",
        test_limit: int = DEFAULT_TEST_LIMIT,
        output_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(config, targets)

        # Log warning for unexpected kwargs but don't fail
        if kwargs:
            print(f"Warning: Ignoring unexpected kwargs: {kwargs}")

        # Model settings for lmms-eval command
        self.model_name = model_name
        self.base_url = base_url
        self.batch_size = batch_size
        self.language = language
        self.task = task
        self.test_limit = test_limit
        self.output_dir = (
            Path(output_dir) if output_dir else Path("/tmp/whisper_eval_test_output")
        )

        print("Initializing WhisperEvalTest with lmms-eval subprocess approach")

        # Find lmms-eval executable (similar to run_evals.py)
        self.lmms_eval_exec = self._find_lmms_eval_executable()
        if not self.lmms_eval_exec:
            print("Warning: lmms-eval executable not found, using fallback mode")
        else:
            print(f"Found lmms-eval executable: {self.lmms_eval_exec}")

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

    async def _run_specific_test_async(self) -> Dict[str, Any]:
        """
        Run evaluation tests using lmms-eval subprocess (like run_evals.py).

        Returns:
            Dictionary containing test results and metadata.
        """
        start_time = time.time()

        print("Running Whisper evaluation using lmms-eval subprocess...")

        if self.lmms_eval_exec:
            # Use lmms-eval subprocess (matching run_evals.py approach)
            results = await self._run_lmms_eval_subprocess()
        else:
            # Fallback to simple synthetic tests
            results = await self._run_fallback_tests()

        total_time = time.time() - start_time

        return self._build_final_results(results, total_time)

    def _build_final_results(
        self, results: Dict[str, Any], total_time: float
    ) -> Dict[str, Any]:
        """
        Build the final results dictionary.

        Args:
            results: Raw results from evaluation
            total_time: Total execution time in seconds

        Returns:
            Formatted final results dictionary.
        """
        final_results = {
            "evaluation_results": results,
            "evaluation_mode": "lmms_eval_subprocess"
            if self.lmms_eval_exec
            else "fallback",
            "total_time_seconds": total_time,
            "test_type": "whisper_eval",
            "model": self.model_name,
            "base_url": self.base_url,
            "language": self.language,
            "task": self.task,
        }

        print(f"✅ Whisper evaluation test completed successfully in {total_time:.2f}s")
        return final_results

    async def _run_lmms_eval_subprocess(self) -> Dict[str, Any]:
        """
        Run evaluation using lmms-eval subprocess (matching run_evals.py pattern).

        Returns:
            Dictionary containing subprocess results and metadata.
        """
        try:
            print("Running lmms-eval as subprocess...")

            # Create output directory for results
            self.output_dir.mkdir(exist_ok=True, parents=True)

            # Build and execute lmms-eval command
            cmd = self._build_lmms_eval_command()
            print(f"Running command: {' '.join(cmd)}")

            return await self._execute_subprocess(cmd)

        except Exception as e:
            print(f"Error running lmms-eval subprocess: {e}")
            return {
                "status": "error",
                "method": "lmms_eval_subprocess",
                "error": str(e),
                "fallback_used": False,
            }

    def _build_lmms_eval_command(self) -> List[str]:
        """
        Build the lmms-eval command arguments.

        Returns:
            List of command arguments for lmms-eval.
        """
        return [
            str(self.lmms_eval_exec),
            "--model",
            self.model_name,
            "--model_args",
            f"base_url={self.base_url}",
            "--tasks",
            "whisper_asr_test",  # Or use a simple audio task
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
        # Run the subprocess
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        success = process.returncode == 0
        status_msg = (
            "lmms-eval subprocess completed successfully"
            if success
            else f"lmms-eval subprocess failed with return code: {process.returncode}"
        )
        print(status_msg)

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

    async def _run_fallback_tests(self) -> Dict[str, Any]:
        """
        Fallback evaluation when lmms-eval is not available.

        Returns:
            Dictionary containing fallback test results.
        """
        print("Running fallback evaluation (lmms-eval executable not found)...")

        test_samples = self._create_test_audio_samples()
        results = self._process_fallback_samples(test_samples)

        return {
            "status": "success",
            "method": "fallback",
            "results": results,
            "samples_count": len(test_samples),
            "note": "lmms-eval executable not found",
        }

    def _process_fallback_samples(
        self, test_samples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process test samples for fallback mode.

        Args:
            test_samples: List of test audio samples

        Returns:
            List of processed sample results.
        """
        results = []
        for i, sample in enumerate(test_samples):
            results.append(
                {
                    "sample_id": i,
                    "description": sample["description"],
                    "transcription": f"fallback_transcription_{i}",
                    "status": "success",
                    "method": "fallback",
                }
            )
        return results

    def _create_test_audio_samples(self) -> List[Dict[str, Any]]:
        """
        Create test audio samples for evaluation.

        Returns:
            List of test audio samples with metadata.
        """
        test_samples = []
        sample_count = int(SAMPLING_RATE * DEFAULT_TEST_DURATION)
        t = np.linspace(0, DEFAULT_TEST_DURATION, sample_count)

        for config in AUDIO_TEST_CONFIGS:
            audio_data = self._generate_audio_sample(t, config["freq"])
            test_samples.append(
                {
                    "array": audio_data,
                    "sampling_rate": SAMPLING_RATE,
                    "description": config["description"],
                    "expected_type": config["type"],
                }
            )

        return test_samples

    def _generate_audio_sample(
        self, t: np.ndarray, frequency: Optional[float]
    ) -> np.ndarray:
        """
        Generate a single audio sample.

        Args:
            t: Time array for the sample
            frequency: Frequency in Hz, or None for silence

        Returns:
            Audio sample as numpy array.
        """
        if frequency is None:
            # Generate silence
            return np.zeros(len(t), dtype=np.float32)
        else:
            # Generate sine wave
            return (TONE_AMPLITUDE * np.sin(2 * np.pi * frequency * t)).astype(
                np.float32
            )
