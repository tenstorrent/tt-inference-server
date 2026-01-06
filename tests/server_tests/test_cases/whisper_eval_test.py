# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import json
import logging
import re
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
    hf_model_repo: str = "distil-whisper/distil-large-v3"
    api_key: str = "your-secret-key"
    base_url: str = "http://127.0.0.1:8000"
    batch_size: int = 1
    test_limit: int = None  # Remove limit to match real run_evals.py behavior
    task_name: str = "librispeech_test_other"  # Same as run_evals.py
    num_concurrent: int = 1  # Added missing parameter

    # Note: output_dir is now computed dynamically in _construct_output_directory()
    # instead of being a class variable

    # Debug mode - set to True for fast testing with --limit 2
    debug_mode: bool = False  # Disable for real dataset download and evaluation

    # Mock mode - set to True to simulate lmms-eval calls without actual execution
    mock_mode: bool = False

    def __init__(self, config=None, targets=None, **kwargs):
        """Initialize with lmms-eval executable from config or discovery."""
        super().__init__(config, targets)

        # Set mock_mode based on config if provided
        if config:
            mock_mode_from_config = config.get("mock_mode")
            if mock_mode_from_config is not None:
                self.mock_mode = mock_mode_from_config
                logger.info(f"Mock mode set from config: {self.mock_mode}")

        # Set up output directory from config (with fallback)
        if config and config.get("output_path"):
            # Get base output path from config
            base_output_path = config.get("output_path")
            self.base_output_path = base_output_path
            logger.info(f"Using output path from config: {base_output_path}")
        else:
            # Fallback to workflow_logs/evals_output (matching CI path)
            repo_root = self._find_repo_root()
            self.base_output_path = str(repo_root / "workflow_logs" / "evals_output")
            logger.info(f"Using fallback output path: {self.base_output_path}")

        # Initialize lmms-eval executable
        logger.info("Initializing WhisperEvalTest with lmms-eval subprocess approach")

        # Use lmms-eval executable provided by run_evals.py if available,
        # otherwise try to discover it
        self.lmms_eval_exec = getattr(self, "lmms_eval_exec", None)
        if not self.lmms_eval_exec:
            self.lmms_eval_exec = self._find_lmms_eval_executable()

        if not self.lmms_eval_exec:
            logger.warning("⚠️ lmms-eval executable not found, test will fail")
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
        Run evaluation tests using lmms-eval subprocess (like run_evals.py) or mock mode.

        Returns:
            Dictionary containing test results and metadata.
        """
        start_time = time.time()

        if self.mock_mode:
            logger.info("Running Whisper evaluation in MOCK MODE (simulated)...")
            results = await self._run_mock_evaluation()
        else:
            logger.info("Running Whisper evaluation using lmms-eval subprocess...")
            if self.lmms_eval_exec:
                # Use lmms-eval subprocess (matching run_evals.py approach)
                results = await self._run_lmms_eval_subprocess(self.lmms_eval_exec)
            else:
                logger.error(
                    " ❌ lmms-eval executable not found, cannot run evaluation"
                )
                results = {
                    "status": "error",
                    "method": "fallback",
                    "error": "lmms-eval executable not found",
                }

        total_time = time.time() - start_time

        return self._build_final_results(
            results, total_time, self.lmms_eval_exec if not self.mock_mode else "mock"
        )

    def _build_final_results(
        self, results: Dict[str, Any], total_time: float, lmms_eval_exec: Optional[str]
    ) -> Dict[str, Any]:
        """
        Build the final results dictionary.

        Args:
            results: Raw results from evaluation
            total_time: Total execution time in seconds
            lmms_eval_exec: Path to lmms-eval executable (for mode determination) or "mock"

        Returns:
            Formatted final results dictionary.
        """
        final_results = {
            "evaluation_results": results,
            "evaluation_mode": "mock"
            if lmms_eval_exec == "mock"
            else ("lmms_eval_subprocess" if lmms_eval_exec else "fallback"),
            "total_time_seconds": total_time,
            "test_type": "whisper_eval",
            "model": self.model_name,
            "base_url": self.base_url,
        }

        # Extract and promote evaluation metrics to top level for better visibility
        if "evaluation_metrics" in results:
            final_results["metrics"] = results["evaluation_metrics"]
            logger.info(f"Evaluation metrics: {results['evaluation_metrics']}")

        if "json_results" in results:
            final_results["results_files"] = list(results["json_results"].keys())
            # Extract key metrics from JSON results if available
            for filename, data in results["json_results"].items():
                if isinstance(data, dict) and "results" in data:
                    final_results["detailed_results"] = data["results"]
                    break

        mode_emoji = "🎭" if self.mock_mode else "✅"
        mode_text = " (MOCK MODE)" if self.mock_mode else ""
        logger.info(
            f"{mode_emoji} Whisper evaluation test completed successfully in %.2fs{mode_text}",
            total_time,
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

        This method is bypassed in mock mode.

        Args:
            lmms_eval_exec: Path to the lmms-eval executable

        Returns:
            Dictionary containing subprocess results and metadata.
        """
        # Safety check - should not reach here in mock mode
        if self.mock_mode:
            logger.warning(
                "⚠️ lmms-eval subprocess called while in mock mode - returning mock results"
            )
            return await self._run_mock_evaluation()

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

    def _parse_evaluation_results(self, stdout_lines: List[str]) -> Dict[str, Any]:
        """
        Parse evaluation results from lmms-eval stdout output.

        Args:
            stdout_lines: List of stdout lines from lmms-eval

        Returns:
            Dictionary containing parsed evaluation metrics.
        """
        results = {}

        try:
            # Look for the results table in stdout
            # Example format from logs:
            # |librispeech_test_other|Yaml   |none  |     0|wer   |↓  |6.4613|±  |   N/A|

            for line in stdout_lines:
                # Match evaluation results table rows
                if "|" in line and ("wer" in line.lower() or "bleu" in line.lower()):
                    parts = [p.strip() for p in line.split("|") if p.strip()]
                    if len(parts) >= 7:  # Ensure we have enough parts
                        try:
                            task_name = parts[0]
                            metric_name = parts[4] if len(parts) > 4 else "unknown"
                            metric_value = float(parts[6]) if len(parts) > 6 else None

                            if metric_value is not None:
                                results[f"{task_name}_{metric_name}"] = metric_value
                                logger.info(
                                    f"Parsed metric: {task_name} {metric_name} = {metric_value}"
                                )
                        except (ValueError, IndexError) as e:
                            logger.debug(f"Could not parse line: {line}, error: {e}")

            # Also look for summary lines like "wer: 6.4613"
            for line in stdout_lines:
                wer_match = re.search(r"wer[\s:]+([\d.]+)", line.lower())
                if wer_match:
                    results["wer"] = float(wer_match.group(1))

                bleu_match = re.search(r"bleu[\s:]+([\d.]+)", line.lower())
                if bleu_match:
                    results["bleu"] = float(bleu_match.group(1))

        except Exception as e:
            logger.error(f"Error parsing evaluation results: {e}")

        return results

    def _find_and_parse_results_files(self) -> Dict[str, Any]:
        """
        Find and parse JSON results files generated by lmms-eval.

        Returns:
            Dictionary containing parsed JSON results from files.
        """
        results = {}

        try:
            # Look for results files in the expected output directory structure
            # Format: {output_dir}/{hf_model_repo.replace('/', '__')}/*_results.json
            output_path = Path(self.output_dir)
            repo_dir_name = self.hf_model_repo.replace("/", "__")
            results_dir = output_path / repo_dir_name

            logger.info(f"Looking for results files in: {results_dir}")

            if results_dir.exists():
                # Find all JSON files with "results" in the name
                json_files = list(results_dir.glob("*results*.json"))
                logger.info(
                    f"Found {len(json_files)} results files: {[f.name for f in json_files]}"
                )

                for json_file in json_files:
                    try:
                        with open(json_file, "r") as f:
                            data = json.load(f)
                            results[json_file.name] = data
                            logger.info(f"Successfully parsed {json_file.name}")
                    except Exception as e:
                        logger.error(f"Error parsing {json_file}: {e}")
            else:
                logger.warning(f"Results directory not found: {results_dir}")
                # List what actually exists
                if output_path.exists():
                    existing = list(output_path.iterdir())
                    logger.info(
                        f"Contents of {output_path}: {[p.name for p in existing]}"
                    )

        except Exception as e:
            logger.error(f"Error finding results files: {e}")

        return results

    async def _run_mock_evaluation(self) -> Dict[str, Any]:
        """
        Run a mock evaluation that simulates lmms-eval output and file generation.

        This method creates realistic mock data based on the actual lmms-eval output format
        and generates the expected JSON files without running the actual evaluation.

        Returns:
            Dictionary containing mock evaluation results.
        """
        try:
            logger.info("🎭 MOCK MODE: Simulating lmms-eval evaluation...")

            # Simulate some processing time
            await asyncio.sleep(1.0)

            # Create output directory structure
            output_dir = Path(self.output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)

            repo_dir_name = self.hf_model_repo.replace("/", "__")
            results_dir = output_dir / repo_dir_name
            results_dir.mkdir(exist_ok=True, parents=True)

            # Generate mock timestamp
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create mock evaluation metrics (simulate CI logs output)
            mock_wer_value = 5.663552815702886  # From actual results JSON
            mock_stdout_lines = [
                "whisper_tt (model=distil-whisper/distil-large-v3,base_url=http://127.0.0.1:8000,num_concurrent=1), gen_kwargs: (), limit: 0.5, num_fewshot: None, batch_size: 1",
                "|        Tasks         |Version|Filter|n-shot|Metric|   |Value |   |Stderr|",
                "|----------------------|-------|------|-----:|------|---|-----:|---|------|",
                f"|librispeech_test_other|Yaml   |none  |     0|wer   |↓  |{mock_wer_value}|±  |   N/A|",
                "",
            ]

            # Generate mock JSON results file (based on actual format)
            mock_json_results = {
                "results": {
                    "librispeech_test_other": {
                        "alias": "librispeech_test_other",
                        "wer,none": mock_wer_value,
                        "wer_stderr,none": "N/A",
                    }
                },
                "group_subtasks": {"librispeech_test_other": []},
                "configs": {
                    "librispeech_test_other": {
                        "task": "librispeech_test_other",
                        "dataset_path": "lmms-lab/librispeech",
                        "dataset_name": "librispeech_test_other",
                        "dataset_kwargs": {"token": True},
                        "test_split": "librispeech_test_other",
                        "full_docs": False,
                        "process_results_use_image": False,
                        "doc_to_visual": "<function librispeech_doc_to_audio at 0x7f49f7bce8c0>",
                        "doc_to_text": "<function librispeech_doc_to_text at 0x7f49f64f39a0>",
                        "doc_to_target": "gt",
                        "process_results": "<function librispeech_process_result at 0x7f49f63c80d0>",
                        "description": "",
                        "target_delimiter": " ",
                        "fewshot_delimiter": "\n\n",
                        "num_fewshot": 0,
                        "metric_list": [
                            {
                                "metric": "wer",
                                "aggregation": "<function librispeech_wer at 0x7f49f63c8b80>",
                                "higher_is_better": False,
                            }
                        ],
                        "output_type": "generate_until",
                        "generation_kwargs": {
                            "max_new_tokens": 256,
                            "temperature": 0.0,
                            "top_p": 1.0,
                            "num_beams": 1,
                            "do_sample": False,
                            "until": ["\n\n"],
                        },
                        "repeats": 1,
                        "should_decontaminate": False,
                        "metadata": [{"version": 0.0}],
                        "lmms_eval_specific_kwargs": {
                            "default": {"pre_prompt": "", "post_prompt": ""},
                            "qwen2_audio": {"pre_prompt": "", "post_prompt": " <|en|>"},
                            "pre_prompt": "",
                            "post_prompt": "",
                        },
                    }
                },
                "versions": {"librispeech_test_other": "Yaml"},
                "n-shot": {"librispeech_test_other": 0},
                "higher_is_better": {"librispeech_test_other": {"wer": False}},
                "n-samples": {
                    "librispeech_test_other": {
                        "original": 2939,
                        "effective": int(self.test_limit) if self.test_limit else 2939,
                    }
                },
                "config": {
                    "model": self.model_name,
                    "model_args": f"model={self.hf_model_repo},base_url={self.base_url},num_concurrent={self.num_concurrent}",
                    "batch_size": str(self.batch_size),
                    "batch_sizes": [],
                    "device": None,
                    "use_cache": None,
                    "limit": float(self.test_limit) if self.test_limit else None,
                    "bootstrap_iters": 100000,
                    "gen_kwargs": "",
                    "random_seed": 0,
                    "numpy_seed": 1234,
                    "torch_seed": 1234,
                    "fewshot_seed": 1234,
                },
                "git_hash": "mock_hash",
                "date": timestamp,
                "task_hashes": {
                    "librispeech_test_other": "d8ea0af461c5ec8681ea71e3d22a9e1a9506c12c2dfc9defdff737b5ee902259"
                },
                "model_source": self.model_name,
                "model_name": self.hf_model_repo,
                "model_name_sanitized": repo_dir_name,
                "system_instruction": None,
                "system_instruction_sha": None,
                "fewshot_as_multiturn": False,
                "chat_template": None,
                "chat_template_sha": None,
                "start_time": 533768.078025213,
                "end_time": 533773.364108704,
                "total_evaluation_time_seconds": "5.286083491053432",
            }

            # Write mock JSON results file
            json_filename = f"{timestamp}_results.json"
            json_file_path = results_dir / json_filename

            with open(json_file_path, "w") as f:
                json.dump(mock_json_results, f, indent=2)

            # Write mock samples file
            samples_filename = f"{timestamp}_samples_{self.task_name}.jsonl"
            samples_file_path = results_dir / samples_filename

            mock_sample = {
                "doc_id": 0,
                "doc": {"text": "mock audio text"},
                "target": "mock target text",
                "arguments": [],
                "resps": [["mock response text"]],
                "filtered_resps": ["mock response text"],
            }

            with open(samples_file_path, "w") as f:
                f.write(json.dumps(mock_sample) + "\n")

            logger.info(f"🎭 MOCK: Generated {json_filename} and {samples_filename}")

            # Parse the mock evaluation results
            parsed_metrics = self._parse_evaluation_results(mock_stdout_lines)

            # Simulate the same structure as real lmms-eval results
            mock_results = {
                "status": "success",
                "method": "lmms_eval_mock",
                "command": f"MOCK: lmms-eval --model {self.model_name} --model_args model={self.hf_model_repo},base_url={self.base_url},num_concurrent={self.num_concurrent} --tasks {self.task_name} --batch_size {self.batch_size} --output_path {self.output_dir} --log_samples"
                + (f" --limit {self.test_limit}" if self.test_limit else ""),
                "return_code": 0,
                "output_dir": str(self.output_dir),
                "evaluation_metrics": parsed_metrics,
                "json_results": {json_filename: mock_json_results},
            }

            logger.info(f"🎭 MOCK: Evaluation completed with WER={mock_wer_value}")
            return mock_results

        except Exception as e:
            logger.error(f"Error in mock evaluation: {e}")
            return {
                "status": "error",
                "method": "lmms_eval_mock",
                "error": str(e),
                "fallback_used": False,
            }

    def _execute_subprocess_sync(self, cmd: List[str]) -> Dict[str, Any]:
        """
        Execute the lmms-eval subprocess and capture evaluation results.

        Args:
            cmd: Command arguments to execute

        Returns:
            Dictionary containing execution results and parsed evaluation metrics.
        """
        logger.info("Starting lmms-eval subprocess with output capture...")

        # Import threading and subprocess for run_evals.py style execution
        import shlex
        import subprocess
        import threading

        # Capture stdout for parsing while still logging
        captured_stdout = []
        captured_stderr = []

        def stream_and_capture_output(pipe, logger, level, capture_list):
            """Stream output to logger while capturing for parsing"""
            with pipe:
                for line in iter(pipe.readline, ""):
                    line_stripped = line.strip()
                    if line_stripped:  # Only capture non-empty lines
                        capture_list.append(line_stripped)
                        logger.log(level, line_stripped, extra={"raw": True})

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
            target=stream_and_capture_output,
            args=(
                process.stdout,
                logger,
                logging.INFO,
                captured_stdout,
            ),
        )
        stderr_thread = threading.Thread(
            target=stream_and_capture_output,
            args=(process.stderr, logger, logging.ERROR, captured_stderr),
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
            # Parse evaluation results from captured output and generated files
            eval_results = self._parse_evaluation_results(captured_stdout)
            if eval_results:
                result["evaluation_metrics"] = eval_results
                logger.info(f"Parsed evaluation metrics: {eval_results}")
            else:
                logger.warning("No evaluation metrics found in output")

            # Try to find and parse JSON results files
            json_results = self._find_and_parse_results_files()
            if json_results:
                result["json_results"] = json_results
                logger.info(f"Found JSON results files: {list(json_results.keys())}")
        else:
            result.update(
                {
                    "fallback_used": False,
                    "captured_stderr": captured_stderr[-10:]
                    if captured_stderr
                    else [],  # Last 10 error lines
                }
            )

        return result
