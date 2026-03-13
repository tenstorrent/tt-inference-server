# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
"""Eval test for Whisper audio models using lmms-eval subprocess."""

import asyncio
import datetime
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Optional

from tests.server_tests.base_test import BaseTest

logger = logging.getLogger(__name__)

# Metrics supported for parsing from lmms-eval stdout output.
SUPPORTED_METRICS: frozenset[str] = frozenset({"wer", "bleu"})

# Indices for parsing the lmms-eval table format:
# |task|Version|Filter|n-shot|Metric|↓|Value|±|Stderr|
_TABLE_METRIC_INDEX = 4
_TABLE_VALUE_INDEX = 6
_MIN_TABLE_PARTS = 7

# Maximum directory depth when searching for repository root.
_MAX_REPO_ROOT_DEPTH = 15

# Thread join timeout (seconds) after subprocess completes.
_THREAD_JOIN_TIMEOUT = 10


@dataclass(frozen=True)
class WhisperEvalConfig:
    """Whisper evaluation configuration."""

    MODEL_NAME: str = "whisper_tt"
    HF_MODEL_REPO: str = "distil-whisper/distil-large-v3"
    API_KEY: str = "your-secret-key"
    BASE_URL: str = "http://127.0.0.1:8000"
    BATCH_SIZE: int = 1
    TASK_NAME: str = "librispeech_test_other"
    NUM_CONCURRENT: int = 1
    EXECUTABLE_TIMEOUT: int = 10
    LMMS_EVAL_PATHS: tuple = ("lmms-eval", "/usr/local/bin/lmms-eval")
    MOCK_WER_VALUE: float = 5.663552815702886


CONFIG = WhisperEvalConfig()


class WhisperEvalTest(BaseTest):
    """Whisper audio model eval test using lmms-eval subprocess."""

    model_name: str = CONFIG.MODEL_NAME
    hf_model_repo: str = CONFIG.HF_MODEL_REPO
    api_key: str = CONFIG.API_KEY
    base_url: str = CONFIG.BASE_URL
    batch_size: int = CONFIG.BATCH_SIZE
    test_limit: int = None
    task_name: str = CONFIG.TASK_NAME
    num_concurrent: int = CONFIG.NUM_CONCURRENT
    debug_mode: bool = False
    mock_mode: bool = False

    def __init__(self, config=None, targets=None, **kwargs):
        """Initialize with lmms-eval executable from config or discovery."""
        super().__init__(config, targets)

        if config and config.get("mock_mode") is not None:
            self.mock_mode = config.get("mock_mode")
            logger.info(f"Mock mode set from config: {self.mock_mode}")

        self.base_output_path = self._resolve_output_path(config)

        self.lmms_eval_exec = getattr(self, "lmms_eval_exec", None)
        if not self.lmms_eval_exec:
            self.lmms_eval_exec = self._find_lmms_eval_executable()

        if self.lmms_eval_exec:
            logger.info(f"Found lmms-eval executable: {self.lmms_eval_exec}")
        else:
            logger.warning("⚠️ lmms-eval executable not found, test will fail")

    def _resolve_output_path(self, config: Optional[dict]) -> str:
        """Resolve base output path from config or fallback to default."""
        if config and config.get("output_path"):
            path = config.get("output_path")
            logger.info(f"Using output path from config: {path}")
            return path

        fallback = str(self._find_repo_root() / "workflow_logs" / "evals_output")
        logger.info(f"Using fallback output path: {fallback}")
        return fallback

    def _construct_output_directory(self) -> str:
        """Construct the output directory path matching run_reports.py expectations."""
        if self.config and self.config.get("model_id"):
            model_id = self.config.get("model_id")
        else:
            repo_short = (
                self.hf_model_repo.split("/")[-1]
                if "/" in self.hf_model_repo
                else self.hf_model_repo
            )
            model_id = f"whisper_{repo_short}_n150"

        output_dir = Path(self.base_output_path) / f"eval_{model_id}"
        logger.info(f"Output directory: {output_dir}")
        return str(output_dir)

    @property
    def output_dir(self) -> str:
        """Get the output directory, constructed dynamically from configuration."""
        return self._construct_output_directory()

    async def _run_specific_test_async(self) -> dict[str, Any]:
        """Run evaluation using lmms-eval subprocess or mock mode."""
        start_time = time.time()

        if self.mock_mode:
            logger.info("Running Whisper evaluation in MOCK MODE (simulated)...")
            results = await self._run_mock_evaluation()
        elif self.lmms_eval_exec:
            logger.info("Running Whisper evaluation using lmms-eval subprocess...")
            results = await self._run_lmms_eval_subprocess(self.lmms_eval_exec)
        else:
            logger.error("❌ lmms-eval executable not found, cannot run evaluation")
            return self._error("lmms-eval executable not found")

        total_time = time.time() - start_time

        return self._build_final_results(
            results, total_time, self.lmms_eval_exec if not self.mock_mode else "mock"
        )

    def _build_final_results(
        self, results: dict[str, Any], total_time: float, lmms_eval_exec: Optional[str]
    ) -> dict[str, Any]:
        """Build the final results dictionary."""
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

        if "evaluation_metrics" in results:
            final_results["metrics"] = results["evaluation_metrics"]
            logger.info(f"Evaluation metrics: {results['evaluation_metrics']}")

        if "json_results" in results:
            final_results["results_files"] = list(results["json_results"].keys())
            for filename, data in results["json_results"].items():
                if isinstance(data, dict) and "results" in data:
                    final_results["detailed_results"] = data["results"]
                    break

        mode_emoji = "🎭" if self.mock_mode else "✅"
        mode_text = " (MOCK MODE)" if self.mock_mode else ""
        logger.info(
            f"{mode_emoji} Whisper evaluation test completed successfully in {total_time:.2f}s{mode_text}"
        )
        return final_results

    @staticmethod
    def _error(message: str) -> dict[str, Any]:
        """Create error response."""
        return {"success": False, "error": message}

    def _find_lmms_eval_executable(self) -> Optional[str]:
        """Find the lmms-eval executable from known paths."""
        possible_paths = [
            *CONFIG.LMMS_EVAL_PATHS,
            str(Path.home() / ".local/bin/lmms-eval"),
        ]

        workflow_path = self._get_workflow_venv_path()
        if workflow_path:
            possible_paths.insert(0, str(workflow_path))

        return self._test_executable_paths(possible_paths)

    def _get_workflow_venv_path(self) -> Optional[Path]:
        """Get lmms-eval path from workflow virtual environment."""
        try:
            repo_root = self._find_repo_root()
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))

            from workflows.workflow_types import WorkflowVenvType
            from workflows.workflow_venvs import VENV_CONFIGS

            if WorkflowVenvType.EVALS_AUDIO in VENV_CONFIGS:
                venv_config = VENV_CONFIGS[WorkflowVenvType.EVALS_AUDIO]
                bin_dir = "Scripts" if sys.platform == "win32" else "bin"
                return venv_config.venv_path / bin_dir / "lmms-eval"

        except (ImportError, Exception):
            pass

        return None

    def _test_executable_paths(self, paths: list[str]) -> Optional[str]:
        """Return first valid executable path, or None."""
        for path in paths:
            if self._is_valid_executable(path):
                return path
        return None

    def _is_valid_executable(self, path: str) -> bool:
        """Test if a path is a valid lmms-eval executable."""
        try:
            result = subprocess.run(
                [path, "--help"],
                capture_output=True,
                timeout=CONFIG.EXECUTABLE_TIMEOUT,
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            return False

    def _find_repo_root(self) -> Path:
        """Find repository root by traversing up to locate the workflows directory."""
        current = Path(__file__).resolve()
        for _ in range(_MAX_REPO_ROOT_DEPTH):
            if current.parent == current:
                break
            if (current / "workflows").exists() and (current / "workflows").is_dir():
                return current
            current = current.parent
        return Path(__file__).parent.parent.parent.parent.parent

    async def _run_lmms_eval_subprocess(self, lmms_eval_exec: str) -> dict[str, Any]:
        """Run evaluation using lmms-eval subprocess."""
        if self.mock_mode:
            logger.warning(
                "⚠️ lmms-eval subprocess called while in mock mode - returning mock results"
            )
            return await self._run_mock_evaluation()

        try:
            os.environ["OPENAI_API_BASE"] = self.base_url
            os.environ["OPENAI_API_KEY"] = self.api_key

            Path(self.output_dir).mkdir(exist_ok=True, parents=True)

            cmd = self._build_lmms_eval_command(lmms_eval_exec)
            logger.info(f"Running command: {' '.join(cmd)}")
            logger.info(f"  Model: {self.model_name}")
            logger.info(f"  HF repo: {self.hf_model_repo}")
            logger.info(f"  Task: {self.task_name}")
            logger.info(f"  Limit: {self.test_limit} samples")

            return await asyncio.get_running_loop().run_in_executor(
                None, self._execute_subprocess_sync, cmd
            )

        except FileNotFoundError:
            logger.error(f"❌ lmms-eval executable not found: {lmms_eval_exec}")
            return {
                "status": "error",
                "method": "lmms_eval_subprocess",
                "error": f"Executable not found: {lmms_eval_exec}",
                "fallback_used": False,
            }
        except subprocess.TimeoutExpired as e:
            logger.error(f"❌ lmms-eval subprocess timed out: {e}")
            return {
                "status": "error",
                "method": "lmms_eval_subprocess",
                "error": f"Subprocess timed out: {e}",
                "fallback_used": False,
            }
        except Exception as e:
            logger.exception(f"Unexpected error running lmms-eval subprocess: {e}")
            return {
                "status": "error",
                "method": "lmms_eval_subprocess",
                "error": str(e),
                "fallback_used": False,
            }

    def _build_lmms_eval_command(self, lmms_eval_exec: str) -> list[str]:
        """Build lmms-eval command arguments."""
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

        if self.test_limit is not None:
            cmd_args.extend(["--limit", str(self.test_limit)])
        elif self.debug_mode:
            cmd_args.extend(["--limit", "2"])

        return cmd_args

    def _parse_evaluation_results(self, stdout_lines: list[str]) -> dict[str, float]:
        """Parse evaluation metrics from lmms-eval stdout output.

        Parses table format: |task|Version|Filter|n-shot|Metric|dir|Value|±|Stderr|
        Falls back to regex patterns for simple ``metric: value`` formats.
        """
        results: dict[str, float] = {}

        try:
            for line in stdout_lines:
                # Try table format first
                if "|" in line:
                    parsed = self._parse_table_line(line)
                    if parsed:
                        results.update(parsed)
                        continue

                # Fallback: regex for "metric: value" patterns
                for metric in SUPPORTED_METRICS:
                    match = re.search(rf"{metric}[\s:]+([\d.]+)", line.lower())
                    if match:
                        results[metric] = float(match.group(1))

        except Exception as e:
            logger.error(f"Error parsing evaluation results: {e}")

        return results

    @staticmethod
    def _parse_table_line(line: str) -> Optional[dict[str, float]]:
        """Parse a single pipe-delimited table line for supported metrics.

        Returns a dict ``{task_metric: value}`` on success, or ``None``.
        """
        line_lower = line.lower()
        if not any(metric in line_lower for metric in SUPPORTED_METRICS):
            return None

        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) < _MIN_TABLE_PARTS:
            return None

        try:
            task_name = parts[0]
            metric_name = parts[_TABLE_METRIC_INDEX]
            metric_value = float(parts[_TABLE_VALUE_INDEX])
        except (ValueError, IndexError) as e:
            logger.debug(f"Could not parse table line: {line!r}, error: {e}")
            return None

        logger.info(f"Parsed metric: {task_name} {metric_name} = {metric_value}")
        return {f"{task_name}_{metric_name}": metric_value}

    def _find_and_parse_results_files(self) -> dict[str, Any]:
        """Find and parse JSON results files generated by lmms-eval."""
        results = {}

        try:
            output_path = Path(self.output_dir)
            repo_dir_name = self.hf_model_repo.replace("/", "__")
            results_dir = output_path / repo_dir_name

            logger.info(f"Looking for results files in: {results_dir}")

            if results_dir.exists():
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
                    except json.JSONDecodeError as e:
                        logger.error(f"Corrupted JSON in {json_file.name}: {e}")
                    except OSError as e:
                        logger.error(f"Could not read {json_file.name}: {e}")
            else:
                logger.warning(f"Results directory not found: {results_dir}")
                if output_path.exists():
                    existing = list(output_path.iterdir())
                    logger.info(
                        f"Contents of {output_path}: {[p.name for p in existing]}"
                    )

        except Exception as e:
            logger.error(f"Error finding results files: {e}")

        return results

    async def _run_mock_evaluation(self) -> dict[str, Any]:
        """Run a mock evaluation simulating lmms-eval output and file generation."""
        try:
            logger.info("🎭 MOCK MODE: Simulating lmms-eval evaluation...")
            await asyncio.sleep(1.0)

            output_dir = Path(self.output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)

            repo_dir_name = self.hf_model_repo.replace("/", "__")
            results_dir = output_dir / repo_dir_name
            results_dir.mkdir(exist_ok=True, parents=True)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            mock_wer_value = CONFIG.MOCK_WER_VALUE
            mock_stdout_lines = [
                "whisper_tt (model=distil-whisper/distil-large-v3,base_url=http://127.0.0.1:8000,num_concurrent=1), gen_kwargs: (), limit: 0.5, num_fewshot: None, batch_size: 1",
                "|        Tasks         |Version|Filter|n-shot|Metric|   |Value |   |Stderr|",
                "|----------------------|-------|------|-----:|------|---|-----:|---|------|",
                f"|librispeech_test_other|Yaml   |none  |     0|wer   |↓  |{mock_wer_value}|±  |   N/A|",
                "",
            ]

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

            json_filename = f"{timestamp}_results.json"
            json_file_path = results_dir / json_filename

            with open(json_file_path, "w") as f:
                json.dump(mock_json_results, f, indent=2)

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

            parsed_metrics = self._parse_evaluation_results(mock_stdout_lines)
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

    def _execute_subprocess_sync(self, cmd: list[str]) -> dict[str, Any]:
        """Execute lmms-eval subprocess and capture results."""
        logger.info(f"Running command: {shlex.join(cmd)}")

        captured_stdout: list[str] = []
        captured_stderr: list[str] = []

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            text=True,
        )

        stdout_thread = threading.Thread(
            target=self._stream_and_capture,
            args=(process.stdout, logging.INFO, captured_stdout),
        )
        stderr_thread = threading.Thread(
            target=self._stream_and_capture,
            args=(process.stderr, logging.ERROR, captured_stderr),
        )

        stdout_thread.start()
        stderr_thread.start()

        process.wait()
        stdout_thread.join(timeout=_THREAD_JOIN_TIMEOUT)
        stderr_thread.join(timeout=_THREAD_JOIN_TIMEOUT)

        return_code = process.returncode
        success = return_code == 0

        if success:
            logger.info("✅ lmms-eval subprocess completed successfully")
            return self._build_success_result(cmd, captured_stdout)

        logger.error(f"❌ lmms-eval subprocess failed with return code: {return_code}")
        return {
            "status": "error",
            "method": "lmms_eval_subprocess",
            "command": " ".join(cmd),
            "return_code": return_code,
            "fallback_used": False,
            "captured_stderr": captured_stderr[-10:] if captured_stderr else [],
        }

    def _build_success_result(
        self, cmd: list[str], captured_stdout: list[str]
    ) -> dict[str, Any]:
        """Build result dict for a successful subprocess run."""
        result: dict[str, Any] = {
            "status": "success",
            "method": "lmms_eval_subprocess",
            "command": " ".join(cmd),
            "return_code": 0,
            "output_dir": str(self.output_dir),
        }

        eval_results = self._parse_evaluation_results(captured_stdout)
        if eval_results:
            result["evaluation_metrics"] = eval_results
            logger.info(f"Parsed evaluation metrics: {eval_results}")
        else:
            logger.warning("No evaluation metrics found in output")

        json_results = self._find_and_parse_results_files()
        if json_results:
            result["json_results"] = json_results
            logger.info(f"Found JSON results files: {list(json_results.keys())}")

        return result

    @staticmethod
    def _stream_and_capture(pipe: IO[str], level: int, capture_list: list[str]) -> None:
        """Stream subprocess output to logger while capturing lines."""
        with pipe:
            for line in iter(pipe.readline, ""):
                stripped = line.strip()
                if stripped:
                    capture_list.append(stripped)
                    logger.log(level, stripped, extra={"raw": True})
