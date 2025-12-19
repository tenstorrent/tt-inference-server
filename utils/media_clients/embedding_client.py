# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# Standard library imports
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from workflows.utils import get_num_calls
from workflows.workflow_types import WorkflowVenvType
from workflows.workflow_venvs import VENV_CONFIGS

# Local imports
from .base_strategy_interface import BaseMediaStrategy
from .test_status import AudioTestStatus, EmbeddingTestStatus

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

BENCHMARK_RESULT_START = "============ Serving Benchmark Result ============"
BENCHMARK_RESULT_END = "=================================================="
OPENAI_API_KEY = "your-secret-key"


class EmbeddingClientStrategy(BaseMediaStrategy):
    """Strategy for embedding models."""

    def __init__(self, all_params, model_spec, device, output_path, service_port):
        super().__init__(all_params, model_spec, device, output_path, service_port)
        self.model = self.model_spec.hf_model_repo
        self.isl = int(
            model_spec.device_model_spec.env_vars.get("MAX_MODEL_LENGTH", 1024)
        )
        self.concurrency = self.model_spec.device_model_spec.max_concurrency

    def run_eval(self) -> None:
        """Run evaluations for the model."""
        logger.info(
            f"Running evals for model: {self.model_spec.model_name} on device: {self.device.name}"
        )
        try:
            health_status, runner_in_use = self.get_health()
            if health_status:
                logger.info("Health check passed.")
            else:
                logger.error("Health check failed.")
                raise

            logger.info(f"Runner in use: {runner_in_use}")

            return True

        except Exception as e:
            logger.error(f"Eval execution encountered an error: {e}")
            raise

    def run_benchmark(self, attempt=0) -> list[AudioTestStatus]:
        """Run benchmarks for the model."""
        logger.info(
            f"Running benchmarks for model: {self.model_spec.model_name} on device: {self.device.name}"
        )
        try:
            health_status, runner_in_use = self.get_health()
            if health_status:
                logger.info(f"Health check passed. Runner in use: {runner_in_use}")
            else:
                logger.error("Health check failed.")
                raise

            logger.info(f"Runner in use: {runner_in_use}")

            # Get num_calls from benchmark parameters
            num_calls = get_num_calls(self)

            status_list = []
            status_list = self._run_embedding_transcription_benchmark(num_calls)

            return self._generate_report(status_list)
        except Exception as e:
            logger.error(f"Benchmark execution encountered an error: {e}")
            raise

    def _run_embedding_transcription_benchmark(
        self, num_calls: int
    ) -> list[EmbeddingTestStatus]:
        """Run embedding transcription benchmark."""

        # Use the venv's python and vllm executable directly
        venv_config = VENV_CONFIGS.get(WorkflowVenvType.BENCHMARKS_EMBEDDING)
        vllm_exec = venv_config.venv_path / "bin" / "vllm"

        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

        cmd = [
            str(vllm_exec),
            "bench",
            "serve",
            "--model",
            self.model,
            "--random-input-len",
            str(self.isl),
            "--num-prompts",
            str(num_calls),
            "--backend",
            "openai-embeddings",
            "--endpoint",
            "/v1/embeddings",
            "--dataset-name",
            "random",
            "--save-result",
            "--result-dir",
            "benchmark",
        ]

        logger.info(f"Running embedding benchmark with {num_calls} calls...")

        output = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout

        return self._parse_embedding_benchmark_output(output)

    def _parse_embedding_benchmark_output(self, output: str) -> dict:
        """Parse benchmark metrics from output."""
        if BENCHMARK_RESULT_START not in output:
            logger.warning("Benchmark result section not found in output.")
            return {}

        section = output.split(BENCHMARK_RESULT_START, 1)[1]
        # Optionally, stop at the next '====' line after the section
        if BENCHMARK_RESULT_END in section:
            section = section.split(BENCHMARK_RESULT_END, 1)[0]
        section = section.strip()

        # Handles empty string after strip
        if not section:
            logger.warning("Benchmark result section is empty after parsing.")
            return {}

        # Parse the section into a dictionary, stripping units from keys
        metrics = {}
        for line in section.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                key_clean = re.sub(r"\s*\([^)]*\)", "", key).strip()
                metrics[key_clean] = value.strip()
        logger.info(f"Parsed benchmark metrics: {metrics}")

        return metrics

    def _generate_report(self, metrics: dict):
        """Generate benchmark report."""
        logger.info("Generating benchmark report...")
        result_filename = (
            Path(self.output_path)
            / f"benchmark_{self.model_spec.model_id}_{time.time()}.json"
        )
        result_filename.parent.mkdir(parents=True, exist_ok=True)

        total_input_tokens = float(metrics.get("Total input tokens", 0))
        benchmark_duration = float(metrics.get("Benchmark duration", 1.0))
        successful_requests = int(metrics.get("Successful requests", 0))
        failed_requests = int(metrics.get("Failed requests", 0))
        mean_e2el = float(metrics.get("Mean E2EL", 0.0))
        req_tput = float(metrics.get("Request throughput", 0.0))

        tput_prefill = (
            total_input_tokens / benchmark_duration if benchmark_duration else 0.0
        )

        report_data = {
            "benchmarks": {
                "isl": self.isl,
                "concurrency": self.concurrency,
                "num_requests": successful_requests + failed_requests,
                "tput_user": tput_prefill / float(self.concurrency)
                if self.concurrency
                else 0.0,
                "tput_prefill": tput_prefill,
                "e2el": mean_e2el,
                "req_tput": req_tput,
            },
            "model": self.model_spec.model_name,
            "device": self.device.name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "task_type": "embedding",
        }

        with open(result_filename, "w") as f:
            json.dump(report_data, f, indent=4)
        logger.info(f"Report generated: {result_filename}")
