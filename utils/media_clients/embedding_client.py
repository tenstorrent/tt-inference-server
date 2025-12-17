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

# Third-party imports
# Local imports
from .base_strategy_interface import BaseMediaStrategy
from .test_status import AudioTestStatus, EmbeddingTestStatus

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class EmbeddingClientStrategy(BaseMediaStrategy):
    """Strategy for embedding models."""

    def __init__(self, all_params, model_spec, device, output_path, service_port):
        super().__init__(all_params, model_spec, device, output_path, service_port)
        self.model = self.model_spec.hf_model_repo
        self.isl = 120  # input sequence length
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
        venv_dir = Path(".workflow_venvs/.venv_benchmarks_embedding")
        vllm_exec = venv_dir / "bin" / "vllm"
        os.environ["OPENAI_API_KEY"] = "your-secret-key"

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

        results = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = results.stdout

        # Extract the relevant benchmark result section
        start_marker = "============ Serving Benchmark Result ============"
        section = None
        if start_marker in output:
            section = output.split(start_marker, 1)[1]
            # Optionally, stop at the next '====' line after the section
            end_marker = "=================================================="
            if end_marker in section:
                section = section.split(end_marker, 1)[0]
            section = section.strip()
        else:
            logger.warning("Benchmark result section not found in output.")

        # Parse the section into a dictionary, stripping units from keys

        metrics = {}
        if section:
            for line in section.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    key_clean = re.sub(r"\s*\([^)]*\)", "", key).strip()
                    metrics[key_clean] = value.strip()
        logger.info(f"Parsed benchmark metrics: {metrics}")

        return metrics

    def _generate_report(self, metrics: dict) -> list[EmbeddingTestStatus]:
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

        return True
