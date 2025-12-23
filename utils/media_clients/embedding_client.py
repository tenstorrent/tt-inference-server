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
MTEB_TASKS = ["STS12"]  # add AmazonCounterfactualClassification for classification


class EmbeddingClientStrategy(BaseMediaStrategy):
    """Strategy for embedding models."""

    def __init__(self, all_params, model_spec, device, output_path, service_port):
        super().__init__(all_params, model_spec, device, output_path, service_port)
        self.model = self.model_spec.hf_model_repo
        self.isl = int(
            model_spec.device_model_spec.env_vars.get("MAX_MODEL_LENGTH", 1024)
        )
        self.num_calls = 1000
        self.dimensions = 1000
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

            logger.info("Running embedding eval...")

            status_list = self._run_embedding_transcription_eval()

            self._generate_evals_report(status_list)

        except Exception as e:
            logger.error(f"Eval execution encountered an error: {e}")
            raise

    def run_benchmark(self) -> list[AudioTestStatus]:
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

            status_list = []
            status_list = self._run_embedding_transcription_benchmark()

            self._generate_benchmarking_report(status_list)

        except Exception as e:
            logger.error(f"Benchmark execution encountered an error: {e}")
            raise

    def _run_embedding_transcription_benchmark(self) -> list[EmbeddingTestStatus]:
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
            str(self.num_calls),
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

        logger.info(f"Running embedding benchmark with {self.num_calls} calls...")

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

    def _generate_benchmarking_report(self, metrics: dict):
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

    def _run_embedding_transcription_eval(self) -> None:
        """Run embedding transcription evaluation."""

        import mteb
        import numpy as np
        from mteb.models.model_implementations.openai_models import OpenAIModel
        from openai import OpenAI

        model_name = self.model

        # Currently only single string encoding is supported
        def single_string_encode(self, inputs, **kwargs):
            sentences = [text for batch in inputs for text in batch["text"]]
            all_embeddings = []
            for sentence in sentences:
                response = self._client.embeddings.create(
                    input=sentence,
                    model=model_name,
                    encoding_format="float",
                    dimensions=self._embed_dim if self._embed_dim else None,
                )
                all_embeddings.extend(self._to_numpy(response))
            return np.array(all_embeddings)

        client = OpenAI(
            base_url=f"{self.base_url}/v1",
            api_key=OPENAI_API_KEY,
        )

        # Create the model wrapper
        model = OpenAIModel(
            model_name=model_name,
            max_tokens=self.isl,
            embed_dim=self.dimensions,
            client=client,
        )
        model.encode = single_string_encode.__get__(model, type(model))

        # Select tasks and run evaluation
        tasks = mteb.get_tasks(tasks=MTEB_TASKS)

        logger.info("Running embedding transcription evaluation with STS12...")
        results = mteb.evaluate(
            model, tasks=tasks, show_progress_bar=True, encode_kwargs={"batch_size": 1}
        )
        return self._parse_embedding_evals_output(results)

    def _parse_embedding_evals_output(self, results: dict) -> dict:
        """Parse embedding evaluation results and extract key metrics from scores['test']."""
        scores = {}
        try:
            scores = results.task_results[0].scores["test"]
        except Exception as e:
            logger.error(f"Could not extract scores['test']: {e}")
            raise

        # Extract the required metrics
        keys = [
            "pearson",
            "spearman",
            "cosine_pearson",
            "cosine_spearman",
            "manhattan_pearson",
            "manhattan_spearman",
            "euclidean_pearson",
            "euclidean_spearman",
            "main_score",
            "languages",
        ]
        report_data = {k: scores.get(k) for k in keys if k in scores}
        return report_data

    def _generate_evals_report(self, metrics: dict):
        """Generate evals report, attaching metrics to report_data."""
        logger.info("Generating evals report...")
        result_filename = (
            Path(self.output_path)
            / f"eval_{self.model_spec.model_id}"
            / self.model_spec.hf_model_repo.replace("/", "__")
            / f"results_{time.time()}.json"
        )
        result_filename.parent.mkdir(parents=True, exist_ok=True)

        report_data = {
            "model": self.model_spec.model_name,
            "device": self.device.name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "task_type": "embedding",
            "task_name": self.all_params.tasks[0].task_name,
        }
        # Attach metrics dict
        report_data.update(metrics)

        report_data = [report_data]

        with open(result_filename, "w") as f:
            json.dump(report_data, f, indent=4)
        logger.info(f"Report generated: {result_filename}")
