# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# Standard library imports
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from workflows.workflow_types import ReportCheckTypes, WorkflowVenvType
from workflows.workflow_venvs import VENV_CONFIGS

# Local imports
from .base_strategy_interface import BaseMediaStrategy, PerfCheck
from .test_status import AudioTestStatus
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

BENCHMARK_RESULT_START = "============ Serving Benchmark Result ============"
BENCHMARK_RESULT_END = "=================================================="
OPENAI_API_KEY = "your-secret-key"
MTEB_TASKS = ["STS12"]  # add AmazonCounterfactualClassification for classification

# vLLM ``bench serve`` writes its detailed result JSON under this dir/filename
# combination; we set both flags explicitly so the location is deterministic
# (vLLM auto-generates a timestamped filename otherwise).
VLLM_RESULT_DIR = "benchmark"
VLLM_RESULT_FILENAME = "embedding_bench_result.json"
# Percentiles we want vLLM to compute on the per-request E2EL samples. vLLM
# does not save the raw e2el array for the openai-embeddings backend, but it
# does serialise the aggregate p{N}_e2el_ms fields when --metric-percentiles
# is passed, which is enough to populate latency_p50/p90/p95 in our report.
VLLM_E2EL_PERCENTILES = (50, 90, 95)


def _ms_to_seconds_or_none(value_ms):
    """Convert a millisecond reading to seconds, preserving ``None`` for missing values."""
    return value_ms / 1000.0 if value_ms is not None else None


class EmbeddingClientStrategy(BaseMediaStrategy):
    """Strategy for embedding models."""

    def __init__(self, all_params, model_spec, device, output_path, service_port):
        super().__init__(all_params, model_spec, device, output_path, service_port)
        self.model = self.model_spec.hf_model_repo
        self.isl = int(
            model_spec.device_model_spec.env_vars.get("VLLM__MAX_MODEL_LENGTH", 1024)
        )
        self.num_calls = 1000
        self.dimensions = 1000
        self.concurrency = int(
            self.model_spec.device_model_spec.env_vars.get("VLLM__MAX_NUM_SEQS", 1)
        )

    def run_eval(self) -> None:
        """Run evaluations for the model."""
        logger.info(
            f"Running evals for model: {self.model_spec.model_name} on device: {self.device.name}"
        )
        try:
            self.require_health()
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
            self.require_health()
            status_list = self._run_embedding_transcription_benchmark()
            self._generate_benchmarking_report(status_list)
        except Exception as e:
            logger.error(f"Benchmark execution encountered an error: {e}")
            raise

    def _run_embedding_transcription_benchmark(self) -> dict:
        """Run embedding transcription benchmark."""

        # Use the venv's python and vllm executable directly
        venv_config = VENV_CONFIGS.get(WorkflowVenvType.BENCHMARKS_VLLM)
        vllm_exec = venv_config.venv_path / "bin" / "vllm"

        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

        percentiles_arg = ",".join(str(p) for p in VLLM_E2EL_PERCENTILES)
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
            "--percentile-metrics",
            "e2el",
            "--metric-percentiles",
            percentiles_arg,
            "--save-result",
            "--result-dir",
            VLLM_RESULT_DIR,
            "--result-filename",
            VLLM_RESULT_FILENAME,
        ]

        logger.info(f"Running embedding benchmark with {self.num_calls} calls...")

        output = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout

        metrics = self._parse_embedding_benchmark_output(output)
        metrics.update(self._read_vllm_result_file_percentiles())
        return metrics

    def _read_vllm_result_file_percentiles(self) -> dict:
        """Pull aggregate E2EL percentiles out of vLLM's ``--save-result`` JSON."""
        result_path = Path(VLLM_RESULT_DIR) / VLLM_RESULT_FILENAME
        try:
            with result_path.open("r", encoding="utf-8") as fp:
                payload = json.load(fp)
        except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Could not read vLLM result file at %s: %s. "
                "Tail latency percentiles will be reported as null.",
                result_path,
                exc,
            )
            return {}

        keys = [f"p{p}_e2el_ms" for p in VLLM_E2EL_PERCENTILES]
        return {key: payload[key] for key in keys if key in payload}

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

    def _calculate_performance_check(
        self,
        tput_user_value: Optional[float] = None,
        tput_prefill_value: Optional[float] = None,
        e2el_ms_value: Optional[float] = None,
    ) -> ReportCheckTypes:
        """Embedding perf check: compares throughput / E2EL vs configured targets.

        ``e2el_ms_value`` is already in milliseconds (vLLM ``Mean E2EL``), and
        ``targets.e2el_ms`` is also in ms — no unit conversion needed.
        """
        targets = self.get_performance_targets()
        logger.info(f"Performance targets: {targets}")
        return self.calculate_performance_check(
            checks=[
                PerfCheck(
                    "tput_user",
                    tput_user_value,
                    targets.tput_user,
                    lower_is_better=False,
                ),
                PerfCheck(
                    "tput_prefill",
                    tput_prefill_value,
                    targets.tput_prefill,
                    lower_is_better=False,
                ),
                PerfCheck(
                    "e2el_ms", e2el_ms_value, targets.e2el_ms, lower_is_better=True
                ),
            ],
            tolerance=targets.tolerance,
        )

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
        tput_user = tput_prefill / float(self.concurrency) if self.concurrency else 0.0
        performance_check = self._calculate_performance_check(
            tput_user_value=tput_user,
            tput_prefill_value=tput_prefill,
            e2el_ms_value=mean_e2el,
        )

        tail_latencies = {
            f"latency_p{p}": _ms_to_seconds_or_none(metrics.get(f"p{p}_e2el_ms"))
            for p in VLLM_E2EL_PERCENTILES
        }

        report_data = {
            "benchmarks": {
                "isl": self.isl,
                "concurrency": self.concurrency,
                "num_requests": successful_requests + failed_requests,
                "tput_user": tput_user,
                "tput_prefill": tput_prefill,
                "e2el": mean_e2el,
                "req_tput": req_tput,
                **tail_latencies,
            },
            "model": self.model_spec.model_name,
            "device": self.device.name.lower(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "task_type": "embedding",
            "performance_check": performance_check,
        }

        with open(result_filename, "w") as f:
            json.dump(report_data, f, indent=4)
        logger.info(f"Report generated: {result_filename}")

    def _run_embedding_transcription_eval(self) -> None:
        """Run embedding transcription evaluation."""

        import mteb
        import numpy as np
        from mteb.models.model_implementations.openai_models import OpenAIModel
        from mteb.models.model_meta import ModelMeta
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

        # Attach ModelMeta so MTEB can extract model name and revision
        model_meta = ModelMeta(
            name=model_name,
            revision=None,
            embed_dim=self.dimensions,
            max_tokens=self.isl,
            open_weights=False,
            loader=None,
            loader_kwargs={},
            framework=[],
            similarity_fn_name=None,
            use_instructions=None,
            # Required fields added for MTEB schema compatibility
            release_date=None,
            languages=[],
            n_parameters=None,
            memory_usage_mb=None,
            license=None,
            public_training_code=None,
            public_training_data=None,
            training_datasets=None,
        )
        model.mteb_model_meta = model_meta

        # Select tasks and run evaluation
        tasks = mteb.get_tasks(tasks=MTEB_TASKS)

        logger.info("Running embedding transcription evaluation with STS12...")
        results = mteb.evaluate(
            model,
            tasks=tasks,
            encode_kwargs={"batch_size": 1},
            cache=None,
            overwrite_strategy="always",
        )
        logger.info(f"Evaluation results: {results}")
        return self._parse_embedding_evals_output(results)

    def _parse_embedding_evals_output(self, results: dict) -> dict:
        """Parse embedding evaluation results and extract key metrics from scores['test']."""
        scores = {}
        try:
            scores = results.task_results[0].scores["test"]
            if isinstance(scores, list) and len(scores) > 0:
                scores = scores[0]
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
        logger.info(f"Parsed evaluation results: {report_data}")
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

        task = self.all_params.tasks[0]
        report_data = {
            "model": self.model_spec.model_name,
            "device": self.device.name.lower(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "task_type": "embedding",
            "task_name": task.task_name,
            "tolerance": task.score.tolerance,
            "published_score": task.score.published_score,
            "score": metrics.get("main_score"),
            "published_score_ref": task.score.published_score_ref,
            "accuracy_check": ReportCheckTypes.NA,
            "performance_check": self._calculate_performance_check(),
        }
        report_data.update(metrics)

        report_data = [report_data]

        with open(result_filename, "w") as f:
            json.dump(report_data, f, indent=4)
        logger.info(f"Report generated: {result_filename}")
