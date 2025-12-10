# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# Standard library imports
import json
import logging
import sys
import time
from pathlib import Path

# Third-party imports
from transformers import AutoTokenizer

# Local imports
from .base_strategy_interface import BaseMediaStrategy
from .test_status import AudioTestStatus

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class EmbeddingClientStrategy(BaseMediaStrategy):
    """Strategy for embedding models."""

    def __init__(self, all_params, model_spec, device, output_path, service_port):
        super().__init__(all_params, model_spec, device, output_path, service_port)

        # Initialize tokenizer
        self.tokenizer = None
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_spec.hf_model_repo)
            logger.info(f"✅ Loaded tokenizer for {model_spec.hf_model_repo}")
        except Exception as e:
            logger.warning(
                f"⚠️ Could not load tokenizer for {model_spec.hf_model_repo}: {e}"
            )
            logger.info("📝 Falling back to word-based token counting")

    def run_eval(self) -> None:
        """Run evaluations for the model."""
        status_list = []

        logger.info(
            f"Running evals for model: {self.model_spec.model_name} on device: {self.device.name}"
        )
        try:
            health_status, runner_in_use = self.get_health()
            if health_status:
                logger.info("Health check passed.")
            else:
                logger.error("Health check failed.")
                return

            logger.info(f"Runner in use: {runner_in_use}")

        except Exception as e:
            logger.error(f"Eval execution encountered an error: {e}")
            return

        logger.info("Generating eval report...")
        benchmark_data = {}

        # Write benchmark_data to JSON file
        eval_filename = (
            Path(self.output_path)
            / f"eval_{self.model_spec.model_id}"
            / self.model_spec.hf_model_repo.replace("/", "__")
            / f"results_{time.time()}.json"
        )
        # Create directory structure if it doesn't exist
        eval_filename.parent.mkdir(parents=True, exist_ok=True)

        with open(eval_filename, "w") as f:
            json.dump(benchmark_data, f, indent=4)
        logger.info(f"Evaluation data written to: {eval_filename}")

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
                return []

            logger.info(f"Runner in use: {runner_in_use}")

            return True
        except Exception as e:
            logger.error(f"Benchmark execution encountered an error: {e}")
            return []
