# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# Standard library imports
import logging
import sys
from pathlib import Path

# Third-party imports
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

            return True
        except Exception as e:
            logger.error(f"Benchmark execution encountered an error: {e}")
            raise
