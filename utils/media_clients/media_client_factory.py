# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import logging
from enum import Enum
from typing import Dict, Type

from .audio_client import AudioClientStrategy
from .base_strategy_interface import BaseMediaStrategy
from .cnn_client import CnnClientStrategy
from .embedding_client import EmbeddingClientStrategy
from .image_client import ImageClientStrategy
from .tts_client import TtsClientStrategy
from .video_client import VideoClientStrategy

logger = logging.getLogger(__name__)

STRATEGY_MAP: Dict[str, Type[BaseMediaStrategy]] = {
    "CNN": CnnClientStrategy,
    "IMAGE": ImageClientStrategy,
    "AUDIO": AudioClientStrategy,
    "EMBEDDING": EmbeddingClientStrategy,
    "TEXT_TO_SPEECH": TtsClientStrategy,
    "VIDEO": VideoClientStrategy,
}


class MediaTaskType(Enum):
    """Enumeration of supported media task types."""

    EVALUATION = "evaluation"
    BENCHMARK = "benchmark"


class MediaClientFactory:
    """Factory class for creating media client instances based on configuration."""

    @staticmethod
    def _create_strategy(
        model_spec, all_params, device, output_path, service_port
    ) -> BaseMediaStrategy:
        """
        Create appropriate strategy based on model type.

        Args:
            model_spec: Model specification containing model type
            all_params: All parameters for the strategy
            device: Device information
            output_path: Output path for results
            service_port: Service port number

        Returns:
            BaseMediaStrategy: Appropriate strategy instance

        Raises:
            ValueError: If model type is not supported
        """
        logger.info(
            f"Creating media strategy for model type: {model_spec.model_type.name}"
        )
        strategy = STRATEGY_MAP.get(model_spec.model_type.name)
        if strategy:
            logger.info(f"Using strategy: {strategy.__name__} for client.")
            return strategy(all_params, model_spec, device, output_path, service_port)

        raise ValueError(
            f"Unsupported model type: {model_spec.model_type.name}. Supported types: {', '.join(STRATEGY_MAP.keys())}"
        )

    @staticmethod
    def run_media_task(
        model_spec,
        all_params,
        device,
        output_path,
        service_port,
        task_type: MediaTaskType,
    ) -> int:
        """
        Generic function to run media tasks (evaluation or benchmarking).

        Args:
            model_spec: Model specification containing model type
            all_params: All parameters for the strategy
            device: Device information
            output_path: Output path for results
            service_port: Service port number
            task_type: MediaTaskType enum value (EVALUATION or BENCHMARK)

        Returns:
            int: Return code (0 for success, 1 for failure)
        """
        task_name = f"{model_spec.model_type.name} {task_type}"
        logger.info(
            f"Running {task_name} for model: {model_spec.model_name} on device: {device.name}"
        )

        try:
            # Create appropriate test case
            test_case = MediaClientFactory._create_strategy(
                model_spec, all_params, device, output_path, service_port
            )

            # Run the specified task using test_case
            if task_type == MediaTaskType.EVALUATION:
                test_case.run_eval()
            else:
                test_case.run_benchmark()

            logger.info(f"✅ Completed {task_name}")
            return 0  # Success
        except Exception as e:
            logger.error(f"❌ {model_spec.model_type.name} {task_type} failed: {e}")
            return 1  # Failure
