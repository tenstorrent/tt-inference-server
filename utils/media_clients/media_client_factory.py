# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import logging
from typing import Dict, Type
from .base_strategy_interface import BaseMediaStrategy
from .image_client import ImageClientStrategy
from .audio_client import AudioClientStrategy

logger = logging.getLogger(__name__)

STRATEGY_MAP: Dict[str, Type[BaseMediaStrategy]] = {
    "CNN": ImageClientStrategy,
    "AUDIO": AudioClientStrategy,
}

class MediaClientFactory:
    """Factory class for creating media client instances based on configuration."""

    @staticmethod
    def create_strategy(
        model_spec,
        all_params,
        device,
        output_path,
        service_port
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
        logger.info(f"Creating media strategy for model type: {model_spec.model_type.name}")
        strategy = STRATEGY_MAP.get(model_spec.model_type.name)
        if strategy:
            return strategy(all_params, model_spec, device, output_path, service_port)

        raise ValueError(
                f"Unsupported model type: {model_spec.model_type.name}. Supported types: {', '.join(STRATEGY_MAP.keys())}"
            )