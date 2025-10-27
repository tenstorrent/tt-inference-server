# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from .media_strategy_interface import MediaStrategyInterface
from .image_client import ImageClientStrategy
from .audio_client import AudioClientStrategy
from workflows.model_spec import ModelType


class MediaClientFactory:
    """Factory class for creating media client instances based on configuration."""

    @staticmethod
    def create_strategy(
        model_spec,
        all_params,
        device,
        output_path,
        service_port
    ) -> MediaStrategyInterface:
        """
        Create appropriate strategy based on model type.

        Args:
            model_spec: Model specification containing model type
            all_params: All parameters for the strategy
            device: Device information
            output_path: Output path for results
            service_port: Service port number

        Returns:
            MediaStrategyInterface: Appropriate strategy instance

        Raises:
            ValueError: If model type is not supported
        """
        if model_spec.model_type.name == "CNN":
            return ImageClientStrategy(
                all_params, model_spec, device, output_path, service_port
            )
        elif model_spec.model_type.name == "AUDIO":
            return AudioClientStrategy(
                all_params, model_spec, device, output_path, service_port
            )
        else:
            raise ValueError(
                f"Unsupported model type: {model_spec.model_type.name}. "
                f"Supported types: CNN, AUDIO"
            )