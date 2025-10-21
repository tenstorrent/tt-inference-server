# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from model_services.base_service import BaseService
from config.settings import settings
from utils.logger import TTLogger


class CNNService(BaseService):
    """CNN-specific service extending BaseService.
    
    This service handles CNN model-specific preprocessing, configuration,
    and inference coordination. It extends the base service with CNN-specific
    functionality while maintaining the common service interface.
    """

    def __init__(self):
        """Initialize CNN service with CNN-specific configuration."""
        super().__init__()
        self.logger = TTLogger()
        self.logger.info(f"Initialized CNNService for model: {settings.model_runner}")
        # CNN-specific initialization can be added here
        self._setup_cnn_configuration()
    
    def _setup_cnn_configuration(self):
        """Setup CNN-specific configuration.
        
        This method can be extended to add CNN-specific preprocessing pipelines,
        model configurations, or optimization settings.
        """
        # CNN-specific configuration setup
        self.logger.debug("CNN-specific configuration setup completed")
        pass
    
    async def pre_process(self, request):
        """CNN-specific preprocessing.
        
        Args:
            request: Input request to preprocess
            
        Returns:
            Preprocessed request
        """
        # CNN models typically don't require additional preprocessing beyond base
        # but this can be extended for CNN-specific needs
        return await super().pre_process(request)
    
    def post_process(self, result):
        """CNN-specific postprocessing.
        
        Args:
            result: Result from inference
            
        Returns:
            Postprocessed result
        """
        # CNN-specific postprocessing can be added here
        return super().post_process(result)