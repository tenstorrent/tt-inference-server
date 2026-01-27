# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from domain.text_embedding_request import TextEmbeddingRequest

from tt_model_runners.bge_large_en.ttnn_implementation import TTNNBGELargeENRunner


class VLLMBGELargeENRunner:
    """
    Facade class that switches between VLLM and TTNN implementations based on settings.

    If use_vllm_bge is True, uses VLLM implementation.
    If use_vllm_bge is False, uses TTNN implementation.
    """

    def __init__(self, device_id: str, num_torch_threads: int = 1):
        from config.settings import get_settings

        settings = get_settings()
        self.use_vllm = settings.use_vllm_bge

        if self.use_vllm:
            # Import VLLM implementation only when needed
            from tt_model_runners.bge_large_en.vllm_implementation import (
                VLLMBGEImplementation,
            )

            self._impl = VLLMBGEImplementation(device_id, num_torch_threads)
        else:
            self._impl = TTNNBGELargeENRunner(device_id, num_torch_threads)

    @property
    def logger(self):
        """Delegate logger access to implementation."""
        return self._impl.logger

    @property
    def settings(self):
        """Delegate settings access to implementation."""
        return self._impl.settings

    @property
    def device_id(self):
        """Delegate device_id access to implementation."""
        return self._impl.device_id

    async def warmup(self) -> bool:
        """Delegate warmup to the selected implementation."""
        return await self._impl.warmup()

    def run(self, requests: list[TextEmbeddingRequest]):
        """Delegate run to the selected implementation."""
        return self._impl.run(requests)

    def set_device(self):
        """Delegate set_device to the selected implementation."""
        return self._impl.set_device()

    def is_request_batchable(self, request, batch=None):
        """Delegate is_request_batchable to the selected implementation."""
        return self._impl.is_request_batchable(request, batch)

    def get_pipeline_device_params(self):
        """Delegate get_pipeline_device_params to the selected implementation if available."""
        if hasattr(self._impl, "get_pipeline_device_params"):
            return self._impl.get_pipeline_device_params()
        return {}

    def close_device(self) -> bool:
        """Delegate close_device to the selected implementation if available."""
        if hasattr(self._impl, "close_device"):
            return self._impl.close_device()
        return True
