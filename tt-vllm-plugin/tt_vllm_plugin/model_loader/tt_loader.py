# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect

from torch import nn

from vllm.config import ModelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import get_model_architecture

logger = init_logger("vllm.tt_vllm_plugin.model_loader.tt_loader")


def _method_accepts_param(method, param_name: str) -> bool:
    """Check if a method accepts a specific parameter name."""
    try:
        sig = inspect.signature(method)
        # Check if param is explicitly defined or if **kwargs is present
        for name, param in sig.parameters.items():
            if name == param_name:
                return True
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                return True  # Method has **kwargs, so it accepts any keyword arg
        return False
    except (ValueError, TypeError):
        return False


class TTModelLoader(BaseModelLoader):

    def load_model(self, vllm_config: VllmConfig,
                   model_config: ModelConfig) -> nn.Module:
        """Load a model with the given configurations."""

        logger.info("Loading model on TT platform...")

        device_config = vllm_config.device_config
        scheduler_config = vllm_config.scheduler_config

        model_class, _ = get_model_architecture(model_config)
        logger.info(f"Resolved model class: {model_class.__name__}")
        
        # Check if model class has initialize_vllm_model method
        # If not, it's likely vLLM's native implementation and we should let vLLM handle it
        if not hasattr(model_class, 'initialize_vllm_model'):
            logger.error(
                f"Model class {model_class.__name__} does not have initialize_vllm_model method. "
                f"Architecture: {model_config.hf_config.architectures}. "
                "This might be vLLM's native implementation instead of TT-specific implementation."
            )
            raise ValueError(
                f"Model class {model_class.__name__} does not have initialize_vllm_model method. "
                f"Architecture resolved from: {model_config.hf_config.architectures}. "
                "TT plugin requires TT-specific model implementations with initialize_vllm_model. "
                "Ensure your model is registered in vLLM's ModelRegistry with a TT-prefixed architecture name. "
                "For BGE model, ensure 'TTBertModel' -> 'BGEForEmbedding' is registered in ModelRegistry."
            )
        
        # Fix: Check if override_tt_config exists before calling .get()
        optimizations = None
        if model_config.override_tt_config:
            optimizations = model_config.override_tt_config.get("optimizations", None)
        
        if optimizations is not None:
            assert optimizations in [
                "performance", "accuracy"
            ], f"""Invalid optimizations configuration `{optimizations}`, 
            allowed values are 'performance' or 'accuracy'"""

        data_parallel = vllm_config.parallel_config.data_parallel_size
        max_batch_size = scheduler_config.max_num_seqs * data_parallel

        # Build kwargs for initialize_vllm_model
        # Only pass vllm_config if the method accepts it (needed for pooling models like BGE)
        init_kwargs = {
            "max_seq_len": model_config.max_model_len,
            "tt_data_parallel": data_parallel,
            "optimizations": optimizations,
        }
        
        if _method_accepts_param(model_class.initialize_vllm_model, "vllm_config"):
            init_kwargs["vllm_config"] = vllm_config

        model = model_class.initialize_vllm_model(
            model_config.hf_config,
            device_config.device,
            max_batch_size,
            **init_kwargs,
        )
        return model

    def download_model(self, model_config: ModelConfig) -> None:
        """Download a model so that it can be immediately loaded."""
        raise NotImplementedError

    def load_weights(self, model: nn.Module,
                     model_config: ModelConfig) -> None:
        """Load weights into a model. This standalone API allows 
        inplace weights loading for an already-initialized model"""
        raise NotImplementedError

