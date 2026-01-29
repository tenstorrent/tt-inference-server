"""
SGLang TT-Metal Plugin
"""

import logging
import os
import sys

# Set version at module level
__version__ = "0.1.0"

# Force CPU-only mode for sgl_kernel to avoid CUDA dependency issues
os.environ["SGL_DISABLE_CUDA_KERNEL"] = "1" 
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Prevent torch compilation issues
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

logger = logging.getLogger(__name__)

def register_tt_models():
    """Register TT-Metal models with SGLang's model registry."""
    try:
        # Check if TT-Metal is available
        import ttnn
        logger.info("[TT-Plugin] TT-Metal (ttnn) is available")
        
        # Import all TT model classes
        from .models.tt_llm import (
            TTLlamaForCausalLM,
            TTQwenForCausalLM,
            TTMistralForCausalLM,
            TTGptOssForCausalLM,
        )
        
        # Mapping from HuggingFace architecture names to TT model classes
        TT_MODEL_REGISTRY = {
            "LlamaForCausalLM": TTLlamaForCausalLM,      # Llama-3.1-8B, Llama-3.1-70B, etc.
            "Qwen2ForCausalLM": TTQwenForCausalLM,       # Qwen2.5-7B, Qwen2.5-14B, etc.
            "MistralForCausalLM": TTMistralForCausalLM,  # Mistral-7B
            "GptOssForCausalLM": TTGptOssForCausalLM,    # GPT-OSS
        }
        
        # CRITICAL: Directly patch SGLang's ModelRegistry
        try:
            from sglang.srt.models.registry import ModelRegistry
            
            # Override all supported architectures in the registry
            for arch_name, tt_class in TT_MODEL_REGISTRY.items():
                ModelRegistry.models[arch_name] = tt_class
                logger.info(f"[TT-Plugin] Registered {arch_name} -> {tt_class.__name__}")
            
            # Also patch _try_load_model_cls to intercept model loading
            original_try_load = ModelRegistry._try_load_model_cls
            
            @staticmethod
            def patched_try_load_model_cls(architectures):
                for arch in architectures:
                    if arch in TT_MODEL_REGISTRY:
                        tt_class = TT_MODEL_REGISTRY[arch]
                        logger.info(f"[TT-Plugin] Intercepted load for {arch}, returning {tt_class.__name__}")
                        return tt_class
                return original_try_load(architectures)
            
            ModelRegistry._try_load_model_cls = patched_try_load_model_cls
            logger.info(f"[TT-Plugin] ✓ Successfully patched ModelRegistry with {len(TT_MODEL_REGISTRY)} TT models")
            
        except ImportError as e:
            logger.warning(f"[TT-Plugin] Could not import ModelRegistry: {e}")
        
    except ImportError as e:
        logger.warning(f"[TT-Plugin] TT-Metal not available: {e}")
    except Exception as e:
        logger.error(f"[TT-Plugin] Error registering TT models: {e}")

# CRITICAL: Register IMMEDIATELY on import, before SGLang can load anything
register_tt_models()

from .models.tt_llm import (
    TTModels,
    TTLlamaForCausalLM,
    TTQwenForCausalLM,
    TTMistralForCausalLM,
    TTGptOssForCausalLM,
)
from .utils.tt_utils import open_mesh_device

__all__ = [
    "TTModels",
    "TTLlamaForCausalLM",
    "TTQwenForCausalLM",
    "TTMistralForCausalLM",
    "TTGptOssForCausalLM",
    "open_mesh_device", 
    "register_tt_models",
    "__version__",
]