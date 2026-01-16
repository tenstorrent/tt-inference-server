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
        
        # CRITICAL: Import and patch BEFORE SGLang can load its own modules
        # This must happen before any SGLang imports
        
        # Method 1: Preemptively patch sys.modules
        from .models.tt_llama import TTLlamaForCausalLM
        
        # Create a module object that SGLang will import
        import types
        fake_tt_module = types.ModuleType('tt_llama')
        fake_tt_module.TTLlamaForCausalLM = TTLlamaForCausalLM
        fake_tt_module.LlamaForCausalLM = TTLlamaForCausalLM  # Override both
        fake_tt_module.EntryClass = [TTLlamaForCausalLM]  # Match SGLang's pattern
        
        # Patch all possible module paths BEFORE SGLang loads them
        module_paths = [
            'sglang.srt.models.tt_llama',
            'tt_llama',
        ]
        
        for module_path in module_paths:
            sys.modules[module_path] = fake_tt_module
            logger.info(f"[TT-Plugin] Pre-patched {module_path}")
        
        # Method 2: Hook the model loader directly
        try:
            # This will run when SGLang tries to load models
            import importlib.util
            original_find_spec = importlib.util.find_spec
            
            def patched_find_spec(name, package=None):
                if name == 'sglang.srt.models.tt_llama' or name.endswith('.tt_llama'):
                    logger.info(f"[TT-Plugin] INTERCEPTED find_spec for {name}")
                    # Return our fake module spec
                    spec = importlib.util.spec_from_loader(name, loader=None)
                    return spec
                return original_find_spec(name, package)
            
            importlib.util.find_spec = patched_find_spec
            
        except Exception as e:
            logger.warning(f"[TT-Plugin] Could not patch importlib: {e}")
        
        logger.info("[TT-Plugin] Successfully pre-registered TT-Metal models")
        
    except ImportError as e:
        logger.warning(f"[TT-Plugin] TT-Metal not available: {e}")
    except Exception as e:
        logger.error(f"[TT-Plugin] Error registering TT models: {e}")

# CRITICAL: Register IMMEDIATELY on import, before SGLang can load anything
register_tt_models()

from .models.tt_llama import TTLlamaForCausalLM, TTModels
from .utils.tt_utils import open_mesh_device

__all__ = [
    "TTLlamaForCausalLM",
    "TTModels",
    "open_mesh_device", 
    "register_tt_models",
    "__version__",
]