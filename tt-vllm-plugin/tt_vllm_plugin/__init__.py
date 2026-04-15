# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


def register():
    # Delegate to the vLLM fork's built-in TTPlatform rather than the plugin's
    # own platform.py.  The fork's TTPlatform is maintained in lockstep with
    # the fork's vLLM APIs (arg_utils, envs, scheduler, worker) and uses
    # PlatformEnum.TT so that vLLM's built-in TT-specific defaults (e.g.
    # disabling chunked prefill via `is_tt()` in arg_utils) are applied
    # correctly.  The plugin's platform.py targeted an older vLLM version and
    # references removed APIs (envs.VLLM_USE_V1) and out-of-tree workers.
    return "vllm.platforms.tt.TTPlatform"


def register_models():
    """Register custom models with ModelRegistry for online inference.

    This function is called automatically by vLLM when the plugin is loaded,
    ensuring models are registered before the API server or engine starts.
    """
    from vllm import ModelRegistry

    # Register TT Llama model
    ModelRegistry.register_model(
        "TTLlamaForCausalLM",
        "models.tt_transformers.tt.generator_vllm:LlamaForCausalLM",
    )

    # Register BGE embedding model (TTBertModel)
    # This allows vLLM to find the TT-specific BGE implementation
    try:
        ModelRegistry.register_model(
            "TTBertModel",
            "models.demos.wormhole.bge_large_en.demo.generator_vllm:BGEForEmbedding",
        )
        print("Registered BGE embedding model")
    except Exception as e:
        # If registration fails (e.g., module not found), log warning but continue
        # This allows the plugin to work even if BGE model isn't available
        import logging

        logging.warning(
            f"Failed to register TTBertModel (BGE): {e}. "
            "BGE model may not be available. Ensure tt-metal is in Python path."
        )

    # Register Qwen3-Embedding model (TTQwen3Model)
    # This allows vLLM to find the TT-specific Qwen3-Embedding implementation
    # Note: Qwen3-Embedding may be detected as Qwen3ForCausalLM by vLLM,
    # so we register both TTQwen3Model and TTQwen3ForCausalLM
    try:
        ModelRegistry.register_model(
            "TTQwen3Model",
            "models.demos.wormhole.qwen3_embedding_8b.demo.generator_vllm:Qwen3ForEmbedding",
        )
        # Also register TTQwen3ForCausalLM as fallback (in case vLLM detects it as causal LM)
        ModelRegistry.register_model(
            "TTQwen3ForCausalLM",
            "models.demos.wormhole.qwen3_embedding_8b.demo.generator_vllm:Qwen3ForEmbedding",
        )
        print("Registered Qwen3-Embedding model")
    except Exception as e:
        # If registration fails (e.g., module not found), log warning but continue
        import logging

        logging.warning(
            f"Failed to register TTQwen3Model/TTQwen3ForCausalLM (Qwen3-Embedding): {e}. "
            "Qwen3-Embedding model may not be available. Ensure tt-metal is in Python path."
        )

    # Register all tt_symbiote models from the central registry in tt-metal.
    # Adding a new symbiote model only requires updating SYMBIOTE_MODEL_REGISTRY
    # in models/experimental/tt_symbiote/vllm/__init__.py.
    try:
        from models.experimental.tt_symbiote.vllm import SYMBIOTE_MODEL_REGISTRY

        for arch, module_path in SYMBIOTE_MODEL_REGISTRY.items():
            ModelRegistry.register_model(arch, module_path)
            print(f"Registered symbiote model: {arch}")
    except ImportError:
        import logging

        logging.warning(
            "tt_symbiote.vllm registry not importable (tt-metal not on PYTHONPATH?). "
            "Symbiote models will not be available."
        )
    except Exception as e:
        import logging

        logging.warning(
            f"Failed to register tt_symbiote models: {e}. "
            "Ensure tt-metal is in Python path."
        )
