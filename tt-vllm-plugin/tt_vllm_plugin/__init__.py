# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Register the qwen3_5 config at import time so that a bare
# `import tt_vllm_plugin` is sufficient for AutoConfig to parse Qwen3.6-27B
# checkpoints.  Must run before register_models() (which needs vllm) so that
# off-device import tests can verify config registration without vllm present.
try:
    from tt_vllm_plugin.qwen3_5_config import register_qwen3_5_config

    register_qwen3_5_config()
except Exception as _e:
    import logging as _logging

    _logging.warning(f"tt_vllm_plugin: failed to register qwen3_5 config at import time: {_e}")


def register():
    # At first we used ttnn.get_device_ids() to truly understand if the TT platform is supported.
    # This caused the offline inference to hang and never complete, so for now we just assume that we always have TT support.
    return "tt_vllm_plugin.platform.TTPlatform"


def register_models():
    """Register custom models with ModelRegistry for online inference.

    This function is called automatically by vLLM when the plugin is loaded,
    ensuring models are registered before the API server or engine starts.
    """
    from vllm import ModelRegistry

    # Register the qwen3_5 config so vLLM's AutoConfig can parse Qwen3.6-27B
    # (qwen3_5 is not in public transformers). Also called at import time above
    # so tests work without vllm; the registration is idempotent.
    try:
        from tt_vllm_plugin.qwen3_5_config import register_qwen3_5_config

        register_qwen3_5_config()
        print("Registered qwen3_5 config")
    except Exception as e:
        import logging

        logging.warning(f"Failed to register qwen3_5 config: {e}")

    # Register TT Qwen3.6-27B (galaxy v2, text-only)
    try:
        ModelRegistry.register_model(
            "TTQwen3_5ForConditionalGeneration",
            "models.demos.qwen3_6_galaxy_v2.tt.generator_vllm:Qwen3_5ForConditionalGeneration",
        )
        print("Registered TT Qwen3.6-27B")
    except Exception as e:
        import logging

        logging.warning(f"Failed to register TTQwen3_5ForConditionalGeneration: {e}")

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

    # Add additional model registrations here as needed
    # ModelRegistry.register_model("AnotherModel", "path.to:ModelClass")
