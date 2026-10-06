# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


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

    # Register TT Qwen text-generation model (Qwen2.5 / Qwen3 families)
    ModelRegistry.register_model(
        "TTQwen3ForCausalLM",
        "models.tt_transformers.tt.generator_vllm:QwenForCausalLM",
    )

    # Register Qwen3-Embedding model (TTQwen3Model)
    # This allows vLLM to find the TT-specific Qwen3-Embedding implementation
    # Note: Qwen3-Embedding also declares Qwen3ForCausalLM in its HF config, so
    # the architecture name alone does not distinguish it from a text generator.
    # TTPlatform.check_and_update_config rewrites the arch to TTQwen3Model when
    # the runner is a pooling one, which is what routes it here.
    try:
        ModelRegistry.register_model(
            "TTQwen3Model",
            "models.demos.wormhole.qwen3_embedding_8b.demo.generator_vllm:Qwen3ForEmbedding",
        )
        print("Registered Qwen3-Embedding model")
    except Exception as e:
        # If registration fails (e.g., module not found), log warning but continue
        import logging

        logging.warning(
            f"Failed to register TTQwen3Model (Qwen3-Embedding): {e}. "
            "Qwen3-Embedding model may not be available. Ensure tt-metal is in Python path."
        )

    # Register the TT device-socket weight-transfer backend so vLLM's native RL
    # weight-sync API (--weight-transfer-config '{"backend": "device_socket"}')
    # can construct it. Lazy (module-path) registration -- no ttnn import here.
    # The co-located TT worker can also build the engine directly without this.
    try:
        from tt_vllm_plugin.weight_transfer import register_tt_weight_transfer_engine

        register_tt_weight_transfer_engine()
    except Exception as e:  # noqa: BLE001 - never block plugin load on this
        import logging

        logging.warning(f"Failed to register TT weight-transfer engine: {e}")

    # Add additional model registrations here as needed
    # ModelRegistry.register_model("AnotherModel", "path.to:ModelClass")
