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
    import os

    llama_text_version = os.getenv("TT_LLAMA_TEXT_VER", "tt_transformers")
    if llama_text_version == "tt_transformers":
        path_llama_text = "models.tt_transformers.tt.generator_vllm:LlamaForCausalLM"
    elif llama_text_version == "llama3_70b_galaxy":
        path_llama_text = (
            "models.demos.llama3_70b_galaxy.tt.generator_vllm:LlamaForCausalLM"
        )
    elif llama_text_version == "llama2_70b":
        path_llama_text = (
            "models.demos.t3000.llama2_70b.tt.generator_vllm:TtLlamaForCausalLM"
        )
    else:
        raise ValueError(
            f"Unsupported TT Llama version: {llama_text_version}, "
            "pick one of [tt_transformers, llama3_70b_galaxy, llama2_70b]"
        )

    # Llama3.1/3.2 - Text
    ModelRegistry.register_model("TTLlamaForCausalLM", path_llama_text)

    ## Llama3.2 - Vision
    # ModelRegistry.register_model(
    #    "TTMllamaForConditionalGeneration",
    #    "models.tt_transformers.tt.generator_vllm:MllamaForConditionalGeneration",
    #    )
    # try:
    #    ModelRegistry.register_model(
    #        "TTBertModel",
    #        "models.demos.wormhole.bge_large_en.demo.generator_vllm:BGEForEmbedding",
    #    )
    #    print("Registered BGE embedding model")
    # except Exception as e:
    #    # If registration fails (e.g., module not found), log warning but continue
    #    # This allows the plugin to work even if BGE model isn't available
    #    import logging

    #    logging.warning(
    #        f"Failed to register TTBertModel (BGE): {e}. "
    #        "BGE model may not be available. Ensure tt-metal is in Python path."
    #    )

    # Qwen2.5 - Text
    path_qwen_text = "models.tt_transformers.tt.generator_vllm:QwenForCausalLM"
    ModelRegistry.register_model("TTQwen2ForCausalLM", path_qwen_text)
    ModelRegistry.register_model("TTQwen3ForCausalLM", path_qwen_text)

    ## Qwen2.5 - Vision
    # ModelRegistry.register_model(
    #    "TTQwen2_5_VLForConditionalGeneration",
    #    "models.demos.qwen25_vl.tt.generator_vllm:Qwen2_5_VLForConditionalGeneration",
    # )

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

    # Mistral
    ModelRegistry.register_model(
        "TTMistralForCausalLM",
        "models.tt_transformers.tt.generator_vllm:MistralForCausalLM",
    )

    ## Gemma3
    # ModelRegistry.register_model(
    #    "TTGemma3ForConditionalGeneration",
    #    "models.tt_transformers.tt.generator_vllm:Gemma3ForConditionalGeneration",
    # )

    ## DeepseekV3
    # ModelRegistry.register_model(
    #    "TTDeepseekV3ForCausalLM",
    #    "models.demos.deepseek_v3.tt.generator_vllm:DeepseekV3ForCausalLM",
    # )

    # GPT-OSS
    ModelRegistry.register_model(
        "TTGptOssForCausalLM",
        "models.tt_transformers.tt.generator_vllm:GptOssForCausalLM",
    )

    # Add additional model registrations here as needed
    # ModelRegistry.register_model("AnotherModel", "path.to:ModelClass")
