# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import sys
from typing import TYPE_CHECKING, ClassVar, Literal

import torch

from vllm.logger import init_logger
from vllm.platforms.interface import Platform, PlatformEnum
from vllm_tt_plugin.config import get_tt_config

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.inputs import ProcessorInputs, PromptType
    from vllm.pooling_params import PoolingParams
    from vllm.renderers.inputs import DictPrompt, TokPrompt
    from vllm.sampling_params import SamplingParams
    from vllm.utils.argparse_utils import FlexibleArgumentParser
else:
    FlexibleArgumentParser = object

logger = init_logger(__name__)

TT_SCHEDULER_CLS = "vllm_tt_plugin.scheduler.TTScheduler"
_warned_cli_plugin_config = False


def _warn_cli_plugin_config() -> None:
    global _warned_cli_plugin_config
    if _warned_cli_plugin_config:
        return
    logger.warning(
        "TT config passed through --plugin-config is deprecated. "
        "Use --additional-config '{\"tt\": {...}}' instead."
    )
    _warned_cli_plugin_config = True


def _register_model_if_missing(ModelRegistry, model_arch: str, model_path: str) -> None:
    """Register `model_arch` only if not already registered.

    This keeps TT model registration idempotent across multiple call sites
    (e.g. APIServer pre-register, TT worker import, and platform config hook).
    """
    if model_arch not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(model_arch, model_path)


def _should_pre_register_tt_test_models_from_cli() -> bool:
    """Return True iff CLI TT config enables TT test models.

    `TTPlatform.pre_register_and_update()` runs before `VllmConfig` is
    constructed, but ModelConfig may inspect architectures early.
    """
    argv = list(sys.argv[1:])

    def _parse_namespaced_config(raw: str) -> dict | None:
        try:
            parsed = json.loads(raw)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None

    canonical_flags = {"--additional-config", "--plugin-config"}
    parsed_configs: dict[str, dict] = {}
    for i, arg in enumerate(argv):
        if "=" in arg:
            flag, value = arg.split("=", 1)
            normalized_flag = flag.replace("_", "-")
            if normalized_flag in canonical_flags:
                if normalized_flag == "--plugin-config":
                    _warn_cli_plugin_config()
                cfg = _parse_namespaced_config(value)
                if cfg:
                    parsed_configs[normalized_flag] = cfg
        else:
            normalized_arg = arg.replace("_", "-")
            if normalized_arg in canonical_flags and i + 1 < len(argv):
                if normalized_arg == "--plugin-config":
                    _warn_cli_plugin_config()
                cfg = _parse_namespaced_config(argv[i + 1])
                if cfg:
                    parsed_configs[normalized_arg] = cfg

    cfg = parsed_configs.get("--additional-config") or parsed_configs.get(
        "--plugin-config"
    )
    tt_config = cfg.get("tt", {}) if cfg else {}
    return bool(
        isinstance(tt_config, dict) and tt_config.get("register_test_models") is True
    )


def _install_tt_harmony_truncation_patch() -> None:
    """Use right truncation for TT GPT-OSS tokenizers.

    GPT-OSS harmony prompts have important template/control tokens at the
    beginning. Left truncation can remove those tokens when prompt truncation is
    requested, so TT keeps the prefix and truncates from the right for these
    models.

    TODO: remove this once fixed in vLLM core.
    """
    import vllm.tokenizers.registry as tokenizer_registry

    if hasattr(tokenizer_registry, "_tt_original_tokenizer_args_from_config"):
        return

    original = tokenizer_registry.tokenizer_args_from_config
    tokenizer_registry._tt_original_tokenizer_args_from_config = original

    def tokenizer_args_from_config_tt(config, **kwargs):
        tokenizer_mode, tokenizer_name, args, tokenizer_kwargs = original(
            config, **kwargs
        )
        if (
            "truncation_side" not in tokenizer_kwargs
            and config.runner_type in ("generate", "draft")
            and "gpt-oss" in str(tokenizer_name or "").lower()
        ):
            tokenizer_kwargs["truncation_side"] = "right"
        return tokenizer_mode, tokenizer_name, args, tokenizer_kwargs

    tokenizer_registry.tokenizer_args_from_config = tokenizer_args_from_config_tt

    renderer_registry = sys.modules.get("vllm.renderers.registry")
    if renderer_registry is not None:
        renderer_registry.tokenizer_args_from_config = tokenizer_args_from_config_tt


def register_tt_models(register_test_models=False) -> None:
    from vllm.model_executor.models.registry import ModelRegistry

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
    _register_model_if_missing(ModelRegistry, "TTLlamaForCausalLM", path_llama_text)

    # Llama3.2 - Vision
    _register_model_if_missing(
        ModelRegistry,
        "TTMllamaForConditionalGeneration",
        "models.tt_transformers.tt.generator_vllm:MllamaForConditionalGeneration",
    )

    # Qwen2.5 - Text
    path_qwen_text = "models.tt_transformers.tt.generator_vllm:QwenForCausalLM"
    _register_model_if_missing(ModelRegistry, "TTQwen2ForCausalLM", path_qwen_text)

    # Qwen3 - Text
    qwen3_text_version = os.getenv("TT_QWEN3_TEXT_VER", "tt_transformers")
    if qwen3_text_version == "tt_transformers":
        path_qwen3_text = "models.tt_transformers.tt.generator_vllm:QwenForCausalLM"
    elif qwen3_text_version == "qwen3_32b_galaxy":
        path_qwen3_text = (
            "models.demos.llama3_70b_galaxy.tt.generator_vllm:QwenForCausalLM"
        )
    else:
        raise ValueError(
            f"Unsupported TT Qwen3 version: {qwen3_text_version}, "
            "pick one of [tt_transformers, qwen3_32b_galaxy]"
        )

    _register_model_if_missing(ModelRegistry, "TTQwen3ForCausalLM", path_qwen3_text)

    # Qwen2.5 - Vision
    _register_model_if_missing(
        ModelRegistry,
        "TTQwen2_5_VLForConditionalGeneration",
        "models.demos.qwen25_vl.tt.generator_vllm:Qwen2_5_VLForConditionalGeneration",
    )

    # Qwen3 - Vision
    _register_model_if_missing(
        ModelRegistry,
        "TTQwen3VLForConditionalGeneration",
        "models.demos.qwen3_vl.tt.generator_vllm:Qwen3VLForConditionalGeneration",
    )

    # Mistral - Text only
    _register_model_if_missing(
        ModelRegistry,
        "TTMistralForCausalLM",
        "models.tt_transformers.tt.generator_vllm:MistralForCausalLM",
    )

    # Mistral 3 - Multimodal (Vision + Text)
    _register_model_if_missing(
        ModelRegistry,
        "TTMistral3ForConditionalGeneration",
        "models.tt_transformers.tt.generator_vllm:Mistral3ForConditionalGeneration",
    )

    # Gemma3
    _register_model_if_missing(
        ModelRegistry,
        "TTGemma3ForConditionalGeneration",
        "models.tt_transformers.tt.generator_vllm:Gemma3ForConditionalGeneration",
    )

    # Gemma4 — text-only TT bridge.
    #
    # Gemma4 isn't in vLLM's upstream registry, so without an entry here
    # the upstream architecture resolver falls back to
    # ``TransformersMultiModalForCausalLM`` (because ``hf_config !=
    # hf_text_config`` for Gemma4's nested config — see
    # ``ModelConfig._get_transformers_backend_cls``) and crashes on the
    # ``_processor_factory`` assertion in the multimodal registry. The
    # plugin's later ``TT``-prefix logic runs after that resolution, so
    # it can't help.
    #
    # We register the plain HF arch names directly so upstream resolution
    # finds our class. Since ``Gemma4ForCausalLM`` (the TT class) does not
    # use ``SupportsMultiModal``, vLLM's ``_model_info.supports_multimodal``
    # is False, ``multimodal_config`` is not populated, and the request
    # path stays text-only — which matches what the TT model implements.
    # The ``TT``-prefixed aliases satisfy the plugin's later validation
    # in ``check_and_update_config`` so no override is needed.
    _gemma4_target = "models.demos.gemma4.tt.generator_vllm:Gemma4ForCausalLM"
    for arch in (
        "Gemma4ForCausalLM",
        "Gemma4ForConditionalGeneration",
        "Gemma4UnifiedForConditionalGeneration",
        "TTGemma4ForCausalLM",
        "TTGemma4ForConditionalGeneration",
        "TTGemma4UnifiedForConditionalGeneration",
    ):
        _register_model_if_missing(ModelRegistry, arch, _gemma4_target)

    # DeepseekV3
    _register_model_if_missing(
        ModelRegistry,
        "TTDeepseekV3ForCausalLM",
        "models.demos.deepseek_v3.tt.generator_vllm:DeepseekV3ForCausalLM",
    )

    # GPT-OSS
    _register_model_if_missing(
        ModelRegistry,
        "TTGptOssForCausalLM",
        "models.tt_transformers.tt.generator_vllm:GptOssForCausalLM",
    )

    # Optionally register test models if explicitly enabled
    if register_test_models:
        register_tt_test_models()


def register_tt_test_models():
    """Register non-production TT models which are only used for testing."""
    from vllm.model_executor.models.registry import ModelRegistry

    # Fake model for testing multi-process inference on T3000
    _register_model_if_missing(
        ModelRegistry,
        "TTDummyT3000MultiProcessModel",
        "models.vllm_test_utils.t3000_multiproc_test.test_model:DummyT3000MultiProcessModel",
    )

    # Fake model which does nothing, for measuring vLLM host overheads
    _register_model_if_missing(
        ModelRegistry,
        "TTDummyNoOpModel",
        "models.vllm_test_utils.no_op_test.test_model:DummyNoOpModel",
    )

    # Fake model for testing multi-host inference on dual Galaxy
    _register_model_if_missing(
        ModelRegistry,
        "TTDummyDualGlxModel",
        "models.vllm_test_utils.dual_glx_ccl_test.test_model:DummyDualGlxModel",
    )


class TTPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name: str = "tt"
    device_type: str = "tt"
    sample_on_device_mode: ClassVar[Literal["all", "decode_only"] | None] = None
    # Disable torch.compile on TT platform - the triton version in tt-metal
    # is incompatible with torch's inductor backend.
    simple_compile_backend: str = "eager"

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        # Hybrid models (Gemma3/4, GPT-OSS) opt in to upstream's HMA via
        # ``HybridAttentionForCausalLM.get_kv_cache_spec`` so layers from
        # different attention groups can share DRAM tensors. Without this
        # override the base ``Platform`` returns ``False`` and HMA collapses
        # every ``SlidingWindowSpec`` back to ``FullAttentionSpec`` in
        # ``unify_hybrid_kv_cache_specs`` — defeating the entire point.
        return True

    @classmethod
    def pre_register_and_update(
        cls, parser: FlexibleArgumentParser | None = None
    ) -> None:
        # Called during CLI/parser setup (APIServer). ModelConfig may
        # validate/inspect architectures before VllmConfig is constructed in
        # this process, so we must ensure TT test models are registered early
        # when explicitly requested via CLI override.
        super().pre_register_and_update(parser)
        _install_tt_harmony_truncation_patch()
        if _should_pre_register_tt_test_models_from_cli():
            register_tt_test_models()

    @classmethod
    def import_kernels(cls) -> None:
        # Do not import vllm._C or vllm._moe_C
        pass

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        return True

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        _install_tt_harmony_truncation_patch()
        if vllm_config.scheduler_config.enable_chunked_prefill:
            logger.info("Chunked prefill is not yet supported for TT backend")
            vllm_config.scheduler_config.enable_chunked_prefill = False
        assert not vllm_config.speculative_config, (
            "Speculative decoding is not yet supported for TT backend"
        )
        assert (
            vllm_config.parallel_config.tensor_parallel_size == 1
            and vllm_config.parallel_config.pipeline_parallel_size == 1
        ), "TT backend does not support distributed execution"
        assert not vllm_config.lora_config, "LoRA is not supported for TT backend"

        # Device computes top-32 logprobs but the OpenAI API limits to 20
        MAX_TOP_K = 20

        model_config = vllm_config.model_config
        if model_config.max_logprobs > MAX_TOP_K:
            logger.warning(
                "max_logprobs=%d exceeds TT device limit of %d, clamping to %d",
                model_config.max_logprobs,
                MAX_TOP_K,
                MAX_TOP_K,
            )
            model_config.max_logprobs = MAX_TOP_K

        # Import and register models from tt-metal.
        #
        # NOTE: We also register TT models early in `vllm_tt_plugin.worker`
        # (at module import time). That registration is required to handle
        # engine/worker subprocess startup ordering where model architectures
        # may be inspected (e.g. multimodal processor cache init) before this
        # `check_and_update_config()` hook is reached in that process.
        tt_config = get_tt_config(vllm_config)
        register_test_models = False
        if tt_config and "register_test_models" in tt_config:
            register_test_models = tt_config["register_test_models"]
            assert register_test_models in [True, False], (
                f"Invalid option register_test_models: {register_test_models}"
            )
        register_tt_models(register_test_models)

        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm_tt_plugin.worker.TTWorker"
        parallel_config.engine_core_cls = "vllm_tt_plugin.engine.TTEngineCore"
        parallel_config.engine_core_proc_cls = "vllm_tt_plugin.engine.TTEngineCoreProc"
        parallel_config.dp_engine_core_proc_cls = (
            "vllm_tt_plugin.engine.TTDPEngineCoreProc"
        )
        parallel_config.engine_core_launcher_cls = (
            "vllm_tt_plugin.launcher.TTCoreEngineLauncher"
        )

        # For TT models, prepend "TT" to the architecture name,
        # e.g. "TTLlamaForCausalLM"
        arch_names = vllm_config.model_config.hf_config.architectures
        for i in range(len(arch_names)):
            if not arch_names[i].startswith("TT"):
                arch_names[i] = "TT" + arch_names[i]

        # Verify that the TT architecture is registered in the model registry
        from vllm.model_executor.models.registry import ModelRegistry

        supported_archs = ModelRegistry.get_supported_archs()
        if not any(arch_name in supported_archs for arch_name in arch_names):
            tt_archs = sorted(
                [arch for arch in supported_archs if arch.startswith("TT")]
            )
            raise ValueError(
                f"No TT model architecture is registered for "
                f"model: '{vllm_config.model_config.model}'. "
                f"Available TT architectures: {tt_archs}"
            )

        # Setting attributes on the class level is kind of hacky, but
        # it's the only way to make validate_request depend on vllm_config
        # This is needed to catch incompatible requests early enough
        # to return an error instead of crashing.
        # TODO move this to tt_model_runner when request validation
        # stops depending on vllm_config

        if tt_config is not None and "sample_on_device_mode" in tt_config:
            sample_on_device_mode = tt_config["sample_on_device_mode"]
            assert sample_on_device_mode in [
                "all",
                "decode_only",
            ], f"Invalid sample_on_device_mode: {sample_on_device_mode}"
        else:
            sample_on_device_mode = None
        cls.sample_on_device_mode = sample_on_device_mode  # type: ignore[attr-defined]

        # Compat sampling uses the full vLLM sampling pipeline,
        # with logit processors and sampler, instead of our custom sampling.
        # It is enabled only if any of the requests in the batch requires it,
        # or if always_compat_sampling is enabled.

        always_compat_sampling = False
        if tt_config is not None and "always_compat_sampling" in tt_config:
            always_compat_sampling = tt_config["always_compat_sampling"]
            assert always_compat_sampling in [True, False], (
                "always_compat_sampling must be a boolean"
            )
            if always_compat_sampling:
                raise ValueError(
                    "always_compat_sampling is not yet supported for V1 TT backend."
                )
        cls.always_compat_sampling = always_compat_sampling  # type: ignore[attr-defined]

        # must perform local import to get around circular import
        from vllm.model_executor.model_loader.utils import get_model_architecture

        model_class, _ = get_model_architecture(vllm_config.model_config)

        # Get model capabilities from the class
        model_capabilities: dict | None = getattr(
            model_class, "model_capabilities", None
        )

        # A model either supports the full on-device sampling pipeline or it
        # doesn't — there is no greedy-only mode. Models opt in by setting
        # `supports_sample_on_device` in their `model_capabilities` dict.
        supports_sample_on_device = (
            model_capabilities.get("supports_sample_on_device", False)
            if model_capabilities
            else False
        )
        if sample_on_device_mode is not None and not supports_sample_on_device:
            raise ValueError(
                f"sample_on_device_mode={sample_on_device_mode!r} was requested, "
                f"but model {model_class.__name__} "
                f"({model_class.__module__}) does not support on-device sampling. "
                "Unset sample_on_device_mode or use a model that supports it."
            )

        # Model-gated async scheduling. Async overlap requires generators that
        # support split decode submission via `decode_forward(...,
        # read_from_device=False)` followed by `read_decode_output(...,
        # async_read=True)`.
        supports_async_decode = (
            model_capabilities.get("supports_async_decode", False)
            if model_capabilities
            else False
        )
        if vllm_config.scheduler_config.async_scheduling and not supports_async_decode:
            logger.warning(
                "Async scheduling was requested, but TT model %s (%s) does not "
                "declare support (`model_capabilities['supports_async_decode']`). "
                "Disabling async scheduling.",
                model_class.__name__,
                model_class.__module__,
            )
            vllm_config.scheduler_config.async_scheduling = False

        # TT uses a single scheduler implementation for both sync and async
        # execution modes; async_scheduling only controls execution overlap.
        vllm_config.scheduler_config.scheduler_cls = TT_SCHEDULER_CLS

        if vllm_config.cache_config.enable_prefix_caching:
            # Check prefix caching support from capabilities (default to False)
            supports_prefix_caching = (
                model_capabilities.get("supports_prefix_caching", False)
                if model_capabilities
                else False
            )

            if not supports_prefix_caching:
                vllm_config.cache_config.enable_prefix_caching = False
                logger.warning(
                    "Prefix caching is not supported in TT backend for %s, "
                    "disabling it",
                    model_class.__module__,
                )
            else:
                # Check if the model architecture uses sliding window
                uses_sliding_window = (
                    vllm_config.model_config.get_sliding_window() is not None
                )
                if uses_sliding_window:
                    vllm_config.cache_config.enable_prefix_caching = False
                    logger.warning(
                        "Prefix caching is not supported in TT backend for "
                        "models with sliding window, disabling it"
                    )

        logger.info(
            "Automatic prefix caching is %s",
            "enabled" if vllm_config.cache_config.enable_prefix_caching else "disabled",
        )

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        # The regular v0 vLLM sampling code tries
        # to use pinned memory in case we're using GPUs.
        return False

    @classmethod
    def uses_host_device_handling(cls) -> bool:
        return True

    @classmethod
    def validate_request(
        cls,
        prompt: "PromptType | DictPrompt | TokPrompt",
        params: "SamplingParams | PoolingParams",
        processed_inputs: "ProcessorInputs",
    ) -> None:
        """Raises if this request is unsupported on this platform"""
        from vllm.sampling_params import SamplingParams

        dev = cls.device_name

        if isinstance(params, SamplingParams) and params.prompt_logprobs is not None:
            raise ValueError(f"Not yet supporting prompt_logprobs on {dev}")

    @staticmethod
    def compat_sampling_required(sampling_params, num_devices) -> bool:
        # Device logprobs only supported on multi-device setups and only
        # the sampled token's logprob is returned (not top-k alternatives).
        # Single device: any logprobs require host sampling.
        # Multi-device: logprobs > 1 requires host sampling because device
        # can only return the sampled token's logprob.
        # https://github.com/tenstorrent/tt-metal/issues/34077
        if (
            sampling_params.logprobs is not None
            and sampling_params.logprobs > 0
            and (num_devices == 1 or sampling_params.logprobs > 1)
        ):
            return True

        # all of the following sampling params require compat sampling
        return (
            sampling_params.min_p != 0.0
            or (
                sampling_params.bad_words is not None
                and len(sampling_params.bad_words) > 0
            )
            or sampling_params.prompt_logprobs is not None
            or sampling_params.structured_outputs is not None
            or sampling_params.logit_bias is not None
            or sampling_params.allowed_token_ids is not None
            or sampling_params.min_tokens != 0
        )
