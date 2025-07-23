# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import sys
import runpy
import logging
from pathlib import Path

from vllm import ModelRegistry

from utils.logging_utils import set_vllm_logging_config
from utils.vllm_run_utils import (
    get_vllm_override_args,
    get_override_tt_config,
    resolve_commit,
    is_head_eq_or_after_commit,
    create_model_symlink,
    get_encoded_api_key,
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def get_hf_model_id():
    model = os.getenv("HF_MODEL_REPO_ID")
    assert model, "Must set environment variable: HF_MODEL_REPO_ID"
    return model


def handle_code_versions():
    tt_metal_home = os.getenv("TT_METAL_HOME")
    vllm_dir = os.getenv("vllm_dir")

    tt_metal_sha = resolve_commit("HEAD", tt_metal_home)
    logger.info(f"TT_METAL_HOME: {tt_metal_home}")
    logger.info(f"commit SHA: {tt_metal_sha}")

    vllm_sha = resolve_commit("HEAD", vllm_dir)
    logger.info(f"vllm_dir: {vllm_dir}")
    logger.info(f"commit SHA: {vllm_sha}")

    metal_tt_transformers_commit = "8815f46aa191d0b769ed1cc1eeb59649e9c77819"
    if os.getenv("MODEL_IMPL") == "tt-transformers":
        assert is_head_eq_or_after_commit(
            commit=metal_tt_transformers_commit, repo_path=tt_metal_home
        ), "tt-transformers model_impl requires tt-metal: v0.57.0-rc1 or later"


# Copied from vllm/examples/offline_inference_tt.py
def register_tt_models():
    model_impl = os.getenv("MODEL_IMPL", "tt-transformers")
    if model_impl == "tt-transformers":
        path_ttt_generators = "models.tt_transformers.tt.generator_vllm"
        path_llama_text = f"{path_ttt_generators}:LlamaForCausalLM"

        try:
            ModelRegistry.register_model(
                "TTQwen2ForCausalLM", f"{path_ttt_generators}:QwenForCausalLM"
            )
            ModelRegistry.register_model(
                "TTQwen3ForCausalLM", f"{path_ttt_generators}:QwenForCausalLM"
            )
        except (AttributeError) as e:
            logger.warning(f"Failed to register TTQwenForCausalLM: {e}, attempting to register older model signature")
            # Fallback registration without TT-specific implementation
            ModelRegistry.register_model(
                "TTQwen2ForCausalLM", f"{path_ttt_generators}:Qwen2ForCausalLM"
            )

        ModelRegistry.register_model(
            "TTMllamaForConditionalGeneration",
            f"{path_ttt_generators}:MllamaForConditionalGeneration",
        )
        if os.getenv("HF_MODEL_REPO_ID") == "mistralai/Mistral-7B-Instruct-v0.3":
            ModelRegistry.register_model(
                "TTMistralForCausalLM", f"{path_ttt_generators}:MistralForCausalLM"
            )
    elif model_impl == "subdevices":
        path_llama_text = (
            "models.demos.llama3_subdevices.tt.generator_vllm:LlamaForCausalLM"
        )
    elif model_impl == "t3000-llama2-70b":
        path_llama_text = (
            "models.demos.t3000.llama2_70b.tt.generator_vllm:TtLlamaForCausalLM"
        )
    else:
        raise ValueError(
            f"Unsupported model_impl: {model_impl}, pick one of [tt-transformers, subdevices, llama2-t3000]"
        )

    ModelRegistry.register_model("TTLlamaForCausalLM", path_llama_text)


register_tt_models()  # Import and register models from tt-metal


def ensure_mesh_device(hf_model_id):
    # model specific MESH_DEVICE management
    default_mesh_device = {
        "Qwen/Qwen3-32B": "T3K",
        "Qwen/QwQ-32B": "T3K",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": "T3K",
        "Qwen/Qwen2.5-72B-Instruct": "T3K",
        "Qwen/Qwen2.5-7B-Instruct": "N300",
        "meta-llama/Llama-3.1-70B-Instruct": "T3K",
        "meta-llama/Llama-3.3-70B-Instruct": "T3K",
        "meta-llama/Llama-3.2-1B-Instruct": "N150",
        "meta-llama/Llama-3.2-3B-Instruct": "N150",
        "meta-llama/Llama-3.1-8B-Instruct": "N300",
        "meta-llama/Llama-3.2-11B-Vision-Instruct": "N300",
    }
    valid_mesh_devices = {
        # only T3K_RING available for this 70B model implementation
        # see: https://github.com/tenstorrent/tt-metal/blob/main/models/demos/t3000/llama2_70b/tt/generator_vllm.py#L47
        # TG implementation will be impl in: https://github.com/tenstorrent/tt-metal/blob/main/models/demos/llama3/tt/generator_vllm.py#L136
        "meta-llama/Llama-3.1-70B-Instruct": ["T3K", "TG", "P150x4"],
        "meta-llama/Llama-3.3-70B-Instruct": ["T3K", "TG", "P150x4"],
        "Qwen/Qwen3-32B": ["T3K"],
        "Qwen/QwQ-32B": ["T3K"],
        "Qwen/Qwen2.5-72B-Instruct": ["T3K"],
        "Qwen/Qwen2.5-7B-Instruct": ["N300", "T3K"],
        "meta-llama/Llama-3.2-11B-Vision-Instruct": [
            "N300",
            "T3K",
        ],
    }
    cur_mesh_device = os.getenv("MESH_DEVICE")
    if hf_model_id in default_mesh_device.keys():
        if cur_mesh_device is None:
            # set good default
            os.environ["MESH_DEVICE"] = default_mesh_device[hf_model_id]
            cur_mesh_device = os.getenv("MESH_DEVICE")

    if hf_model_id in valid_mesh_devices.keys():
        assert (
            cur_mesh_device in valid_mesh_devices[hf_model_id]
        ), f"Invalid MESH_DEVICE for {hf_model_id}, {cur_mesh_device}"

    logger.info(f"using MESH_DEVICE:={os.getenv('MESH_DEVICE')}")


def runtime_settings(hf_model_id):
    # step 1: validate env vars passed in
    ensure_mesh_device(hf_model_id)
    model_impl = os.getenv("MODEL_IMPL")
    logger.info(f"MODEL_IMPL:={model_impl}")
    logging.info(f"MODEL_SOURCE: {os.getenv('MODEL_SOURCE')}")

    cache_root = Path(os.getenv("CACHE_ROOT"))
    assert cache_root.exists(), f"CACHE_ROOT: {cache_root} does not exist"
    symlinks_dir = cache_root / "model_file_symlinks_map"
    symlinks_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"MODEL_WEIGHTS_PATH: {os.getenv('MODEL_WEIGHTS_PATH')}")
    assert os.getenv("MODEL_WEIGHTS_PATH") is not None, "MODEL_WEIGHTS_PATH must be set"
    weights_dir = Path(os.getenv("MODEL_WEIGHTS_PATH"))
    assert weights_dir.exists(), f"MODEL_WEIGHTS_PATH: {weights_dir} does not exist"

    logging.info(f"TT_CACHE_PATH: {os.getenv('TT_CACHE_PATH')}")
    assert os.getenv("TT_CACHE_PATH") is not None, "TT_CACHE_PATH must be set"

    # step 2: set default runtime env vars
    # set up logging
    config_path, log_path = set_vllm_logging_config(level="DEBUG")
    logger.info(f"setting vllm logging config at: {config_path}")
    logger.info(f"setting vllm logging file at: {log_path}")

    env_vars = {
        # note: the vLLM logging environment variables do not cause the configuration
        # to be loaded in all cases, so it is loaded manually in set_vllm_logging_config
        "VLLM_CONFIGURE_LOGGING": "1",
        "VLLM_LOGGING_CONFIG": str(config_path),
        # stop timeout during long sequential prefill batches
        # e.g. 32x 2048 token prefills taking longer than default 30s timeout
        # timeout is 3x VLLM_RPC_TIMEOUT
        "VLLM_RPC_TIMEOUT": "900000",  # 200000ms = 200s
    }

    if os.getenv("MESH_DEVICE") in ["N150", "N300", "T3K"]:
        env_vars["WH_ARCH_YAML"] = "wormhole_b0_80_arch_eth_dispatch.yaml"
    else:
        # remove WH_ARCH_YAML if it was set
        env_vars["WH_ARCH_YAML"] = None

    if hf_model_id.startswith("meta-llama"):
        logging.info(f"Llama setup for {hf_model_id}")

        model_dir_name = hf_model_id.split("/")[-1]
        # the mapping in: models/tt_transformers/tt/model_config.py
        # uses e.g. Llama3.2 instead of Llama-3.2
        model_dir_name = model_dir_name.replace("Llama-", "Llama")
        file_symlinks_map = {}
        if hf_model_id.startswith("meta-llama/Llama-3.2-11B-Vision"):
            # Llama-3.2-11B-Vision requires specific file symlinks with different names
            # The loading code in:
            # https://github.com/tenstorrent/tt-metal/blob/v0.57.0-rc71/models/tt_transformers/demo/simple_vision_demo.py#L55
            # does not handle this difference in naming convention for the weights
            file_symlinks_map = {
                "consolidated.00.pth": "consolidated.pth",
                "params.json": "params.json",
                "tokenizer.model": "tokenizer.model",
            }
        elif model_dir_name.startswith("Llama3.3"):
            # Only Llama 3.1 70B is defined in models/tt_transformers/tt/model_config.py
            if os.getenv("MESH_DEVICE") == "T3K":
                env_vars["MAX_PREFILL_CHUNK_SIZE"] = "32"

        llama_dir = create_model_symlink(
            symlinks_dir,
            model_dir_name,
            weights_dir,
            file_symlinks_map=file_symlinks_map,
        )

        env_vars["LLAMA_DIR"] = str(llama_dir)
        env_vars.update({"HF_MODEL": None})
    else:
        logging.info(f"HF model setup for {hf_model_id}")
        model_dir_name = hf_model_id.split("/")[-1]
        hf_dir = create_model_symlink(symlinks_dir, model_dir_name, weights_dir)
        env_vars["HF_MODEL"] = hf_dir
        env_vars.update({"LLAMA_DIR": None})

    if model_impl == "tt-transformers":
        env_vars.update(
            {
                "meta-llama/Llama-3.1-70B-Instruct": {},
                "meta-llama/Llama-3.3-70B-Instruct": {},
                "Qwen/Qwen3-32B": {
                    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": 1,
                },
                "Qwen/QwQ-32B": {
                    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": 1,
                },
                "Qwen/Qwen2.5-72B-Instruct": {
                    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": 1,
                    # TODO: remove after this is closed https://github.com/tenstorrent/tt-metal/issues/19890#issuecomment-3081077938
                    "MAX_PREFILL_CHUNK_SIZE": 16,
                },
                "Qwen/Qwen2.5-7B-Instruct": {
                    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": 1,
                },
                "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": {
                    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": 1,
                },
            }.get(hf_model_id, {})
        )
    elif model_impl == "subdevices":
        env_vars["LLAMA_VERSION"] = "subdevices"
    elif model_impl == "llama2-t3000":
        env_vars.update(
            {
                "meta-llama/Llama-3.1-70B-Instruct": {
                    "LLAMA_VERSION": "llama3",
                    "LLAMA_DIR": os.getenv("MODEL_WEIGHTS_PATH"),
                },
                "meta-llama/Llama-3.3-70B-Instruct": {
                    "LLAMA_VERSION": "llama3",
                    "LLAMA_DIR": os.getenv("MODEL_WEIGHTS_PATH"),
                },
            }.get(hf_model_id, {})
        )

    # Set each environment variable
    logger.info("setting runtime environment variables:")
    for key, value in env_vars.items():
        logger.info(f"setting env var: {key}={value}")
        if value is not None:
            os.environ[key] = str(value)
        elif key in os.environ:
            del os.environ[key]


def model_setup(hf_model_id):
    # TODO: check HF repo access with HF_TOKEN supplied
    logger.info(f"using model: {hf_model_id}")

    # Check if HF_TOKEN is set
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        logger.info("HF_TOKEN is set")
    else:
        logger.warning(
            "HF_TOKEN is not set - this may cause issues accessing private models or models requiring authorization"
        )

    # check if JWT_SECRET is set
    jwt_secret = os.getenv("JWT_SECRET")
    if jwt_secret:
        logger.info(
            "JWT_SECRET is set: HTTP requests to vLLM API require bearer token in 'Authorization' header. See docs for how to get bearer token."
        )
    else:
        logger.warning(
            "JWT_SECRET is not set: HTTP requests to vLLM API will not require authorization"
        )

    runtime_settings(hf_model_id)
    args = {
        "model": hf_model_id,
        "block_size": os.getenv("VLLM_BLOCK_SIZE", "64"),
        "max_num_seqs": os.getenv("VLLM_MAX_NUM_SEQS", "32"),
        "max_model_len": os.getenv("VLLM_MAX_MODEL_LEN", "131072"),
        "max_num_batched_tokens": os.getenv("VLLM_MAX_NUM_BATCHED_TOKENS", "131072"),
        "num_scheduler_steps": os.getenv("VLLM_NUM_SCHEDULER_STEPS", "10"),
        "max-log-len": os.getenv("VLLM_MAX_LOG_LEN", "64"),
        "port": os.getenv("SERVICE_PORT", "7000"),
        "api-key": get_encoded_api_key(os.getenv("JWT_SECRET", None)),
        "override_tt_config": get_override_tt_config(),
    }

    if 'ENABLE_AUTO_TOOL_CHOICE' in os.environ:
        raise AssertionError("setting ENABLE_AUTO_TOOL_CHOICE has been deprecated, use the VLLM_OVERRIDE_ARGS env var directly or via --vllm-override-args in run.py CLI.\n" \
                             "Enable auto tool choice by adding --vllm-override-args \'{\"enable-auto-tool-choice\": true, \"tool-call-parser\": <parser-name>}\' when calling run.py")

    # Apply vLLM argument overrides
    override_args = get_vllm_override_args()
    if override_args:
        args.update(override_args)

    return args


def main():
    handle_code_versions()
    hf_model_id = get_hf_model_id()
    # vLLM CLI arguments
    args = model_setup(hf_model_id)
    for key, value in args.items():
        if value is not None:
            # Handle boolean flags
            if isinstance(value, bool):
                if value:  # Only add the flag if True
                    sys.argv.append("--" + key)
            else:
                sys.argv.extend(["--" + key, str(value)])

    # runpy uses the same process and environment so the registered models are available
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
