# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

from dataclasses import dataclass
from typing import List, Optional
import os


@dataclass
class PromptConfig:
    input_seq_len: int
    max_prompt_length: int
    num_prompts: int
    distribution: str = "fixed"
    dataset: str = "random"
    tokenizer_model: str = os.environ.get(
        "HF_MODEL_REPO_ID", "meta-llama/Llama-3.1-70B-Instruct"
    )
    template: Optional[str] = None
    save_path: Optional[str] = None
    print_prompts: bool = False
    include_images: bool = False
    images_per_prompt: int = 1
    image_width: int = 256
    image_height: int = 256
    use_chat_api: bool = False


@dataclass
class BatchConfig:
    max_concurrent: int
    output_seq_lens: List[int]
    num_full_iterations: int = 1
    vary_max_concurrent: bool = False
    inter_batch_delay: int = 0
    stream: bool = True
    use_chat_api: bool = False


def get_mesh_device():
    mesh_device = os.environ.get("MESH_DEVICE")
    return mesh_device


@dataclass
class EnvironmentConfig:
    vllm_model: str = os.environ.get(
        "HF_MODEL_REPO_ID", "meta-llama/Llama-3.1-70B-Instruct"
    )
    vllm_api_key: Optional[str] = os.environ.get("VLLM_API_KEY")
    jwt_secret: Optional[str] = os.environ.get("JWT_SECRET")
    deploy_url: str = os.environ.get("DEPLOY_URL", "http://127.0.0.1")
    service_port: str = os.environ.get("SERVICE_PORT", "7000")
    cache_root: str = os.environ.get("CACHE_ROOT", ".")
    mesh_device: str = get_mesh_device()
