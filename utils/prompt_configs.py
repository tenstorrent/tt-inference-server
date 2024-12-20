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
        "VLLM_MODEL", "meta-llama/Llama-3.1-70B-Instruct"
    )
    template: Optional[str] = None
    save_path: Optional[str] = None
    print_prompts: bool = False


@dataclass
class BatchConfig:
    batch_size: int
    output_seq_lens: List[int]
    num_full_iterations: int = 1
    vary_batch_size: bool = False
    inter_batch_delay: int = 0
    stream: bool = True


@dataclass
class EnvironmentConfig:
    vllm_model: str = os.environ.get("VLLM_MODEL", "meta-llama/Llama-3.1-70B-Instruct")
    authorization: Optional[str] = os.environ.get("AUTHORIZATION")
    jwt_secret: Optional[str] = os.environ.get("JWT_SECRET")
    deploy_url: str = os.environ.get("DEPLOY_URL", "http://127.0.0.1")
    service_port: str = os.environ.get("SERVICE_PORT", "7000")
    cache_root: str = os.environ.get("CACHE_ROOT", ".")
