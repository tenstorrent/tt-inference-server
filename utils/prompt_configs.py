# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

from dataclasses import dataclass, field
from typing import List, Optional
import os
from urllib.parse import urlsplit, urlunsplit


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


def _strip_api_suffix(path: str) -> str:
    normalized_path = path.rstrip("/")
    for suffix in (
        "/v1/chat/completions",
        "/v1/completions",
        "/chat/completions",
        "/completions",
        "/v1",
    ):
        if normalized_path.endswith(suffix):
            return normalized_path[: -len(suffix)]
    return normalized_path


def _normalize_explicit_url(url: str) -> str:
    normalized_url = url.strip().rstrip("/")
    if "://" not in normalized_url:
        normalized_url = f"http://{normalized_url}"

    parsed = urlsplit(normalized_url)
    path = _strip_api_suffix(parsed.path)
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", "")).rstrip("/")


def _build_netloc_with_port(parsed, service_port: Optional[str]) -> str:
    if parsed.port is not None or not service_port:
        return parsed.netloc

    hostname = parsed.hostname
    if not hostname:
        raise ValueError(f"Invalid deploy URL: {parsed.geturl()}")

    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"

    auth = ""
    if parsed.username:
        auth = parsed.username
        if parsed.password:
            auth += f":{parsed.password}"
        auth += "@"

    return f"{auth}{hostname}:{service_port}"


def resolve_server_root_url(
    service_port: Optional[str] = None,
    base_url: Optional[str] = None,
    deploy_url: Optional[str] = None,
) -> str:
    """Resolve the server root URL without the OpenAI `/v1` suffix."""
    if base_url:
        return _normalize_explicit_url(base_url)

    resolved_deploy_url = (deploy_url or "http://127.0.0.1").strip().rstrip("/")
    if "://" not in resolved_deploy_url:
        resolved_deploy_url = f"http://{resolved_deploy_url}"

    parsed = urlsplit(resolved_deploy_url)
    netloc = _build_netloc_with_port(parsed, service_port)
    path = _strip_api_suffix(parsed.path)
    return urlunsplit((parsed.scheme, netloc, path, "", "")).rstrip("/")


def resolve_api_base_url(
    service_port: Optional[str] = None,
    base_url: Optional[str] = None,
    deploy_url: Optional[str] = None,
    include_v1: bool = True,
) -> str:
    root_url = resolve_server_root_url(
        service_port=service_port,
        base_url=base_url,
        deploy_url=deploy_url,
    )
    return f"{root_url}/v1" if include_v1 else root_url


@dataclass
class EnvironmentConfig:
    vllm_model: str = field(
        default_factory=lambda: os.environ.get(
            "HF_MODEL_REPO_ID", "meta-llama/Llama-3.1-70B-Instruct"
        )
    )
    vllm_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("VLLM_API_KEY")
    )
    jwt_secret: Optional[str] = field(
        default_factory=lambda: os.environ.get("JWT_SECRET")
    )
    base_url: Optional[str] = field(default_factory=lambda: os.environ.get("BASE_URL"))
    deploy_url: str = field(
        default_factory=lambda: os.environ.get("DEPLOY_URL", "http://127.0.0.1")
    )
    service_port: str = field(
        default_factory=lambda: os.environ.get("SERVICE_PORT", "7000")
    )
    cache_root: str = field(default_factory=lambda: os.environ.get("CACHE_ROOT", "."))
    mesh_device: str = field(default_factory=get_mesh_device)
