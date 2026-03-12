# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files
from utils.logger import TTLogger

logger = TTLogger()


def _find_safetensors_filename(repo_id: str) -> str:
    """Find the .safetensors file in a HuggingFace LoRA repository.

    Raises FileNotFoundError if no safetensors file is found.
    When multiple exist, prefers root-level files over nested ones.
    """
    files = list_repo_files(repo_id)
    safetensors_files = [f for f in files if f.endswith(".safetensors")]

    if not safetensors_files:
        raise FileNotFoundError(
            f"No .safetensors file found in HuggingFace repo '{repo_id}'"
        )

    if len(safetensors_files) == 1:
        return safetensors_files[0]

    root_files = [f for f in safetensors_files if "/" not in f]
    if root_files:
        return root_files[0]

    return safetensors_files[0]


def resolve_lora_path(lora_path: str) -> str:
    """Resolve a LoRA path to a guaranteed local file.

    If *lora_path* is an existing local file it is returned as-is.
    Otherwise it is treated as an HF repo ID (``org/name``) and
    downloaded via ``hf_hub_download`` into the standard HF cache
    (``$HF_HOME/hub/``), same as base model weights.

    HuggingFace handles caching, deduplication, and concurrent-download
    locking internally so no extra copy or lock is needed here.
    """
    if Path(lora_path).is_file():
        return str(Path(lora_path).resolve())

    filename = _find_safetensors_filename(lora_path)

    logger.info(f"Resolving LoRA {lora_path}/{filename} via HF cache")
    local_path = hf_hub_download(repo_id=lora_path, filename=filename)
    logger.info(f"LoRA resolved: {local_path}")
    return local_path
