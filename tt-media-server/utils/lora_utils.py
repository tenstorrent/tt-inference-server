# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files


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

    If lora_path is an existing local file, returns the resolved absolute path.
    Otherwise treats it as an HF repo ID, discovers the safetensors filename,
    and downloads via hf_hub_download (cached by HF hub).
    """
    if Path(lora_path).is_file():
        return str(Path(lora_path).resolve())

    filename = _find_safetensors_filename(lora_path)
    cache_dir = os.environ.get("HF_HOME")
    return hf_hub_download(repo_id=lora_path, filename=filename, cache_dir=cache_dir)
