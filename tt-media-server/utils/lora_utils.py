# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import re
from functools import lru_cache
from pathlib import Path

from huggingface_hub import ModelCard, hf_hub_download, list_repo_files
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

from utils.logger import TTLogger

logger = TTLogger()

_SAFETENSORS_TRIGGER_KEYS = (
    "trigger_word",
    "trigger_words",
    "ss_trigger_words",
    "activation_text",
    "instance_prompt",
)

_README_TRIGGER_PATTERNS = (
    re.compile(
        r"\*{0,2}trigger\s*words?\*{0,2}\s*[:=]\s*\*{0,2}\s*`?([^`\n]+?)`?\*{0,2}\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r"\*{0,2}activation\s*(?:tokens?|words?)\*{0,2}\s*[:=]\s*\*{0,2}\s*`?([^`\n]+?)`?\*{0,2}\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r'[Uu]se\s+[`"\']([^`"\']+)[`"\']\s+(?:in\s+(?:your\s+)?prompt|to\s+activate)',
        re.MULTILINE,
    ),
)

_TRIGGER_CACHE_SIZE = 64


def _parse_trigger_list(raw: str) -> list[str]:
    return [t.strip() for t in raw.split(",") if t.strip()]


def _find_safetensors_filename(repo_id: str) -> str:
    try:
        files = list_repo_files(repo_id)
    except RepositoryNotFoundError as e:
        raise FileNotFoundError(f"Repository not found: {repo_id}") from e

    safetensors_files = [f for f in files if f.endswith(".safetensors")]

    if not safetensors_files:
        raise FileNotFoundError(f"No .safetensors file in repository: {repo_id}")

    if len(safetensors_files) == 1:
        return safetensors_files[0]

    root_files = [f for f in safetensors_files if "/" not in f]
    return root_files[0] if root_files else safetensors_files[0]


def _get_triggers_from_safetensors(local_path: str) -> list[str] | None:
    try:
        from safetensors import safe_open

        with safe_open(local_path, framework="numpy") as f:
            metadata = f.metadata()
    except (ImportError, FileNotFoundError, OSError) as e:
        logger.debug(f"Failed to read safetensors file {local_path}: {e}")
        return None

    if not metadata:
        return None

    for key in _SAFETENSORS_TRIGGER_KEYS:
        if value := metadata.get(key):
            return _parse_trigger_list(value)

    return None


def _get_triggers_from_model_card(repo_id: str) -> list[str] | None:
    try:
        card = ModelCard.load(repo_id)
        if card.data and (prompt := getattr(card.data, "instance_prompt", None)):
            return _parse_trigger_list(prompt)
    except Exception as e:
        logger.debug(f"Failed to load model card for {repo_id}: {e}")

    return None


def _get_triggers_from_readme(repo_id: str) -> list[str] | None:
    try:
        readme_path = hf_hub_download(repo_id=repo_id, filename="README.md")
        text = Path(readme_path).read_text(encoding="utf-8")
    except EntryNotFoundError:
        return None
    except Exception as e:
        logger.debug(f"Failed to fetch README for {repo_id}: {e}")
        return None

    for pattern in _README_TRIGGER_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        triggers = _parse_trigger_list(match.group(1).rstrip("."))
        if triggers:
            return triggers

    return None


def _get_triggers_from_safetensors_repo(repo_id: str) -> list[str] | None:
    try:
        filename = _find_safetensors_filename(repo_id)
        local_file = hf_hub_download(repo_id=repo_id, filename=filename)
        return _get_triggers_from_safetensors(local_file)
    except Exception:
        return None


def resolve_lora_path(lora_path: str) -> str:
    local = Path(lora_path)

    if local.is_file():
        return str(local.resolve())

    filename = _find_safetensors_filename(lora_path)
    logger.info(f"Resolving LoRA {lora_path}/{filename} via HF cache")

    try:
        local_path = hf_hub_download(repo_id=lora_path, filename=filename)
    except Exception as e:
        raise FileNotFoundError(f"Failed to download LoRA: {lora_path}") from e

    return local_path


@lru_cache(maxsize=_TRIGGER_CACHE_SIZE)
def get_lora_trigger_words(lora_path: str) -> tuple[str, ...] | None:
    if Path(lora_path).is_file():
        triggers = _get_triggers_from_safetensors(lora_path)
        return tuple(triggers) if triggers else None

    sources = [
        ("model card", lambda: _get_triggers_from_model_card(lora_path)),
        ("safetensors", lambda: _get_triggers_from_safetensors_repo(lora_path)),
        ("README", lambda: _get_triggers_from_readme(lora_path)),
    ]

    for source_name, get_triggers in sources:
        if triggers := get_triggers():
            logger.info(f"Found triggers for {lora_path} ({source_name}): {triggers}")
            return tuple(triggers)

    logger.debug(f"No trigger words found for {lora_path}")
    return None


def prepare_prompt_with_lora(
    prompt: str,
    lora_path: str | None,
    *,
    auto_inject: bool = True,
) -> str:
    if not lora_path or not prompt or not auto_inject:
        return prompt

    triggers = get_lora_trigger_words(lora_path)
    if not triggers:
        return prompt

    trigger = triggers[0]

    if trigger.lower() in prompt.lower():
        return prompt

    logger.info(f"Injecting LoRA trigger '{trigger}' into prompt")
    return f"{prompt}, {trigger}"
