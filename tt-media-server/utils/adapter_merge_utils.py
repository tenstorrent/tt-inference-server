# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Merging of LoRA adapters back into their base model.

The merge owns every weight-related file and emits exactly one
`model.safetensors`; the config, tokenizer and all remaining repo files are
copied from the base model directory the fine-tune started from.
"""

import fnmatch
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Optional, Tuple

from utils.logger import TTLogger

logger = TTLogger()

MERGED_WEIGHTS_FILE_NAME = "model.safetensors"

# Checkpoint formats, i.e. the files the merge owns: never copied over from the
# base model, and never emitted besides MERGED_WEIGHTS_FILE_NAME.
WEIGHT_FILE_PATTERNS = (
    "*.safetensors",
    "*.safetensors.index.json",
    "*.bin",
    "*.bin.index.json",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.h5",
    "*.msgpack",
    "*.gguf",
)


@dataclass
class MergeResult:
    output_dir: str
    # Commit SHA of the base snapshot the copied files came from (None for a
    # plain local base directory), so a merged checkpoint can be traced back to
    # the exact revision it inherited its config and tokenizer from.
    base_revision: Optional[str]


def merge_adapter(
    base_model_name: str,
    adapter_path: str,
    output_dir: str,
    dtype_str: str = "torch.bfloat16",
) -> MergeResult:
    # Heavy deps imported lazily so they only load inside the merge subprocess.
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from tt_model_runners.forge_training_runners.torch_utils import resolve_dtype

    torch_dtype = resolve_dtype(dtype_str)

    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

    base_model_dir, base_revision = _resolve_base_model(base_model_name)

    logger.info(f"Loading base model '{base_model_name}' (dtype={torch_dtype})")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True
    )

    logger.info(f"Merging LoRA adapter from {adapter_path}")
    merged_model = PeftModel.from_pretrained(
        base_model, adapter_path
    ).merge_and_unload()

    # Checked before anything is written, so a merge that outgrew the base
    # metadata fails fast instead of leaving a mislabelled checkpoint on disk.
    _check_base_metadata_applies(merged_model, base_model_dir, torch_dtype)

    os.makedirs(output_dir, exist_ok=True)
    _save_merged_weights(merged_model, output_dir)
    _copy_base_model_files(base_model_dir, output_dir)

    logger.info(f"Adapter merge complete: {output_dir}")
    return MergeResult(output_dir=output_dir, base_revision=base_revision)


def _resolve_base_model(base_model_name: str) -> Tuple[str, Optional[str]]:
    """Locate the base model directory and the revision it pins.

    Metadata the fine-tune never pulled is fetched here, with the weights
    excluded so this stays cheap. No `cache_dir` is passed, so the snapshot
    resolves to the same one `from_pretrained` loads the base model from.
    """
    if os.path.isdir(base_model_name):
        base_model_dir = base_model_name
    else:
        from huggingface_hub import snapshot_download

        try:
            base_model_dir = snapshot_download(
                repo_id=base_model_name, ignore_patterns=list(WEIGHT_FILE_PATTERNS)
            )
        except Exception as e:
            logger.warning(
                f"Could not refresh '{base_model_name}' from the hub ({e}); "
                "falling back to the locally cached snapshot"
            )
            base_model_dir = snapshot_download(
                repo_id=base_model_name, local_files_only=True
            )

    # The hub caches as `<repo>/snapshots/<commit sha>`; a plain local
    # directory carries no revision.
    parent, name = os.path.split(os.path.normpath(base_model_dir))
    revision = name if os.path.basename(parent) == "snapshots" else None
    return base_model_dir, revision


def _check_base_metadata_applies(
    merged_model, base_model_dir: str, torch_dtype
) -> None:
    """Refuse to copy base metadata that no longer describes the merged model.

    Reusing the base config and tokenizer is only valid while the merge leaves
    the vocabulary untouched. An adapter that saves embedding layers or adds
    tokens resizes the merged model, and the copied metadata would then point
    loaders at the wrong vocab size.
    """
    config = {}
    config_path = os.path.join(base_model_dir, "config.json")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            config = json.load(f)

    # Multimodal configs (Gemma 3 and friends) nest the LM config.
    text_config = config.get("text_config") or {}
    vocab_size = config.get("vocab_size") or text_config.get("vocab_size")
    embeddings = merged_model.get_input_embeddings()
    embedding_rows = None if embeddings is None else embeddings.weight.shape[0]

    if vocab_size is None or embedding_rows is None:
        logger.warning(
            f"Could not compare vocab size against {base_model_dir}; "
            "copying base config and tokenizer unchecked"
        )
    elif vocab_size != embedding_rows:
        raise RuntimeError(
            f"Base config declares vocab_size={vocab_size} but the merged model has "
            f"{embedding_rows} embedding rows. The merge changed the vocabulary, so "
            "the base config and tokenizer no longer describe it and would have to "
            "be written by the merge itself."
        )

    # Only the config's dtype claim can go stale; the weights are written at the
    # dtype they were merged in either way, so this is not worth failing over.
    base_dtype = config.get("torch_dtype") or config.get("dtype")
    merged_dtype = str(torch_dtype).split(".")[-1]
    if base_dtype and base_dtype != merged_dtype:
        logger.warning(
            f"Base config reports dtype '{base_dtype}' but the merge ran in "
            f"'{merged_dtype}'; the copied config will understate the saved weights"
        )


def _save_merged_weights(merged_model, output_dir: str) -> None:
    """Write the merged weights as a single `model.safetensors` in `output_dir`.

    Staging on the same filesystem keeps the configs `save_pretrained` insists
    on writing out of the output, and lets the weights be moved into place
    rather than copied.
    """
    logger.info(f"Saving merged weights to {output_dir}")
    with tempfile.TemporaryDirectory(dir=output_dir) as staging_dir:
        merged_model.save_pretrained(
            staging_dir,
            safe_serialization=True,
            # Large enough that the save never shards, so the weights always
            # land in one file with no index alongside them.
            max_shard_size="1TB",
        )
        weights_path = os.path.join(staging_dir, MERGED_WEIGHTS_FILE_NAME)
        if not os.path.isfile(weights_path):
            raise RuntimeError(
                f"Merged model was not saved as a single {MERGED_WEIGHTS_FILE_NAME}; "
                f"got {sorted(os.listdir(staging_dir))}"
            )
        os.replace(weights_path, os.path.join(output_dir, MERGED_WEIGHTS_FILE_NAME))


def _copy_base_model_files(base_model_dir: str, output_dir: str) -> None:
    """Copy every non-weight file of the base model into `output_dir`."""
    logger.info(f"Copying base model files from {base_model_dir}")
    copied = 0
    for root, dirs, files in os.walk(base_model_dir):
        # Skips hub bookkeeping dirs such as .cache/ and .locks/.
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        rel_root = os.path.relpath(root, base_model_dir)
        for name in files:
            is_weight = any(fnmatch.fnmatch(name, p) for p in WEIGHT_FILE_PATTERNS)
            if name.startswith(".") or is_weight:
                continue
            destination = os.path.normpath(os.path.join(output_dir, rel_root, name))
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            # Snapshot entries are symlinks into the blob store, and copy2
            # follows them so the merged checkpoint is self-contained.
            shutil.copy2(os.path.join(root, name), destination)
            copied += 1

    if not copied:
        raise RuntimeError(
            f"No base model files found to copy from {base_model_dir}; "
            "the merged model would be missing its config and tokenizer"
        )
    logger.info(f"Copied {copied} base model file(s) to {output_dir}")
