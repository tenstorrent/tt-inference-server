# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Merging of LoRA adapters back into their base model.

The merge owns every weight-related file and emits exactly one
`model.safetensors`; the config, tokenizer and all remaining repo files are
copied from the base model directory the fine-tune started from.

The merge is meant to run under the *inference* `transformers` version (4.x),
not the training/forge one (5.x): the checkpoint is written by the same major
version that vLLM later serves it with, so the artifact is natively loadable
instead of relying on the copied base metadata to disguise a newer artifact as
an older one. `run_merge_subprocess` launches this module as a CLI in a
dedicated 4.x venv; `_verify_merged_model_loads` then gates completion by
loading the result back in that same version before it is marked servable.
"""

import argparse
import fnmatch
import gc
import json
import os
import shutil
import subprocess
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


def _resolve_dtype(dtype_str: str):
    """Map a `"torch.<dtype>"` string to the actual dtype.

    Inlined (rather than imported from the forge training package) so this
    module carries no dependency on the tt-forge / torch-xla stack and can run
    in the minimal transformers-4.x merge venv.
    """
    import torch

    mapping = {
        "torch.bfloat16": torch.bfloat16,
        "torch.float32": torch.float32,
        "torch.float16": torch.float16,
    }
    if dtype_str not in mapping:
        raise ValueError(
            f"Unsupported dtype '{dtype_str}', must be one of {list(mapping)}"
        )
    return mapping[dtype_str]


def merge_adapter(
    base_model_name: str,
    adapter_path: str,
    output_dir: str,
    dtype_str: str = "torch.bfloat16",
    verify_load: bool = True,
) -> MergeResult:
    # Heavy deps imported lazily so they only load inside the merge subprocess.
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    torch_dtype = _resolve_dtype(dtype_str)

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

    # Free the in-memory model before the gate reloads the checkpoint, so peak
    # host memory stays at roughly one model rather than two.
    del merged_model, base_model
    gc.collect()

    if verify_load:
        _verify_merged_model_loads(output_dir, torch_dtype)

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


def _verify_merged_model_loads(output_dir: str, torch_dtype) -> None:
    """Gate: prove the merged checkpoint loads in this (inference) transformers.

    Copying the base config makes `config.json`/tokenizer parseable, but says
    nothing about whether the safetensors *tensor keys* written here match what
    the serving loader expects. Running this under the same major `transformers`
    version that vLLM serves with turns a late, silent serving failure into an
    immediate merge failure.

    `unexpected_keys` / `mismatched_keys` mean the on-disk weights don't line up
    with the model definition (real format drift), so they fail hard.
    `missing_keys` is only warned about because it is usually tied weights
    (e.g. `lm_head` sharing the embedding) rather than a genuine problem.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Verifying merged checkpoint loads back from {output_dir}")
    model, loading_info = AutoModelForCausalLM.from_pretrained(
        output_dir,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        output_loading_info=True,
    )
    del model
    gc.collect()

    unexpected = loading_info.get("unexpected_keys") or []
    mismatched = loading_info.get("mismatched_keys") or []
    error_msgs = loading_info.get("error_msgs") or []
    missing = loading_info.get("missing_keys") or []

    if unexpected or mismatched or error_msgs:
        raise RuntimeError(
            "Merged checkpoint did not load cleanly under the inference "
            f"transformers version: unexpected_keys={unexpected}, "
            f"mismatched_keys={mismatched}, error_msgs={error_msgs}. The weights "
            "written by the merge do not match what the serving loader expects."
        )
    if missing:
        logger.warning(
            f"Merged checkpoint reports missing_keys on reload: {missing}. This is "
            "usually tied weights, but confirm against the served model."
        )

    # Config + tokenizer must also parse under this version, not just the weights.
    AutoTokenizer.from_pretrained(output_dir)
    logger.info("Merged checkpoint verified: loads under the inference transformers")


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


def run_merge_subprocess(
    base_model_name: str,
    adapter_path: str,
    output_dir: str,
    *,
    python_executable: str,
    cwd: str,
    dtype_str: str = "torch.bfloat16",
) -> MergeResult:
    """Run `merge_adapter` in a separate interpreter and return its result.

    The child is `python_executable` — the dedicated transformers-4.x merge
    venv — so the merge (and its load-test gate) run under the version that
    serves the model, not the forge 5.x stack the caller lives in. A fresh
    process also fully reclaims the base-model memory on exit and isolates a
    merge crash/OOM from the API process (the reasons the old in-process pool
    existed).

    The child writes its `MergeResult` to a private temp file rather than
    stdout, so transformers/hub log noise can't corrupt the parsed result.
    """
    os.makedirs(output_dir, exist_ok=True)
    result_fd, result_path = tempfile.mkstemp(suffix=".json", prefix="merge_result_")
    os.close(result_fd)
    try:
        cmd = [
            python_executable,
            "-m",
            "utils.adapter_merge_utils",
            "--base-model",
            base_model_name,
            "--adapter-path",
            adapter_path,
            "--output-dir",
            output_dir,
            "--dtype",
            dtype_str,
            "--result-json",
            result_path,
        ]
        # Point PYTHONPATH only at the app dir so the child imports the merge
        # venv's transformers/peft, and drop TT_METAL_HOME so nothing pulls the
        # forge (5.x) site-packages back onto the path.
        env = {**os.environ, "PYTHONPATH": cwd}
        env.pop("TT_METAL_HOME", None)

        logger.info(f"Launching adapter merge subprocess: {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Adapter merge subprocess failed (exit {proc.returncode}).\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )

        with open(result_path) as f:
            payload = json.load(f)
        return MergeResult(
            output_dir=payload["output_dir"],
            base_revision=payload.get("base_revision"),
        )
    finally:
        try:
            os.remove(result_path)
        except OSError:
            pass


def main() -> None:
    """CLI entrypoint invoked as `python -m utils.adapter_merge_utils`.

    Runs inside the transformers-4.x merge venv; see `run_merge_subprocess`.
    """
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into its base model")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dtype", default="torch.bfloat16")
    parser.add_argument("--result-json", required=True)
    parser.add_argument(
        "--no-verify-load",
        action="store_true",
        help="Skip the load-test gate (not recommended).",
    )
    args = parser.parse_args()

    result = merge_adapter(
        args.base_model,
        args.adapter_path,
        args.output_dir,
        args.dtype,
        verify_load=not args.no_verify_load,
    )
    with open(args.result_json, "w") as f:
        json.dump(
            {"output_dir": result.output_dir, "base_revision": result.base_revision}, f
        )


if __name__ == "__main__":
    main()
