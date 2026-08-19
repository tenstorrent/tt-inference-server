# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Merge a LoRA adapter into its base model, producing a servable checkpoint.

This runs under the *inference* transformers version, in a dedicated venv (see
`scripts/build_merge_venv.sh`). 
"""

import argparse
import gc
import os
import subprocess
import sys
from typing import Optional

from utils.logger import TTLogger

logger = TTLogger()


def _merge_venv_python() -> str:
    """Interpreter that runs the merge.

    Falls back to the current interpreter when `ADAPTER_MERGE_PYTHON` is unset
    (e.g. local dev), so the merge still runs, just under whatever `transformers`
    this process has.
    """
    return os.getenv("ADAPTER_MERGE_PYTHON", sys.executable)


def _app_root() -> str:
    """Dir from which `python -m utils.adapter_merge_utils` resolves, used as the
    child's cwd. Derived from *this* module's location so it stays correct
    regardless of which caller launches the subprocess."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_dtype(dtype_str: str):
    """Map a `"torch.<dtype>"` string to the actual dtype (torch imported lazily
    so this module stays importable outside the merge venv)."""
    import torch

    mapping = {
        "torch.bfloat16": torch.bfloat16,
        "torch.float16": torch.float16,
        "torch.float32": torch.float32,
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
) -> None:
    """Merge `adapter_path` into `base_model_name`, writing to `output_dir`."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

    torch_dtype = _resolve_dtype(dtype_str)

    logger.info(f"Loading base model '{base_model_name}' (dtype={torch_dtype})")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True
    )

    logger.info(f"Merging LoRA adapter from {adapter_path}")
    merged_model = PeftModel.from_pretrained(base_model, adapter_path).merge_and_unload()

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving merged model to {output_dir}")
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    # save_pretrained writes the config + weights but not the tokenizer, so pull
    # it from the base model to keep the checkpoint self-contained.
    AutoTokenizer.from_pretrained(base_model_name).save_pretrained(output_dir)

    # Free the model before the gate reloads it, keeping peak memory at ~one model.
    del merged_model, base_model
    gc.collect()

    if verify_load:
        _verify_merged_model_loads(output_dir, torch_dtype)
    logger.info(f"Adapter merge complete: {output_dir}")


def _verify_merged_model_loads(output_dir: str, torch_dtype) -> None:
    """Reload the merged checkpoint to prove it is servable before completion."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Verifying merged checkpoint loads back from {output_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        output_dir, torch_dtype=torch_dtype, low_cpu_mem_usage=True
    )
    del model
    gc.collect()
    AutoTokenizer.from_pretrained(output_dir)
    logger.info("Merged checkpoint verified: loads under the inference transformers")


def run_merge_subprocess(
    base_model_name: str,
    adapter_path: str,
    output_dir: str,
    *,
    python_executable: Optional[str] = None,
    cwd: Optional[str] = None,
    dtype_str: str = "torch.bfloat16",
) -> None:
    """Run `merge_adapter` in the merge venv (a separate interpreter).
    """
    if python_executable is None:
        python_executable = _merge_venv_python()
    if cwd is None:
        cwd = _app_root()
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
    ]
    # Point PYTHONPATH only at the app dir so the child imports the merge venv's
    # transformers/peft, and drop TT_METAL_HOME so the forge site-packages
    # can't leak back onto the path.
    env = {**os.environ, "PYTHONPATH": cwd}
    env.pop("TT_METAL_HOME", None)

    logger.info(f"Launching adapter merge subprocess: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Adapter merge subprocess failed (exit {proc.returncode}).\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )


def main() -> None:
    """CLI entrypoint, invoked as `python -m utils.adapter_merge_utils` in the
    merge venv (see `run_merge_subprocess`)."""
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into its base model")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dtype", default="torch.bfloat16")
    parser.add_argument(
        "--no-verify-load",
        action="store_true",
        help="Skip the load-test gate (not recommended).",
    )
    args = parser.parse_args()

    merge_adapter(
        args.base_model,
        args.adapter_path,
        args.output_dir,
        args.dtype,
        verify_load=not args.no_verify_load,
    )


if __name__ == "__main__":
    main()
