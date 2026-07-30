# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import os

from utils.logger import TTLogger

logger = TTLogger()


def merge_adapter(
    base_model_name: str,
    adapter_path: str,
    output_dir: str,
    dtype_str: str = "torch.bfloat16",
) -> str:
    """Merge a LoRA adapter into its base model and write a full HF checkpoint.

    Runs purely on CPU. The output is a standard HuggingFace checkpoint (config
    + safetensors + tokenizer) servable via the vLLM container's
    --host-weights-dir. Designed to run in a spawned subprocess: it is a
    top-level function with picklable (string) arguments.
    """
    # Heavy deps imported lazily so they only load inside the merge subprocess.
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tt_model_runners.forge_training_runners.torch_utils import resolve_dtype

    torch_dtype = resolve_dtype(dtype_str)

    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

    logger.info(f"Loading base model '{base_model_name}' (dtype={torch_dtype})")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True
    )

    logger.info(f"Merging LoRA adapter from {adapter_path}")
    merged_model = PeftModel.from_pretrained(base_model, adapter_path).merge_and_unload()

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving merged model to {output_dir}")
    # save_pretrained also writes config.json / generation_config.json; the
    # tokenizer is saved separately since peft does not persist it.
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(base_model_name).save_pretrained(output_dir)

    logger.info(f"Adapter merge complete: {output_dir}")
    return output_dir
