#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Offline plugin test for the optimized single-chip BGE-M3 embedding model.

Runs ``llm.embed([...])`` through the TT vLLM plugin and confirms that the
optimized class (``BgeM3ForEmbeddingOptimized``, registered as
``TTXLMRobertaModel``) loads and returns embeddings.

Fixed serving case: batch 1, ISL 512, single chip (TT_VISIBLE_DEVICES=0).
"""

import os

# Single chip + vLLM v1 architecture.
os.environ.setdefault("TT_VISIBLE_DEVICES", "0")
os.environ["VLLM_USE_V1"] = "1"
os.environ["HF_MODEL"] = "BAAI/bge-m3"

from vllm import LLM


def main():
    from vllm.platforms import current_platform

    assert current_platform.device_name == "tt", (
        f"Expected 'tt' platform, got {current_platform.device_name}"
    )
    assert current_platform.is_out_of_tree(), "Platform should be OOT"
    print(
        f"Using platform: {current_platform.device_name} (OOT: {current_platform.is_out_of_tree()})"
    )

    print("Initializing optimized BGE-M3 embedding model with TT platform...")

    llm = LLM(
        model="BAAI/bge-m3",
        max_model_len=512,
        max_num_seqs=1,
        max_num_batched_tokens=512,
    )

    print("Model loaded successfully!")

    # A couple of distinct prompts; the optimized model is fixed at B1, so vLLM
    # processes them one at a time.
    prompts = [
        "What is the capital of France?",
        "Tenstorrent builds AI accelerators.",
    ]

    outputs = llm.embed(prompts)
    print(f"Got {len(outputs)} embeddings")
    for i, output in enumerate(outputs):
        emb = output.outputs.embedding
        print(f"  [{i}] dim={len(emb)} first5={emb[:5]}")


if __name__ == "__main__":
    main()
