#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import os

# Enable vLLM v1 architecture
os.environ["VLLM_USE_V1"] = "1"
os.environ["HF_MODEL"] = "BAAI/bge-large-en-v1.5"

from vllm import LLM


def main():
    from vllm.platforms import current_platform

    # Verify your platform is detected
    assert current_platform.device_name == "tt", (
        f"Expected 'tt' platform, got {current_platform.device_name}"
    )
    assert current_platform.is_out_of_tree(), "Platform should be OOT"
    print(
        f"Using platform: {current_platform.device_name} (OOT: {current_platform.is_out_of_tree()})"
    )

    print("Initializing BGE embedding model with TT platform...")
    print("Note: The TT plugin should be automatically discovered via entry points")

    llm = LLM(
        model="BAAI/bge-large-en-v1.5",
        max_model_len=384,
        max_num_seqs=8,
        max_num_batched_tokens=4096,
    )

    print("Model loaded successfully!")

    # Test prompts for embedding generation
    prompts = ["What is the capital of France?"] * 64

    # 8 by 8
    for i in range(0, len(prompts), 8):
        batch = prompts[i : i + 8]
        result = llm.embed(batch)
        print(f"Batch {i // 8 + 1} result length: {len(result)}")


if __name__ == "__main__":
    main()
