#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import os

# Enable vLLM v1 architecture
os.environ["VLLM_USE_V1"] = "1"
os.environ["HF_MODEL"] = "Qwen/Qwen3-Embedding-8B"

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

    print("Initializing Qwen3-Embedding-8B model with TT platform...")
    print("Note: The TT plugin should be automatically discovered via entry points")

    llm = LLM(
        model="Qwen/Qwen3-Embedding-8B",
        max_model_len=8192,  # Qwen3-Embedding supports up to 8192 tokens
        max_num_seqs=8,
        max_num_batched_tokens=65536,  # 8 * 8192
        # Force vLLM to use Qwen3Model architecture (embedding) instead of Qwen3ForCausalLM
        hf_overrides={"architectures": ["Qwen3Model"]},
    )

    print("Model loaded successfully!")

    # Test prompts for embedding generation
    prompts = ["What is the capital of France?"] * 64

    # Process in batches of 8
    for i in range(0, len(prompts), 8):
        batch = prompts[i : i + 8]
        outputs = llm.embed(batch)
        print(f"Batch {i // 8 + 1} result length: {len(outputs)}")
        if len(outputs) > 0:
            # Access embedding via output.outputs.embedding (list of floats)
            # or output.outputs.data if it's a PoolingOutput
            output = outputs[0]
            if hasattr(output.outputs, "embedding"):
                # EmbeddingOutput has embedding attribute (list of floats)
                embedding = output.outputs.embedding
                print(f"  Embedding dimension: {len(embedding)}")
                print(f"  Embedding (first 10 dims): {embedding[:10]}")
            elif hasattr(output.outputs, "data"):
                # PoolingOutput has data attribute (torch.Tensor)
                embedding = output.outputs.data
                print(f"  Embedding shape: {embedding.shape}")
                print(f"  Embedding dimension: {embedding.shape[0]}")
                print(f"  Embedding (first 10 dims): {embedding[:10]}")
            else:
                print(f"  Output type: {type(output.outputs)}")
                print(f"  Output attributes: {dir(output.outputs)}")


if __name__ == "__main__":
    main()
