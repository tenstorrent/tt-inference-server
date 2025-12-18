#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Minimal example for running Llama-3.1-8B-Instruct on Tenstorrent hardware
using the TT vLLM plugin.

This example demonstrates basic offline inference with the plugin.

Usage:
    python examples/llama_3_1_8b_instruct.py
"""

import os

# Enable vLLM v1 architecture
os.environ["VLLM_USE_V1"] = "1"
os.environ["HF_MODEL"] = "meta-llama/Llama-3.1-8B-Instruct"

from vllm import LLM, SamplingParams


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

    print("Initializing LLM with TT platform...")
    print("Note: The TT plugin should be automatically discovered via entry points")

    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        max_model_len=65536,
        max_num_seqs=32,
        enable_chunked_prefill=False,
        block_size=64,
        max_num_batched_tokens=65536,
        seed=9472,
    )

    print("Model loaded successfully!")

    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=50,
    )

    # Test prompts
    prompts = [
        "Tell me a joke.",
        "What is the capital of Italy?",
        "Explain quantum computing in simple terms.",
        "How do you make pancakes?",
        "What is the tallest mountain?",
        "Describe a sunset.",
        "Who wrote 1984?",
        "What is the speed of light?",
        "Where is the Great Barrier Reef located?",
        "Write a short poem about rain.",
        "What causes seasons?",
        "Define artificial intelligence.",
        "List three primary colors.",
        "Translate 'hello' to Spanish.",
        "Why do cats purr?",
        "What is photosynthesis?",
        "Name a famous scientist.",
        "What is the nearest planet to Earth?",
        "Summarize the plot of Romeo and Juliet.",
        "Give a synonym for 'happy'.",
        "What is the formula for water?",
        "List the continents.",
        "What is blockchain?",
        "Describe the taste of chocolate.",
        "Who painted the Mona Lisa?",
        "How old is the universe?",
        "Name a musical instrument.",
        "What is the freezing point of water?",
        "Explain gravity in one sentence.",
        "Who is the president of the USA?",
        "How many hours are in a day?",
        "What language is spoken in Brazil?",
        "What is pi?",
    ]

    print("\nGenerating responses...")
    outputs = llm.generate(prompts, sampling_params)

    # Print results
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    for i, output in enumerate(outputs):
        prompt = prompts[i]
        generated_text = output.outputs[0].text
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated_text}")
        print(f"Token IDs: {output.outputs[0].token_ids[:10]}...")  # First 10 tokens
    print("=" * 60)
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
