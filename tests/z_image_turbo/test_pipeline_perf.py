# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
"""
Full Z-Image-Turbo pipeline perf test -- warmup once, then run 5 perf
iterations with distinct prompts and report timing.

This is a perf-only test: it verifies that the pipeline produces valid
512x512 images and measures end-to-end latency. No PCC checks.
"""

import time

import pytest
from PIL import Image

from models.demos.z_image_turbo.tt.z_image_turbo import ZImageTurbo

PROMPTS = [
    "Cinematic cyberpunk street, neon rain puddles, glowing signs, 8k resolution.",
    "Whimsical treehouse village, golden hour lighting, floating lanterns, digital art.",
    "Majestic snow leopard, piercing blue eyes, mountain peak, hyper-realistic fur.",
    "Vintage oil painting, stormy sea, wooden ship, dramatic lightning strikes.",
    "Astronaut sitting on moon, drinking coffee, earth in background, surreal.",
]

NUM_PERF_ITERATIONS = 5
INFERENCE_STEPS = 9


@pytest.mark.parametrize(
    "mesh_device",
    [ZImageTurbo.DEFAULT_MESH_SHAPE],
    indirect=True,
)
def test_pipeline_perf(mesh_device, tmp_path):
    pipeline = ZImageTurbo(mesh_device=mesh_device)

    # Warmup: compiles all models, captures traces
    print("\n--- Warmup ---")
    t0 = time.time()
    warmup_image = pipeline.warmup(steps=INFERENCE_STEPS, seed=42)
    warmup_elapsed = time.time() - t0
    assert isinstance(warmup_image, Image.Image), "Warmup did not return a PIL Image"
    assert warmup_image.size == (512, 512), f"Expected 512x512, got {warmup_image.size}"
    print(f"Warmup completed in {warmup_elapsed:.2f}s")

    # Perf iterations
    print(f"\n--- Perf ({NUM_PERF_ITERATIONS} iterations) ---")
    iteration_times = []
    for i in range(NUM_PERF_ITERATIONS):
        prompt = PROMPTS[i % len(PROMPTS)]
        seed = 42 + i

        t0 = time.time()
        image = pipeline(prompt, steps=INFERENCE_STEPS, seed=seed)
        elapsed = time.time() - t0
        iteration_times.append(elapsed)

        assert isinstance(image, Image.Image), (
            f"Iteration {i} did not return a PIL Image"
        )
        assert image.size == (512, 512), (
            f"Iteration {i}: expected 512x512, got {image.size}"
        )

        out_path = tmp_path / f"perf_iter_{i}.png"
        image.save(str(out_path))

        print(
            f"  Iteration {i + 1}/{NUM_PERF_ITERATIONS}: "
            f"{elapsed * 1000:.0f} ms  [{prompt[:50]}...]"
        )

    avg_ms = sum(iteration_times) / len(iteration_times) * 1000
    min_ms = min(iteration_times) * 1000
    max_ms = max(iteration_times) * 1000
    print(f"\nPerf summary ({NUM_PERF_ITERATIONS} iterations):")
    print(f"  Avg: {avg_ms:.0f} ms")
    print(f"  Min: {min_ms:.0f} ms")
    print(f"  Max: {max_ms:.0f} ms")
