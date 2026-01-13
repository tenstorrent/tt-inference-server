# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from __future__ import annotations

from benchmarking.benchmark_config import expand_concurrency_sweep_params
from workflows.utils_report import BenchmarkTaskParams


def test_expand_concurrency_sweeps_text_includes_powers_of_two_and_allowed_max():
    # allowed_max by context: 4096 // (128+128) = 16, and model_max=32 => allowed_max=16
    params = [
        BenchmarkTaskParams(
            isl=128,
            osl=128,
            max_concurrency=1,
            num_prompts=8,
            task_type="text",
        )
    ]
    expanded = expand_concurrency_sweep_params(
        params,
        max_context=4096,
        model_max_concurrency=32,
        model_name="Qwen3-8B",
        candidate_concurrencies=[1, 2, 4, 8, 16, 32],
    )
    got = sorted(p.max_concurrency for p in expanded if p.isl == 128 and p.osl == 128)
    assert got == [1, 2, 4, 8, 16]


def test_expand_concurrency_sweeps_text_includes_non_power_of_two_allowed_max():
    # allowed_max by context: 5000 // 256 = 19, and model_max=32 => allowed_max=19
    params = [
        BenchmarkTaskParams(
            isl=128,
            osl=128,
            max_concurrency=1,
            num_prompts=8,
            task_type="text",
        )
    ]
    expanded = expand_concurrency_sweep_params(
        params,
        max_context=5000,
        model_max_concurrency=32,
        model_name="Qwen3-8B",
        candidate_concurrencies=[1, 2, 4, 8, 16, 32],
    )
    got = sorted(p.max_concurrency for p in expanded if p.isl == 128 and p.osl == 128)
    # powers-of-2 <= 19 plus allowed_max itself
    assert got == [1, 2, 4, 8, 16, 19]


def test_expand_concurrency_sweeps_image_accounts_for_vision_tokens():
    # For gemma-3, calculate_vision_tokens returns 256 tokens/image.
    # total_seq_len = isl + osl + vision_tokens = 128+128+256 = 512
    # allowed_max by context: 4096 // 512 = 8, model_max=32 => allowed_max=8
    params = [
        BenchmarkTaskParams(
            isl=128,
            osl=128,
            max_concurrency=1,
            num_prompts=8,
            task_type="image",
            image_height=512,
            image_width=512,
            images_per_prompt=1,
        )
    ]
    expanded = expand_concurrency_sweep_params(
        params,
        max_context=4096,
        model_max_concurrency=32,
        model_name="google/gemma-3-4b-it",
        candidate_concurrencies=[1, 2, 4, 8, 16, 32],
    )
    got = sorted(p.max_concurrency for p in expanded if p.task_type == "image")
    assert got == [1, 2, 4, 8]

