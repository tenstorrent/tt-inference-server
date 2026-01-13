#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import importlib
import sys
from typing import Iterable, List, Optional, Set, Tuple

import pytest

from workflows.model_spec import MODEL_SPECS
from workflows.workflow_types import DeviceTypes


def _find_model_id(
    *, model_name: str, device: DeviceTypes, impl_name: Optional[str] = None
) -> str:
    for model_id, spec in MODEL_SPECS.items():
        if spec.model_name != model_name:
            continue
        if spec.device_type != device:
            continue
        if impl_name is not None and spec.impl.impl_name != impl_name:
            continue
        return model_id
    raise AssertionError(
        f"Could not find model_id for model_name={model_name!r}, device={device}, impl_name={impl_name!r}"
    )


def _import_benchmark_config(monkeypatch):
    # benchmark_config builds BENCHMARK_CONFIGS at import-time and optionally
    # skips sweeps based on ONLY_BENCHMARK_TARGETS. We want sweeps enabled here.
    #
    # NOTE: bool(os.getenv("ONLY_BENCHMARK_TARGETS")) treats any non-empty string
    # as True (including "0"), so we must *unset* the env var.
    monkeypatch.delenv("ONLY_BENCHMARK_TARGETS", raising=False)

    module_name = "benchmarking.benchmark_config"
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


def _allowed_max_concurrency(
    *, isl: int, osl: int, max_context: int, model_max_concurrency: int
) -> int:
    total_seq_len = isl + osl
    if total_seq_len > max_context:
        return 1
    return min(max_context // total_seq_len, model_max_concurrency)


def _extract_sweep_triplets(
    sweep_params: Iterable, *, include_images: bool = True
) -> List[Tuple[int, int, int]]:
    triplets: Set[Tuple[int, int, int]] = set()
    for p in sweep_params:
        # CNN / audio / embedding sweeps may not have isl/osl at all
        if p.isl is None or p.osl is None or p.max_concurrency is None:
            continue
        if not include_images and getattr(p, "task_type", "text") == "image":
            continue
        triplets.add((int(p.isl), int(p.osl), int(p.max_concurrency)))
    return sorted(triplets)


@pytest.mark.parametrize(
    "model_name,device,expect_image_sweeps",
    [
        ("Qwen3-8B", DeviceTypes.N150, False),
        ("gemma-3-4b-it", DeviceTypes.N150, True),
    ],
)
def test_benchmark_configs_selected_models_print_sweeps(
    monkeypatch, model_name: str, device: DeviceTypes, expect_image_sweeps: bool
):
    benchmark_config = _import_benchmark_config(monkeypatch)

    model_id = _find_model_id(
        model_name=model_name, device=device, impl_name="tt-transformers"
    )
    assert model_id in benchmark_config.BENCHMARK_CONFIGS

    config = benchmark_config.BENCHMARK_CONFIGS[model_id]
    assert config.model_id == model_id
    assert len(config.tasks) == 2  # perf_reference + sweeps

    perf_ref_task = config.tasks[0]
    sweep_task = config.tasks[1]

    assert device in perf_ref_task.param_map
    assert device in sweep_task.param_map
    assert isinstance(sweep_task.param_map[device], list)
    assert len(sweep_task.param_map[device]) > 0

    model_spec = MODEL_SPECS[model_id]
    max_context = model_spec.device_model_spec.max_context
    model_max_concurrency = model_spec.device_model_spec.max_concurrency

    # Validate that text sweep params never exceed context/model limits.
    for p in sweep_task.param_map[device]:
        if p.isl is None or p.osl is None or p.max_concurrency is None:
            continue
        if getattr(p, "task_type", "text") != "text":
            continue
        allowed = _allowed_max_concurrency(
            isl=p.isl,
            osl=p.osl,
            max_context=max_context,
            model_max_concurrency=model_max_concurrency,
        )
        assert p.max_concurrency <= allowed

    # Spot-check concurrency capping for a couple of representative pairs on Qwen3-8B
    if model_name == "Qwen3-8B":
        sweep_params = sweep_task.param_map[device]
        pairs_to_expected = {
            (128, 128): [1, 32],
            (65536, 128): [1],
        }
        for (isl, osl), expected_concurrency in pairs_to_expected.items():
            got = sorted(
                p.max_concurrency
                for p in sweep_params
                if p.isl == isl and p.osl == osl and getattr(p, "task_type", "text") == "text"
            )
            assert got == expected_concurrency

    # For VLMs, confirm we actually got image sweeps and that they are capped properly
    if expect_image_sweeps:
        image_params = [
            p
            for p in sweep_task.param_map[device]
            if getattr(p, "task_type", "text") == "image"
        ]
        assert len(image_params) > 0

        for p in image_params:
            # gemma-3 vision tokens are fixed at 256 per image (see calculate_vision_tokens)
            vision_tokens = 256 * int(p.images_per_prompt or 0)
            total_seq_len = int(p.isl) + int(p.osl) + vision_tokens
            if total_seq_len > max_context:
                allowed = 1
            else:
                allowed = min(max_context // total_seq_len, model_max_concurrency)
            assert int(p.max_concurrency) <= int(allowed)

    # Print all (isl, osl, concurrency) combinations for visibility when running with `pytest -s`.
    triplets = _extract_sweep_triplets(
        sweep_task.param_map[device], include_images=expect_image_sweeps
    )
    print(f"\nModel: {model_name} ({model_id}) device={device.name}")
    print("isl,osl,max_concurrency:")
    for isl, osl, c in triplets:
        print(f"{isl},{osl},{c}")

