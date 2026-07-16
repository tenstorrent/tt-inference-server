#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import importlib
import sys
from dataclasses import replace
from typing import Iterable, List, Optional, Set, Tuple

import pytest

from workflows.model_spec import MODEL_SPECS
from workflows.utils_report import BenchmarkTaskParams, BenchmarkTaskParamsCNN
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
    # benchmark_config optionally skips sweeps based on ONLY_BENCHMARK_TARGETS.
    # We want sweeps enabled here.
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
    config = benchmark_config.get_benchmark_config(MODEL_SPECS[model_id])
    assert config.model_id == model_id
    assert len(config.tasks) == 3  # perf_reference + sweeps + structured_output

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
            (65536, 128): [],
        }
        for (isl, osl), expected_concurrency in pairs_to_expected.items():
            got = sorted(
                p.max_concurrency
                for p in sweep_params
                if p.isl == isl
                and p.osl == osl
                and getattr(p, "task_type", "text") == "text"
            )
            assert got == expected_concurrency

    # For VLMs, confirm we actually got image sweeps and that they are capped properly
    if expect_image_sweeps:
        image_params = [
            p
            for p in sweep_task.param_map[device]
            if getattr(p, "task_type", "text") == "vlm"
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


@pytest.mark.parametrize(
    "model_name,device",
    [
        ("Qwen3-8B", DeviceTypes.N150),
        ("gemma-3-4b-it", DeviceTypes.N150),
    ],
)
def test_select_smoke_test_benchmark_config(
    monkeypatch, model_name: str, device: DeviceTypes
):
    benchmark_config = _import_benchmark_config(monkeypatch)
    model_id = _find_model_id(
        model_name=model_name, device=device, impl_name="tt-transformers"
    )

    config = benchmark_config.get_benchmark_config(MODEL_SPECS[model_id])
    smoke_config = benchmark_config.select_smoke_test_benchmark_config(config, device)

    assert smoke_config.model_id == config.model_id
    assert len(smoke_config.tasks) == 1
    assert smoke_config.tasks[0].param_map[device] == config.tasks[0].param_map[device]


def test_select_smoke_test_benchmark_config_adds_smoke_pair_without_targets(
    monkeypatch,
):
    benchmark_config = _import_benchmark_config(monkeypatch)

    perf_task = benchmark_config.BenchmarkTask(param_map={DeviceTypes.N150: []})
    sweep_task = benchmark_config.BenchmarkTask(
        param_map={
            DeviceTypes.N150: [
                BenchmarkTaskParams(
                    isl=128,
                    osl=128,
                    max_concurrency=32,
                    num_prompts=256,
                )
            ]
        }
    )
    config = benchmark_config.BenchmarkConfig(
        model_id="smoke-test-model",
        tasks=[perf_task, sweep_task],
    )

    smoke_config = benchmark_config.select_smoke_test_benchmark_config(
        config, DeviceTypes.N150
    )

    assert len(smoke_config.tasks) == 1
    smoke_sweep_params = smoke_config.tasks[0].param_map[DeviceTypes.N150]
    assert len(smoke_sweep_params) == 1
    assert (
        smoke_sweep_params[0].isl,
        smoke_sweep_params[0].osl,
    ) == benchmark_config.SMOKE_TEST_BENCHMARK_PAIR
    assert smoke_sweep_params[0].max_concurrency == 1
    assert smoke_sweep_params[0].num_prompts == benchmark_config.get_num_prompts(
        *benchmark_config.SMOKE_TEST_BENCHMARK_PAIR, 1
    )
    assert getattr(smoke_sweep_params[0], "task_type", "text") == "text"


def test_get_benchmark_config_uses_runtime_spec_even_when_model_id_collides(
    monkeypatch,
):
    benchmark_config = _import_benchmark_config(monkeypatch)

    source_id = _find_model_id(
        model_name="Qwen3-8B", device=DeviceTypes.N150, impl_name="tt-transformers"
    )
    source_spec = MODEL_SPECS[source_id]
    runtime_spec = replace(
        source_spec,
        device_model_spec=replace(
            source_spec.device_model_spec,
            max_context=256,
            max_concurrency=32,
        ),
    )

    config = benchmark_config.get_benchmark_config(runtime_spec)

    assert config.model_id == source_id
    assert len(config.tasks) == 3
    assert runtime_spec.device_type in config.tasks[0].param_map
    assert config.tasks[0].param_map[runtime_spec.device_type]

    sweep_task = config.tasks[1]
    sweep_params = sweep_task.param_map[runtime_spec.device_type]
    assert [
        p.max_concurrency
        for p in sweep_params
        if p.isl == 128 and p.osl == 128 and getattr(p, "task_type", "text") == "text"
    ] == [1]
    assert not [
        p
        for p in sweep_params
        if p.isl == 128 and p.osl == 1024 and getattr(p, "task_type", "text") == "text"
    ]


def test_select_smoke_test_benchmark_config_skips_non_text_sweeps(monkeypatch):
    benchmark_config = _import_benchmark_config(monkeypatch)

    perf_task = benchmark_config.BenchmarkTask(param_map={DeviceTypes.N150: []})
    sweep_task = benchmark_config.BenchmarkTaskCNN(
        param_map={DeviceTypes.N150: [BenchmarkTaskParamsCNN(num_eval_runs=5)]}
    )
    config = benchmark_config.BenchmarkConfig(
        model_id="smoke-test-model",
        tasks=[perf_task, sweep_task],
    )

    smoke_config = benchmark_config.select_smoke_test_benchmark_config(
        config, DeviceTypes.N150
    )

    assert smoke_config.tasks == []


def test_get_num_prompts_min_floor(monkeypatch):
    benchmark_config = _import_benchmark_config(monkeypatch)

    # High-ISL point: base is 1x concurrency (=64); the floor raises it to 256.
    assert benchmark_config.get_num_prompts(245760, 128, 64) == 64
    assert benchmark_config.get_num_prompts(245760, 128, 64, min_num_prompts=256) == 256

    # Low-ISL point: base is 8x concurrency (=512), already above the floor.
    assert benchmark_config.get_num_prompts(128, 128, 64) == 512
    assert benchmark_config.get_num_prompts(128, 128, 64, min_num_prompts=256) == 512

    # The floor never lowers a base count that already exceeds it.
    assert (
        benchmark_config.get_num_prompts(245760, 128, 512, min_num_prompts=256) == 512
    )


def _make_super_cluster_runtime_spec():
    source_id = _find_model_id(
        model_name="Qwen3-8B", device=DeviceTypes.N150, impl_name="tt-transformers"
    )
    source_spec = MODEL_SPECS[source_id]
    return source_id, replace(
        source_spec,
        device_type=DeviceTypes.SUPER_CLUSTER,
        device_model_spec=replace(
            source_spec.device_model_spec,
            device=DeviceTypes.SUPER_CLUSTER,
            max_context=262144,  # 256K, matching Kimi's SUPER_CLUSTER spec
            max_concurrency=64,
        ),
    )


def test_super_cluster_sweep_enforces_min_num_prompts(monkeypatch):
    benchmark_config = _import_benchmark_config(monkeypatch)

    _, runtime_spec = _make_super_cluster_runtime_spec()
    config = benchmark_config.get_benchmark_config(runtime_spec)

    sweep_params = config.tasks[1].param_map[DeviceTypes.SUPER_CLUSTER]
    text_params = [
        p for p in sweep_params if getattr(p, "task_type", "text") == "text"
    ]
    assert text_params

    floor = benchmark_config.SUPER_CLUSTER_MIN_NUM_PROMPTS

    for p in text_params:
        # A sweep point must never issue fewer prompts than its concurrency.
        assert p.num_prompts >= p.max_concurrency
        # Multi-user points must honor the SUPER_CLUSTER floor.
        if p.max_concurrency > 1:
            assert p.num_prompts >= floor

    # Single-user latency points use the normal length-based count (1-8);
    # applying the throughput floor here would waste 256 serial requests.
    single_user = [p for p in text_params if p.max_concurrency == 1]
    assert single_user
    for p in single_user:
        assert p.num_prompts == benchmark_config.get_num_prompts(p.isl, p.osl, 1)
        assert p.num_prompts < floor

    # The high-ISL extension points must be present and respect the floor
    # (their base 1x concurrency count would otherwise fall well below it).
    high_isl = [p for p in text_params if p.isl >= 196608 and p.max_concurrency > 1]
    assert high_isl
    for p in high_isl:
        assert p.num_prompts >= floor


def test_non_super_cluster_sweep_has_no_min_num_prompts_floor(monkeypatch):
    benchmark_config = _import_benchmark_config(monkeypatch)

    model_id = _find_model_id(
        model_name="Qwen3-8B", device=DeviceTypes.N150, impl_name="tt-transformers"
    )
    config = benchmark_config.get_benchmark_config(MODEL_SPECS[model_id])

    sweep_params = config.tasks[1].param_map[DeviceTypes.N150]
    text_params = [
        p for p in sweep_params if getattr(p, "task_type", "text") == "text"
    ]
    assert text_params

    # num_prompts must match the unfloored helper for a non-SUPER_CLUSTER device.
    for p in text_params:
        assert p.num_prompts == benchmark_config.get_num_prompts(
            p.isl, p.osl, p.max_concurrency
        )

    # At least one point falls below the SUPER_CLUSTER floor (e.g. the
    # concurrency=1 sweep points), proving the floor is not applied here.
    assert any(
        p.num_prompts < benchmark_config.SUPER_CLUSTER_MIN_NUM_PROMPTS
        for p in text_params
    )
