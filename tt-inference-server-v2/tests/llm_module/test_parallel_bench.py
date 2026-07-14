# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_module.agentic.live_isl import LiveISLTracker
from llm_module.agentic.parallel_bench import (
    HARBOR_PARALLEL_OSL,
    RANDOM_CONTEXT_BUFFER,
    RANDOM_RANGE_RATIO,
    SWEBENCH_PARALLEL_ISL,
    SWEBENCH_PARALLEL_OSL,
    ParallelBenchLoad,
    ParallelBenchStrategy,
    _random_workload_lengths,
    start_parallel_bench,
)
from workflows.workflow_types import DeviceTypes


@pytest.mark.parametrize("max_context", [128 * 1024, 256 * 1024])
def test_random_workload_lengths_reserve_buffer_and_split_maximum(max_context):
    isl, osl = _random_workload_lengths(max_context)

    assert isl == osl
    sampled_isl_max = math.ceil(isl * (1 + RANDOM_RANGE_RATIO))
    sampled_osl_max = math.ceil(osl * (1 + RANDOM_RANGE_RATIO))
    usable_context = max_context - RANDOM_CONTEXT_BUFFER

    assert sampled_isl_max <= usable_context // 2
    assert sampled_osl_max <= usable_context // 2
    assert sampled_isl_max + sampled_osl_max <= usable_context


def test_random_workload_lengths_reject_too_small_context():
    with pytest.raises(ValueError, match="too small"):
        _random_workload_lengths(RANDOM_CONTEXT_BUFFER + 1)


def _context(*, max_context=256 * 1024, max_concurrency=64):
    device_spec = SimpleNamespace(
        max_concurrency=max_concurrency,
        max_context=max_context,
    )
    model_spec = SimpleNamespace(
        device_type=DeviceTypes.SUPER_CLUSTER,
        device_model_spec=device_spec,
        hf_model_repo="moonshotai/Kimi-K2.6",
    )
    return SimpleNamespace(
        model_spec=model_spec,
        server_url="https://console.tenstorrent.com",
        remote_server=True,
        server_host="unused",
        server_port=443,
        device=DeviceTypes.SUPER_CLUSTER,
    )


def _task(*, swebench=False, n_concurrent_trials=4):
    eval_config = SimpleNamespace(n_concurrent_trials=n_concurrent_trials)
    return SimpleNamespace(
        task_name="swe_bench_verified" if swebench else "terminal_bench_2_1",
        agentic_eval_config=None if swebench else eval_config,
        swebench_eval_config=eval_config if swebench else None,
    )


def _start_without_thread(monkeypatch, ctx, task, **kwargs):
    monkeypatch.setattr(ParallelBenchLoad, "start", lambda self: self)
    monkeypatch.setattr(
        "llm_module.agentic.parallel_bench._resolve_vllm_binary", lambda: "vllm"
    )
    return start_parallel_bench(
        ctx,
        task,
        watch_dir=Path("/tmp/agentic"),
        sidecar_dir=Path("/tmp/parallel_bench"),
        **kwargs,
    )


@pytest.mark.parametrize("swebench", [False, True])
def test_random_range_is_default_for_all_agentic_tasks(monkeypatch, swebench):
    ctx = _context()
    load = _start_without_thread(monkeypatch, ctx, _task(swebench=swebench))

    assert load is not None
    assert load._random_range_ratio == RANDOM_RANGE_RATIO
    assert load._fixed_isl == load._osl
    assert load._tracker is None
    assert load._bench_concurrency == 60

    sampled_max = math.ceil(load._fixed_isl * (1 + RANDOM_RANGE_RATIO))
    assert 2 * sampled_max <= 256 * 1024 - RANDOM_CONTEXT_BUFFER


def test_random_loop_passes_range_and_ignore_eos_to_driver(tmp_path):
    configs = []
    load = None

    class FakeDriver:
        def run(self, config, server, context, result_filename=None):
            configs.append(config)
            load._stop.set()

    load = ParallelBenchLoad(
        driver=FakeDriver(),
        server=SimpleNamespace(),
        context=SimpleNamespace(output_dir=tmp_path),
        bench_concurrency=60,
        max_context=256 * 1024,
        osl=80 * 1024,
        fixed_isl=80 * 1024,
        tracker=None,
        random_range_ratio=RANDOM_RANGE_RATIO,
    )

    load._loop()

    assert len(configs) == 1
    assert configs[0].random_range_ratio == RANDOM_RANGE_RATIO
    assert configs[0].ignore_eos is True
    assert configs[0].num_prompts == configs[0].max_concurrency == 60


def test_task_aware_harbor_uses_live_isl_tracker(monkeypatch):
    load = _start_without_thread(
        monkeypatch,
        _context(),
        _task(),
        strategy=ParallelBenchStrategy.TASK_AWARE,
    )

    assert load is not None
    assert load._random_range_ratio is None
    assert load._fixed_isl is None
    assert load._osl == HARBOR_PARALLEL_OSL
    assert isinstance(load._tracker, LiveISLTracker)


def test_task_aware_swebench_keeps_fixed_shape(monkeypatch):
    load = _start_without_thread(
        monkeypatch,
        _context(),
        _task(swebench=True),
        strategy=ParallelBenchStrategy.TASK_AWARE,
    )

    assert load is not None
    assert load._random_range_ratio is None
    assert load._fixed_isl == SWEBENCH_PARALLEL_ISL
    assert load._osl == SWEBENCH_PARALLEL_OSL
    assert load._tracker is None


def test_random_range_skips_context_smaller_than_buffer(monkeypatch):
    load = _start_without_thread(
        monkeypatch,
        _context(max_context=RANDOM_CONTEXT_BUFFER + 1),
        _task(),
    )

    assert load is None
