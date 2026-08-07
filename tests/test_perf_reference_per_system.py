# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
"""Tests for per-system vs per-replica performance-target resolution.

Milestone-0 RFP §5.4 / Appendix B.0: for a 32-chip Blackhole Galaxy the graded
bar must be a single per-system value that the operator's data-parallel choice
cannot move. These tests pin that behaviour and guard the existing per-replica
scaling used by the Wormhole Galaxy data-parallel specs.
"""

from workflows.model_spec import (
    DeviceModelSpec,
    get_perf_reference,
    scale_llm_perf_targets,
)
from workflows.utils_report import BenchmarkTaskParams, PerformanceTarget
from workflows.workflow_types import DeviceTypes


def _task(tput, tput_user=50.0, ttft_ms=100.0, max_concurrency=128):
    return BenchmarkTaskParams(
        isl=128,
        osl=128,
        max_concurrency=max_concurrency,
        num_prompts=8,
        targets={
            "target": PerformanceTarget(
                ttft_ms=ttft_ms, tput_user=tput_user, tput=tput
            )
        },
    )


def test_expresses_targets_per_system_only_for_blackhole_galaxy():
    assert DeviceTypes.BLACKHOLE_GALAXY.expresses_targets_per_system() is True
    for device in (
        DeviceTypes.GALAXY,
        DeviceTypes.T3K,
        DeviceTypes.P300X2,
        DeviceTypes.P150X4,
        DeviceTypes.N150,
    ):
        assert device.expresses_targets_per_system() is False


def test_per_system_device_ignores_data_parallel_scaling():
    """A data-parallel choice must not move the per-system bar: the target is
    read from the BLACKHOLE_GALAXY key verbatim, not the DP=8 subdevice ×8."""
    spec = DeviceModelSpec(
        device=DeviceTypes.BLACKHOLE_GALAXY,
        max_concurrency=128,
        max_context=262144,
        vllm_args={"data_parallel_size": 8},
    )
    perf_reference_map = {
        # The published per-system target.
        DeviceTypes.BLACKHOLE_GALAXY: [_task(tput=1000.0, max_concurrency=128)],
        # A per-replica subdevice table that must NOT be selected/scaled.
        DeviceTypes.P150X4: [_task(tput=200.0, max_concurrency=16)],
    }

    result = get_perf_reference(spec, perf_reference_map)

    assert len(result) == 1
    assert result[0].targets["target"].tput == 1000.0  # unscaled, not 200 × 8
    assert result[0].max_concurrency == 128  # unscaled, not 16 × 8


def test_per_system_device_without_data_parallel_reads_device_key():
    spec = DeviceModelSpec(
        device=DeviceTypes.BLACKHOLE_GALAXY,
        max_concurrency=128,
        max_context=262144,
    )
    perf_reference_map = {
        DeviceTypes.BLACKHOLE_GALAXY: [_task(tput=1000.0, max_concurrency=128)]
    }

    result = get_perf_reference(spec, perf_reference_map)

    assert result[0].targets["target"].tput == 1000.0
    assert result[0].max_concurrency == 128


def test_galaxy_data_parallel_still_scales_from_subdevice():
    """Regression guard: per-replica devices keep the existing subdevice lookup
    and ×data_parallel aggregation."""
    spec = DeviceModelSpec(
        device=DeviceTypes.GALAXY,
        max_concurrency=8,
        max_context=131072,
        vllm_args={"data_parallel_size": 4},
    )
    perf_reference_map = {
        # DP=4 subdevice for GALAXY is T3K; this per-replica table is scaled ×4.
        DeviceTypes.T3K: [_task(tput=100.0, max_concurrency=2)],
        # The system key must be ignored on a per-replica device.
        DeviceTypes.GALAXY: [_task(tput=999.0, max_concurrency=999)],
    }

    result = get_perf_reference(spec, perf_reference_map)

    assert result[0].targets["target"].tput == 400.0  # 100 × 4
    assert result[0].max_concurrency == 8  # 2 × 4


def test_scale_llm_perf_targets_scales_tput_and_concurrency():
    scaled = scale_llm_perf_targets(_task(tput=100.0, max_concurrency=4), 4)
    assert scaled.targets["target"].tput == 400.0
    assert scaled.max_concurrency == 16
    # tput_user (per-user) and ttft are not aggregate metrics: unchanged.
    assert scaled.targets["target"].tput_user == 50.0
    assert scaled.targets["target"].ttft_ms == 100.0


def test_scale_llm_perf_targets_leaves_single_stream_concurrency():
    scaled = scale_llm_perf_targets(_task(tput=100.0, max_concurrency=1), 4)
    assert scaled.max_concurrency == 1  # concurrency==1 is not multiplied
