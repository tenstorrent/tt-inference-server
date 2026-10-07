# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for the informational "% of DRAM Floor" benchmark column.

``report_module.roofline`` turns a model's parameter count and the device it
runs on into a decode TPOT floor; ``llm_module.target_checks`` stamps the
measured TPOT's percentage of that floor onto the benchmark row. The column
must be purely additive: unknown inputs give ``None`` (rendered "N/A") and the
graded ``target_checks`` / ``target_check`` never change.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_module.config import LLMRunConfig
from llm_module.target_checks import apply_target_checks
from report_module.display import decimal_places, display_name
from report_module.renderers import DRAM_FLOOR_NOTE, render_benchmarks
from report_module.roofline import (
    DEFAULT_WEIGHT_BYTES_PER_PARAM,
    dram_floor_ms,
    dram_floor_ms_for_spec,
    pct_of_dram_floor,
    tp_chips,
)
from report_module.schema import Block
from workflows.utils_report import PerformanceTarget
from workflows.workflow_types import DeviceTypes, ReportCheckTypes

# 70B params x 1.0625 B/param over 8 Wormhole chips at 288 GB/s each.
T3K_70B_FLOOR_MS = 70 * 1.0625 / 8 / 288 * 1000  # ~32.28 ms

TARGETS = {
    "functional": PerformanceTarget(ttft_ms=890.0, tput_user=1.7, tput=1.7),
    "complete": PerformanceTarget(ttft_ms=178.0, tput_user=8.5, tput=8.5),
    "target": PerformanceTarget(ttft_ms=89.0, tput_user=17.0, tput=17.0),
}


def _record(tpot=64.56, **extra):
    record = {
        "concurrency": 1,
        "num_requests": 8,
        "input_sequence_length": 128,
        "output_sequence_length": 128.0,
        "mean_ttft_ms": 350.0,
        "mean_tpot_ms": tpot,
        "tps_output_throughput": 13.9,
    }
    record.update(extra)
    return record


def _cfg(targets=None, floor_ms=None):
    return LLMRunConfig(
        isl=128,
        osl=128,
        max_concurrency=1,
        num_prompts=8,
        targets=dict(targets or {}),
        dram_floor_ms=floor_ms,
    )


def _block(record=None, **kwargs):
    return Block(
        kind="benchmarks",
        data=record if record is not None else _record(),
        title="vLLM Benchmark",
        **kwargs,
    )


class TestFloor:
    def test_known_device_matches_the_formula(self):
        assert dram_floor_ms(70, DeviceTypes.T3K) == pytest.approx(T3K_70B_FLOOR_MS)
        # Device names are accepted too (the string the report envelope carries).
        assert dram_floor_ms(70, "t3k") == pytest.approx(T3K_70B_FLOOR_MS)

    def test_default_weight_format_is_bfp8(self):
        assert DEFAULT_WEIGHT_BYTES_PER_PARAM == 1.0625

    def test_weight_bytes_override_scales_the_floor(self):
        bf16 = dram_floor_ms(70, DeviceTypes.T3K, weight_bytes_per_param=2.0)
        assert bf16 == pytest.approx(T3K_70B_FLOOR_MS * 2.0 / 1.0625)

    def test_blackhole_uses_512_gbps(self):
        # 8B on one P150: 8 x 1.0625 / 1 chip / 512 GB/s.
        assert dram_floor_ms(8, DeviceTypes.P150) == pytest.approx(
            8 * 1.0625 / 512 * 1000
        )

    @pytest.mark.parametrize(
        "device,chips",
        [
            (DeviceTypes.N150, 1),
            (DeviceTypes.N300, 2),
            (DeviceTypes.T3K, 8),
            (DeviceTypes.GALAXY, 32),
            (DeviceTypes.P150X4, 4),
            (DeviceTypes.P150X8, 8),
            (DeviceTypes.P300X2, 4),
            (DeviceTypes.BLACKHOLE_GALAXY, 32),
        ],
    )
    def test_tensor_parallel_chip_counts(self, device, chips):
        assert tp_chips(device) == chips

    def test_data_parallel_divides_the_chips(self):
        # BH Galaxy with four engines: 8 chips per engine, via the enum's own
        # DP subdevice mapping (BLACKHOLE_GALAXY, 4) -> P150X8.
        assert tp_chips(DeviceTypes.BLACKHOLE_GALAXY, 4) == 8
        assert tp_chips(DeviceTypes.GALAXY, 4) == 8
        # String devices fall back to plain division.
        assert tp_chips("BLACKHOLE_GALAXY", 4) == 8
        assert dram_floor_ms(
            31, DeviceTypes.BLACKHOLE_GALAXY, data_parallel_size=4
        ) == pytest.approx(31 * 1.0625 / 8 / 512 * 1000)

    @pytest.mark.parametrize("device", [DeviceTypes.CPU, DeviceTypes.GPU, "", None])
    def test_unknown_device_has_no_floor(self, device):
        assert tp_chips(device) is None
        assert dram_floor_ms(70, device) is None

    @pytest.mark.parametrize("params", [None, 0, -1, "seventy"])
    def test_missing_param_count_has_no_floor(self, params):
        assert dram_floor_ms(params, DeviceTypes.T3K) is None

    def test_floor_for_spec_reads_catalog_fields(self):
        spec = SimpleNamespace(
            param_count=70,
            device_type=DeviceTypes.T3K,
            device_model_spec=SimpleNamespace(
                vllm_args={}, weight_bytes_per_param=None
            ),
        )
        assert dram_floor_ms_for_spec(spec) == pytest.approx(T3K_70B_FLOOR_MS)

        dp_spec = SimpleNamespace(
            param_count=31,
            device_type=DeviceTypes.BLACKHOLE_GALAXY,
            device_model_spec=SimpleNamespace(
                vllm_args={"data_parallel_size": 4}, weight_bytes_per_param=2.0
            ),
        )
        assert dram_floor_ms_for_spec(dp_spec) == pytest.approx(
            31 * 2.0 / 8 / 512 * 1000
        )
        # Explicit device wins over the spec's own (ctx.device may differ).
        assert dram_floor_ms_for_spec(spec, DeviceTypes.N150) == pytest.approx(
            70 * 1.0625 / 1 / 288 * 1000
        )
        assert dram_floor_ms_for_spec(None) is None


class TestPercentage:
    def test_pct_is_floor_over_tpot(self):
        assert pct_of_dram_floor(T3K_70B_FLOOR_MS, T3K_70B_FLOOR_MS * 2) == 50.0
        assert pct_of_dram_floor(10.0, 10.0) == 100.0

    @pytest.mark.parametrize(
        "floor,tpot", [(None, 50.0), (10.0, None), (10.0, 0), (0, 50.0), (10.0, "x")]
    )
    def test_unusable_inputs_give_none(self, floor, tpot):
        assert pct_of_dram_floor(floor, tpot) is None


class TestColumnStamping:
    def test_graded_row_carries_the_percentage(self):
        block = apply_target_checks(
            _block(_record(tpot=T3K_70B_FLOOR_MS * 2)),
            _cfg(TARGETS, floor_ms=T3K_70B_FLOOR_MS),
        )
        assert block.data["pct_of_dram_floor"] == pytest.approx(50.0)
        # Sits just before the verdict so the table reads ... TPOT | % | Check.
        keys = list(block.data)
        assert keys.index("pct_of_dram_floor") == keys.index("target_check") - 1

    def test_unknown_floor_stamps_none_on_graded_and_ungraded_rows(self):
        graded = apply_target_checks(_block(), _cfg(TARGETS))
        ungraded = apply_target_checks(_block(), _cfg())
        assert graded.data["pct_of_dram_floor"] is None
        assert ungraded.data["pct_of_dram_floor"] is None

    def test_column_does_not_change_target_checks(self):
        without = apply_target_checks(_block(), _cfg(TARGETS))
        with_floor = apply_target_checks(
            _block(), _cfg(TARGETS, floor_ms=T3K_70B_FLOOR_MS)
        )
        assert with_floor.data["target_checks"] == without.data["target_checks"]
        assert with_floor.data["target_check"] == without.data["target_check"]
        assert with_floor.data["target_check"] == ReportCheckTypes.FAIL
        assert with_floor.title == without.title
        # The percentage is the only difference between the two rows.
        diff = {
            key
            for key in set(with_floor.data) | set(without.data)
            if with_floor.data.get(key) != without.data.get(key)
        }
        assert diff == {"pct_of_dram_floor"}

    def test_ungraded_row_keeps_its_na_status(self):
        block = apply_target_checks(
            _block(_record(tpot=T3K_70B_FLOOR_MS * 2)),
            _cfg(floor_ms=T3K_70B_FLOOR_MS),
        )
        assert block.data["status"] == "na"
        assert block.data["target_check"] == ReportCheckTypes.NA
        assert block.data["pct_of_dram_floor"] == pytest.approx(50.0)


class TestRendering:
    def test_display_header_and_precision(self):
        assert display_name("pct_of_dram_floor") == "% of DRAM Floor"
        assert decimal_places("pct_of_dram_floor") == 1

    def test_known_floor_renders_value_and_note(self):
        rows = [
            {**_record(), "pct_of_dram_floor": 64.56, "target_check": 2},
            {**_record(), "concurrency": 32, "pct_of_dram_floor": None},
        ]
        out = render_benchmarks(_block({"records": rows}), {})
        header, _, first, second = [
            line for line in out.splitlines() if line.startswith("|")
        ][:4]
        assert "% of DRAM Floor" in header
        assert header.index("% of DRAM Floor") < header.index("Target Check")
        assert "64.6" in first
        assert "N/A" in second
        assert DRAM_FLOOR_NOTE in out

    def test_all_unknown_floors_change_nothing(self):
        plain = [{**_record(), "target_check": 2}]
        stamped = [{**row, "pct_of_dram_floor": None} for row in plain]
        assert render_benchmarks(_block({"records": stamped}), {}) == render_benchmarks(
            _block({"records": plain}), {}
        )
        assert DRAM_FLOOR_NOTE not in render_benchmarks(
            _block({"records": stamped}), {}
        )
