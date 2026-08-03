# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for the agentic trace-replay config, run expansion, and CLI build.

The load-bearing invariants here are the ones a broken run would only reveal an
hour in (or worse, as a plausible-looking row of numbers):

* every configured model pins an InferenceX ref and covers every mode,
* no mode can drop below the scenario's 900s profiling floor,
* the derived knobs (max context, trust-remote-code) track the ModelSpec,
* a trace source with no driver still fails loudly instead of producing an
  empty sweep, while the two implemented sources (InferenceX, SwarmOne) build,
* SwarmOne's per-mode scoping runs all tasks at load in ``full`` and a single
  short task at concurrency 1 in ``ci``,
* the emitted CLI (AIPerf and swo-bench) carries the flags the scenario requires.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_module.agentic_traces import (
    build_runs,
    estimated_run_seconds,
    summarize_runs,
    total_planned_seconds,
)
from llm_module.drivers.aiperf_agentic_traces import (
    _invalid_result_reason,
    build_aiperf_cmd,
    parse_aiperf_output,
)
from llm_module.drivers.swo_bench_agentic_traces import (
    build_swo_bench_cmd,
    parse_swo_bench_output,
)
from llm_module.drivers.swo_bench_agentic_traces import (
    _invalid_result_reason as _swo_invalid_result_reason,
)
from llm_module.parsers.aiperf_agentic_traces import AIPerfAgenticTracesParser
from llm_module.parsers.swo_bench_agentic_traces import SwoBenchAgenticTracesParser
from reference_config.agentic_traces.agentic_traces_config import (
    AGENTIC_TRACES_CONFIGS,
    AGENTIC_TRACES_MIN_PROFILE_SECONDS,
    AgenticTracesConfig,
    AgenticTracesModeSettings,
    AgenticTracesRunSpec,
    TraceSource,
    get_agentic_traces_config,
    resolve_run_specs,
)
from workflows.workflow_types import AgenticTracesMode

KIMI_MODEL_ID = "id_tt-transformers_Kimi-K2.7-Code_super_cluster"
KIMI_PINNED_REF = "ddeb02eb9c5c89f44e2e4950e741b499d0b8190a"


class _FakeDeviceModelSpec:
    def __init__(self, max_context: int) -> None:
        self.max_context = max_context


class _FakeModelSpec:
    """Minimal stand-in for the attributes the run builder reads."""

    def __init__(
        self,
        *,
        model_id: str = KIMI_MODEL_ID,
        max_context: int = 262144,
        metadata: dict = None,
    ) -> None:
        self.model_id = model_id
        self.model_name = "Kimi-K2.7-Code"
        self.hf_model_repo = "moonshotai/Kimi-K2.7-Code"
        self.device_model_spec = _FakeDeviceModelSpec(max_context)
        self.metadata = metadata if metadata is not None else {}


class TestConfigRegistry:
    def test_kimi_is_registered_with_the_pinned_ref(self):
        config = AGENTIC_TRACES_CONFIGS[KIMI_MODEL_ID]
        assert config.inferencex_git_ref == KIMI_PINNED_REF

    def test_every_config_covers_every_mode(self):
        for model_id, config in AGENTIC_TRACES_CONFIGS.items():
            for mode in AgenticTracesMode:
                assert config.settings_for_mode(mode) is not None, (
                    f"{model_id} is missing settings for {mode.name}"
                )

    def test_every_config_pins_a_ref_and_has_runs(self):
        for model_id, config in AGENTIC_TRACES_CONFIGS.items():
            assert config.inferencex_git_ref.strip(), f"{model_id} has no git ref"
            assert config.runs, f"{model_id} has no runs"

    def test_registry_is_keyed_by_the_config_model_id(self):
        for model_id, config in AGENTIC_TRACES_CONFIGS.items():
            assert config.model_id == model_id

    def test_lookup_by_model_spec(self):
        assert get_agentic_traces_config(_FakeModelSpec()) is not None

    def test_lookup_returns_none_for_unconfigured_model(self):
        assert get_agentic_traces_config(_FakeModelSpec(model_id="id_nope")) is None

    def test_lookup_tolerates_a_spec_without_model_id(self):
        assert get_agentic_traces_config(object()) is None


class TestMinDurationFloor:
    def test_config_below_the_floor_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="below the 900s minimum"):
            AgenticTracesConfig(
                model_id="id_test",
                inferencex_git_ref="abc123",
                mode_settings={
                    AgenticTracesMode.FULL: AgenticTracesModeSettings(
                        benchmark_duration=3600,
                        warmup_requests_per_lane=14,
                        warmup_grace_period=1800,
                        num_dataset_entries=393,
                    ),
                    AgenticTracesMode.CI: AgenticTracesModeSettings(
                        benchmark_duration=300,
                        warmup_requests_per_lane=3,
                        warmup_grace_period=120,
                        num_dataset_entries=8,
                    ),
                },
            )

    def test_ci_mode_sits_exactly_on_the_floor(self):
        settings = AGENTIC_TRACES_CONFIGS[KIMI_MODEL_ID].settings_for_mode(
            AgenticTracesMode.CI
        )
        assert settings.benchmark_duration == AGENTIC_TRACES_MIN_PROFILE_SECONDS

    def test_a_scenario_without_the_floor_may_go_shorter(self):
        config = AgenticTracesConfig(
            model_id="id_test",
            inferencex_git_ref="abc123",
            runs=(AgenticTracesRunSpec(scenario="some-other-scenario"),),
            mode_settings={
                AgenticTracesMode.FULL: AgenticTracesModeSettings(
                    benchmark_duration=600,
                    warmup_requests_per_lane=3,
                    warmup_grace_period=120,
                    num_dataset_entries=8,
                ),
                AgenticTracesMode.CI: AgenticTracesModeSettings(
                    benchmark_duration=120,
                    warmup_requests_per_lane=2,
                    warmup_grace_period=60,
                    num_dataset_entries=4,
                ),
            },
        )
        assert config.settings_for_mode(AgenticTracesMode.CI).benchmark_duration == 120

    def test_config_missing_a_mode_is_rejected(self):
        with pytest.raises(ValueError, match="missing mode_settings for: CI"):
            AgenticTracesConfig(
                model_id="id_test",
                inferencex_git_ref="abc123",
                mode_settings={
                    AgenticTracesMode.FULL: AgenticTracesModeSettings(
                        benchmark_duration=3600,
                        warmup_requests_per_lane=14,
                        warmup_grace_period=1800,
                        num_dataset_entries=393,
                    )
                },
            )

    def test_empty_git_ref_is_rejected(self):
        with pytest.raises(ValueError, match="inferencex_git_ref"):
            AgenticTracesConfig(model_id="id_test", inferencex_git_ref="  ")


class TestModeResolution:
    def test_full_mode_uses_the_reference_durations(self):
        config = AGENTIC_TRACES_CONFIGS[KIMI_MODEL_ID]
        run = build_runs(config, _FakeModelSpec(), mode=AgenticTracesMode.FULL)[0]
        assert run.benchmark_duration == 3600
        assert run.warmup_requests_per_lane == 14
        assert run.num_dataset_entries == 393
        assert run.mode is AgenticTracesMode.FULL

    def test_ci_mode_shortens_the_run(self):
        config = AGENTIC_TRACES_CONFIGS[KIMI_MODEL_ID]
        full = build_runs(config, _FakeModelSpec(), mode=AgenticTracesMode.FULL)[0]
        ci = build_runs(config, _FakeModelSpec(), mode=AgenticTracesMode.CI)[0]
        assert ci.benchmark_duration < full.benchmark_duration
        assert ci.warmup_requests_per_lane < full.warmup_requests_per_lane
        assert ci.num_dataset_entries < full.num_dataset_entries

    def test_mode_is_part_of_the_run_label(self):
        config = AGENTIC_TRACES_CONFIGS[KIMI_MODEL_ID]
        ci = build_runs(config, _FakeModelSpec(), mode=AgenticTracesMode.CI)[0]
        assert ci.label.endswith("_ci")

    def test_duration_override_wins_over_the_mode(self):
        config = AGENTIC_TRACES_CONFIGS[KIMI_MODEL_ID]
        run = build_runs(
            config,
            _FakeModelSpec(),
            mode=AgenticTracesMode.CI,
            duration_override=1200,
        )[0]
        assert run.benchmark_duration == 1200

    def test_mode_settings_concurrency_overrides_the_run_spec(self):
        config = AgenticTracesConfig(
            model_id="id_test",
            inferencex_git_ref="abc123",
            runs=(AgenticTracesRunSpec(concurrency=8),),
            mode_settings={
                AgenticTracesMode.FULL: AgenticTracesModeSettings(
                    benchmark_duration=3600,
                    warmup_requests_per_lane=14,
                    warmup_grace_period=1800,
                    num_dataset_entries=393,
                ),
                AgenticTracesMode.CI: AgenticTracesModeSettings(
                    benchmark_duration=900,
                    warmup_requests_per_lane=3,
                    warmup_grace_period=600,
                    num_dataset_entries=32,
                    concurrency=2,
                ),
            },
        )
        assert (
            build_runs(config, _FakeModelSpec(), mode=AgenticTracesMode.CI)[
                0
            ].concurrency
            == 2
        )
        assert (
            build_runs(config, _FakeModelSpec(), mode=AgenticTracesMode.FULL)[
                0
            ].concurrency
            == 8
        )

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("full", AgenticTracesMode.FULL),
            ("ci", AgenticTracesMode.CI),
            ("CI", AgenticTracesMode.CI),
            (None, None),
        ],
    )
    def test_mode_from_string(self, raw, expected):
        assert AgenticTracesMode.from_string(raw) is expected

    def test_invalid_mode_string_raises(self):
        with pytest.raises(ValueError, match="Invalid AgenticTracesMode"):
            AgenticTracesMode.from_string("nightly")


class TestDerivedFromModelSpec:
    def test_max_context_length_comes_from_the_spec(self):
        config = AGENTIC_TRACES_CONFIGS[KIMI_MODEL_ID]
        run = build_runs(config, _FakeModelSpec(max_context=131072))[0]
        assert run.max_context_length == 131072

    def test_explicit_max_context_length_overrides_the_spec(self):
        config = AgenticTracesConfig(
            model_id="id_test",
            inferencex_git_ref="abc123",
            runs=(AgenticTracesRunSpec(max_context_length=65536),),
        )
        run = build_runs(config, _FakeModelSpec(max_context=262144))[0]
        assert run.max_context_length == 65536

    def test_missing_max_context_is_a_clear_error(self):
        config = AGENTIC_TRACES_CONFIGS[KIMI_MODEL_ID]
        with pytest.raises(ValueError, match="max_context_length"):
            build_runs(config, _FakeModelSpec(max_context=0))

    def test_trust_remote_code_comes_from_spec_metadata(self):
        config = AGENTIC_TRACES_CONFIGS[KIMI_MODEL_ID]
        spec = _FakeModelSpec(metadata={"tokenizer_trust_remote_code": True})
        assert build_runs(config, spec)[0].tokenizer_trust_remote_code is True
        assert build_runs(config, _FakeModelSpec())[0].tokenizer_trust_remote_code is (
            False
        )

    def test_run_metadata_carries_the_pinned_ref(self):
        config = AGENTIC_TRACES_CONFIGS[KIMI_MODEL_ID]
        run = build_runs(config, _FakeModelSpec())[0]
        assert run.metadata["inferencex_git_ref"] == KIMI_PINNED_REF
        assert run.metadata["model_id"] == KIMI_MODEL_ID


class TestTraceSourceSelection:
    def test_from_string_accepts_hyphens_and_case(self):
        assert TraceSource.from_string("INFERENCEX-AGENTX") is (
            TraceSource.INFERENCEX_AGENTX
        )

    def test_from_string_rejects_unknown(self):
        with pytest.raises(ValueError, match="Invalid TraceSource"):
            TraceSource.from_string("mystery_traces")

    def test_swarmone_run_builds(self):
        """SwarmOne is a supported source; building its runs must not raise."""
        config = AgenticTracesConfig(
            model_id="id_test",
            inferencex_git_ref="abc123",
            runs=(
                AgenticTracesRunSpec(
                    trace_source=TraceSource.SWARMONE,
                    scenario="claude-code-swe-bench-python-kimi-k2.7-code",
                    public_dataset="",
                ),
            ),
        )
        runs = build_runs(config, _FakeModelSpec())
        assert len(runs) == 1
        assert runs[0].trace_source is TraceSource.SWARMONE

    def test_unimplemented_source_still_fails_loudly(self):
        """A source outside SUPPORTED_TRACE_SOURCES fails at plan time."""
        import llm_module.agentic_traces.runs as runs_mod

        config = AgenticTracesConfig(
            model_id="id_test",
            inferencex_git_ref="abc123",
            runs=(AgenticTracesRunSpec(trace_source=TraceSource.SWARMONE),),
        )
        original = runs_mod.SUPPORTED_TRACE_SOURCES
        runs_mod.SUPPORTED_TRACE_SOURCES = (TraceSource.INFERENCEX_AGENTX,)
        try:
            with pytest.raises(NotImplementedError, match="swarmone"):
                build_runs(config, _FakeModelSpec())
        finally:
            runs_mod.SUPPORTED_TRACE_SOURCES = original

    def test_narrowing_to_a_configured_source_keeps_its_runs(self):
        config = AGENTIC_TRACES_CONFIGS[KIMI_MODEL_ID]
        _, runs = resolve_run_specs(
            config, trace_sources=(TraceSource.INFERENCEX_AGENTX,)
        )
        assert runs
        assert all(r.trace_source is TraceSource.INFERENCEX_AGENTX for r in runs)

    def test_narrowing_to_an_unconfigured_source_raises(self):
        # A config that only has an InferenceX run rejects a swarmone selection.
        config = AgenticTracesConfig(
            model_id="id_test",
            inferencex_git_ref="abc123",
            runs=(AgenticTracesRunSpec(trace_source=TraceSource.INFERENCEX_AGENTX),),
        )
        with pytest.raises(ValueError, match="do not match|No agentic-trace runs"):
            resolve_run_specs(config, trace_sources=(TraceSource.SWARMONE,))

    def test_git_ref_override_replaces_the_pin(self):
        config = AGENTIC_TRACES_CONFIGS[KIMI_MODEL_ID]
        effective, _ = resolve_run_specs(config, git_ref_override="cafebabe")
        assert effective.inferencex_git_ref == "cafebabe"
        assert config.inferencex_git_ref == KIMI_PINNED_REF


class TestRunSpecValidation:
    def test_zero_concurrency_is_rejected(self):
        with pytest.raises(ValueError, match="concurrency"):
            AgenticTracesRunSpec(concurrency=0)

    def test_threshold_outside_unit_interval_is_rejected(self):
        with pytest.raises(ValueError, match="failed_request_threshold"):
            AgenticTracesRunSpec(failed_request_threshold=10)

    def test_inverted_trajectory_ratios_are_rejected(self):
        with pytest.raises(ValueError, match="trajectory ratios"):
            AgenticTracesRunSpec(
                trajectory_start_min_ratio=0.8, trajectory_start_max_ratio=0.2
            )


class TestAiperfCommand:
    def _cmd(self, **kwargs):
        config = AGENTIC_TRACES_CONFIGS[KIMI_MODEL_ID]
        spec = _FakeModelSpec(metadata={"tokenizer_trust_remote_code": True})
        run = build_runs(config, spec, mode=AgenticTracesMode.CI)[0]
        defaults = dict(
            run=run,
            venv_python=Path("/venv/bin/python"),
            model_name="moonshotai/Kimi-K2.7-Code",
            tokenizer="moonshotai/Kimi-K2.7-Code",
            url="http://localhost:8000",
            artifact_dir=Path("/tmp/artifacts"),
        )
        defaults.update(kwargs)
        return build_aiperf_cmd(**defaults)

    def test_invokes_the_venv_aiperf_module(self):
        cmd = self._cmd()
        assert cmd[:5] == [
            "/venv/bin/python",
            "-m",
            "aiperf",
            "profile",
            "--scenario",
        ]

    def test_carries_the_scenario_and_dataset(self):
        cmd = self._cmd()
        assert "inferencex-agentx-mvp" in cmd
        assert "semianalysis_cc_traces_weka_062126_256k" in cmd

    def test_emits_the_flags_the_scenario_requires(self):
        cmd = self._cmd()
        for flag in (
            "--streaming",
            "--use-server-token-count",
            "--no-gpu-telemetry",
            "--tokenizer-trust-remote-code",
            "--max-context-length",
            "--warmup-requests-per-lane",
            "--warmup-grace-period",
        ):
            assert flag in cmd, f"{flag} missing from {cmd}"

    def test_the_superseded_duration_warmup_flag_is_not_emitted(self):
        """It is mutually exclusive with --warmup-requests-per-lane."""
        assert "--agentic-cache-warmup-duration" not in self._cmd()

    def test_durations_match_the_selected_mode(self):
        cmd = self._cmd()
        assert cmd[cmd.index("--benchmark-duration") + 1] == str(
            AGENTIC_TRACES_MIN_PROFILE_SECONDS
        )
        assert cmd[cmd.index("--warmup-requests-per-lane") + 1] == "3"

    def test_bare_host_gets_a_scheme(self):
        cmd = self._cmd(url="localhost:8000")
        assert cmd[cmd.index("--url") + 1] == "http://localhost:8000"

    def test_api_key_only_present_with_a_token(self):
        assert "--api-key" not in self._cmd()
        cmd = self._cmd(auth_token="tok123")
        assert cmd[cmd.index("--api-key") + 1] == "tok123"

    def test_gpu_telemetry_flag_is_omitted_when_enabled(self):
        config = AgenticTracesConfig(
            model_id="id_test",
            inferencex_git_ref="abc123",
            runs=(AgenticTracesRunSpec(gpu_telemetry=True),),
        )
        run = build_runs(config, _FakeModelSpec())[0]
        cmd = build_aiperf_cmd(
            run=run,
            venv_python=Path("/venv/bin/python"),
            model_name="m",
            tokenizer="m",
            url="http://localhost:8000",
            artifact_dir=Path("/tmp/artifacts"),
        )
        assert "--no-gpu-telemetry" not in cmd

    def test_no_server_metrics_flag_without_explicit_urls(self):
        """AIPerf already derives <url>/metrics; the flag is only for extras."""
        assert "--server-metrics" not in self._cmd()

    def test_extra_metrics_endpoints_are_passed_after_one_flag(self):
        cmd = self._cmd(
            metrics_urls=(
                "http://worker-a:9000/metrics",
                "http://worker-b:9000/metrics",
            )
        )
        idx = cmd.index("--server-metrics")
        assert cmd[idx + 1 : idx + 3] == [
            "http://worker-a:9000/metrics",
            "http://worker-b:9000/metrics",
        ]


class TestPlanSummary:
    def test_summary_names_every_run(self):
        config = AGENTIC_TRACES_CONFIGS[KIMI_MODEL_ID]
        runs = build_runs(config, _FakeModelSpec())
        summary = summarize_runs(runs)
        for run in runs:
            assert run.label in summary

    def test_summary_handles_no_runs(self):
        assert "No runs planned" in summarize_runs([])

    def test_total_planned_seconds_sums_profiling_and_warmup_allowance(self):
        """A request-bounded warmup has no duration, so grace is the allowance."""
        config = AGENTIC_TRACES_CONFIGS[KIMI_MODEL_ID]
        runs = build_runs(config, _FakeModelSpec(), mode=AgenticTracesMode.CI)
        assert total_planned_seconds(runs) == sum(
            r.benchmark_duration + r.warmup_grace_period for r in runs
        )


def _latency_block(**overrides):
    """A metric block shaped like the fork's export (every stat present)."""
    block = {
        "unit": "ms",
        "avg": 6130.49,
        "p1": 831.94,
        "p5": 1813.5,
        "p10": 2215.77,
        "p25": 3019.83,
        "p50": 4808.69,
        "p75": 7254.54,
        "p90": 10803.95,
        "p95": 15830.15,
        "p99": 27015.14,
        "min": 273.99,
        "max": 36797.26,
        "std": 4998.71,
        "count": 336,
        "sum": 2059845.73,
    }
    block.update(overrides)
    return block


def _summary_export(**overrides):
    """A trimmed copy of a real ``profile_export_aiperf.json`` from the fork.

    Values are taken from a completed 3600s Kimi run so the count semantics stay
    honest: 336 successful + 35 errors = 371 completed.
    """
    summary = {
        "schema_version": "1.2",
        "aiperf_version": "0.8.0",
        "time_to_first_token": _latency_block(),
        "inter_token_latency": _latency_block(avg=15.02, p50=9.6, p95=44.49, p99=62.73),
        "request_latency": _latency_block(avg=14588.45, p50=10016.63, p99=58096.55),
        "time_to_second_token": _latency_block(avg=83.36),
        "output_token_throughput_per_user": _latency_block(avg=119.49, p50=104.12),
        "e2e_output_token_throughput": _latency_block(avg=40.41),
        "output_token_throughput": {"unit": "tokens/sec", "avg": 57.19},
        "input_token_throughput": {"unit": "tokens/sec", "avg": 7878.22},
        "total_token_throughput": {"unit": "tokens/sec", "avg": 7935.41},
        "request_throughput": {"unit": "requests/sec", "avg": 0.0925},
        "request_count": {"unit": "requests", "avg": 336.0},
        "completed_request_count": {"unit": "requests", "avg": 371.0},
        "error_request_count": {"unit": "requests", "avg": 35.0},
        "request_error_rate": {"unit": "%", "avg": 9.433962264150944},
        "input_sequence_length": _latency_block(avg=85113.06),
        "output_sequence_length": _latency_block(avg=617.92),
        "total_isl": {"unit": "tokens", "avg": 28597989.0},
        "total_osl": {"unit": "tokens", "avg": 207620.0},
        "benchmark_duration": {"unit": "sec", "avg": 3605.47},
        "context_overflow_count": {"unit": "requests", "avg": 0.0},
        "osl_mismatch_count": {"unit": "requests", "avg": 0.0},
        "osl_mismatch_diff_pct": _latency_block(unit="%", avg=0.0),
        "theoretical_prefix_cache_hit": {"unit": "%", "avg": 95.74},
        "was_cancelled": False,
        "time_to_first_output_token": _latency_block(avg=9660.41),
        "effective_latency": _latency_block(avg=14592.3, p50=10019.1, p99=58101.4),
        "adj_time_to_first_token": {"unit": "ms", "p50": 5012.3, "p90": 11987.6},
        "adj_request_latency": {"unit": "ms", "p50": 11004.7, "p99": 61122.8},
        "effective_prefill_throughput": {"unit": "tokens/sec", "avg": 7878.22},
        "active_prefill_throughput": {"unit": "tokens/sec", "avg": 14882.3},
        "effective_decode_throughput": {"unit": "tokens/sec", "avg": 57.19},
        "active_decode_throughput": {"unit": "tokens/sec", "avg": 67.86},
        "effective_concurrency": {"unit": "requests", "avg": 7.42},
        "tokens_in_flight": _latency_block(unit="tokens", avg=173742.75, max=243462.0),
        "credit_drop_latency": _latency_block(avg=5.17, count=2),
        "credit_to_start_latency": _latency_block(avg=3.78),
        "http_req_connection_reused": _latency_block(unit="ratio", avg=0.923),
        "branch_stats": {
            "children_spawned": 12,
            "children_completed": 9,
            "children_errored": 1,
            "children_truncated": 2,
            "children_delayed": 3,
            "parents_failed_due_to_child_error": 1,
            "joins_suppressed": 2,
        },
        "error_summary": [
            {
                "error_details": {"type": "InvalidInferenceResultError"},
                "count": 1,
            },
            {"error_details": {"type": "ServerDisconnectedError"}, "count": 34},
        ],
        "metadata": {
            "dataset": {
                "loader": "semianalysis_cc_traces_weka_062126",
                "hf_dataset_name": "semianalysisai/cc-traces-weka-062126",
                "num_dataset_entries": 393,
            },
            "scenario": "inferencex-agentx-mvp",
            "submission_valid": True,
        },
    }
    summary.update(overrides)
    return summary


def _write_summary(tmp_path, **overrides):
    (tmp_path / "profile_export_aiperf.json").write_text(
        json.dumps(_summary_export(**overrides))
    )
    return tmp_path


class TestOutputParsing:
    def test_missing_artifact_dir_yields_no_metrics(self, tmp_path):
        assert parse_aiperf_output(tmp_path / "nope") == {}

    def test_reads_latency_and_throughput_stats(self, tmp_path):
        metrics = parse_aiperf_output(_write_summary(tmp_path))
        assert metrics["mean_ttft_ms"] == 6130.49
        assert metrics["median_ttft_ms"] == 4808.69
        assert metrics["p99_ttft_ms"] == 27015.14
        assert metrics["std_ttft_ms"] == 4998.71
        assert metrics["mean_tpot_ms"] == 15.02
        assert metrics["median_e2el_ms"] == 10016.63
        assert metrics["output_token_throughput_per_user"] == 119.49
        assert metrics["e2e_output_token_throughput_per_user"] == 40.41
        assert metrics["measured_benchmark_duration"] == 3605.47

    def test_distinguishes_successful_from_completed_counts(self, tmp_path):
        """``request_count`` is successes only; ``completed_request_count`` adds errors."""
        metrics = parse_aiperf_output(_write_summary(tmp_path))
        assert metrics["completed"] == 336
        assert metrics["completed_with_errors"] == 371
        assert metrics["error_request_count"] == 35
        assert metrics["error_rate_pct"] == pytest.approx(9.4339, abs=1e-3)

    def test_token_totals_come_from_the_exact_tags(self, tmp_path):
        """Not mean_isl * request_count, which undercounts when requests error."""
        metrics = parse_aiperf_output(_write_summary(tmp_path))
        assert metrics["total_input_tokens"] == 28597989
        assert metrics["total_output_tokens"] == 207620

    def test_surfaces_scenario_validity_and_dataset_identity(self, tmp_path):
        metrics = parse_aiperf_output(_write_summary(tmp_path))
        assert metrics["submission_valid"] is True
        assert metrics["scenario"] == "inferencex-agentx-mvp"
        assert metrics["dataset_loader"] == "semianalysis_cc_traces_weka_062126"
        assert metrics["dataset_num_entries"] == 393
        assert metrics["theoretical_prefix_cache_hit_pct"] == 95.74

    def test_reads_the_prefill_decode_split_and_applied_load(self, tmp_path):
        """active/effective is the duty cycle, so both halves have to be kept."""
        metrics = parse_aiperf_output(_write_summary(tmp_path))
        assert metrics["effective_prefill_throughput"] == 7878.22
        assert metrics["active_prefill_throughput"] == 14882.3
        assert metrics["effective_decode_throughput"] == 57.19
        assert metrics["active_decode_throughput"] == 67.86
        assert metrics["effective_concurrency"] == 7.42
        assert metrics["mean_tokens_in_flight"] == 173742.75
        assert metrics["max_tokens_in_flight"] == 243462.0

    def test_reads_replay_fidelity_signals(self, tmp_path):
        metrics = parse_aiperf_output(_write_summary(tmp_path))
        assert metrics["mean_ttfot_ms"] == 9660.41
        assert metrics["mean_effective_latency_ms"] == 14592.3
        # `count` is the number of dropped credits, not a mean.
        assert metrics["credit_drop_count"] == 2
        assert metrics["connection_reuse_rate"] == 0.923
        assert metrics["osl_mismatch_diff_pct"] == 0.0

    def test_error_adjusted_percentiles_skip_the_absent_average(self, tmp_path):
        """These blocks carry no ``avg``, so fetching one like the rest reads 0."""
        metrics = parse_aiperf_output(_write_summary(tmp_path))
        assert metrics["p50_adj_ttft_ms"] == 5012.3
        assert metrics["p90_adj_ttft_ms"] == 11987.6
        assert metrics["p50_adj_e2el_ms"] == 11004.7
        assert "p90_adj_e2el_ms" not in metrics

    def test_carries_client_provenance(self, tmp_path):
        metrics = parse_aiperf_output(_write_summary(tmp_path))
        assert metrics["aiperf_version"] == "0.8.0"

    def test_surfaces_branch_stats_and_ranks_errors(self, tmp_path):
        metrics = parse_aiperf_output(_write_summary(tmp_path))
        assert metrics["branch_children_spawned"] == 12
        assert metrics["branch_children_errored"] == 1
        assert metrics["branch_children_truncated"] == 2
        assert metrics["branch_children_delayed"] == 3
        assert metrics["branch_parents_failed_due_to_child_error"] == 1
        assert metrics["branch_joins_suppressed"] == 2
        assert metrics["error_summary"] == [
            {"type": "ServerDisconnectedError", "count": 34},
            {"type": "InvalidInferenceResultError", "count": 1},
        ]

    def test_absent_optional_sections_are_omitted(self, tmp_path):
        """A non-scenario run has no submission verdict to report."""
        (tmp_path / "profile_export_aiperf.json").write_text(
            json.dumps(
                {
                    "time_to_first_token": _latency_block(),
                    "request_count": {"avg": 10},
                }
            )
        )
        metrics = parse_aiperf_output(tmp_path)
        assert "submission_valid" not in metrics
        assert "error_summary" not in metrics
        assert "branch_children_spawned" not in metrics
        assert metrics["completed"] == 10


def _server_metrics_export(**overrides):
    """The fork's aggregated scrape: phase-scoped, with in-window deltas."""
    export = {
        "schema_version": "1.1",
        "summary": {
            "endpoints_configured": ["http://127.0.0.1:8000/metrics"],
            "endpoints_successful": ["http://127.0.0.1:8000/metrics"],
        },
        "metrics_phase": "profiling",
        "metrics": {
            "vllm:prefix_cache_hits": {
                "type": "counter",
                "series": [
                    {
                        "endpoint_url": "http://127.0.0.1:8000/metrics",
                        "stats": {"total": 4544928.0},
                    }
                ],
            },
            "vllm:prefix_cache_queries": {
                "type": "counter",
                "series": [
                    {
                        "endpoint_url": "http://127.0.0.1:8000/metrics",
                        "stats": {"total": 4771106.0},
                    }
                ],
            },
        },
        # Warmup primes the cache and so hits far less; including it would drag
        # the reported rate down.
        "warmup_metrics": {
            "vllm:prefix_cache_hits": {"series": [{"stats": {"total": 1875040.0}}]},
            "vllm:prefix_cache_queries": {"series": [{"stats": {"total": 2206963.0}}]},
        },
    }
    export.update(overrides)
    return export


def _write_server_metrics(tmp_path, **kwargs):
    (tmp_path / "server_metrics_export.json").write_text(
        json.dumps(_server_metrics_export(**kwargs))
    )
    return tmp_path


class TestPrefixCacheMeasurement:
    def test_reads_the_engines_measured_hit_rate(self, tmp_path):
        _write_server_metrics(tmp_path)
        metrics = parse_aiperf_output(_write_summary(tmp_path))
        assert metrics["measured_prefix_cache_hit_pct"] == pytest.approx(
            95.26, abs=1e-2
        )
        assert metrics["prefix_cache_hits_measured"] == 4544928.0
        assert metrics["prefix_cache_queries_measured"] == 4771106.0

    def test_warmup_traffic_is_excluded(self, tmp_path):
        """``metrics`` is the profiling phase; ``warmup_metrics`` must not count."""
        _write_server_metrics(tmp_path)
        metrics = parse_aiperf_output(_write_summary(tmp_path))
        combined_rate = 100 * (4544928 + 1875040) / (4771106 + 2206963)
        assert metrics["measured_prefix_cache_hit_pct"] != pytest.approx(
            combined_rate, abs=1e-2
        )

    def test_sums_across_workers_before_dividing(self, tmp_path):
        """A token-weighted rate, so an idle worker cannot skew it."""
        _write_server_metrics(
            tmp_path,
            metrics={
                "vllm:prefix_cache_hits": {
                    "series": [
                        {"stats": {"total": 900.0}},
                        {"stats": {"total": 0.0}},
                    ]
                },
                "vllm:prefix_cache_queries": {
                    "series": [
                        {"stats": {"total": 1000.0}},
                        {"stats": {"total": 100.0}},
                    ]
                },
            },
        )
        metrics = parse_aiperf_output(_write_summary(tmp_path))
        assert metrics["measured_prefix_cache_hit_pct"] == pytest.approx(
            81.81, abs=1e-2
        )

    def test_absent_scrape_omits_the_metric_rather_than_reporting_zero(self, tmp_path):
        metrics = parse_aiperf_output(_write_summary(tmp_path))
        assert "measured_prefix_cache_hit_pct" not in metrics
        # The theoretical bound comes from the summary and stays regardless.
        assert metrics["theoretical_prefix_cache_hit_pct"] == 95.74

    def test_a_server_without_cache_counters_omits_the_metric(self, tmp_path):
        _write_server_metrics(
            tmp_path, metrics={"vllm:num_requests_running": {"series": []}}
        )
        metrics = parse_aiperf_output(_write_summary(tmp_path))
        assert "measured_prefix_cache_hit_pct" not in metrics

    def test_explicit_endpoints_exclude_the_load_target(self, tmp_path):
        """AIPerf keeps scraping the load target; a frontend that also exports
        the counters would double-count every prompt if it were summed in."""
        _write_server_metrics(
            tmp_path,
            metrics={
                "vllm:prefix_cache_hits": {
                    "series": [
                        {
                            "endpoint_url": "http://frontend:8000/metrics",
                            "stats": {"total": 500.0},
                        },
                        {
                            "endpoint_url": "http://worker-a:9000/metrics",
                            "stats": {"total": 900.0},
                        },
                    ]
                },
                "vllm:prefix_cache_queries": {
                    "series": [
                        {
                            "endpoint_url": "http://frontend:8000/metrics",
                            "stats": {"total": 1000.0},
                        },
                        {
                            "endpoint_url": "http://worker-a:9000/metrics",
                            "stats": {"total": 1000.0},
                        },
                    ]
                },
            },
        )
        metrics = parse_aiperf_output(
            _write_summary(tmp_path), metrics_urls=("http://worker-a:9000/metrics",)
        )
        assert metrics["measured_prefix_cache_hit_pct"] == pytest.approx(90.0)
        assert metrics["prefix_cache_queries_measured"] == 1000.0
        assert metrics["prefix_cache_metrics_endpoints"] == [
            "http://worker-a:9000/metrics"
        ]

    def test_unmatched_endpoint_omits_the_metric(self, tmp_path):
        """A typo'd worker URL must not silently fall back to the load target."""
        _write_server_metrics(tmp_path)
        metrics = parse_aiperf_output(
            _write_summary(tmp_path), metrics_urls=("http://typo:9999/metrics",)
        )
        assert "measured_prefix_cache_hit_pct" not in metrics

    def test_zero_queries_does_not_divide_by_zero(self, tmp_path):
        _write_server_metrics(
            tmp_path,
            metrics={
                "vllm:prefix_cache_hits": {"series": [{"stats": {"total": 0.0}}]},
                "vllm:prefix_cache_queries": {"series": [{"stats": {"total": 0.0}}]},
            },
        )
        metrics = parse_aiperf_output(_write_summary(tmp_path))
        assert "measured_prefix_cache_hit_pct" not in metrics


class TestResultValidity:
    def test_a_healthy_run_is_usable(self, tmp_path):
        metrics = parse_aiperf_output(_write_summary(tmp_path))
        assert _invalid_result_reason(metrics) is None

    def test_an_invalid_submission_is_rejected(self, tmp_path):
        metrics = parse_aiperf_output(
            _write_summary(
                tmp_path,
                metadata={
                    "scenario": "inferencex-agentx-mvp",
                    "submission_valid": False,
                    "submission_invalid_reasons": ["context_overflow_rate_exceeded"],
                },
            )
        )
        reason = _invalid_result_reason(metrics)
        assert reason and "context_overflow_rate_exceeded" in reason

    def test_a_cancelled_run_is_rejected(self, tmp_path):
        metrics = parse_aiperf_output(_write_summary(tmp_path, was_cancelled=True))
        assert "cancelled" in (_invalid_result_reason(metrics) or "")

    def test_zero_successful_requests_is_rejected(self, tmp_path):
        """aiperf exits 0 when every request fails; that must not report as zeros."""
        metrics = parse_aiperf_output(
            _write_summary(tmp_path, request_count={"unit": "requests", "avg": 0.0})
        )
        assert "no request completed" in (_invalid_result_reason(metrics) or "")

    def test_zero_ttft_is_rejected(self, tmp_path):
        metrics = parse_aiperf_output(
            _write_summary(tmp_path, time_to_first_token=_latency_block(avg=0.0))
        )
        assert "TTFT" in (_invalid_result_reason(metrics) or "")


class TestParser:
    def test_block_kind_and_error_rate(self, tmp_path):
        metrics = parse_aiperf_output(_write_summary(tmp_path))
        block = AIPerfAgenticTracesParser().parse(
            {
                "model_id": "moonshotai/Kimi-K2.7-Code",
                "date": "20260727-120000",
                "metadata": {"inferencex_git_ref": KIMI_PINNED_REF},
                **metrics,
            },
            device="super_cluster",
        )
        assert block.kind == "agentic_traces"
        # The fork reports a percentage; the fraction is derived for comparison
        # against failed_request_threshold, which is a ratio.
        assert block.data["error_rate_pct"] == pytest.approx(9.4339, abs=1e-3)
        assert block.data["error_rate"] == pytest.approx(0.094339, abs=1e-5)
        assert block.data["submission_status"] == "valid"
        assert block.data["error_summary"] == (
            "34xServerDisconnectedError, 1xInvalidInferenceResultError"
        )
        assert block.data["inferencex_git_ref"] == KIMI_PINNED_REF
        assert block.targets["device"] == "super_cluster"
        assert block.targets["timestamp"] == "2026-07-27 12:00:00"

    def test_invalid_submission_is_flagged_in_the_report(self):
        block = AIPerfAgenticTracesParser().parse(
            {
                "model_id": "m",
                "submission_valid": False,
                "submission_invalid_reasons": ["unsafe_override", "run_cancelled"],
            }
        )
        assert block.data["submission_status"] == (
            "INVALID: unsafe_override, run_cancelled"
        )

    def test_missing_counts_leave_error_rate_absent(self):
        block = AIPerfAgenticTracesParser().parse({"model_id": "m"})
        assert "error_rate" not in block.data
        assert "submission_status" not in block.data


SWO_SCENARIO = "claude-code-swe-bench-python-kimi-k2.7-code"


class TestSwarmOneModeScoping:
    """The Kimi config runs SwarmOne all-tasks at load in full, sympy c1 in ci."""

    def _swo_runs(self, mode):
        config = AGENTIC_TRACES_CONFIGS[KIMI_MODEL_ID]
        return [
            run
            for run in build_runs(config, _FakeModelSpec(), mode=mode)
            if run.trace_source is TraceSource.SWARMONE
        ]

    def test_full_runs_every_task_at_load(self):
        runs = self._swo_runs(AgenticTracesMode.FULL)
        assert len(runs) == 1
        run = runs[0]
        assert run.scenario == SWO_SCENARIO
        assert run.task is None
        assert run.concurrency == 8

    def test_ci_narrows_to_one_task_at_concurrency_one(self):
        runs = self._swo_runs(AgenticTracesMode.CI)
        assert len(runs) == 1
        run = runs[0]
        assert run.task == "sympy-bugfix"
        assert run.concurrency == 1

    def test_inferencex_run_is_unchanged_by_swarmone_ci(self):
        """SwarmOne's CI c1 scoping must not shrink the InferenceX CI run.

        The InferenceX run keeps its own configured concurrency (from its spec /
        CI mode settings), never SwarmOne's concurrency of 1.
        """
        config = AGENTIC_TRACES_CONFIGS[KIMI_MODEL_ID]
        inferencex_spec = next(
            spec
            for spec in config.runs
            if spec.trace_source is TraceSource.INFERENCEX_AGENTX
        )
        expected = (
            config.settings_for_mode(AgenticTracesMode.CI).concurrency
            or inferencex_spec.concurrency
        )
        inferencex = [
            run
            for run in build_runs(config, _FakeModelSpec(), mode=AgenticTracesMode.CI)
            if run.trace_source is TraceSource.INFERENCEX_AGENTX
        ]
        assert len(inferencex) == 1
        assert inferencex[0].concurrency == expected
        assert inferencex[0].concurrency != 1

    def test_default_selection_holds_swarmone_back(self):
        """swarmone is opt-in, so an unnarrowed sweep must not pull it in and
        make a SwarmOne license a precondition for every Kimi run."""
        config = AGENTIC_TRACES_CONFIGS[KIMI_MODEL_ID]
        _, specs = resolve_run_specs(config)
        sources = {spec.trace_source for spec in specs}
        assert sources == {TraceSource.INFERENCEX_AGENTX}

    def test_explicitly_selecting_both_sources_runs_both(self):
        config = AGENTIC_TRACES_CONFIGS[KIMI_MODEL_ID]
        _, specs = resolve_run_specs(
            config,
            trace_sources=(TraceSource.INFERENCEX_AGENTX, TraceSource.SWARMONE),
        )
        sources = {
            run.trace_source
            for run in build_runs(
                config,
                _FakeModelSpec(),
                mode=AgenticTracesMode.FULL,
                run_specs=specs,
            )
        }
        assert sources == {TraceSource.INFERENCEX_AGENTX, TraceSource.SWARMONE}

    def test_a_swarmone_only_config_still_runs_by_default(self):
        """Holding back the only configured source would leave an empty sweep."""
        config = AgenticTracesConfig(
            model_id="id_test",
            inferencex_git_ref=None,
            runs=(
                AgenticTracesRunSpec(
                    trace_source=TraceSource.SWARMONE, scenario=SWO_SCENARIO
                ),
            ),
        )
        _, specs = resolve_run_specs(config)
        assert [spec.trace_source for spec in specs] == [TraceSource.SWARMONE]

    def test_swarmone_timeout_estimate_is_generous(self):
        full = self._swo_runs(AgenticTracesMode.FULL)[0]
        ci = self._swo_runs(AgenticTracesMode.CI)[0]
        assert estimated_run_seconds(full) == 8 * 3600
        assert estimated_run_seconds(ci) == 3600

    def test_swarmone_label_keys_on_scenario_and_task(self):
        ci = self._swo_runs(AgenticTracesMode.CI)[0]
        assert "swarmone" in ci.label
        assert SWO_SCENARIO in ci.label
        assert "sympy-bugfix" in ci.label
        assert ci.label.endswith("_ci")


class TestSwarmOneRunSpecValidation:
    def test_bad_cache_mode_is_rejected(self):
        with pytest.raises(ValueError, match="cache_mode"):
            AgenticTracesRunSpec(trace_source=TraceSource.SWARMONE, cache_mode="turbo")

    def test_bad_history_mode_is_rejected(self):
        with pytest.raises(ValueError, match="history_mode"):
            AgenticTracesRunSpec(
                trace_source=TraceSource.SWARMONE, history_mode="rewind"
            )

    def test_bad_max_tokens_mode_is_rejected(self):
        with pytest.raises(ValueError, match="max_tokens_mode"):
            AgenticTracesRunSpec(
                trace_source=TraceSource.SWARMONE, max_tokens_mode="dynamic"
            )

    def test_zero_max_tokens_is_rejected(self):
        with pytest.raises(ValueError, match="max_tokens"):
            AgenticTracesRunSpec(trace_source=TraceSource.SWARMONE, max_tokens=0)

    def test_zero_resident_is_rejected(self):
        with pytest.raises(ValueError, match="resident"):
            AgenticTracesRunSpec(trace_source=TraceSource.SWARMONE, resident=0)


class TestSwarmOneConfigWithoutInferencex:
    def test_swarmone_only_config_needs_no_git_ref(self):
        config = AgenticTracesConfig(
            model_id="id_swo_only",
            inferencex_git_ref="",
            runs=(
                AgenticTracesRunSpec(
                    trace_source=TraceSource.SWARMONE,
                    scenario=SWO_SCENARIO,
                    public_dataset="",
                ),
            ),
        )
        runs = build_runs(config, _FakeModelSpec())
        assert runs[0].trace_source is TraceSource.SWARMONE


class TestSwoBenchCommand:
    def _run(self, mode=AgenticTracesMode.CI):
        config = AGENTIC_TRACES_CONFIGS[KIMI_MODEL_ID]
        return [
            run
            for run in build_runs(config, _FakeModelSpec(), mode=mode)
            if run.trace_source is TraceSource.SWARMONE
        ][0]

    def _cmd(self, mode=AgenticTracesMode.CI, **kwargs):
        defaults = dict(
            run=self._run(mode),
            venv_python=Path("/venv/bin/python"),
            model_name="moonshotai/Kimi-K2.7-Code",
            url="http://localhost:8000",
            results_path=Path("/tmp/artifacts/swo_bench_results.json"),
        )
        defaults.update(kwargs)
        return build_swo_bench_cmd(**defaults)

    def test_invokes_the_venv_swo_bench_module(self):
        cmd = self._cmd()
        assert cmd[:4] == ["/venv/bin/python", "-m", "swo_bench", "replay"]

    def test_carries_scenario_task_and_context(self):
        cmd = self._cmd()
        assert cmd[cmd.index("--scenario") + 1] == SWO_SCENARIO
        assert cmd[cmd.index("--task") + 1] == "sympy-bugfix"
        assert cmd[cmd.index("--model-context-length") + 1] == "262144"
        assert "--no-resolve-model-context" in cmd

    def test_endpoint_gets_v1_suffix(self):
        cmd = self._cmd()
        assert cmd[cmd.index("--endpoint") + 1] == "http://localhost:8000/v1"

    def test_bare_host_gets_scheme_and_v1(self):
        cmd = self._cmd(url="localhost:8000")
        assert cmd[cmd.index("--endpoint") + 1] == "http://localhost:8000/v1"

    def test_concurrency_and_resident_default_together(self):
        cmd = self._cmd(mode=AgenticTracesMode.CI)
        assert cmd[cmd.index("--concurrent") + 1] == "1"
        # resident defaults to concurrency when unset in the spec.
        assert cmd[cmd.index("--resident") + 1] == "1"

    def test_replay_knobs_and_output(self):
        cmd = self._cmd()
        assert cmd[cmd.index("--cache-mode") + 1] == "realistic"
        assert cmd[cmd.index("--history-mode") + 1] == "faithful"
        assert cmd[cmd.index("--max-tokens") + 1] == "4096"
        assert cmd[cmd.index("--max-tokens-mode") + 1] == "flat"
        assert "--verbose-text" in cmd
        assert cmd[cmd.index("--json-output") + 1] == (
            "/tmp/artifacts/swo_bench_results.json"
        )

    def test_full_mode_replays_all_tasks(self):
        cmd = self._cmd(mode=AgenticTracesMode.FULL)
        assert "--task" not in cmd
        assert cmd[cmd.index("--concurrent") + 1] == "8"

    def test_api_key_only_present_with_a_token(self):
        assert "--api-key" not in self._cmd()
        cmd = self._cmd(auth_token="tok123")
        assert cmd[cmd.index("--api-key") + 1] == "tok123"

    def test_license_never_appears_on_argv(self):
        cmd = self._cmd(auth_token="tok123")
        joined = " ".join(cmd)
        assert "--license-key" not in joined
        assert "SWO_LICENSE_KEY" not in joined


def _swo_results(**overrides):
    """A trimmed swo-bench ``--json-output`` file (real 3.x schema)."""
    results = {
        "session_id": "bench_session_abc",
        "source": SWO_SCENARIO,
        "endpoint": "http://localhost:8000/v1",
        "model": "moonshotai/Kimi-K2.7-Code",
        "config": {"concurrency": 1, "max_tokens": 4096, "max_requests": None},
        "wall_clock_ms": 67053.4,
        "timings": [
            {"index": 0, "status": "success", "prompt_tokens": 1015},
            {"index": 1, "status": "success", "prompt_tokens": 21834},
        ],
        "report": {
            "metrics": {
                "aggregate_throughput_tok_s": 107.59,
                "decode_tok_per_sec": {
                    "max": 142.7,
                    "mean": 106.09,
                    "min": 52.13,
                    "p50": 113.31,
                    "p90": 127.64,
                    "p99": 142.7,
                },
                "failed": 0,
                "itl_ms_p50": {"mean": 0.02, "p50": 0.02},
                "latency_ms": {
                    "max": 7324.25,
                    "mean": 2481.6,
                    "min": 485.18,
                    "p50": 1692.27,
                    "p90": 5078.3,
                    "p99": 7324.25,
                },
                "prefill_tok_per_sec": {
                    "mean": 500.81,
                    "p50": 276.16,
                    "p90": 985.43,
                },
                "successful": 26,
                "total_output_tokens": 7214,
                "total_requests": 26,
                "ttft_ms": {
                    "max": 7314.14,
                    "mean": 2553.08,
                    "min": 484.15,
                    "p50": 1691.08,
                    "p90": 5078.19,
                    "p99": 7314.14,
                },
                "wall_clock_s": 67.05,
            },
            "summary": "26/26 requests succeeded",
        },
    }
    results.update(overrides)
    return results


def _write_swo_results(tmp_path, **overrides):
    path = tmp_path / "swo_bench_results.json"
    path.write_text(json.dumps(_swo_results(**overrides)))
    return path


class TestSwoBenchOutputParsing:
    def test_missing_file_yields_no_metrics(self, tmp_path):
        assert parse_swo_bench_output(tmp_path / "nope.json") == {}

    def test_reads_latency_and_throughput(self, tmp_path):
        metrics = parse_swo_bench_output(_write_swo_results(tmp_path))
        assert metrics["mean_ttft_ms"] == 2553.08
        assert metrics["median_ttft_ms"] == 1691.08
        assert metrics["p99_ttft_ms"] == 7314.14
        assert metrics["median_e2el_ms"] == 1692.27
        assert metrics["output_token_throughput_per_user"] == 106.09
        assert metrics["output_token_throughput"] == 107.59
        assert metrics["prefill_tok_per_sec_p50"] == 276.16
        assert metrics["measured_benchmark_duration"] == 67.05

    def test_counts_and_error_rate(self, tmp_path):
        metrics = parse_swo_bench_output(_write_swo_results(tmp_path))
        assert metrics["completed"] == 26
        assert metrics["completed_with_errors"] == 26
        assert metrics["error_request_count"] == 0
        assert metrics["error_rate_pct"] == 0.0

    def test_input_tokens_summed_from_timings(self, tmp_path):
        metrics = parse_swo_bench_output(_write_swo_results(tmp_path))
        assert metrics["total_input_tokens"] == 1015 + 21834
        assert metrics["total_output_tokens"] == 7214

    def test_provenance_is_carried(self, tmp_path):
        metrics = parse_swo_bench_output(_write_swo_results(tmp_path))
        assert metrics["swo_session_id"] == "bench_session_abc"
        assert metrics["swo_source_label"] == SWO_SCENARIO

    def test_missing_report_block_yields_no_metrics(self, tmp_path):
        path = tmp_path / "swo_bench_results.json"
        path.write_text(json.dumps({"session_id": "x", "timings": []}))
        assert parse_swo_bench_output(path) == {}


class TestSwoBenchResultValidity:
    def test_healthy_run_is_usable(self, tmp_path):
        metrics = parse_swo_bench_output(_write_swo_results(tmp_path))
        assert _swo_invalid_result_reason(metrics) is None

    def test_zero_successful_requests_is_rejected(self, tmp_path):
        results = _swo_results()
        results["report"]["metrics"]["successful"] = 0
        results["report"]["metrics"]["failed"] = 26
        path = tmp_path / "swo_bench_results.json"
        path.write_text(json.dumps(results))
        metrics = parse_swo_bench_output(path)
        assert "no request completed" in (_swo_invalid_result_reason(metrics) or "")


class TestSwoBenchParser:
    def test_block_kind_and_error_rate(self, tmp_path):
        metrics = parse_swo_bench_output(_write_swo_results(tmp_path))
        block = SwoBenchAgenticTracesParser().parse(
            {
                "model_id": "moonshotai/Kimi-K2.7-Code",
                "date": "20260727-120000",
                "trace_source": "swarmone",
                "backend": "swo-bench",
                "metadata": {"mode": "ci"},
                **metrics,
            },
            device="super_cluster",
        )
        assert block.kind == "agentic_traces"
        assert block.data["trace_source"] == "swarmone"
        assert block.data["error_rate"] == pytest.approx(0.0)
        assert block.data["mode"] == "ci"
        assert block.targets["device"] == "super_cluster"
        assert block.targets["timestamp"] == "2026-07-27 12:00:00"
