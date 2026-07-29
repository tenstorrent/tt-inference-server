# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for the agentic trace-replay config, run expansion, and CLI build.

The load-bearing invariants here are the ones a broken run would only reveal an
hour in (or worse, as a plausible-looking row of numbers):

* every configured model pins an InferenceX ref and covers every mode,
* no mode can drop below the scenario's 900s profiling floor,
* the derived knobs (max context, trust-remote-code) track the ModelSpec,
* selecting the unimplemented SwarmOne source fails loudly instead of
  producing an empty sweep,
* the emitted CLI carries the flags the scenario requires.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_module.agentic_traces import build_runs, summarize_runs, total_planned_seconds
from llm_module.drivers.aiperf_agentic_traces import (
    _invalid_result_reason,
    build_aiperf_cmd,
    parse_aiperf_output,
)
from llm_module.parsers.aiperf_agentic_traces import AIPerfAgenticTracesParser
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
KIMI_PINNED_REF = "e2dcfa91c86936cc011e3be0668eb3b1ca17288f"


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
                        agentic_cache_warmup_duration=600,
                        warmup_grace_period=1800,
                        num_dataset_entries=393,
                    ),
                    AgenticTracesMode.CI: AgenticTracesModeSettings(
                        benchmark_duration=300,
                        agentic_cache_warmup_duration=60,
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
                    agentic_cache_warmup_duration=60,
                    warmup_grace_period=120,
                    num_dataset_entries=8,
                ),
                AgenticTracesMode.CI: AgenticTracesModeSettings(
                    benchmark_duration=120,
                    agentic_cache_warmup_duration=30,
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
                        agentic_cache_warmup_duration=600,
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
        assert run.agentic_cache_warmup_duration == 600
        assert run.num_dataset_entries == 393
        assert run.mode is AgenticTracesMode.FULL

    def test_ci_mode_shortens_the_run(self):
        config = AGENTIC_TRACES_CONFIGS[KIMI_MODEL_ID]
        full = build_runs(config, _FakeModelSpec(), mode=AgenticTracesMode.FULL)[0]
        ci = build_runs(config, _FakeModelSpec(), mode=AgenticTracesMode.CI)[0]
        assert ci.benchmark_duration < full.benchmark_duration
        assert ci.agentic_cache_warmup_duration < full.agentic_cache_warmup_duration
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
                    agentic_cache_warmup_duration=600,
                    warmup_grace_period=1800,
                    num_dataset_entries=393,
                ),
                AgenticTracesMode.CI: AgenticTracesModeSettings(
                    benchmark_duration=900,
                    agentic_cache_warmup_duration=120,
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

    def test_swarmone_run_raises_not_implemented(self):
        config = AgenticTracesConfig(
            model_id="id_test",
            inferencex_git_ref="abc123",
            runs=(AgenticTracesRunSpec(trace_source=TraceSource.SWARMONE),),
        )
        with pytest.raises(NotImplementedError, match="swarmone"):
            build_runs(config, _FakeModelSpec())

    def test_narrowing_to_a_configured_source_keeps_its_runs(self):
        config = AGENTIC_TRACES_CONFIGS[KIMI_MODEL_ID]
        _, runs = resolve_run_specs(
            config, trace_sources=(TraceSource.INFERENCEX_AGENTX,)
        )
        assert len(runs) == len(config.runs)

    def test_narrowing_to_an_unconfigured_source_raises(self):
        config = AGENTIC_TRACES_CONFIGS[KIMI_MODEL_ID]
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
            "--agentic-cache-warmup-duration",
            "--warmup-grace-period",
        ):
            assert flag in cmd, f"{flag} missing from {cmd}"

    def test_durations_match_the_selected_mode(self):
        cmd = self._cmd()
        assert cmd[cmd.index("--benchmark-duration") + 1] == str(
            AGENTIC_TRACES_MIN_PROFILE_SECONDS
        )

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


class TestPlanSummary:
    def test_summary_names_every_run(self):
        config = AGENTIC_TRACES_CONFIGS[KIMI_MODEL_ID]
        runs = build_runs(config, _FakeModelSpec())
        summary = summarize_runs(runs)
        for run in runs:
            assert run.label in summary

    def test_summary_handles_no_runs(self):
        assert "No runs planned" in summarize_runs([])

    def test_total_planned_seconds_sums_warmup_and_profiling(self):
        config = AGENTIC_TRACES_CONFIGS[KIMI_MODEL_ID]
        runs = build_runs(config, _FakeModelSpec(), mode=AgenticTracesMode.CI)
        assert total_planned_seconds(runs) == sum(
            r.benchmark_duration
            + r.agentic_cache_warmup_duration
            + r.warmup_grace_period
            for r in runs
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
        "theoretical_prefix_cache_hit": {"unit": "%", "avg": 95.74},
        "was_cancelled": False,
        "branch_stats": {
            "children_spawned": 12,
            "children_completed": 9,
            "children_errored": 1,
            "children_truncated": 2,
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

    def test_surfaces_branch_stats_and_ranks_errors(self, tmp_path):
        metrics = parse_aiperf_output(_write_summary(tmp_path))
        assert metrics["branch_children_spawned"] == 12
        assert metrics["branch_children_errored"] == 1
        assert metrics["branch_children_truncated"] == 2
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
