# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
"""Feature-extraction and encoder throughput on the Whisper path.

`whisper_runner` probes for tt-metal's `PerfMetrics` (#53717) at import time, so
the stub below carries it. Its absence is the older-tt-metal case covered by
`test_absent_perf_metrics_*`.
"""

import asyncio
import importlib
import sys
import types
from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest
from prometheus_client import REGISTRY

FEATURE_INPUT = "tt_media_server_audio_feature_extraction_input_seconds_total"
FEATURE_DURATION = "tt_media_server_audio_feature_extraction_duration_seconds"
ENCODER_INPUT = "tt_media_server_audio_encoder_input_seconds_total"
ENCODER_DURATION = "tt_media_server_audio_encoder_duration_seconds"

_METRIC_ATTRS = {
    "audio_feature_extraction_input_seconds": FEATURE_INPUT,
    "audio_feature_extraction_duration": FEATURE_DURATION,
    "audio_encoder_input_seconds": ENCODER_INPUT,
    "audio_encoder_duration": ENCODER_DURATION,
    "audio_chunk_first_token_duration": "tt_media_server_audio_chunk_first_token_seconds",
    "audio_chunk_processing_duration": "tt_media_server_audio_chunk_processing_seconds",
}
_AUDIO_STUBS = (
    "models.demos.audio",
    "models.demos.audio.whisper",
    "models.demos.audio.whisper.tt",
    "models.demos.audio.whisper.tt.ttnn_optimized_functional_whisper",
    "models.demos.audio.whisper.tt.whisper_generator",
)

_FEATURE_LABELS = {
    "model_type": "tt-whisper",
    "device_id": "0",
    "sample_rate": "16000",
    "channels": "1",
    "batch": "1",
}
_ENCODER_LABELS = {
    "model_type": "tt-whisper",
    "device_id": "0",
    "model_name": "distil-large-v3",
    "language": "English",
    "batch": "1",
    "trace_hit": "true",
}


@dataclass
class StubPerfMetrics:
    """Mirrors the fields whisper_runner reads off the real PerfMetrics.

    Defaults match tt-metal's dataclass exactly, zeros included. They used to
    be 0.01/0.2, which made a bare StubPerfMetrics() record where a bare
    PerfMetrics() records nothing — so a test could assert positive recording
    against a value the real generator never produces. Every test that wants a
    recording now has to say so.
    """

    feature_extract_s: float = 0.0
    encoder_s: float = 0.0
    total_audio_s: float = 0.0
    ttft: float = 0.0
    decode_throughput: float = 0.0
    encoder_trace_hit: bool = True


def _pin_real_collectors():
    """Pin the real collectors onto telemetry.telemetry_client.

    Other test modules leave a Mock there that swallows every observation; the
    registry keeps the one reference surviving that swap.
    """
    module = sys.modules.get("telemetry.telemetry_client")
    if module is None:
        # A bare stub, not a re-import: that would re-register every collector.
        module = types.ModuleType("telemetry.telemetry_client")
        module.TelemetryEvent = MagicMock()
        sys.modules["telemetry.telemetry_client"] = module

    saved = []
    for attr, name in _METRIC_ATTRS.items():
        saved.append((attr, hasattr(module, attr), getattr(module, attr, None)))
        setattr(module, attr, REGISTRY._names_to_collectors[name])
    return module, saved


def _whisper_module(with_perf_metrics=True):
    # Every tt_model_runners entry goes, not just whisper_runner: other modules
    # park a Mock at `tt_model_runners.base_device_runner`, reached through the
    # base class, and re-importing over that yields a Mock TTWhisperRunner.
    saved_modules = {
        name: sys.modules.get(name)
        for name in list(_AUDIO_STUBS)
        + [name for name in sys.modules if name.startswith("tt_model_runners")]
    }
    telemetry_module, saved_attrs = _pin_real_collectors()

    for name in _AUDIO_STUBS:
        stub = types.ModuleType(name)
        stub.__dict__.update(
            {
                "WHISPER_L1_SMALL_SIZE": 0,
                "WHISPER_TRACE_REGION_SIZE": 0,
                "convert_to_ttnn": MagicMock(),
                "create_custom_mesh_preprocessor": MagicMock(),
                "init_kv_cache": MagicMock(),
                "GenerationParams": MagicMock(),
                "WhisperGenerator": MagicMock(),
            }
        )
        if with_perf_metrics:
            stub.PerfMetrics = StubPerfMetrics
        sys.modules[name] = stub

    for name in list(sys.modules):
        if name.startswith("tt_model_runners"):
            sys.modules.pop(name, None)
    module = importlib.import_module("tt_model_runners.whisper_runner")
    return module, (telemetry_module, saved_attrs, saved_modules)


def _restore(state):
    telemetry_module, saved_attrs, saved_modules = state
    for attr, had_attr, previous in saved_attrs:
        if had_attr:
            setattr(telemetry_module, attr, previous)
        else:
            delattr(telemetry_module, attr)
    for name in list(sys.modules):
        if name.startswith("tt_model_runners"):
            sys.modules.pop(name, None)
    for name, module in saved_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


@pytest.fixture
def whisper_module():
    module, state = _whisper_module()
    try:
        yield module
    finally:
        _restore(state)


@pytest.fixture
def whisper_module_without_perf_metrics():
    module, state = _whisper_module(with_perf_metrics=False)
    try:
        yield module
    finally:
        _restore(state)


def _counter(metric, labels):
    value = REGISTRY.get_sample_value(metric, labels)
    return 0.0 if value is None else value


def _histogram(metric, field, labels):
    value = REGISTRY.get_sample_value(f"{metric}_{field}", labels)
    return 0.0 if value is None else value


def _runner(whisper_module, frame_seconds=30.0):
    """Carries only what the throughput helpers touch."""
    runner = MagicMock()
    runner.settings.default_sample_rate = 16000
    runner.settings.model_runner = "tt-whisper"
    runner.settings.model_weights_path = "distil-whisper/distil-large-v3"
    runner.settings.audio_language = "English"
    runner.device_id = 0
    runner._encoder_frame_seconds = frame_seconds

    runner_class = whisper_module.TTWhisperRunner
    for name in (
        "_audio_stage_context",
        "_record_audio_stage_throughput",
        "_stream_with_stage_metrics",
    ):
        setattr(runner, name, getattr(runner_class, name).__get__(runner))
    runner._find_perf_metrics = runner_class._find_perf_metrics
    return runner


def _batch(seconds, sample_rate=16000, count=1):
    array = np.zeros(int(sample_rate * seconds), dtype=np.float32)
    return [(sample_rate, array)] * count


def test_stage_metrics_recorded_once_per_batch(whisper_module):
    runner = _runner(whisper_module)
    feature_before = _counter(FEATURE_INPUT, _FEATURE_LABELS)
    encoder_before = _counter(ENCODER_INPUT, _ENCODER_LABELS)
    feature_count_before = _histogram(FEATURE_DURATION, "count", _FEATURE_LABELS)
    encoder_count_before = _histogram(ENCODER_DURATION, "count", _ENCODER_LABELS)

    context = runner._audio_stage_context(_batch(8.0))
    result = (
        "text",
        None,
        None,
        StubPerfMetrics(feature_extract_s=0.01, encoder_s=0.2),
    )
    assert runner._record_audio_stage_throughput(result, context) is True

    assert _counter(FEATURE_INPUT, _FEATURE_LABELS) == pytest.approx(
        feature_before + 8.0
    )
    assert _counter(ENCODER_INPUT, _ENCODER_LABELS) == pytest.approx(
        encoder_before + 8.0
    )
    assert _histogram(FEATURE_DURATION, "count", _FEATURE_LABELS) == (
        feature_count_before + 1
    )
    assert _histogram(ENCODER_DURATION, "count", _ENCODER_LABELS) == (
        encoder_count_before + 1
    )


def test_throughput_derives_from_the_counter_pair(whisper_module):
    """Both halves of the ratio must move together in the same units."""
    runner = _runner(whisper_module)
    input_before = _counter(ENCODER_INPUT, _ENCODER_LABELS)
    sum_before = _histogram(ENCODER_DURATION, "sum", _ENCODER_LABELS)

    context = runner._audio_stage_context(_batch(15.0))
    runner._record_audio_stage_throughput(
        ("text", None, None, StubPerfMetrics(encoder_s=0.5)), context
    )

    audio_delta = _counter(ENCODER_INPUT, _ENCODER_LABELS) - input_before
    time_delta = _histogram(ENCODER_DURATION, "sum", _ENCODER_LABELS) - sum_before
    assert audio_delta / time_delta == pytest.approx(30.0)


def test_audio_seconds_are_capped_at_one_encoder_frame(whisper_module):
    """The extractor truncates to the frame; crediting the full submission
    would inflate throughput by that ratio."""
    runner = _runner(whisper_module)

    assert runner._audio_stage_context(_batch(90.0))["audio_seconds"] == pytest.approx(
        30.0
    )
    assert runner._audio_stage_context(_batch(12.0))["audio_seconds"] == pytest.approx(
        12.0
    )


def test_batch_audio_seconds_sum_per_item(whisper_module):
    runner = _runner(whisper_module)
    context = runner._audio_stage_context(_batch(10.0, count=2))

    assert context["audio_seconds"] == pytest.approx(20.0)
    assert context["feature_labels"]["batch"] == "2"
    assert context["encoder_labels"]["batch"] == "2"


def test_batch_padding_is_not_credited_as_audio(whisper_module):
    """A 2-item batch is zero-padded to the longer item before submission.

    The arrays therefore both measure 25s, but only 30s of real audio went in.
    Deriving duration from the submitted array would credit the pad and report
    50s, inflating both stage throughputs by 1.67x.
    """
    runner = _runner(whisper_module)
    padded = _batch(25.0, count=2)

    assert runner._audio_stage_context(padded)["audio_seconds"] == pytest.approx(50.0)

    context = runner._audio_stage_context(padded, audio_durations=[5.0, 25.0])
    assert context["audio_seconds"] == pytest.approx(30.0)
    assert context["feature_labels"]["batch"] == "2"


def test_supplied_durations_are_still_capped_at_the_frame(whisper_module):
    """The clamp has to survive the override, or an over-long submitted
    duration reintroduces the inflation the extractor's truncation causes."""
    runner = _runner(whisper_module)
    context = runner._audio_stage_context(
        _batch(90.0, count=2), audio_durations=[90.0, 10.0]
    )

    assert context["audio_seconds"] == pytest.approx(40.0)


def test_missing_durations_fall_back_to_the_array(whisper_module):
    """Every non-batched caller passes nothing; the array stays the source."""
    runner = _runner(whisper_module)

    assert runner._audio_stage_context(_batch(7.0), audio_durations=None)[
        "audio_seconds"
    ] == pytest.approx(7.0)
    assert runner._audio_stage_context(_batch(7.0, count=2), audio_durations=[3.0])[
        "audio_seconds"
    ] == pytest.approx(10.0)


def test_labels_carry_the_operating_point(whisper_module):
    runner = _runner(whisper_module)
    context = runner._audio_stage_context(
        _batch(4.0), audio_profile={"sample_rate": "8000", "channels": "2"}
    )

    assert context["feature_labels"] == {
        "model_type": "tt-whisper",
        "device_id": "0",
        "sample_rate": "8000",
        "channels": "2",
        "batch": "1",
    }
    assert context["encoder_labels"] == {
        "model_type": "tt-whisper",
        "device_id": "0",
        "model_name": "distil-large-v3",
        "language": "English",
        "batch": "1",
    }


def test_labels_fall_back_to_the_pipeline_operating_point(whisper_module):
    """Warmup and any caller that has no request pass no profile."""
    runner = _runner(whisper_module)
    labels = runner._audio_stage_context(_batch(4.0))["feature_labels"]

    assert labels["sample_rate"] == "16000"
    assert labels["channels"] == "1"


def test_source_profile_reports_what_was_submitted(whisper_module):
    """The array is always mono at the default rate by the time the runner sees
    it, so these labels have to come off the request, not the array."""
    profile = whisper_module.TTWhisperRunner._audio_profile

    request = MagicMock()
    request._source_sample_rate = 44100
    request._source_channels = 2
    assert profile(request) == {"sample_rate": "44100", "channels": "2"}


def test_unobservable_source_profile_is_not_reported_as_real(whisper_module):
    """ffmpeg normalises before the WAV header is read, so the source is None."""
    profile = whisper_module.TTWhisperRunner._audio_profile

    request = MagicMock()
    request._source_sample_rate = None
    request._source_channels = None
    assert profile(request) == {"sample_rate": "unknown", "channels": "unknown"}


def test_disagreeing_batch_reports_mixed_rather_than_one_item(whisper_module):
    profile = whisper_module.TTWhisperRunner._audio_profile

    first, second = MagicMock(), MagicMock()
    first._source_sample_rate, first._source_channels = 44100, 2
    second._source_sample_rate, second._source_channels = 16000, 2
    assert profile([first, second]) == {"sample_rate": "mixed", "channels": "2"}


def test_trace_capture_call_is_labelled_not_dropped(whisper_module):
    """Panels filter the capture call out on the label, so it must be recorded
    under trace_hit="false" rather than dropped."""
    runner = _runner(whisper_module)
    miss_labels = dict(_ENCODER_LABELS, trace_hit="false")
    hit_before = _counter(ENCODER_INPUT, _ENCODER_LABELS)
    miss_before = _counter(ENCODER_INPUT, miss_labels)

    context = runner._audio_stage_context(_batch(5.0))
    runner._record_audio_stage_throughput(
        # The capture call runs the encoder twice, hence the outlier timing.
        (
            "text",
            None,
            None,
            StubPerfMetrics(encoder_s=0.9, encoder_trace_hit=False),
        ),
        context,
    )

    assert _counter(ENCODER_INPUT, miss_labels) == pytest.approx(miss_before + 5.0)
    assert _counter(ENCODER_INPUT, _ENCODER_LABELS) == pytest.approx(hit_before)


def test_zero_duration_stage_is_not_credited_with_audio(whisper_module):
    """generate()'s no-valid-output paths return zeroed timings."""
    runner = _runner(whisper_module)
    feature_before = _counter(FEATURE_INPUT, _FEATURE_LABELS)
    encoder_before = _counter(ENCODER_INPUT, _ENCODER_LABELS)

    context = runner._audio_stage_context(_batch(5.0))
    recorded = runner._record_audio_stage_throughput(
        ("text", None, None, StubPerfMetrics(feature_extract_s=0.0, encoder_s=0.0)),
        context,
    )

    assert recorded is False
    assert _counter(FEATURE_INPUT, _FEATURE_LABELS) == pytest.approx(feature_before)
    assert _counter(ENCODER_INPUT, _ENCODER_LABELS) == pytest.approx(encoder_before)


def test_result_without_perf_metrics_records_nothing(whisper_module):
    runner = _runner(whisper_module)
    before = _counter(FEATURE_INPUT, _FEATURE_LABELS)

    context = runner._audio_stage_context(_batch(5.0))
    assert runner._record_audio_stage_throughput(("text", 0.0, 1.0), context) is False
    assert _counter(FEATURE_INPUT, _FEATURE_LABELS) == pytest.approx(before)


def test_streaming_records_once_across_every_yield(whisper_module):
    """Every yield repeats the timings; per-item recording would multiply one
    batch by its token count."""
    runner = _runner(whisper_module)
    perf = StubPerfMetrics(feature_extract_s=0.01, encoder_s=0.2)
    items = [
        ("a", None, None, perf, False),
        ("b", None, None, perf, False),
        ("final", None, None, perf, True),
    ]
    before = _counter(FEATURE_INPUT, _FEATURE_LABELS)
    count_before = _histogram(FEATURE_DURATION, "count", _FEATURE_LABELS)

    context = runner._audio_stage_context(_batch(6.0))
    drained = list(runner._stream_with_stage_metrics(iter(items), context))

    assert drained == items
    assert _counter(FEATURE_INPUT, _FEATURE_LABELS) == pytest.approx(before + 6.0)
    assert _histogram(FEATURE_DURATION, "count", _FEATURE_LABELS) == count_before + 1


def test_streaming_keeps_looking_past_a_metric_free_first_yield(whisper_module):
    runner = _runner(whisper_module)
    before = _counter(FEATURE_INPUT, _FEATURE_LABELS)

    context = runner._audio_stage_context(_batch(3.0))
    list(
        runner._stream_with_stage_metrics(
            iter(
                [
                    ("a", None, None),
                    (
                        "b",
                        None,
                        None,
                        StubPerfMetrics(feature_extract_s=0.01, encoder_s=0.2),
                        True,
                    ),
                ]
            ),
            context,
        )
    )

    assert _counter(FEATURE_INPUT, _FEATURE_LABELS) == pytest.approx(before + 3.0)


def test_is_final_tracks_the_trailing_bool_not_a_fixed_index(whisper_module):
    """A PerfMetrics at the old index 3 must not read as a final marker."""
    is_final = whisper_module.TTWhisperRunner._is_final_result

    assert is_final(("text", None, None, StubPerfMetrics(), True)) is True
    assert is_final(("text", None, None, StubPerfMetrics(), False)) is False
    assert is_final(("text", None, None, True)) is True
    assert is_final(("text", None, None, False)) is False
    # Non-streaming tuples end in a PerfMetrics or a tensor, never a bool.
    assert is_final(("text", None, None, StubPerfMetrics())) is False
    assert is_final(("text", None, None)) is False
    assert is_final("text") is False


def test_frame_seconds_come_from_the_loaded_extractor(whisper_module):
    resolve = whisper_module.TTWhisperRunner._resolve_encoder_frame_seconds

    assert resolve(
        types.SimpleNamespace(n_samples=480000, sampling_rate=16000)
    ) == pytest.approx(30.0)
    assert resolve(types.SimpleNamespace(chunk_length=20)) == pytest.approx(20.0)
    # A MagicMock extractor must not turn the cap into a Mock arithmetic result.
    assert resolve(MagicMock()) == pytest.approx(
        whisper_module.WHISPER_ENCODER_FRAME_SECONDS
    )
    assert resolve(None) == pytest.approx(whisper_module.WHISPER_ENCODER_FRAME_SECONDS)


def test_perf_metrics_are_requested_when_tt_metal_supports_them(whisper_module):
    assert whisper_module.WHISPER_PERF_METRICS_SUPPORTED is True


def test_absent_perf_metrics_leaves_the_request_off(
    whisper_module_without_perf_metrics,
):
    """Without PerfMetrics, asking for metrics only changes the tuple arity."""
    assert whisper_module_without_perf_metrics.WHISPER_PERF_METRICS_SUPPORTED is False


def test_absent_perf_metrics_still_resolves_the_final_marker(
    whisper_module_without_perf_metrics,
):
    is_final = whisper_module_without_perf_metrics.TTWhisperRunner._is_final_result

    assert is_final(("text", 0.0, 1.0, True)) is True
    assert is_final(("text", 0.0, 1.0)) is False


def test_streaming_path_yields_unchanged_items(whisper_module):
    """The wrapper must be transparent to the streaming loop's tuple handling."""
    runner = _runner(whisper_module)
    context = runner._audio_stage_context(_batch(2.0))

    async def drain():
        return [
            item
            for item in runner._stream_with_stage_metrics(
                iter([("only", None, None, StubPerfMetrics(), True)]), context
            )
        ]

    assert asyncio.run(drain()) == [("only", None, None, StubPerfMetrics(), True)]


def test_dashboard_queries_match_the_exposed_series(whisper_module):
    """A rename that misses the panels silently empties them."""
    import json
    import re
    from pathlib import Path

    registered = set(REGISTRY._names_to_collectors)
    dashboard = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "monitoring/grafana/dashboards/tt_media_server_python.json"
        ).read_text()
    )
    queried = set()
    for panel in dashboard["panels"]:
        for target in panel.get("targets", []):
            queried.update(
                re.findall(r"tt_media_server_audio_\w+", target.get("expr", ""))
            )

    assert queried, "dashboard no longer queries the audio metrics"
    assert queried <= registered, sorted(queried - registered)
