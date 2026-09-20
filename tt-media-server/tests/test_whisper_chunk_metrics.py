# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
"""Per-chunk latency metrics on the streaming ASR path.

conftest mocks out `tt_model_runners.whisper_runner`, so this imports the real
module: conftest already stubs everything it needs bar the tt-metal
`models.demos.audio` tree, stubbed below. Both swaps are undone per test.
"""

import asyncio
import importlib
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest
from prometheus_client import REGISTRY

FIRST_TOKEN = "tt_media_server_audio_chunk_first_token_seconds"
PROCESSING = "tt_media_server_audio_chunk_processing_seconds"
_LABELS = {"model_type": "tt-whisper"}
_METRIC_ATTRS = {
    "audio_chunk_first_token_duration": FIRST_TOKEN,
    "audio_chunk_processing_duration": PROCESSING,
}
_AUDIO_STUBS = (
    "models.demos.audio",
    "models.demos.audio.whisper",
    "models.demos.audio.whisper.tt",
    "models.demos.audio.whisper.tt.ttnn_optimized_functional_whisper",
    "models.demos.audio.whisper.tt.whisper_generator",
)
# A chunk normally streams text before the final marker closes it out.
_DEFAULT_ITEMS = (("text", 0.0, 1.0), ("final", 0.0, 1.0, True))


def _pin_real_collectors():
    """Pin the real histograms onto telemetry.telemetry_client.

    Other test modules leave a Mock there, which would swallow every observe.
    The registry keeps the one reference that survives that swap.
    """
    module = sys.modules.get("telemetry.telemetry_client")
    if module is None:
        # A bare stub, not a re-import: that would re-register every histogram.
        module = types.ModuleType("telemetry.telemetry_client")
        module.TelemetryEvent = MagicMock()
        sys.modules["telemetry.telemetry_client"] = module

    saved = []
    for attr, name in _METRIC_ATTRS.items():
        saved.append((attr, hasattr(module, attr), getattr(module, attr, None)))
        setattr(module, attr, REGISTRY._names_to_collectors[name])
    return module, saved


@pytest.fixture
def whisper_module():
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
        sys.modules[name] = stub

    for name in list(sys.modules):
        if name.startswith("tt_model_runners"):
            sys.modules.pop(name, None)
    try:
        yield importlib.import_module("tt_model_runners.whisper_runner")
    finally:
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


def _metric(metric, field):
    value = REGISTRY.get_sample_value(f"{metric}_{field}", _LABELS)
    return 0 if value is None else value


def _request(segments, sample_rate=16000, seconds=10):
    request = MagicMock()
    request._segments = segments
    request._audio_array = np.zeros(sample_rate * seconds, dtype=np.float32)
    request._duration = float(seconds)
    request._task_id = "task-1"
    request.stream = True
    request.prompt = None
    return request


def _runner(whisper_module, items=_DEFAULT_ITEMS):
    """Carries only what the streaming loop touches."""
    runner = MagicMock()
    runner.settings.default_sample_rate = 16000
    runner.settings.model_runner = "tt-whisper"
    runner.device_id = 0

    async def execute_pipeline(
        _audio, _stream, _params, prompt=None, audio_profile=None
    ):
        async def generator():
            for item in items:
                yield item

        return generator()

    runner._execute_pipeline = execute_pipeline
    runner._create_generation_params = MagicMock(return_value={})
    return whisper_module.TTWhisperRunner._process_segments_streaming.__get__(runner)


def _drain(bound_method, request):
    async def run():
        return [item async for item in bound_method(request)]

    return asyncio.run(run())


def test_both_metrics_record_once_per_chunk(whisper_module):
    """ms/chunk needs one observation per chunk, not per request."""
    segments = [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
        {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_01"},
        {"start": 4.0, "end": 6.0, "speaker": "SPEAKER_00"},
    ]
    first_token_before = _metric(FIRST_TOKEN, "count")
    processing_before = _metric(PROCESSING, "count")

    _drain(_runner(whisper_module), _request(segments))

    assert _metric(FIRST_TOKEN, "count") == first_token_before + len(segments)
    assert _metric(PROCESSING, "count") == processing_before + len(segments)


def test_skipped_empty_chunks_are_not_observed(whisper_module):
    """Zero-length chunks feed no inference, so must not be counted."""
    segments = [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
        {"start": 3.0, "end": 3.0, "speaker": "SPEAKER_01"},  # empty slice
        {"start": 4.0, "end": 6.0, "speaker": "SPEAKER_00"},
    ]
    first_token_before = _metric(FIRST_TOKEN, "count")
    processing_before = _metric(PROCESSING, "count")

    _drain(_runner(whisper_module), _request(segments))

    assert _metric(FIRST_TOKEN, "count") == first_token_before + 2
    assert _metric(PROCESSING, "count") == processing_before + 2


def test_first_item_observed_when_final_marker_arrives_first(whisper_module):
    """A chunk that yields only the final marker still paid the encode cost.

    The observation therefore has to sit ahead of the is_final `break`, not
    behind it.
    """
    first_token_before = _metric(FIRST_TOKEN, "count")

    _drain(
        _runner(whisper_module, items=(("final", 0.0, 1.0, True),)),
        _request([{"start": 0.0, "end": 2.0}]),
    )

    assert _metric(FIRST_TOKEN, "count") == first_token_before + 1


def test_first_item_span_is_contained_in_the_processing_span(whisper_module):
    """Both clocks start together, so first-item can never exceed the total."""
    first_token_before = _metric(FIRST_TOKEN, "sum")
    processing_before = _metric(PROCESSING, "sum")

    _drain(_runner(whisper_module), _request([{"start": 0.0, "end": 2.0}]))

    first_token_delta = _metric(FIRST_TOKEN, "sum") - first_token_before
    processing_delta = _metric(PROCESSING, "sum") - processing_before
    assert first_token_delta <= processing_delta


@pytest.mark.parametrize("metric", [FIRST_TOKEN, PROCESSING])
def test_observed_duration_is_finite_and_non_negative(whisper_module, metric):
    """Catches an unset or misordered perf_counter span."""
    sum_before = _metric(metric, "sum")
    count_before = _metric(metric, "count")

    _drain(_runner(whisper_module), _request([{"start": 0.0, "end": 2.0}]))

    delta = _metric(metric, "sum") - sum_before
    assert _metric(metric, "count") == count_before + 1
    assert 0 <= delta < 5.0
