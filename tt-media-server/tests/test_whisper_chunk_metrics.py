# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
"""Per-chunk preparation metric on the streaming ASR path.

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

METRIC = "tt_media_server_audio_chunk_preparation_seconds"
_METRIC_ATTR = "audio_chunk_preparation_duration"
_AUDIO_STUBS = (
    "models.demos.audio",
    "models.demos.audio.whisper",
    "models.demos.audio.whisper.tt",
    "models.demos.audio.whisper.tt.ttnn_optimized_functional_whisper",
    "models.demos.audio.whisper.tt.whisper_generator",
)


def _pin_real_collector():
    """Pin the real histogram onto telemetry.telemetry_client.

    test_device_worker* leave a Mock there, which would swallow every observe.
    The registry keeps the one reference that survives that swap.
    """
    collector = REGISTRY._names_to_collectors[METRIC]
    module = sys.modules.get("telemetry.telemetry_client")
    if module is None:
        # A bare stub, not a re-import: that would re-register every histogram.
        module = types.ModuleType("telemetry.telemetry_client")
        module.TelemetryEvent = MagicMock()
        sys.modules["telemetry.telemetry_client"] = module
    had_attr = hasattr(module, _METRIC_ATTR)
    previous = getattr(module, _METRIC_ATTR, None)
    setattr(module, _METRIC_ATTR, collector)
    return module, had_attr, previous


@pytest.fixture
def whisper_module():
    saved = {name: sys.modules.get(name) for name in _AUDIO_STUBS}
    saved["tt_model_runners.whisper_runner"] = sys.modules.get(
        "tt_model_runners.whisper_runner"
    )
    telemetry_module, had_attr, previous_metric = _pin_real_collector()

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

    sys.modules.pop("tt_model_runners.whisper_runner", None)
    try:
        yield importlib.import_module("tt_model_runners.whisper_runner")
    finally:
        if had_attr:
            setattr(telemetry_module, _METRIC_ATTR, previous_metric)
        else:
            delattr(telemetry_module, _METRIC_ATTR)
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _metric(field, labels={"model_type": "tt-whisper"}):
    value = REGISTRY.get_sample_value(f"{METRIC}_{field}", labels)
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


def _runner(whisper_module):
    """Carries only what the streaming loop touches."""
    runner = MagicMock()
    runner.settings.default_sample_rate = 16000
    runner.settings.model_runner = "tt-whisper"
    runner.device_id = 0

    async def execute_pipeline(_audio, _stream, _params, prompt=None):
        async def generator():
            yield ("text", 0.0, 1.0)
            yield ("final", 0.0, 1.0, True)

        return generator()

    runner._execute_pipeline = execute_pipeline
    runner._create_generation_params = MagicMock(return_value={})
    return whisper_module.TTWhisperRunner._process_segments_streaming.__get__(runner)


def _drain(bound_method, request):
    async def run():
        return [item async for item in bound_method(request)]

    return asyncio.run(run())


def test_records_one_observation_per_prepared_chunk(whisper_module):
    """ms/chunk needs one observation per chunk, not per request."""
    segments = [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
        {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_01"},
        {"start": 4.0, "end": 6.0, "speaker": "SPEAKER_00"},
    ]
    before = _metric("count")

    _drain(_runner(whisper_module), _request(segments))

    assert _metric("count") == before + len(segments)


def test_skipped_empty_chunks_are_not_observed(whisper_module):
    """Zero-length chunks feed no inference, so must not be counted."""
    segments = [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
        {"start": 3.0, "end": 3.0, "speaker": "SPEAKER_01"},  # empty slice
        {"start": 4.0, "end": 6.0, "speaker": "SPEAKER_00"},
    ]
    before = _metric("count")

    _drain(_runner(whisper_module), _request(segments))

    assert _metric("count") == before + 2


def test_observed_duration_is_finite_and_non_negative(whisper_module):
    """Catches an unset or misordered perf_counter span."""
    sum_before = _metric("sum")
    count_before = _metric("count")

    _drain(_runner(whisper_module), _request([{"start": 0.0, "end": 2.0}]))

    delta = _metric("sum") - sum_before
    assert _metric("count") == count_before + 1
    assert 0 <= delta < 5.0
