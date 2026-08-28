# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Tests for the request-level audio (STT/TTS) metrics.

Prometheus collectors are process-global and cumulative, so each test uses its
own ``model_type`` label value.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from config.settings import settings
from domain.audio_text_response import AudioStreamChunk, AudioTextResponse
from fastapi import HTTPException
from open_ai_api.audio import handle_audio_request
from open_ai_api.text_to_speech import handle_tts_request
from prometheus_client import REGISTRY
from telemetry.audio_metrics import (
    STATUS_ERROR,
    STATUS_SUCCESS,
    VOICE_CUSTOM,
    VOICE_DEFAULT,
    SttStreamProgress,
    TtsChunkProgress,
    char_count,
    confidence_from_generator_output,
    record_stt_confidence,
    record_stt_request,
    record_tts_request,
    transcript_compression_ratio,
    tts_voice_label,
)

# Handler tests read the language off the real settings object.
LANGUAGE = settings.audio_language or "unknown"


def sample(name, **labels):
    return REGISTRY.get_sample_value(name, labels)


def stt_labels(model_type, **overrides):
    labels = dict(
        model_type=model_type,
        task=("translation" if settings.audio_task.lower() == "translate" else "transcription"),
        language=LANGUAGE,
        streaming="false",
        status=STATUS_SUCCESS,
    )
    labels.update(overrides)
    return labels


def stt_usage(model_type, **overrides):
    labels = dict(
        model_type=model_type,
        task="transcription",
        language=LANGUAGE,
        streaming="false",
    )
    labels.update(overrides)
    return labels


class TestCharCount:
    def test_str(self):
        assert char_count("hello") == 5

    def test_empty_str(self):
        assert char_count("") == 0

    def test_none(self):
        assert char_count(None) is None

    def test_mock(self):
        assert char_count(MagicMock()) is None


class TestTtsVoiceLabel:
    def test_result_speaker_id_wins(self):
        request = MagicMock()
        request.speaker_id = "asked-for"
        result = MagicMock()
        result.speaker_id = "actually-used"
        assert tts_voice_label(request, result) == "actually-used"

    def test_request_speaker_id_fallback(self):
        request = MagicMock()
        request.speaker_id = "alice"
        request.speaker_embedding = None
        result = MagicMock()
        result.speaker_id = None
        assert tts_voice_label(request, result) == "alice"

    def test_custom_embedding(self):
        request = MagicMock()
        request.speaker_id = None
        request.speaker_embedding = b"embedding-bytes"
        assert tts_voice_label(request, None) == VOICE_CUSTOM

    def test_default(self):
        request = MagicMock()
        request.speaker_id = None
        request.speaker_embedding = None
        assert tts_voice_label(request, None) == VOICE_DEFAULT

    def test_long_id_truncated(self):
        request = MagicMock()
        request.speaker_id = "v" * 200
        request.speaker_embedding = None
        assert tts_voice_label(request, None) == "v" * 64


class TestRecordSttRequest:
    def test_success_records_usage_and_rtf(self):
        model = "stt-success"
        record_stt_request(
            model_type=model,
            task="transcription",
            language="English",
            streaming=False,
            status=STATUS_SUCCESS,
            duration_seconds=2.0,
            audio_seconds=10.0,
            characters=120,
        )
        labels = stt_labels(model, language="English")
        assert sample("tt_media_server_audio_stt_requests_total", **labels) == 1
        assert (
            sample("tt_media_server_audio_stt_request_duration_seconds_sum", **labels)
            == 2.0
        )
        usage = stt_usage(model, language="English")
        assert (
            sample("tt_media_server_audio_stt_input_audio_seconds_total", **usage)
            == 10.0
        )
        assert (
            sample(
                "tt_media_server_audio_stt_input_audio_duration_seconds_count",
                **usage,
            )
            == 1
        )
        assert (
            sample("tt_media_server_audio_stt_output_characters_total", **usage) == 120
        )
        assert sample("tt_media_server_audio_stt_realtime_factor_count", **usage) == 1
        assert sample(
            "tt_media_server_audio_stt_realtime_factor_sum", **usage
        ) == pytest.approx(0.2)

    def test_error_skips_usage(self):
        model = "stt-error"
        record_stt_request(
            model_type=model,
            task="transcription",
            language="English",
            streaming=True,
            status=STATUS_ERROR,
            duration_seconds=1.0,
            audio_seconds=30.0,
            characters=500,
        )
        labels = stt_labels(
            model, language="English", streaming="true", status=STATUS_ERROR
        )
        assert sample("tt_media_server_audio_stt_requests_total", **labels) == 1
        usage = stt_usage(model, language="English", streaming="true")
        assert (
            sample("tt_media_server_audio_stt_input_audio_seconds_total", **usage)
            is None
        )
        assert (
            sample("tt_media_server_audio_stt_output_characters_total", **usage) is None
        )

    def test_unknown_audio_seconds_still_counts_request(self):
        model = "stt-no-audio"
        for audio_seconds in (None, 0.0, object()):
            record_stt_request(
                model_type=model,
                task="transcription",
                language="English",
                streaming=False,
                status=STATUS_SUCCESS,
                duration_seconds=1.0,
                audio_seconds=audio_seconds,
                characters=None,
            )
        assert (
            sample(
                "tt_media_server_audio_stt_requests_total",
                **stt_labels(model, language="English"),
            )
            == 3
        )
        usage = stt_usage(model, language="English")
        assert (
            sample("tt_media_server_audio_stt_input_audio_seconds_total", **usage)
            is None
        )
        assert (
            sample("tt_media_server_audio_stt_realtime_factor_count", **usage) is None
        )

    def test_non_str_language_becomes_unknown(self):
        model = "stt-bad-language"
        record_stt_request(
            model_type=model,
            task="transcription",
            language=None,
            streaming=False,
            status=STATUS_SUCCESS,
            duration_seconds=1.0,
        )
        labels = stt_labels(model, language="unknown")
        assert sample("tt_media_server_audio_stt_requests_total", **labels) == 1


class TestSttStreamProgress:
    def test_first_update_then_intervals(self):
        model = "stt-stream-progress"
        progress = SttStreamProgress(
            model_type=model, task="transcription", language="English"
        )
        progress.on_update()
        progress.on_update()
        progress.on_update()
        labels = dict(model_type=model, task="transcription", language="English")
        assert (
            sample("tt_media_server_audio_stt_first_partial_seconds_count", **labels)
            == 1
        )
        assert (
            sample("tt_media_server_audio_stt_partial_interval_seconds_count", **labels)
            == 2
        )

    def test_no_updates_records_nothing(self):
        model = "stt-stream-quiet"
        SttStreamProgress(model_type=model, task="transcription", language="English")
        labels = dict(model_type=model, task="transcription", language="English")
        assert (
            sample("tt_media_server_audio_stt_first_partial_seconds_count", **labels)
            is None
        )

    def test_final_records_finalization_not_cadence(self):
        model = "stt-stream-final"
        progress = SttStreamProgress(
            model_type=model, task="transcription", language="English"
        )
        progress.on_update()
        progress.on_final()
        labels = dict(model_type=model, task="transcription", language="English")
        assert (
            sample("tt_media_server_audio_stt_finalization_seconds_count", **labels)
            == 1
        )
        assert (
            sample("tt_media_server_audio_stt_partial_interval_seconds_count", **labels)
            is None
        )

    def test_final_without_partials_measures_from_start(self):
        model = "stt-stream-final-only"
        progress = SttStreamProgress(
            model_type=model, task="transcription", language="English"
        )
        progress.on_final()
        labels = dict(model_type=model, task="transcription", language="English")
        assert (
            sample("tt_media_server_audio_stt_finalization_seconds_count", **labels)
            == 1
        )
        assert (
            sample("tt_media_server_audio_stt_first_partial_seconds_count", **labels)
            is None
        )


class TestTtsChunkProgress:
    def test_first_chunk_and_cadence(self):
        model = "tts-chunk-progress"
        progress = TtsChunkProgress(model_type=model)
        progress.on_chunk()
        progress.on_chunk()
        progress.on_chunk()
        labels = dict(model_type=model)
        assert (
            sample("tt_media_server_audio_tts_first_chunk_seconds_count", **labels) == 1
        )
        assert (
            sample("tt_media_server_audio_tts_chunk_generation_seconds_count", **labels)
            == 3
        )

    def test_no_chunks_records_nothing(self):
        model = "tts-chunk-quiet"
        TtsChunkProgress(model_type=model)
        assert (
            sample(
                "tt_media_server_audio_tts_first_chunk_seconds_count", model_type=model
            )
            is None
        )


class FakeTensor:
    """Stand-in for a 0-d torch tensor: float() works, iteration does not."""

    def __init__(self, value):
        self._value = value

    def __float__(self):
        return self._value


class TestConfidenceHelpers:
    def test_extracts_floats_by_position(self):
        item = ("transcript", -0.5, 0.1, True)
        assert confidence_from_generator_output(item) == (-0.5, 0.1)

    def test_extracts_tensor_scalars_and_sequences(self):
        item = ("t", FakeTensor(-0.75), [FakeTensor(0.2), FakeTensor(0.4)], True)
        avg_logprob, no_speech = confidence_from_generator_output(item)
        assert avg_logprob == -0.75
        assert no_speech == pytest.approx(0.3)

    def test_rejects_non_tuples_and_short_tuples(self):
        assert confidence_from_generator_output("text") == (None, None)
        assert confidence_from_generator_output(("text", -0.5)) == (None, None)

    def test_compression_ratio_repetitive_text_is_higher(self):
        varied = transcript_compression_ratio("the quick brown fox jumps over it")
        looped = transcript_compression_ratio("the the the the the the the the the")
        assert varied is not None and looped is not None
        assert looped > varied

    def test_compression_ratio_empty_or_non_str(self):
        assert transcript_compression_ratio("") is None
        assert transcript_compression_ratio(None) is None


class TestRecordSttConfidence:
    def test_records_all_signals(self):
        model = "stt-confidence"
        record_stt_confidence(
            model_type=model,
            language="English",
            avg_logprob=-0.4,
            no_speech_prob=0.05,
            compression_ratio=1.6,
        )
        labels = dict(model_type=model, language="English")
        # No _sum assertion for avg_logprob: prometheus_client omits _sum for
        # histograms holding negative observations (OpenMetrics rule).
        assert sample("tt_media_server_audio_stt_avg_logprob_count", **labels) == 1
        assert (
            sample("tt_media_server_audio_stt_avg_logprob_bucket", le="-0.4", **labels)
            == 1
        )
        assert (
            sample("tt_media_server_audio_stt_no_speech_probability_sum", **labels)
            == 0.05
        )
        assert (
            sample("tt_media_server_audio_stt_compression_ratio_sum", **labels) == 1.6
        )

    def test_partial_signals_record_independently(self):
        model = "stt-confidence-partial"
        record_stt_confidence(
            model_type=model,
            language="English",
            avg_logprob=-1.2,
            no_speech_prob=None,
            compression_ratio=None,
        )
        labels = dict(model_type=model, language="English")
        assert sample("tt_media_server_audio_stt_avg_logprob_count", **labels) == 1
        assert (
            sample("tt_media_server_audio_stt_no_speech_probability_count", **labels)
            is None
        )

    def test_garbage_never_raises(self):
        record_stt_confidence(
            model_type="stt-confidence-garbage",
            language=None,
            avg_logprob=object(),
            no_speech_prob="not-a-number",
            compression_ratio=-2.0,
        )


class TestRecordTtsRequest:
    def test_success_records_usage_and_rtf(self):
        model = "tts-success"
        record_tts_request(
            model_type=model,
            response_format="wav",
            status=STATUS_SUCCESS,
            duration_seconds=1.5,
            voice="alice",
            characters=200,
            audio_seconds=6.0,
        )
        labels = dict(
            model_type=model,
            response_format="wav",
            voice="alice",
            status=STATUS_SUCCESS,
        )
        assert sample("tt_media_server_audio_tts_requests_total", **labels) == 1
        assert (
            sample(
                "tt_media_server_audio_tts_request_duration_seconds_sum",
                model_type=model,
                response_format="wav",
                status=STATUS_SUCCESS,
            )
            == 1.5
        )
        assert (
            sample("tt_media_server_audio_tts_input_characters_total", model_type=model)
            == 200
        )
        voiced = dict(model_type=model, voice="alice")
        assert (
            sample("tt_media_server_audio_tts_output_audio_seconds_total", **voiced)
            == 6.0
        )
        assert sample(
            "tt_media_server_audio_tts_realtime_factor_sum", **voiced
        ) == pytest.approx(0.25)

    def test_error_skips_usage(self):
        model = "tts-error"
        record_tts_request(
            model_type=model,
            response_format="mp3",
            status=STATUS_ERROR,
            duration_seconds=0.5,
            voice="bob",
            characters=100,
            audio_seconds=2.0,
        )
        labels = dict(
            model_type=model,
            response_format="mp3",
            voice="bob",
            status=STATUS_ERROR,
        )
        assert sample("tt_media_server_audio_tts_requests_total", **labels) == 1
        assert (
            sample("tt_media_server_audio_tts_input_characters_total", model_type=model)
            is None
        )
        assert (
            sample(
                "tt_media_server_audio_tts_output_audio_seconds_total",
                model_type=model,
                voice="bob",
            )
            is None
        )

    def test_non_str_response_format_and_voice_fall_back(self):
        model = "tts-bad-format"
        record_tts_request(
            model_type=model,
            response_format=MagicMock(),
            status=STATUS_SUCCESS,
            duration_seconds=1.0,
            voice=MagicMock(),
        )
        labels = dict(
            model_type=model,
            response_format="unknown",
            voice=VOICE_DEFAULT,
            status=STATUS_SUCCESS,
        )
        assert sample("tt_media_server_audio_tts_requests_total", **labels) == 1


@pytest.fixture
def model_runner_label(monkeypatch, request):
    """Point settings.model_runner at a per-test unique label value."""
    label = f"handler-{request.node.name}"
    monkeypatch.setattr(settings, "model_runner", label)
    return label


def make_stt_request(stream=False, response_format="verbose_json"):
    audio_request = MagicMock()
    audio_request.stream = stream
    audio_request.response_format = response_format
    audio_request._duration = 8.0
    return audio_request


def make_tts_request(text="say this", speaker_id=None, speaker_embedding=None):
    tts_request = MagicMock()
    tts_request.response_format = "wav"
    tts_request.text = text
    tts_request.speaker_id = speaker_id
    tts_request.speaker_embedding = speaker_embedding
    return tts_request


class TestHandleAudioRequestMetrics:
    @pytest.mark.asyncio
    async def test_non_streaming_success(self, model_runner_label):
        result = AudioTextResponse(text="hello world", duration=12.0)
        service = MagicMock()
        service.process_request = AsyncMock(return_value=result)

        response = await handle_audio_request(make_stt_request(), service)

        assert response == result.to_dict()
        labels = stt_labels(model_runner_label)
        assert sample("tt_media_server_audio_stt_requests_total", **labels) == 1
        usage = stt_usage(model_runner_label)
        assert (
            sample("tt_media_server_audio_stt_input_audio_seconds_total", **usage)
            == 12.0
        )
        assert sample(
            "tt_media_server_audio_stt_output_characters_total", **usage
        ) == len("hello world")

    @pytest.mark.asyncio
    async def test_non_streaming_error(self, model_runner_label):
        service = MagicMock()
        service.process_request = AsyncMock(side_effect=RuntimeError("boom"))

        with pytest.raises(HTTPException) as exc_info:
            await handle_audio_request(make_stt_request(), service)

        assert exc_info.value.status_code == 500
        labels = stt_labels(model_runner_label, status=STATUS_ERROR)
        assert sample("tt_media_server_audio_stt_requests_total", **labels) == 1
        usage = stt_usage(model_runner_label)
        assert (
            sample("tt_media_server_audio_stt_input_audio_seconds_total", **usage)
            is None
        )

    @pytest.mark.asyncio
    async def test_streaming_success_uses_final_result(self, model_runner_label):
        final = AudioTextResponse(text="one two", duration=5.0)

        async def stream(_request):
            yield AudioStreamChunk(text="one ", chunk_id=1)
            yield AudioStreamChunk(text="two", chunk_id=2)
            yield final

        service = MagicMock()
        service.scheduler.check_is_model_ready = MagicMock()
        service.process_streaming_request = stream

        response = await handle_audio_request(make_stt_request(stream=True), service)
        chunks = [chunk async for chunk in response.body_iterator]

        assert json.loads(chunks[-1]) == final.to_dict()
        labels = stt_labels(model_runner_label, streaming="true")
        assert sample("tt_media_server_audio_stt_requests_total", **labels) == 1
        usage = stt_usage(model_runner_label, streaming="true")
        assert (
            sample("tt_media_server_audio_stt_input_audio_seconds_total", **usage)
            == 5.0
        )
        # The final transcript supersedes the accumulated chunk count.
        assert sample(
            "tt_media_server_audio_stt_output_characters_total", **usage
        ) == len("one two")
        # Two partials then the final: one first-partial, one interval, and
        # the final feeds finalization instead of cadence.
        stream_labels = dict(
            model_type=model_runner_label, task="transcription", language=LANGUAGE
        )
        assert (
            sample(
                "tt_media_server_audio_stt_first_partial_seconds_count",
                **stream_labels,
            )
            == 1
        )
        assert (
            sample(
                "tt_media_server_audio_stt_partial_interval_seconds_count",
                **stream_labels,
            )
            == 1
        )
        assert (
            sample(
                "tt_media_server_audio_stt_finalization_seconds_count",
                **stream_labels,
            )
            == 1
        )

    @pytest.mark.asyncio
    async def test_streaming_text_format_falls_back_to_request_duration(
        self, model_runner_label
    ):
        async def stream(_request):
            yield AudioStreamChunk(text="one ", chunk_id=1)
            yield AudioStreamChunk(text="two", chunk_id=2)

        service = MagicMock()
        service.scheduler.check_is_model_ready = MagicMock()
        service.process_streaming_request = stream

        audio_request = make_stt_request(stream=True, response_format="text")
        response = await handle_audio_request(audio_request, service)
        chunks = [chunk async for chunk in response.body_iterator]

        assert chunks == ["one \n", "two\n"]
        usage = stt_usage(model_runner_label, streaming="true")
        assert (
            sample("tt_media_server_audio_stt_input_audio_seconds_total", **usage)
            == 8.0
        )
        assert sample(
            "tt_media_server_audio_stt_output_characters_total", **usage
        ) == len("one two")

    @pytest.mark.asyncio
    async def test_streaming_mid_stream_error_records_error(self, model_runner_label):
        async def stream(_request):
            yield AudioStreamChunk(text="one ", chunk_id=1)
            raise RuntimeError("device died")

        service = MagicMock()
        service.scheduler.check_is_model_ready = MagicMock()
        service.process_streaming_request = stream

        response = await handle_audio_request(make_stt_request(stream=True), service)
        with pytest.raises(RuntimeError):
            async for _ in response.body_iterator:
                pass

        labels = stt_labels(model_runner_label, streaming="true", status=STATUS_ERROR)
        assert sample("tt_media_server_audio_stt_requests_total", **labels) == 1
        usage = stt_usage(model_runner_label, streaming="true")
        assert (
            sample("tt_media_server_audio_stt_input_audio_seconds_total", **usage)
            is None
        )
        # The one chunk that did reach the client still counts as first partial.
        stream_labels = dict(
            model_type=model_runner_label, task="transcription", language=LANGUAGE
        )
        assert (
            sample(
                "tt_media_server_audio_stt_first_partial_seconds_count",
                **stream_labels,
            )
            == 1
        )

    @pytest.mark.asyncio
    async def test_streaming_model_not_ready_records_error_and_405(
        self, model_runner_label
    ):
        service = MagicMock()
        service.scheduler.check_is_model_ready = MagicMock(
            side_effect=RuntimeError("not ready")
        )

        with pytest.raises(HTTPException) as exc_info:
            await handle_audio_request(make_stt_request(stream=True), service)

        assert exc_info.value.status_code == 405
        labels = stt_labels(model_runner_label, streaming="true", status=STATUS_ERROR)
        assert sample("tt_media_server_audio_stt_requests_total", **labels) == 1


class TestHandleTtsRequestMetrics:
    @pytest.mark.asyncio
    async def test_success(self, model_runner_label):
        result = MagicMock()
        result.output_bytes = b"riff"
        result.format = "wav"
        result.duration = 4.0
        result.speaker_id = "alice"
        service = MagicMock()
        service.process_request = AsyncMock(return_value=result)

        response = await handle_tts_request(make_tts_request(), service)

        assert response.body == b"riff"
        labels = dict(
            model_type=model_runner_label,
            response_format="wav",
            voice="alice",
            status=STATUS_SUCCESS,
        )
        assert sample("tt_media_server_audio_tts_requests_total", **labels) == 1
        assert sample(
            "tt_media_server_audio_tts_input_characters_total",
            model_type=model_runner_label,
        ) == len("say this")
        assert (
            sample(
                "tt_media_server_audio_tts_output_audio_seconds_total",
                model_type=model_runner_label,
                voice="alice",
            )
            == 4.0
        )

    @pytest.mark.asyncio
    async def test_error(self, model_runner_label):
        service = MagicMock()
        service.process_request = AsyncMock(side_effect=RuntimeError("boom"))

        with pytest.raises(HTTPException) as exc_info:
            await handle_tts_request(make_tts_request(), service)

        assert exc_info.value.status_code == 500
        labels = dict(
            model_type=model_runner_label,
            response_format="wav",
            voice=VOICE_DEFAULT,
            status=STATUS_ERROR,
        )
        assert sample("tt_media_server_audio_tts_requests_total", **labels) == 1
        assert (
            sample(
                "tt_media_server_audio_tts_input_characters_total",
                model_type=model_runner_label,
            )
            is None
        )
