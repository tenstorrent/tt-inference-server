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
    char_count,
    record_stt_request,
    record_tts_request,
)


def sample(name, **labels):
    return REGISTRY.get_sample_value(name, labels)


def stt_labels(model_type, **overrides):
    labels = dict(
        model_type=model_type,
        task="transcription",
        streaming="false",
        status=STATUS_SUCCESS,
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


class TestRecordSttRequest:
    def test_success_records_usage_and_rtf(self):
        model = "stt-success"
        record_stt_request(
            model_type=model,
            task="transcription",
            streaming=False,
            status=STATUS_SUCCESS,
            duration_seconds=2.0,
            audio_seconds=10.0,
            characters=120,
        )
        labels = stt_labels(model)
        assert sample("tt_media_server_audio_stt_requests_total", **labels) == 1
        assert (
            sample("tt_media_server_audio_stt_request_duration_seconds_sum", **labels)
            == 2.0
        )
        usage = dict(model_type=model, task="transcription")
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
        rtf = dict(model_type=model, task="transcription", streaming="false")
        assert sample("tt_media_server_audio_stt_realtime_factor_count", **rtf) == 1
        assert sample(
            "tt_media_server_audio_stt_realtime_factor_sum", **rtf
        ) == pytest.approx(0.2)

    def test_error_skips_usage(self):
        model = "stt-error"
        record_stt_request(
            model_type=model,
            task="transcription",
            streaming=True,
            status=STATUS_ERROR,
            duration_seconds=1.0,
            audio_seconds=30.0,
            characters=500,
        )
        labels = stt_labels(model, streaming="true", status=STATUS_ERROR)
        assert sample("tt_media_server_audio_stt_requests_total", **labels) == 1
        usage = dict(model_type=model, task="transcription")
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
                streaming=False,
                status=STATUS_SUCCESS,
                duration_seconds=1.0,
                audio_seconds=audio_seconds,
                characters=None,
            )
        assert (
            sample("tt_media_server_audio_stt_requests_total", **stt_labels(model)) == 3
        )
        usage = dict(model_type=model, task="transcription")
        assert (
            sample("tt_media_server_audio_stt_input_audio_seconds_total", **usage)
            is None
        )
        rtf = dict(model_type=model, task="transcription", streaming="false")
        assert sample("tt_media_server_audio_stt_realtime_factor_count", **rtf) is None


class TestRecordTtsRequest:
    def test_success_records_usage_and_rtf(self):
        model = "tts-success"
        record_tts_request(
            model_type=model,
            response_format="wav",
            status=STATUS_SUCCESS,
            duration_seconds=1.5,
            characters=200,
            audio_seconds=6.0,
        )
        labels = dict(model_type=model, response_format="wav", status=STATUS_SUCCESS)
        assert sample("tt_media_server_audio_tts_requests_total", **labels) == 1
        assert (
            sample("tt_media_server_audio_tts_request_duration_seconds_sum", **labels)
            == 1.5
        )
        usage = dict(model_type=model)
        assert (
            sample("tt_media_server_audio_tts_input_characters_total", **usage) == 200
        )
        assert (
            sample("tt_media_server_audio_tts_output_audio_seconds_total", **usage)
            == 6.0
        )
        assert sample(
            "tt_media_server_audio_tts_realtime_factor_sum", **usage
        ) == pytest.approx(0.25)

    def test_error_skips_usage(self):
        model = "tts-error"
        record_tts_request(
            model_type=model,
            response_format="mp3",
            status=STATUS_ERROR,
            duration_seconds=0.5,
            characters=100,
            audio_seconds=2.0,
        )
        labels = dict(model_type=model, response_format="mp3", status=STATUS_ERROR)
        assert sample("tt_media_server_audio_tts_requests_total", **labels) == 1
        usage = dict(model_type=model)
        assert (
            sample("tt_media_server_audio_tts_input_characters_total", **usage) is None
        )
        assert (
            sample("tt_media_server_audio_tts_output_audio_seconds_total", **usage)
            is None
        )

    def test_non_str_response_format_becomes_unknown(self):
        model = "tts-bad-format"
        record_tts_request(
            model_type=model,
            response_format=MagicMock(),
            status=STATUS_SUCCESS,
            duration_seconds=1.0,
        )
        labels = dict(
            model_type=model, response_format="unknown", status=STATUS_SUCCESS
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
        usage = dict(model_type=model_runner_label, task="transcription")
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
        usage = dict(model_type=model_runner_label, task="transcription")
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
        usage = dict(model_type=model_runner_label, task="transcription")
        assert (
            sample("tt_media_server_audio_stt_input_audio_seconds_total", **usage)
            == 5.0
        )
        # The final transcript supersedes the accumulated chunk count.
        assert sample(
            "tt_media_server_audio_stt_output_characters_total", **usage
        ) == len("one two")

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
        usage = dict(model_type=model_runner_label, task="transcription")
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
        usage = dict(model_type=model_runner_label, task="transcription")
        assert (
            sample("tt_media_server_audio_stt_input_audio_seconds_total", **usage)
            is None
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
        service = MagicMock()
        service.process_request = AsyncMock(return_value=result)

        tts_request = MagicMock()
        tts_request.response_format = "wav"
        tts_request.text = "say this"

        response = await handle_tts_request(tts_request, service)

        assert response.body == b"riff"
        labels = dict(
            model_type=model_runner_label,
            response_format="wav",
            status=STATUS_SUCCESS,
        )
        assert sample("tt_media_server_audio_tts_requests_total", **labels) == 1
        usage = dict(model_type=model_runner_label)
        assert sample(
            "tt_media_server_audio_tts_input_characters_total", **usage
        ) == len("say this")
        assert (
            sample("tt_media_server_audio_tts_output_audio_seconds_total", **usage)
            == 4.0
        )

    @pytest.mark.asyncio
    async def test_error(self, model_runner_label):
        service = MagicMock()
        service.process_request = AsyncMock(side_effect=RuntimeError("boom"))

        tts_request = MagicMock()
        tts_request.response_format = "wav"
        tts_request.text = "say this"

        with pytest.raises(HTTPException) as exc_info:
            await handle_tts_request(tts_request, service)

        assert exc_info.value.status_code == 500
        labels = dict(
            model_type=model_runner_label,
            response_format="wav",
            status=STATUS_ERROR,
        )
        assert sample("tt_media_server_audio_tts_requests_total", **labels) == 1
        usage = dict(model_type=model_runner_label)
        assert (
            sample("tt_media_server_audio_tts_input_characters_total", **usage) is None
        )
