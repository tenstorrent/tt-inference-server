# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from unittest.mock import Mock

import numpy as np
import pytest
from domain.audio_processing_request import AudioProcessingRequest
from model_services.audio_service import AudioService


@pytest.fixture
def service():
    # Bare instance: create_segment_request only needs a logger.
    instance = AudioService.__new__(AudioService)
    instance.logger = Mock()
    return instance


def _request(duration_seconds: float, payload_chars: int) -> AudioProcessingRequest:
    sample_rate = 16000
    request = AudioProcessingRequest(file="A" * payload_chars)
    request._audio_array = np.arange(
        int(duration_seconds * sample_rate), dtype=np.float32
    )
    request._duration = duration_seconds
    return request


class TestCreateSegmentRequest:
    def test_segment_does_not_carry_encoded_clip(self, service, monkeypatch):
        """The base64 clip must not be copied onto each segment.

        model_dump() used to deep-copy the whole payload once per segment (~13MB
        x 32 segments on a 320s clip) before it was cleared, which serialised
        fan-out and made it cost duration x segment-count.
        """
        from config.settings import settings

        monkeypatch.setattr(settings, "default_sample_rate", 16000, raising=False)
        original = _request(duration_seconds=60.0, payload_chars=100_000)

        segment = service.create_segment_request(
            original, {"start": 10.0, "end": 20.0}, 1
        )

        assert segment.file is None
        # The parent keeps its payload; only the copy is avoided.
        assert len(original.file) == 100_000

    def test_segment_slices_audio_and_disables_reprocessing(self, service, monkeypatch):
        from config.settings import settings

        monkeypatch.setattr(settings, "default_sample_rate", 16000, raising=False)
        original = _request(duration_seconds=60.0, payload_chars=1000)

        segment = service.create_segment_request(
            original, {"start": 10.0, "end": 20.0}, 1
        )

        assert segment.is_preprocessing_enabled is False
        assert segment._duration == pytest.approx(10.0)
        assert len(segment._audio_array) == 10 * 16000
        # Slice must start where the segment starts, not at the clip head.
        assert segment._audio_array[0] == pytest.approx(10 * 16000)

    def test_non_file_fields_are_preserved(self, service, monkeypatch):
        from config.settings import settings

        monkeypatch.setattr(settings, "default_sample_rate", 16000, raising=False)
        original = _request(duration_seconds=60.0, payload_chars=10)
        original.response_format = "json"
        original.chunk_duration_seconds = 10
        original.prompt = "keep me"

        segment = service.create_segment_request(
            original, {"start": 0.0, "end": 10.0}, 0
        )

        assert segment.response_format == "json"
        assert segment.chunk_duration_seconds == 10
        assert segment.prompt == "keep me"
