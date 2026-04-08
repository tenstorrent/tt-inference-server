# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import io
import wave

import numpy as np
import pytest
from unittest.mock import patch

# Avoid loading silero_vad/torchaudio (needs libtorchaudio.so in CI). audio_manager
# only imports them when settings.model_service == AUDIO; patch so that branch is skipped.
import config.settings

with patch.object(config.settings.settings, "model_service", None):
    from utils.audio_manager import AudioManager


class DummySettings:
    allow_audio_preprocessing = False
    default_sample_rate = 16000
    max_audio_size_bytes = 1000000
    max_audio_duration_seconds = 60
    max_audio_duration_with_preprocessing_seconds = 120
    audio_chunk_duration_seconds = 10
    model_service = "AUDIO"
    preprocessing_model_weights_path = None


@patch("utils.audio_manager.settings", new=DummySettings())
def generate_dummy_wav_bytes():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00" * 320)  # 160 samples
    return buffer.getvalue()


def test_to_audio_array_base64():
    manager = AudioManager()
    wav_bytes = generate_dummy_wav_bytes()
    import base64

    b64 = base64.b64encode(wav_bytes).decode()
    arr, duration = manager.to_audio_array(b64, False)
    assert isinstance(arr, np.ndarray)
    assert duration > 0


@patch("utils.audio_manager.settings", new=DummySettings())
def test_to_audio_array_bytes():
    manager = AudioManager()
    wav_bytes = generate_dummy_wav_bytes()
    arr, duration = manager.to_audio_array(wav_bytes, False)
    assert isinstance(arr, np.ndarray)
    assert duration > 0


@patch("utils.audio_manager.settings", new=DummySettings())
def test_to_audio_array_invalid_type():
    manager = AudioManager()
    with pytest.raises(ValueError):
        manager.to_audio_array(123, False)


@patch("utils.audio_manager.settings", new=DummySettings())
def test_validate_file_size():
    manager = AudioManager()
    manager._validate_file_size(b"\x00" * 100)
    with pytest.raises(ValueError):
        manager._validate_file_size(b"\x00" * (DummySettings.max_audio_size_bytes + 1))


@patch("utils.audio_manager.settings", new=DummySettings())
def test_validate_and_truncate_duration():
    manager = AudioManager()
    arr = np.zeros(DummySettings.default_sample_rate * 70, dtype=np.float32)
    truncated, duration = manager._validate_and_truncate_duration(arr, False)
    assert duration == DummySettings.max_audio_duration_seconds
    assert (
        len(truncated)
        == DummySettings.default_sample_rate * DummySettings.max_audio_duration_seconds
    )


@patch("utils.audio_manager.settings", new=DummySettings())
def test_normalize_speaker_ids():
    manager = AudioManager()
    segments = [
        {"start": 0, "end": 1, "speaker": "A"},
        {"start": 1, "end": 2, "speaker": "B"},
        {"start": 2, "end": 3, "speaker": "A"},
    ]
    norm = manager._normalize_speaker_ids(segments)
    assert all("speaker" in s for s in norm)
    assert norm[0]["speaker"] == "SPEAKER_00"
    assert norm[1]["speaker"] == "SPEAKER_01"
    assert norm[2]["speaker"] == "SPEAKER_00"
