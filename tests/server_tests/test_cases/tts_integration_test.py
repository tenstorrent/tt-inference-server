# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""
Integration tests for SpeechT5 TTS API endpoint on N150 hardware.

These tests verify:
1. Server startup and health check
2. TS generation
3. Error handling for invalid inputs
4. Audio output validation (WAV format, sample rate, duration)
5. API authentication

To run these tests:
    cd tt-inference-server/tt-media-server
    export ARCH_NAME=wormhole_b0
    export TT_METAL_HOME=/path/to/tt-metal
    export PYTHONPATH=/path/to/tt-metal
    export MODEL_RUNNER=tt-speecht5-tts
    export DEVICE_IDS="(0)"
    export IS_GALAXY=False

    # Start server in background
    uvicorn main:app --lifespan on --port 8000 &

    # Wait for server to be ready
    sleep 60

    # Run tests
    pytest tests/server_tests/test_cases/tts_integration_test.py -v
"""

import io
import time
import wave

import pytest
import requests

from tests.server_tests.base_test import BaseTest

# Test configuration
BASE_URL = "http://localhost:8000"
TTS_ENDPOINT = f"{BASE_URL}/audio/speech"
API_KEY = "your-secret-key"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

# Expected audio properties for SpeechT5 TTS
EXPECTED_SAMPLE_RATE = 16000
EXPECTED_CHANNELS = 1
EXPECTED_BIT_DEPTH = 16


class TestTTSServerHealth(BaseTest):
    """Test server health and availability"""

    def test_server_is_running(self):
        """Verify server is accessible"""
        try:
            response = requests.get(
                f"{BASE_URL}/tt-liveness", headers=HEADERS, timeout=5
            )
            assert response.status_code == 200, (
                f"Server not accessible: {response.status_code}"
            )
        except requests.exceptions.ConnectionError:
            pytest.fail(
                "Server is not running. Start it with: uvicorn main:app --lifespan on --port 8000"
            )

    def test_metrics_endpoint(self):
        """Verify Prometheus metrics endpoint"""
        response = requests.get(f"{BASE_URL}/metrics", timeout=5)
        assert response.status_code == 200
        assert "python_info" in response.text


class TestTTSAuthentication(BaseTest):
    """Test API authentication"""

    def test_missing_auth_token(self):
        """Request without auth token should fail"""
        response = requests.post(TTS_ENDPOINT, json={"text": "Test"})
        assert response.status_code == 401 or response.status_code == 403

    def test_invalid_auth_token(self):
        """Request with invalid auth token should fail"""
        headers = {
            "Authorization": "Bearer invalid-token",
            "Content-Type": "application/json",
        }
        response = requests.post(
            TTS_ENDPOINT,
            headers=headers,
            json={"text": "Test"},
        )
        assert response.status_code == 401 or response.status_code == 403

    def test_valid_auth_token(self):
        """Request with valid auth token should succeed"""
        response = requests.post(
            TTS_ENDPOINT,
            headers=HEADERS,
            json={"text": "Test"},
        )
        # Should get 200 or a processing error, not auth error
        assert response.status_code not in [401, 403]


class TestTTS(BaseTest):
    """Test TTS generation"""

    def test_simple_text_generation(self):
        """Generate speech from simple text"""
        text = "Hello world"
        response = requests.post(
            TTS_ENDPOINT,
            headers=HEADERS,
            json={"text": text},
            timeout=30,
        )

        assert response.status_code == 200, (
            f"Failed with status {response.status_code}: {response.text}"
        )
        assert response.headers.get("content-type") == "audio/wav"

        # Validate WAV format
        audio_data = response.content
        assert len(audio_data) > 0, "Empty audio response"

        with wave.open(io.BytesIO(audio_data)) as wav:
            assert wav.getframerate() == EXPECTED_SAMPLE_RATE, (
                f"Expected {EXPECTED_SAMPLE_RATE}Hz, got {wav.getframerate()}Hz"
            )
            assert wav.getnchannels() == EXPECTED_CHANNELS, (
                f"Expected {EXPECTED_CHANNELS} channel(s), got {wav.getnchannels()}"
            )
            assert wav.getsampwidth() == EXPECTED_BIT_DEPTH // 8, (
                f"Expected {EXPECTED_BIT_DEPTH}-bit, got {wav.getsampwidth() * 8}-bit"
            )

            # Verify audio has reasonable duration (at least 0.1 seconds)
            duration = wav.getnframes() / wav.getframerate()
            assert duration > 0.1, f"Audio too short: {duration}s"
            assert duration < 10.0, f"Audio unexpectedly long: {duration}s"

    def test_longer_text_generation(self):
        """Generate speech from longer text"""
        text = "The quick brown fox jumps over the lazy dog. This is a longer sentence to test TTS generation."
        response = requests.post(
            TTS_ENDPOINT,
            headers=HEADERS,
            json={"text": text},
            timeout=60,
        )

        assert response.status_code == 200
        assert len(response.content) > 1000  # Should be substantial audio

        # Validate it's proper WAV
        with wave.open(io.BytesIO(response.content)) as wav:
            duration = wav.getnframes() / wav.getframerate()
            # Longer text should produce longer audio (rough estimate)
            assert duration > 1.0, f"Audio too short for input text: {duration}s"

    def test_punctuation_handling(self):
        """Test text with various punctuation"""
        text = "Hello! How are you? I'm fine, thank you. Great to meet you."
        response = requests.post(
            TTS_ENDPOINT,
            headers=HEADERS,
            json={"text": text},
            timeout=30,
        )

        assert response.status_code == 200
        assert response.headers.get("content-type") == "audio/wav"

    def test_different_speaker_ids(self):
        """Test with different speaker embeddings"""
        text = "Testing different speakers"

        for speaker_id in [0, 100, 1000]:
            response = requests.post(
                TTS_ENDPOINT,
                headers=HEADERS,
                json={"text": text, "speaker_id": speaker_id},
                timeout=30,
            )

            assert response.status_code == 200, f"Failed for speaker_id {speaker_id}"
            assert len(response.content) > 0

    def test_response_format_verbose_json(self):
        """Test response_format=verbose_json returns JSON with base64 audio."""
        response = requests.post(
            TTS_ENDPOINT,
            headers=HEADERS,
            json={"text": "Hello", "response_format": "verbose_json"},
            timeout=30,
        )
        assert response.status_code == 200
        assert response.headers.get("content-type", "").startswith("application/json")
        data = response.json()
        assert "audio" in data
        assert "duration" in data
        assert "sample_rate" in data
        assert "format" in data

    def test_response_format_mp3(self):
        """Test response_format=mp3 returns binary with Content-Type audio/mpeg (requires ffmpeg)."""
        response = requests.post(
            TTS_ENDPOINT,
            headers=HEADERS,
            json={"text": "Hello", "response_format": "mp3"},
            timeout=30,
        )
        if response.status_code == 500 and "ffmpeg" in response.text:
            pytest.skip("ffmpeg not available; skip mp3 test")
        assert response.status_code == 200, f"Unexpected: {response.status_code} {response.text[:200]}"
        assert response.headers.get("content-type") == "audio/mpeg"
        assert len(response.content) > 0

    def test_response_format_ogg(self):
        """Test response_format=ogg returns binary with Content-Type audio/ogg (requires ffmpeg)."""
        response = requests.post(
            TTS_ENDPOINT,
            headers=HEADERS,
            json={"text": "Hello", "response_format": "ogg"},
            timeout=30,
        )
        if response.status_code == 500 and "ffmpeg" in response.text:
            pytest.skip("ffmpeg not available; skip ogg test")
        assert response.status_code == 200, f"Unexpected: {response.status_code} {response.text[:200]}"
        assert response.headers.get("content-type") == "audio/ogg"
        assert len(response.content) > 0


class TestTTSErrorHandling(BaseTest):
    """Test error handling for invalid inputs"""

    def test_empty_text(self):
        """Empty text should return error"""
        response = requests.post(
            TTS_ENDPOINT, headers=HEADERS, json={"text": ""}
        )
        # Should fail validation
        assert response.status_code in [400, 422]

    def test_missing_text_field(self):
        """Missing required text field should return error"""
        response = requests.post(TTS_ENDPOINT, headers=HEADERS, json={})
        assert response.status_code in [400, 422]

    def test_invalid_json(self):
        """Invalid JSON should return error"""
        response = requests.post(
            TTS_ENDPOINT, headers=HEADERS, data="invalid json"
        )
        assert response.status_code in [400, 422]

    def test_very_long_text(self):
        """Very long text should either work or return clear error"""
        text = "word " * 1000  # Very long text
        response = requests.post(
            TTS_ENDPOINT,
            headers=HEADERS,
            json={"text": text},
            timeout=120,
        )
        # Should either succeed or return clear error (not timeout/crash)
        assert response.status_code in [200, 400, 413, 422]

    def test_invalid_speaker_id(self):
        """Invalid speaker ID should handle gracefully"""
        response = requests.post(
            TTS_ENDPOINT,
            headers=HEADERS,
            json={"text": "Test", "speaker_id": -1},
        )
        # Should either accept and use default or return validation error
        assert response.status_code in [200, 400, 422]


class TestTTSPerformance(BaseTest):
    """Test performance characteristics"""

    def test_generation_latency(self):
        """Test that generation completes in reasonable time"""
        text = "Performance test"

        start_time = time.time()
        response = requests.post(
            TTS_ENDPOINT,
            headers=HEADERS,
            json={"text": text},
            timeout=30,
        )
        end_time = time.time()

        assert response.status_code == 200

        latency = end_time - start_time
        # Should complete within reasonable time (adjust based on hardware)
        assert latency < 10.0, f"Generation too slow: {latency}s"

    def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        text = "Concurrent test"

        # Send multiple requests concurrently (simulated with quick succession)
        responses = []
        for i in range(3):
            response = requests.post(
                TTS_ENDPOINT,
                headers=HEADERS,
                json={"text": f"{text} {i}"},
                timeout=60,
            )
            responses.append(response)

        # All should eventually succeed
        for response in responses:
            assert response.status_code == 200


class TestTTSAudioQuality(BaseTest):
    """Test audio output quality characteristics"""

    def test_audio_not_silent(self):
        """Verify generated audio is not silent"""
        text = "Audio quality test"
        response = requests.post(
            TTS_ENDPOINT,
            headers=HEADERS,
            json={"text": text},
            timeout=30,
        )

        assert response.status_code == 200

        with wave.open(io.BytesIO(response.content)) as wav:
            # Read audio samples
            frames = wav.readframes(wav.getnframes())
            samples = list(frames)

            # Check that audio is not all zeros (silent)
            non_zero = sum(1 for s in samples if s != 0)
            silence_ratio = 1.0 - (non_zero / len(samples))

            assert silence_ratio < 0.9, (
                f"Audio appears to be {silence_ratio * 100}% silent"
            )

    def test_consistent_format_across_requests(self):
        """Verify format consistency across multiple generations"""
        formats = []

        for i in range(3):
            response = requests.post(
                TTS_ENDPOINT,
                headers=HEADERS,
                json={"text": f"Test {i}"},
                timeout=30,
            )

            assert response.status_code == 200

            with wave.open(io.BytesIO(response.content)) as wav:
                formats.append(
                    {
                        "rate": wav.getframerate(),
                        "channels": wav.getnchannels(),
                        "width": wav.getsampwidth(),
                    }
                )

        # All formats should be identical
        assert all(f == formats[0] for f in formats), "Inconsistent audio formats"
