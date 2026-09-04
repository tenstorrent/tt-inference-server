# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from __future__ import annotations

import base64
import os
from typing import TYPE_CHECKING

import aiohttp

from report_module.schema import Block

from .._test_common import BaseTest, TestConfig
from ._metrics_smoke import assert_series_present, fetch_metrics_body

if TYPE_CHECKING:
    from ..context import MediaContext

DEFAULT_API_KEY = "your-secret-key"

HEADERS = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('API_KEY', DEFAULT_API_KEY)}",
}

# Series that must appear on /metrics after one successful generation.
# tts_requests_total / output_audio_seconds are recorded in the API process;
# first_chunk / chunk_generation inside the device worker, so their presence
# also proves the multiprocess aggregation path (see _metrics_smoke).
TTS_METRIC_SERIES = (
    "tt_media_server_audio_tts_requests_total",
    "tt_media_server_audio_tts_output_audio_seconds_total",
    "tt_media_server_audio_tts_first_chunk_seconds",
    "tt_media_server_audio_tts_chunk_generation_seconds",
    "tt_media_server_audio_tts_input_tokens_total",
)


class SpeechT5TTSTest(BaseTest):
    """Test SpeechT5 Text-to-Speech functionality"""

    KIND = "speecht5_tts"
    TASK_TYPE = "integration"

    async def _run_specific_test_async(self):
        """Run SpeechT5 TTS tests"""
        results = {"success": False}
        try:
            basic_result = await self._test_basic_tts()
            results["basic_tts"] = basic_result
        except Exception as e:
            results["basic_tts"] = {"error": str(e)}
            return results

        # Only meaningful after a successful generation has populated them.
        try:
            metrics_result = await self._test_metrics_smoke()
            results["metrics_smoke"] = metrics_result
        except Exception as e:
            results["metrics_smoke"] = {"error": str(e)}
            return results

        results["success"] = (
            basic_result.get("status") == "success"
            and metrics_result.get("status") == "success"
        )
        return results

    async def _test_basic_tts(self):
        """Test basic text-to-speech generation"""
        url = f"{self.base_url}/v1/audio/speech"

        payload = {
            "text": "Hello world, this is a test of SpeechT5 text to speech synthesis.",
            "response_format": "json",
        }

        timeout = aiohttp.ClientTimeout(total=120)  # 2 minute timeout for TTS
        async with aiohttp.ClientSession(headers=HEADERS, timeout=timeout) as session:
            async with session.post(url, json=payload) as response:
                assert response.status == 200, (
                    f"Expected status 200, got {response.status}"
                )

                result = await response.json()

                # Validate response structure
                assert "audio" in result, "Response should contain 'audio' field"
                assert "duration" in result, "Response should contain 'duration' field"
                assert "sample_rate" in result, (
                    "Response should contain 'sample_rate' field"
                )
                assert "format" in result, "Response should contain 'format' field"

                # Validate audio data
                audio_b64 = result["audio"]
                assert isinstance(audio_b64, str), (
                    "Audio should be base64 encoded string"
                )

                # Try to decode to verify it's valid base64
                try:
                    audio_bytes = base64.b64decode(audio_b64)
                    assert len(audio_bytes) > 0, "Decoded audio should not be empty"
                except Exception as e:
                    raise AssertionError(f"Audio data is not valid base64: {e}")

                # Validate duration is reasonable (should be > 0 and not too long)
                duration = result["duration"]
                assert duration > 0, f"Duration should be positive, got {duration}"
                assert duration < 30, (
                    f"Duration seems too long for test text, got {duration}s"
                )

                return {
                    "status": "success",
                    "duration": duration,
                    "sample_rate": result["sample_rate"],
                    "format": result["format"],
                    "audio_size_bytes": len(audio_bytes),
                }

    async def _test_metrics_smoke(self):
        """The TTS metric series must be exported after a real generation."""
        body = await fetch_metrics_body(self.base_url, headers=HEADERS)
        checked = assert_series_present(body, TTS_METRIC_SERIES)
        return {"status": "success", "series_checked": checked}


def run_speecht5_tts(ctx: MediaContext) -> Block:
    """Run SpeechT5TTSTest under ``ctx`` and return its Block."""
    test_config = TestConfig(
        {
            "timeout": 180,
            "retry_attempts": 2,
            "retry_delay": 5,
            "break_on_failure": False,
        }
    )
    return SpeechT5TTSTest(test_config, targets={}, ctx=ctx).run_tests()


__all__ = ["SpeechT5TTSTest", "run_speecht5_tts"]
