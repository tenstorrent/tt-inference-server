# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import base64
import logging

import aiohttp
from server_tests.base_test import BaseTest

logger = logging.getLogger(__name__)

headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "Bearer your-secret-key",
}


class SpeechT5TTSTest(BaseTest):
    """Test SpeechT5 Text-to-Speech functionality"""

    async def _run_specific_test_async(self):
        """Run SpeechT5 TTS tests"""
        results = {}

        # Test: Basic TTS generation
        logger.info("Testing basic TTS generation...")
        try:
            basic_result = await self._test_basic_tts()
            results["basic_tts"] = basic_result
            logger.info("✅ Basic TTS test passed")
        except Exception as e:
            results["basic_tts"] = {"error": str(e)}
            logger.error(f"❌ Basic TTS test failed: {e}")

        return results

    async def _test_basic_tts(self):
        """Test basic text-to-speech generation (verbose_json to validate structure)."""
        url = f"http://localhost:{self.service_port}/audio/speech"

        payload = {
            "text": "Hello world, this is a test of SpeechT5 text to speech synthesis.",
            "response_format": "verbose_json",
        }

        timeout = aiohttp.ClientTimeout(total=120)  # 2 minute timeout for TTS
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload, headers=headers) as response:
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
