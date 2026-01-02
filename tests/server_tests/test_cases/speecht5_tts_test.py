# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import base64
import json

import aiohttp
from server_tests.base_test import BaseTest


class SpeechT5TTSTest(BaseTest):
    """Test SpeechT5 Text-to-Speech functionality"""

    async def _run_specific_test_async(self):
        """Run SpeechT5 TTS tests"""
        results = {}

        # Test 1: Basic TTS generation (non-streaming)
        print("Testing basic TTS generation...")
        try:
            basic_result = await self._test_basic_tts()
            results["basic_tts"] = basic_result
            print("✓ Basic TTS test passed")
        except Exception as e:
            results["basic_tts"] = {"error": str(e)}
            print(f"✗ Basic TTS test failed: {e}")

        # Test 2: TTS with custom speaker ID
        print("Testing TTS with custom speaker ID...")
        try:
            speaker_result = await self._test_tts_with_speaker()
            results["tts_with_speaker"] = speaker_result
            print("✓ Speaker TTS test passed")
        except Exception as e:
            results["tts_with_speaker"] = {"error": str(e)}
            print(f"✗ Speaker TTS test failed: {e}")

        # Test 3: Streaming TTS
        print("Testing streaming TTS...")
        try:
            streaming_result = await self._test_streaming_tts()
            results["streaming_tts"] = streaming_result
            print("✓ Streaming TTS test passed")
        except Exception as e:
            results["streaming_tts"] = {"error": str(e)}
            print(f"✗ Streaming TTS test failed: {e}")

        return results

    async def _test_basic_tts(self):
        """Test basic text-to-speech generation"""
        url = f"http://localhost:{self.service_port}/audio/speech"

        payload = {
            "text": "Hello world, this is a test of SpeechT5 text to speech synthesis.",
            "stream": False
        }

        timeout = aiohttp.ClientTimeout(total=120)  # 2 minute timeout for TTS
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as response:
                assert response.status == 200, f"Expected status 200, got {response.status}"

                result = await response.json()

                # Validate response structure
                assert "audio" in result, "Response should contain 'audio' field"
                assert "duration" in result, "Response should contain 'duration' field"
                assert "sample_rate" in result, "Response should contain 'sample_rate' field"
                assert "format" in result, "Response should contain 'format' field"

                # Validate audio data
                audio_b64 = result["audio"]
                assert isinstance(audio_b64, str), "Audio should be base64 encoded string"

                # Try to decode to verify it's valid base64
                try:
                    audio_bytes = base64.b64decode(audio_b64)
                    assert len(audio_bytes) > 0, "Decoded audio should not be empty"
                except Exception as e:
                    raise AssertionError(f"Audio data is not valid base64: {e}")

                # Validate duration is reasonable (should be > 0 and not too long)
                duration = result["duration"]
                assert duration > 0, f"Duration should be positive, got {duration}"
                assert duration < 30, f"Duration seems too long for test text, got {duration}s"

                return {
                    "status": "success",
                    "duration": duration,
                    "sample_rate": result["sample_rate"],
                    "format": result["format"],
                    "audio_size_bytes": len(audio_bytes)
                }

    async def _test_tts_with_speaker(self):
        """Test TTS with custom speaker ID"""
        url = f"http://localhost:{self.service_port}/audio/speech"

        payload = {
            "text": "This is a test with a specific speaker voice.",
            "speaker_id": "7306",  # Common speaker ID from CMU Arctic dataset
            "stream": False
        }

        timeout = aiohttp.ClientTimeout(total=120)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as response:
                assert response.status == 200, f"Expected status 200, got {response.status}"

                result = await response.json()

                # Validate response
                assert "audio" in result, "Response should contain 'audio' field"
                assert result.get("speaker_id") == "7306", f"Expected speaker_id '7306', got {result.get('speaker_id')}"

                return {
                    "status": "success",
                    "speaker_id": result.get("speaker_id"),
                    "duration": result["duration"]
                }

    async def _test_streaming_tts(self):
        """Test streaming TTS generation"""
        url = f"http://localhost:{self.service_port}/audio/speech"

        payload = {
            "text": "This is a streaming test of text to speech generation.",
            "stream": True
        }

        timeout = aiohttp.ClientTimeout(total=180)  # 3 minute timeout for streaming
        chunks_received = 0
        final_result = None

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as response:
                assert response.status == 200, f"Expected status 200, got {response.status}"

                # Read streaming response line by line
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if not line:
                        continue

                    try:
                        chunk_data = json.loads(line)

                        if chunk_data.get('type') == 'streaming_chunk':
                            chunks_received += 1
                            chunk = chunk_data.get('chunk', {})
                            assert 'audio_chunk' in chunk, "Streaming chunk should contain audio_chunk"
                            assert 'chunk_id' in chunk, "Streaming chunk should contain chunk_id"

                            # Verify audio chunk is valid base64
                            audio_chunk_b64 = chunk['audio_chunk']
                            base64.b64decode(audio_chunk_b64)  # Should not raise exception

                        elif chunk_data.get('type') == 'final_result':
                            final_result = chunk_data.get('result', {})
                            break

                    except json.JSONDecodeError as e:
                        raise AssertionError(f"Invalid JSON in streaming response: {line} - {e}")

        # Validate we received chunks and final result
        assert chunks_received > 0, "Should have received at least one streaming chunk"
        assert final_result is not None, "Should have received final result"
        assert 'audio' in final_result, "Final result should contain audio"

        return {
            "status": "success",
            "chunks_received": chunks_received,
            "final_duration": final_result.get("duration"),
            "final_sample_rate": final_result.get("sample_rate")
        }

