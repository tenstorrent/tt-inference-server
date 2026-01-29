# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""
Integration tests for TTS API - focuses on error handling and input validation.

Tests verify:
1. Error handling for invalid inputs (empty text, missing fields, invalid JSON)
2. Input validation (very long text exceeds max_tts_text_length)

Note: Basic TTS functionality is covered by speecht5_tts_test.py
      Quality testing (WER) is covered by tts_quality_test.py
      Load testing is covered by tts_load_test.py
"""

import logging
import time

import aiohttp
from server_tests.base_test import BaseTest

logger = logging.getLogger(__name__)

HEADERS = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "Bearer your-secret-key",
}


class TTSIntegrationTest(BaseTest):
    """Integration test for TTS API - error handling and input validation.

    This test focuses on edge cases and error scenarios that are not covered
    by other TTS tests (speecht5_tts_test, tts_quality_test, tts_load_test).
    """

    async def _run_specific_test_async(self):
        """Run integration tests for TTS API."""
        test_start_time = time.time()
        self.url = f"http://localhost:{self.service_port}/audio/speech"

        results = {
            "empty_text": None,
            "missing_text_field": None,
            "invalid_json": None,
            "very_long_text": None,
        }

        session_timeout = aiohttp.ClientTimeout(total=120)
        async with aiohttp.ClientSession(timeout=session_timeout) as session:
            logger.info("Testing empty text handling...")
            results["empty_text"] = await self._test_empty_text(session)

            logger.info("Testing missing text field handling...")
            results["missing_text_field"] = await self._test_missing_text_field(session)

            logger.info("Testing invalid JSON handling...")
            results["invalid_json"] = await self._test_invalid_json(session)

            logger.info("Testing very long text handling...")
            results["very_long_text"] = await self._test_very_long_text(session)

        passed_tests = sum(1 for v in results.values() if v and v.get("passed"))
        total_tests = len(results)
        all_passed = passed_tests == total_tests

        total_duration = time.time() - test_start_time
        logger.info(
            f"TTS Integration Test completed: {passed_tests}/{total_tests} passed, "
            f"duration={total_duration:.1f}s"
        )

        return {
            "success": all_passed,
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "results": results,
            "duration": round(total_duration, 2),
        }

    async def _test_empty_text(self, session: aiohttp.ClientSession) -> dict:
        """Empty text should return validation error (400 or 422)."""
        try:
            async with session.post(
                self.url, json={"text": ""}, headers=HEADERS
            ) as response:
                status = response.status
                # Empty text should fail validation
                passed = status in [400, 422]
                return {
                    "passed": passed,
                    "status": status,
                    "expected": "400 or 422",
                    "message": "Empty text correctly rejected"
                    if passed
                    else f"Expected 400/422, got {status}",
                }
        except Exception as e:
            logger.error(f"Empty text test failed: {e}")
            return {"passed": False, "error": str(e)}

    async def _test_missing_text_field(self, session: aiohttp.ClientSession) -> dict:
        """Missing text field should return validation error (400 or 422)."""
        try:
            async with session.post(self.url, json={}, headers=HEADERS) as response:
                status = response.status
                # Missing required field should fail validation
                passed = status in [400, 422]
                return {
                    "passed": passed,
                    "status": status,
                    "expected": "400 or 422",
                    "message": "Missing text field correctly rejected"
                    if passed
                    else f"Expected 400/422, got {status}",
                }
        except Exception as e:
            logger.error(f"Missing text field test failed: {e}")
            return {"passed": False, "error": str(e)}

    async def _test_invalid_json(self, session: aiohttp.ClientSession) -> dict:
        """Invalid JSON should return error (400 or 422)."""
        try:
            async with session.post(
                self.url,
                data="this is not valid json",
                headers={**HEADERS, "Content-Type": "application/json"},
            ) as response:
                status = response.status
                # Invalid JSON should fail
                passed = status in [400, 422]
                return {
                    "passed": passed,
                    "status": status,
                    "expected": "400 or 422",
                    "message": "Invalid JSON correctly rejected"
                    if passed
                    else f"Expected 400/422, got {status}",
                }
        except Exception as e:
            logger.error(f"Invalid JSON test failed: {e}")
            return {"passed": False, "error": str(e)}

    async def _test_very_long_text(self, session: aiohttp.ClientSession) -> dict:
        """Very long text should be rejected with 422 (exceeds max_tts_text_length)."""
        try:
            # Generate text longer than max_tts_text_length (default 600 chars)
            long_text = "word " * 200  # ~1000 characters
            async with session.post(
                self.url,
                json={"text": long_text, "response_format": "verbose_json"},
                headers=HEADERS,
            ) as response:
                status = response.status
                # Should return 422 validation error for text too long
                passed = status == 422
                return {
                    "passed": passed,
                    "status": status,
                    "expected": "422",
                    "message": "Long text correctly rejected with validation error"
                    if passed
                    else f"Expected 422, got {status}",
                }
        except Exception as e:
            logger.error(f"Very long text test failed: {e}")
            return {"passed": False, "error": str(e)}
