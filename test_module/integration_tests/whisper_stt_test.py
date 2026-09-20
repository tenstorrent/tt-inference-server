# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Whisper STT integration test: one real transcription + metrics smoke.

The metrics smoke is the point: the STT confidence signals (avg_logprob,
no_speech_probability, compression_ratio) are extracted from tt-metal's
generator result tuples by position and recorded fail-open, so an interface
drift in tt-metal silences them without any error. This test is the alarm —
see ``_metrics_smoke``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import aiohttp

from report_module.schema import Block

from .._test_common import BaseTest, TestConfig
from ..context import TEST_PAYLOADS_PATH
from ._metrics_smoke import assert_series_present, fetch_metrics_body

if TYPE_CHECKING:
    from ..context import MediaContext

DEFAULT_API_KEY = "your-secret-key"

HEADERS = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('API_KEY', DEFAULT_API_KEY)}",
}

AUDIO_PAYLOAD_FILE = "image_client_audio_payload"

# Series that must appear on /metrics after one successful transcription.
# requests_total / input_audio_seconds / realtime_factor are recorded in the
# API process; the three confidence signals inside the device worker, off the
# raw tt-metal generator tuples — their presence proves both the multiprocess
# aggregation and the tuple contract (see _metrics_smoke).
STT_METRIC_SERIES = (
    "tt_media_server_audio_stt_requests_total",
    "tt_media_server_audio_stt_requests_by_audio_format_total",
    "tt_media_server_audio_stt_input_audio_seconds_total",
    "tt_media_server_audio_stt_realtime_factor",
    "tt_media_server_audio_stt_avg_logprob",
    "tt_media_server_audio_stt_no_speech_probability",
    "tt_media_server_audio_stt_compression_ratio",
)


class WhisperSTTTest(BaseTest):
    """Test Whisper speech-to-text functionality and metric export."""

    KIND = "whisper_stt"
    TASK_TYPE = "integration"

    async def _run_specific_test_async(self):
        """Run Whisper STT tests"""
        results = {"success": False}
        try:
            basic_result = await self._test_basic_transcription()
            results["basic_transcription"] = basic_result
        except Exception as e:
            results["basic_transcription"] = {"error": str(e)}
            return results

        # Only meaningful after a successful transcription has populated them.
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

    def _payloads_path(self) -> Path:
        if self.ctx is not None:
            return Path(self.ctx.test_payloads_path)
        return Path(TEST_PAYLOADS_PATH)

    async def _test_basic_transcription(self):
        """Test one non-streaming transcription of the standard test clip."""
        url = f"{self.base_url}/v1/audio/transcriptions"

        with open(self._payloads_path() / AUDIO_PAYLOAD_FILE) as f:
            audio_file = json.load(f)

        payload = {
            "file": audio_file["file"],
            "stream": False,
        }

        start = time.time()
        timeout = aiohttp.ClientTimeout(total=120)
        async with aiohttp.ClientSession(headers=HEADERS, timeout=timeout) as session:
            async with session.post(url, json=payload) as response:
                assert response.status == 200, (
                    f"Expected status 200, got {response.status}"
                )
                result = await response.json()

        assert "text" in result, "Response should contain 'text' field"
        assert isinstance(result["text"], str) and result["text"].strip(), (
            "Transcript should be a non-empty string"
        )
        assert "duration" in result, "Response should contain 'duration' field"
        duration = result["duration"]
        assert duration > 0, f"Audio duration should be positive, got {duration}"

        return {
            "status": "success",
            "audio_duration": duration,
            "transcript_chars": len(result["text"]),
            "elapsed_s": round(time.time() - start, 2),
        }

    async def _test_metrics_smoke(self):
        """The STT metric series must be exported after a real transcription."""
        body = await fetch_metrics_body(self.base_url, headers=HEADERS)
        checked = assert_series_present(body, STT_METRIC_SERIES)
        return {"status": "success", "series_checked": checked}


def run_whisper_stt(ctx: MediaContext) -> Block:
    """Run WhisperSTTTest under ``ctx`` and return its Block."""
    test_config = TestConfig(
        {
            "timeout": 180,
            "retry_attempts": 2,
            "retry_delay": 5,
            "break_on_failure": False,
        }
    )
    return WhisperSTTTest(test_config, targets={}, ctx=ctx).run_tests()


__all__ = ["WhisperSTTTest", "run_whisper_stt"]
