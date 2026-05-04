# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""
DP2 load test with chunk_duration_seconds=30 and 60s audio.
Start server with: AUDIO_CHUNK_DURATION_SECONDS=30
Fewer segments per request (60/30 = 2) → fewer tasks in queue.
"""

import logging

from .audio_transcription_load_test import (
    AudioTranscriptionLoadTest,
)
from .test_payloads.audio_payload_60s import dataset as dataset60s

logger = logging.getLogger(__name__)

CHUNK_DURATION_SECONDS = 30


class AudioTranscriptionLoadDp2Chunk30Test(AudioTranscriptionLoadTest):
    """DP2 burst load, 60s audio, chunk 30s. Server: AUDIO_CHUNK_DURATION_SECONDS=30."""

    async def _run_specific_test_async(self):
        self.url = f"http://localhost:{self.service_port}/v1/audio/transcriptions"
        num_concurrent = self.targets.get("num_concurrent", 64)
        target_max_duration_s = self.targets.get("max_duration_s", None)

        payload = {
            "file": dataset60s["file"],
            "stream": False,
            "is_preprocessing_enabled": True,
            "prompt": "",
        }

        (
            requests_duration,
            avg_duration,
            num_ok,
        ) = await self._run_burst_concurrent(num_concurrent, payload)

        success = num_ok == num_concurrent
        if target_max_duration_s is not None:
            success = success and requests_duration <= target_max_duration_s

        out = {
            "success": success,
            "num_concurrent": num_concurrent,
            "num_ok": num_ok,
            "requests_duration_s": round(requests_duration, 2),
            "avg_duration_s": round(avg_duration, 2),
            "dataset": "60s",
            "chunk_duration_seconds": CHUNK_DURATION_SECONDS,
        }
        if target_max_duration_s is not None:
            out["target_max_duration_s"] = target_max_duration_s

        logger.info(
            f"DP2 burst (chunk={CHUNK_DURATION_SECONDS}s, 60s audio): "
            f"{num_ok}/{num_concurrent} OK, max={requests_duration:.2f}s, avg={avg_duration:.2f}s"
        )
        return out
