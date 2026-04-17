# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import base64
import shutil
from typing import Any, Optional

from config.constants import AUDIO_RESPONSE_FORMATS, FFMPEG_REQUIRED_FORMATS
from domain.text_to_speech_request import TextToSpeechRequest
from domain.text_to_speech_response import TextToSpeechResponse
from model_services.base_service import BaseService
from model_services.cpu_workload_handler import CpuWorkloadHandler
from utils.decorators import log_execution_time
from utils.ffmpeg_utils import encode_wav_to

FFMPEG_REQUIRED_MSG = (
    "response_format={fmt} requires ffmpeg but ffmpeg is not in PATH. "
    "Install ffmpeg (e.g. apt install ffmpeg) for MP3/OGG output. Falling back to WAV."
)
POST_PROCESS_FAILED_MSG = (
    "TTS post-process failed for response_format={fmt}: {e}. Falling back to WAV."
)
WORKER_NONE_MSG = (
    "TTS worker returned None for response_format={fmt}. Falling back to WAV."
)


def tts_worker_function(
    worker_context: Any,
    base64_audio: str,
    response_format: Optional[str] = None,
) -> Optional[bytes]:
    """Decode base64 WAV and return bytes in requested format (WAV/MP3/OGG)."""
    if (fmt := (response_format or "").strip().lower()) not in AUDIO_RESPONSE_FORMATS:
        return None
    raw = base64.b64decode(base64_audio)
    return raw if fmt == "wav" else encode_wav_to(raw, fmt)


class TextToSpeechService(BaseService):
    def __init__(self):
        super().__init__()
        minimal_wav_base64 = (
            "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="
        )
        self._cpu_workload_handler = CpuWorkloadHandler(
            name="TTSPostprocessing",
            worker_count=self.scheduler.get_worker_count(),
            worker_function=tts_worker_function,
            worker_context_setup=None,
            warmup_task_data=(minimal_wav_base64, "wav"),
        )

    async def _set_result_to_wav(self, result: TextToSpeechResponse) -> None:
        """Encode result.audio to WAV and set result.output_bytes and result.format."""
        result.output_bytes = await self._cpu_workload_handler.execute_task(
            result.audio, "wav"
        )
        result.format = "wav"
        self.logger.debug(
            f"TTS post-process fallback to WAV: {len(result.output_bytes)} bytes"
        )

    async def _warn_and_fallback_to_wav(
        self, result: TextToSpeechResponse, fmt: str, msg_template: str, **kwargs: Any
    ) -> TextToSpeechResponse:
        """Log warning with formatted message and set result to WAV."""
        self.logger.warning(msg_template.format(fmt=fmt, **kwargs))
        await self._set_result_to_wav(result)
        return result

    @log_execution_time("TTS post-processing")
    async def post_process(
        self, result: TextToSpeechResponse, input_request: TextToSpeechRequest
    ) -> TextToSpeechResponse:
        """
        Convert result.audio (base64 WAV from runner) to requested format.
        If mp3/ogg requested but ffmpeg is unavailable or encoding fails, fall back to WAV.
        """
        fmt = input_request.response_format.lower()
        if fmt not in AUDIO_RESPONSE_FORMATS:
            return result

        needs_ffmpeg = fmt in FFMPEG_REQUIRED_FORMATS
        if needs_ffmpeg and not shutil.which("ffmpeg"):
            return await self._warn_and_fallback_to_wav(
                result, fmt, FFMPEG_REQUIRED_MSG
            )

        try:
            output_bytes = await self._cpu_workload_handler.execute_task(
                result.audio, input_request.response_format
            )
        except Exception as e:
            if needs_ffmpeg:
                return await self._warn_and_fallback_to_wav(
                    result, fmt, POST_PROCESS_FAILED_MSG, e=e
                )
            self.logger.error(f"TTS post-process failed: {e}")
            raise ValueError(f"Failed to produce audio ({fmt}): {e!s}") from e

        if output_bytes is not None:
            result.output_bytes = output_bytes
            result.format = fmt
            self.logger.debug(f"TTS post-process {fmt}: {len(output_bytes)} bytes")
            return result

        if needs_ffmpeg:
            return await self._warn_and_fallback_to_wav(result, fmt, WORKER_NONE_MSG)

        result.output_bytes = output_bytes
        result.format = fmt
        self.logger.debug(f"TTS post-process {fmt}: {len(output_bytes)} bytes")
        return result

    def stop_workers(self):
        """Stop CPU workload handler workers."""
        self.logger.info("Shutting down TTS postprocessing workers")
        self._cpu_workload_handler.stop_workers()
        return super().stop_workers()
