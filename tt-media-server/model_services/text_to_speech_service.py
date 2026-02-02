# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import base64

from config.constants import ResponseFormat
from domain.text_to_speech_request import TextToSpeechRequest
from domain.text_to_speech_response import TextToSpeechResponse
from model_services.base_service import BaseService
from model_services.cpu_workload_handler import CpuWorkloadHandler
from utils.decorators import log_execution_time
from utils.ffmpeg_utils import encode_wav_to

# base64 (from runner) + format -> bytes in that format (WAV, MP3 or OGG)
TTS_BINARY_FORMATS = ("audio", "wav", "mp3", "ogg")
TTS_WAV_FORMATS = ("audio", "wav")


def tts_worker_function(worker_context, base64_audio: str, response_format: str = None):
    """
    Decode base64 WAV from runner and return bytes in requested format (WAV/MP3/OGG).
    Uses utils.ffmpeg_utils for MP3/OGG encoding.
    """
    if not response_format:
        return None
    fmt = response_format.lower()
    if fmt not in TTS_BINARY_FORMATS:
        return None

    raw = base64.b64decode(base64_audio)
    if fmt in TTS_WAV_FORMATS:
        return raw
    return encode_wav_to(raw, fmt)


class TextToSpeechService(BaseService):
    def __init__(self):
        super().__init__()

        # Isolates CPU operations in separate processes and enables batch processing
        # Create minimal valid base64 WAV for warmup (44 bytes WAV header + minimal data)
        # This is a minimal valid WAV file: RIFF header + minimal PCM data
        minimal_wav_base64 = (
            "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="
        )

        self._cpu_workload_handler = CpuWorkloadHandler(
            name="TTSPostprocessing",
            worker_count=self.scheduler.get_worker_count(),
            worker_function=tts_worker_function,
            worker_context_setup=None,
            warmup_task_data=(minimal_wav_base64, ResponseFormat.AUDIO.value),
        )

    @log_execution_time("TTS post-processing")
    async def post_process(
        self, result: TextToSpeechResponse, input_request: TextToSpeechRequest
    ) -> TextToSpeechResponse:
        """
        Convert result.audio (base64 WAV from runner) to requested format and set
        result.output_bytes. No-op for non-binary response_format (json/verbose_json).
        """
        fmt = input_request.response_format.lower()
        if fmt not in TTS_BINARY_FORMATS:
            return result

        try:
            output_bytes = await self._cpu_workload_handler.execute_task(
                result.audio, input_request.response_format
            )
        except Exception as e:
            self.logger.error(f"TTS post-process failed: {e}")
            raise ValueError(f"Failed to produce audio ({fmt}): {str(e)}") from e

        result.output_bytes = output_bytes
        self.logger.debug(f"TTS post-process {fmt}: {len(output_bytes)} bytes")
        return result

    def stop_workers(self):
        """Stop CPU workload handler workers."""
        self.logger.info("Shutting down TTS postprocessing workers")
        self._cpu_workload_handler.stop_workers()
        return super().stop_workers()
