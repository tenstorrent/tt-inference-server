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


def tts_worker_function(worker_context, base64_audio: str, response_format: str = None):
    """
    Worker function for TTS post-processing.
    Decodes base64 audio to WAV bytes if response_format is "audio" or "wav".

    Args:
        worker_context: Worker context (None for TTS, not used)
        base64_audio: Base64-encoded audio string
        response_format: Response format ("audio" or "wav" for WAV bytes, otherwise None)

    Returns:
        WAV bytes if response_format is "audio" or "wav", otherwise None
    """
    if response_format and response_format.lower() in ("audio", "wav"):
        return base64.b64decode(base64_audio)
    return None


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
            worker_context_setup=None,  # No context needed for base64 decode
            warmup_task_data=(minimal_wav_base64, ResponseFormat.AUDIO.value),
        )

    # async def pre_process(self, request: TextToSpeechRequest) -> TextToSpeechRequest:
    #     return request

    @log_execution_time("TTS post-processing")
    async def post_process(
        self, result: TextToSpeechResponse, input_request: TextToSpeechRequest
    ) -> TextToSpeechResponse:
        """
        Post-process TTS response using CPU workload handler.
        If response_format is "audio" or "wav", decode base64 audio to WAV bytes.
        Otherwise, return response as-is with base64 audio.

        Uses CPU workers for consistency with ImageService and AudioService,
        and to enable future batch processing capabilities.
        """
        if input_request.response_format.lower() in ("audio", "wav"):
            try:
                wav_bytes = await self._cpu_workload_handler.execute_task(
                    result.audio, input_request.response_format
                )

                result._wav_bytes = wav_bytes
                self.logger.debug(
                    f"Decoded base64 audio to WAV bytes: {len(wav_bytes)} bytes"
                )
            except Exception as e:
                self.logger.error(f"Failed to decode base64 audio: {e}")
                raise ValueError(f"Failed to decode audio data: {str(e)}") from e

        return result

    def stop_workers(self):
        """Stop CPU workload handler workers"""
        self.logger.info("Shutting down TTS postprocessing workers")
        self._cpu_workload_handler.stop_workers()

        return super().stop_workers()
