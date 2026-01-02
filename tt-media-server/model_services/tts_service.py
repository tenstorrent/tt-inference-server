# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from domain.text_to_speech_request import TextToSpeechRequest
from model_services.base_service import BaseService
from utils.decorators import log_execution_time
from telemetry.telemetry_client import TelemetryEvent


class TTSService(BaseService):
    def __init__(self):
        super().__init__()

    @log_execution_time("TTS preprocessing", TelemetryEvent.PRE_PROCESSING, None)
    async def pre_process(self, request: TextToSpeechRequest):
        """Preprocessing for TTS requests - minimal processing needed"""
        try:
            if request.text is None or not request.text.strip():
                raise ValueError("No text provided for TTS")

            # Basic validation and setup
            request._task_id = getattr(request, '_task_id', None)
            request._estimated_duration = len(request.text.split()) * 0.5  # Rough estimate

            return request

        except Exception as e:
            self.logger.error(f"TTS preprocessing failed: {e}")
            raise















