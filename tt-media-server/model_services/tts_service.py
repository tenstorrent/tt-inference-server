# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from domain.text_to_speech_request import TextToSpeechRequest
from model_services.base_service import BaseService
from telemetry.telemetry_client import TelemetryEvent
from utils.decorators import log_execution_time


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

    def create_segment_request(
        self, original_request: TextToSpeechRequest, segment, segment_index: int
    ) -> TextToSpeechRequest:
        """Create a request for processing a single text segment"""
        self.logger.debug(
            f"TTS segment {segment_index}: text='{segment[:50]}...'"
        )

        field_values = original_request.model_dump()
        new_request = type(original_request)(**field_values)
        new_request.text = segment  # Override with segment text

        new_request._task_id = f"{original_request._task_id}_segment_{segment_index}"

        return new_request

    def combine_results(self, results):
        """Combine multiple TTS results into one"""
        if not results:
            return None

        # For now, return the first result (single text processing)
        # Could be extended for multi-segment audio concatenation
        return results[0]

    @log_execution_time("TTS post-processing", TelemetryEvent.POST_PROCESSING, None)
    async def post_process(self, result):
        """Post-processing for TTS results"""
        return result














