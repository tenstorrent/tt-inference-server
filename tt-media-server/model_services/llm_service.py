# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from model_services.base_service import BaseService


class LLMService(BaseService):
    def __init__(self):
        super().__init__()

    def handle_streaming_chunk(self, chunk):
        chunk = chunk["data"]
        if chunk and chunk.text:
            return chunk
        return None

    def handle_final_result(self, result):
        # acts the same as handle_streaming_chunk
        return self.handle_streaming_chunk(result)

    async def post_process(self, result, input_request=None):
        if isinstance(result, dict):
            return result["data"]
        return result
