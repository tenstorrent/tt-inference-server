# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from domain.rerank_request import RerankPairRequest, RerankRequest
from model_services.base_service import BaseService


class RerankService(BaseService):
    """A rerank request is split into one (query, document) pair request per document. The pairs go
    through the scheduler on their own, so a worker batches pairs from any number of requests, and
    combine_results returns their responses in document order."""

    async def pre_process(self, request: RerankRequest):
        # RerankRequest's validation guarantees 1 to MAX_RERANK_DOCUMENTS documents.
        request._segments = request.document_texts()
        return request

    def create_segment_request(
        self, original_request: RerankRequest, segment: str, segment_index: int
    ) -> RerankPairRequest:
        return RerankPairRequest(
            model=original_request.model,
            query=original_request.query,
            document=segment,
        )

    def combine_results(self, results):
        # A failed pair comes back from the scheduler as an Exception object (not raised): raise the
        # first one, so the request fails with the worker's message.
        for result in results:
            if isinstance(result, Exception):
                raise result
        return list(results)
