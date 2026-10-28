# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from domain.rerank_request import RerankPairRequest, RerankRequest
from model_services.base_service import BaseService


class RerankService(BaseService):
    """A rerank request is split into one (query, document) pair request per document. The pairs go
    through the scheduler on their own, so a worker batches pairs from any number of requests, and
    combine_results returns their responses in document order."""

    def __init__(self):
        super().__init__()

    async def pre_process(self, request: RerankRequest):
        texts = request.document_texts()
        if not texts:
            raise ValueError("documents must not be empty")
        request._segments = texts
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
        return list(results)
