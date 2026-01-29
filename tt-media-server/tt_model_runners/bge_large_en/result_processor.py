# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import torch
from domain.embedding_response import EmbeddingResponse
from domain.text_embedding_request import TextEmbeddingRequest


class BGEResultProcessor:
    """Converts BGE TTNN output (torch.Tensor) to EmbeddingResponse list."""

    def process(
        self,
        result: torch.Tensor,
        requests: list[TextEmbeddingRequest],
        token_counts: list[int] | None = None,
    ) -> list[EmbeddingResponse]:
        num_requests = len(requests)
        counts = (token_counts or [0] * num_requests)[:num_requests]
        embeddings = result[:num_requests]
        return [
            EmbeddingResponse(embedding=emb.cpu().numpy().tolist(), total_tokens=tc)
            for emb, tc in zip(embeddings, counts)
        ]
