# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC


class EmbeddingResponse:
    def __init__(self, embedding: list[float], total_tokens: int):
        self.embedding = embedding
        self.total_tokens = total_tokens
