# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC


class RerankPairResponse:
    """Relevance logit of one (query, document) pair and its prompt's token count."""

    def __init__(self, logit: float, total_tokens: int):
        self.logit = logit
        self.total_tokens = total_tokens
