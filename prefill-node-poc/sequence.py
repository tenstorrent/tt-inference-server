# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Prefill-only sequence: a single request (prompt token ids) to be prefilled."""

from __future__ import annotations

from copy import copy
from itertools import count


class PrefillSequence:
    """
    A single prefill request: prompt token IDs and block allocation state.

    Inspired by nano-vllm Sequence; simplified for prefill-only (no decode,
    no append_token). Used by the scheduler and block manager.
    """

    _counter = count()

    def __init__(
        self,
        token_ids: list[int],
        req_id: str | None = None,
        block_size: int = 32,
    ):
        self.req_id = req_id or f"req_{next(PrefillSequence._counter)}"
        self.token_ids = copy(token_ids)
        self.block_size = block_size
        self.block_table: list[int] = []
        self.num_cached_tokens: int = 0

    def __len__(self) -> int:
        return len(self.token_ids)

    def __getitem__(self, key: int | slice):
        return self.token_ids[key]

    @property
    def num_blocks(self) -> int:
        return (len(self.token_ids) + self.block_size - 1) // self.block_size

    @property
    def num_cached_blocks(self) -> int:
        return self.num_cached_tokens // self.block_size

    def block(self, i: int) -> list[int]:
        """Token IDs for block index i (length block_size, last block may be shorter)."""
        start = i * self.block_size
        end = min((i + 1) * self.block_size, len(self.token_ids))
        return self.token_ids[start:end]
