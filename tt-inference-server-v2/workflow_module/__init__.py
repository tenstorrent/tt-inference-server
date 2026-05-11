# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

from .blocks_sink import BlockAccumulator, accept_blocks, get_default_accumulator

__all__ = ["BlockAccumulator", "accept_blocks", "get_default_accumulator"]
