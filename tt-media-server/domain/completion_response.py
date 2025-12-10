# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC


from dataclasses import dataclass


@dataclass
class CompletionStreamChunk:
    text: str
