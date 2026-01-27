# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC


from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, TypedDict


@dataclass
class CompletionResult:
    """Result of a completion operation - used for both streaming chunks and full responses."""

    text: str
    index: Optional[int] = None
    finish_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "index": self.index,
            "finish_reason": self.finish_reason,
        }


class CompletionOutput(TypedDict):
    """Output from the model runner - used for both streaming and non-streaming."""

    type: Literal["streaming_chunk", "final_result"]
    data: CompletionResult
