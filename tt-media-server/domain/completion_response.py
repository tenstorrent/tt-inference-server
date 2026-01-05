# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC


from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, TypedDict, Union


@dataclass
class CompletionStreamChunk:
    text: str
    index: Optional[int] = None
    finish_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "index": self.index,
            "finish_reason": self.finish_reason,
        }


class StreamingChunkOutput(TypedDict):
    """Output yielded during streaming generation."""

    type: Literal["streaming_chunk"]
    chunk: CompletionStreamChunk
    task_id: str


class FinalResultOutput(TypedDict):
    """Final output yielded at the end of streaming generation."""

    type: Literal["final_result"]
    result: CompletionStreamChunk
    task_id: str
    return_result: bool


# Union type for async generator yield type
StreamingOutput = Union[StreamingChunkOutput, FinalResultOutput]
