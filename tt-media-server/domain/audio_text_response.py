# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class AudioTextSegment:
    id: int
    speaker: str
    start_time: float
    end_time: float
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "speaker": self.speaker,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AudioTextSegment":
        return cls(
            id=data["id"],
            speaker=data["speaker"],
            start_time=data["start_time"],
            end_time=data["end_time"],
            text=data["text"],
        )


@dataclass
class AudioTextResponse:
    text: str
    duration: float
    segments: Optional[List[AudioTextSegment]] = None
    speaker_count: Optional[int] = None
    speakers: Optional[List[str]] = None
    start: Optional[float] = None
    end: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "text": self.text,
            "duration": self.duration,
        }

        if self.segments is not None:
            result["segments"] = [segment.to_dict() for segment in self.segments]

        if self.speaker_count is not None:
            result["speaker_count"] = self.speaker_count

        if self.speakers is not None:
            result["speakers"] = self.speakers

        if self.start is not None:
            result["start"] = self.start

        if self.end is not None:
            result["end"] = self.end

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AudioTextResponse":
        kwargs = {
            "text": data["text"],
            "duration": data["duration"],
        }

        segments = None
        if "segments" in data and data["segments"] is not None:
            segments = [AudioTextSegment.from_dict(seg) for seg in data["segments"]]

        if segments is not None:
            kwargs["segments"] = segments

        if "speaker_count" in data and data["speaker_count"] is not None:
            kwargs["speaker_count"] = data["speaker_count"]

        if "speakers" in data and data["speakers"] is not None:
            kwargs["speakers"] = data["speakers"]

        if "start" in data and data["start"] is not None:
            kwargs["start"] = data["start"]

        if "end" in data and data["end"] is not None:
            kwargs["end"] = data["end"]

        return cls(**kwargs)


@dataclass
class AudioStreamChunk:
    text: str
    chunk_id: int

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "chunk_id": self.chunk_id}
