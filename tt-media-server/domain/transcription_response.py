# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class TranscriptionSegment:
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
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriptionSegment":
        return cls(
            id=data["id"],
            speaker=data["speaker"],
            start_time=data["start_time"],
            end_time=data["end_time"],
            text=data["text"],
        )


@dataclass
class TranscriptionResponse:
    text: str
    task: str
    language: str
    duration: float
    segments: Optional[List[TranscriptionSegment]] = None
    speaker_count: Optional[int] = None
    speakers: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "text": self.text,
            "task": self.task,
            "language": self.language,
            "duration": self.duration,
        }

        if self.segments is not None:
            result["segments"] = [segment.to_dict() for segment in self.segments]

        if self.speaker_count is not None:
            result["speaker_count"] = self.speaker_count

        if self.speakers is not None:
            result["speakers"] = self.speakers

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriptionResponse":
        kwargs = {
            "text": data["text"],
            "task": data["task"],
            "language": data["language"],
            "duration": data["duration"],
        }

        segments = None
        if "segments" in data and data["segments"] is not None:
            segments = [TranscriptionSegment.from_dict(seg) for seg in data["segments"]]

        if segments is not None:
            kwargs["segments"] = segments

        if "speaker_count" in data and data["speaker_count"] is not None:
            kwargs["speaker_count"] = data["speaker_count"]

        if "speakers" in data and data["speakers"] is not None:
            kwargs["speakers"] = data["speakers"]

        return cls(**kwargs)


@dataclass
class PartialStreamingTranscriptionResponse:
    text: str
    chunk_id: int

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "chunk_id": self.chunk_id}
