# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from dataclasses import dataclass
from typing import Optional, List, Dict, Any


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
            "text": self.text
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TranscriptionSegment':
        return cls(
            id=data['id'],
            speaker=data['speaker'],
            start_time=data['start_time'],
            end_time=data['end_time'],
            text=data['text']
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
            "duration": self.duration
        }
        
        if self.segments is not None:
            result["segments"] = [segment.to_dict() for segment in self.segments]
        
        if self.speaker_count is not None:
            result["speaker_count"] = self.speaker_count
            
        if self.speakers is not None:
            result["speakers"] = self.speakers
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TranscriptionResponse':
        segments = None
        if 'segments' in data and data['segments'] is not None:
            segments = [
                TranscriptionSegment.from_dict(seg) for seg in data['segments']
            ]
        
        return cls(
            text=data['text'],
            task=data['task'],
            language=data['language'],
            duration=data['duration'],
            segments=segments,
            speaker_count=data.get('speaker_count'),
            speakers=data.get('speakers')
        )
    
    def has_segments(self) -> bool:
        return self.segments is not None and len(self.segments) > 0
    
    def has_speakers(self) -> bool:
        return self.speakers is not None and len(self.speakers) > 0


@dataclass
class PartialStreamingTranscriptionResponse:
    text: str
    chunk_id: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "chunk_id": self.chunk_id
        }