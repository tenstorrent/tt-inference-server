# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from utils.logger import TTLogger


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
    task: str
    language: str
    duration: float
    segments: Optional[List[AudioTextSegment]] = None
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
    def from_dict(cls, data: Dict[str, Any]) -> "AudioTextResponse":
        kwargs = {
            "text": data["text"],
            "task": data["task"],
            "language": data["language"],
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

        return cls(**kwargs)


@dataclass
class PartialStreamingAudioTextResponse:
    text: str
    chunk_id: int

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "chunk_id": self.chunk_id}


def combine_transcription_responses(
    responses: List[AudioTextResponse],
) -> AudioTextResponse:
    """
    Combine multiple AudioTextResponse objects into a single response.
    Returns combined response with summed duration and merged content
    """
    if not responses:
        raise ValueError("No transcription responses to combine")

    if len(responses) == 1:
        return responses[0]

    logger = TTLogger()

    # Combine text from all responses
    combined_text = " ".join(
        response.text.strip() for response in responses if response.text.strip()
    )

    # Sum up all durations
    total_duration = sum(response.duration for response in responses)

    # Use first response's task and language as defaults
    first_response = responses[0]

    # Combine segments if available
    combined_segments = []
    segment_id_counter = 1
    all_speakers = set()

    # Flatten all segments from responses into a single list
    all_segments = [
        segment
        for response in responses
        for segment in response.segments
        if response.segments
    ]

    for segment in all_segments:
        # Create new segment with updated ID to maintain sequence
        combined_segment = AudioTextSegment(
            id=segment_id_counter,
            speaker=segment.speaker,
            start_time=segment.start_time,
            end_time=segment.end_time,
            text=segment.text,
        )
        combined_segments.append(combined_segment)
        all_speakers.add(segment.speaker)
        segment_id_counter += 1

    # Combine speaker information
    combined_speakers = sorted(all_speakers) if all_speakers else None
    combined_speaker_count = len(all_speakers) if all_speakers else None

    # Create combined response
    combined_response = AudioTextResponse(
        text=combined_text,
        task=first_response.task,
        language=first_response.language,
        duration=total_duration,
        segments=combined_segments if combined_segments else None,
        speaker_count=combined_speaker_count,
        speakers=combined_speakers,
    )

    logger.info(
        f"Combined {len(responses)} transcription responses: "
        f"total_duration={total_duration:.2f}s, "
        f"total_segments={len(combined_segments)}, "
        f"speaker_count={combined_speaker_count}"
    )

    return combined_response
