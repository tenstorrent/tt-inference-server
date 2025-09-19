# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import Dict, Any, Union, List

class TranscriptBuilder:
    def __init__(self):
        self.unique_speakers = set()
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Remove EOS tokens and clean text"""
        if not isinstance(text, str):
            return str(text)
        return text.replace("<EOS>", "").strip()
    
    @staticmethod
    def create_partial_result(text: str, chunk_id: int = None) -> Dict[str, Any]:
        """Create a standardized partial streaming result"""
        clean_text = TranscriptBuilder.clean_text(text)
        if not clean_text:  # Don't yield empty chunks
            return None
            
        result = {
            "type": "partial_text",
            "text": clean_text,
            "is_partial": True
        }
        
        if chunk_id is not None:
            result["chunk_id"] = chunk_id
            
        return result
    
    @staticmethod
    def create_final_result(text: str, task: str, language: str, duration: float,
                                        segments: List[Dict] = None, speaker_count: int = None, speakers: List[str] = None) -> Dict[str, Any]:
        """Create a standardized final result with speaker information"""
        response = {
            "type": "final_result",
            "text": TranscriptBuilder.clean_text(text),
            "task": task,
            "language": language,
            "duration": duration,
            "is_final": True
        }
        if segments is not None:
            response["segments"] = segments
        if speaker_count is not None:
            response["speaker_count"] = speaker_count
        if speakers is not None:
            response["speakers"] = speakers
        
        return response

    @staticmethod
    def create_segment_result(segment_id: int, start_time: float, end_time: float, 
                            text: str, speaker: str, partial_transcript: str = None) -> Dict[str, Any]:
        """Create a standardized segment streaming result"""
        return {
            "type": "segment_result",
            "segment_id": segment_id,
            "start_time": start_time,
            "end_time": end_time,
            "text": TranscriptBuilder.clean_text(text),
            "speaker": speaker,
            "partial_transcript": partial_transcript,
            "is_partial": True
        }
    
    @staticmethod
    def concatenate_chunks(chunks: List[Union[str, Dict]]) -> str:
        """Concatenate text chunks into final transcript"""
        texts = []
        for chunk in chunks:
            if isinstance(chunk, str):
                clean_text = TranscriptBuilder.clean_text(chunk)
                if clean_text:
                    texts.append(clean_text)
            elif isinstance(chunk, dict) and "text" in chunk:
                clean_text = TranscriptBuilder.clean_text(chunk["text"])
                if clean_text:
                    texts.append(clean_text)
        
        return " ".join(texts)