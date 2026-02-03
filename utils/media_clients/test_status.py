# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence


def _metrics_from_attrs(obj: Any, attrs: Sequence[str]) -> Dict[str, float]:
    """Build metrics dict from object attributes; only non-None numeric values."""
    return {
        k: float(getattr(obj, k)) for k in attrs if getattr(obj, k, None) is not None
    }


class BaseTestStatus(ABC):
    """Base class for all test status objects."""

    def __init__(self, status: bool, elapsed: float):
        self.status = status
        self.elapsed = elapsed

    def get_metrics(self) -> Dict[str, float]:
        """Numeric metrics for report aggregation. Override in subclasses to add more."""
        return {"elapsed": float(self.elapsed)}

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        pass


class ImageGenerationTestStatus(BaseTestStatus):
    """Test status for image generation models (SDXL, SD3.5, etc.)."""

    def __init__(
        self,
        status: bool,
        elapsed: float,
        num_inference_steps: int = 0,
        inference_steps_per_second: float = 0,
        ttft: Optional[float] = None,
        tpups: Optional[float] = None,
        base64image: Optional[str] = None,
        prompt: Optional[str] = None,
    ):
        super().__init__(status, elapsed)
        self.num_inference_steps = num_inference_steps
        self.inference_steps_per_second = inference_steps_per_second
        self.ttft = ttft
        self.tpups = tpups
        self.base64image = base64image
        self.prompt = prompt

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "elapsed": self.elapsed,
            "num_inference_steps": self.num_inference_steps,
            "inference_steps_per_second": self.inference_steps_per_second,
            "ttft": self.ttft,
            "tpups": self.tpups,
            "base64image": self.base64image,
            "prompt": self.prompt,
        }


class AudioTestStatus(BaseTestStatus):
    """Test status for audio transcription models."""

    def __init__(
        self,
        status: bool,
        elapsed: float,
        ttft: Optional[float] = None,
        tsu: Optional[float] = None,
        rtr: Optional[float] = None,
    ):
        super().__init__(status, elapsed)
        self.ttft = ttft
        self.tsu = tsu
        self.rtr = rtr

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "elapsed": self.elapsed,
            "ttft": self.ttft,
            "t/s/u": self.tsu,
            "rtr": self.rtr,
        }


class CnnGenerationTestStatus(BaseTestStatus):
    """Test status for CNN models (RESNET, etc.)."""

    def __init__(
        self,
        status: bool,
        elapsed: float,
        num_inference_steps: int = 0,
        inference_steps_per_second: float = 0,
        ttft: Optional[float] = None,
        tpups: Optional[float] = None,
        base64image: Optional[str] = None,
        prompt: Optional[str] = None,
    ):
        super().__init__(status, elapsed)
        self.num_inference_steps = num_inference_steps
        self.inference_steps_per_second = inference_steps_per_second
        self.ttft = ttft
        self.tpups = tpups
        self.base64image = base64image
        self.prompt = prompt

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "elapsed": self.elapsed,
            "num_inference_steps": self.num_inference_steps,
            "inference_steps_per_second": self.inference_steps_per_second,
            "ttft": self.ttft,
            "tpups": self.tpups,
            "base64image": self.base64image,
            "prompt": self.prompt,
        }


class EmbeddingTestStatus(BaseTestStatus):
    """Test status for embedding models."""

    def __init__(self, status: bool, elapsed: float, ttft: Optional[float] = None):
        super().__init__(status, elapsed)
        self.ttft = ttft

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "elapsed": self.elapsed, "ttft": self.ttft}


class TtsTestStatus(BaseTestStatus):
    """Test status for text-to-speech models."""

    _METRIC_ATTRS = ("elapsed", "ttft_ms", "rtr", "audio_duration", "wer")

    def __init__(
        self,
        status: bool,
        elapsed: float,
        ttft_ms: Optional[float] = None,
        rtr: Optional[float] = None,
        text: Optional[str] = None,
        audio_duration: Optional[float] = None,
        wer: Optional[float] = None,
        reference_text: Optional[str] = None,
    ):
        super().__init__(status, elapsed)
        self.ttft_ms = ttft_ms
        self.rtr = rtr
        self.text = text
        self.audio_duration = audio_duration
        self.wer = wer
        self.reference_text = reference_text

    def get_metrics(self) -> Dict[str, float]:
        """Numeric metrics for report aggregation (elapsed, ttft_ms, rtr, etc.)."""
        return _metrics_from_attrs(self, self._METRIC_ATTRS)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "elapsed": self.elapsed,
            "ttft_ms": self.ttft_ms,
            "rtr": self.rtr,
            "text": self.text,
            "audio_duration": self.audio_duration,
            "wer": self.wer,
            "reference_text": self.reference_text,
        }


class VideoGenerationTestStatus(BaseTestStatus):
    """Test status for video generation models (Mochi, WAN, etc.)."""

    def __init__(
        self,
        status: bool,
        elapsed: float,
        num_inference_steps: int = 0,
        inference_steps_per_second: float = 0,
        ttft: Optional[float] = None,
        job_id: Optional[str] = None,
        video_path: Optional[str] = None,
        prompt: Optional[str] = None,
    ):
        super().__init__(status, elapsed)
        self.num_inference_steps = num_inference_steps
        self.inference_steps_per_second = inference_steps_per_second
        self.ttft = ttft
        self.job_id = job_id
        self.video_path = video_path
        self.prompt = prompt

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "elapsed": self.elapsed,
            "num_inference_steps": self.num_inference_steps,
            "inference_steps_per_second": self.inference_steps_per_second,
            "ttft": self.ttft,
            "job_id": self.job_id,
            "video_path": self.video_path,
            "prompt": self.prompt,
        }
