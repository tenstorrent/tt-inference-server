# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Per-call result objects emitted by the media client benchmark loops."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseTestStatus(ABC):
    """Base class for all test status objects."""

    def __init__(self, status: bool, elapsed: float):
        self.status = status
        self.elapsed = elapsed

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        pass


class ImageGenerationTestStatus(BaseTestStatus):
    """Test status for image generation models (SDXL, SD3.5, Flux, etc.)."""

    def __init__(
        self,
        status: bool,
        elapsed: float,
        num_inference_steps: int = 0,
        inference_steps_per_second: float = 0,
        base64image: Optional[str] = None,
        prompt: Optional[str] = None,
    ):
        super().__init__(status, elapsed)
        self.num_inference_steps = num_inference_steps
        self.inference_steps_per_second = inference_steps_per_second
        self.base64image = base64image
        self.prompt = prompt

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "elapsed": self.elapsed,
            "num_inference_steps": self.num_inference_steps,
            "inference_steps_per_second": self.inference_steps_per_second,
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
    """Test status for CNN models (ResNet, MobileNetV2, etc.)."""

    def __init__(self, status: bool, elapsed: float):
        super().__init__(status, elapsed)

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "elapsed": self.elapsed}


class EmbeddingTestStatus(BaseTestStatus):
    """Test status for embedding models."""

    def __init__(self, status: bool, elapsed: float):
        super().__init__(status, elapsed)

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "elapsed": self.elapsed}


class TtsTestStatus(BaseTestStatus):
    """Test status for text-to-speech models."""

    def __init__(
        self,
        status: bool,
        elapsed: float,
        latency: Optional[float] = None,
        rtr: Optional[float] = None,
    ):
        super().__init__(status, elapsed)
        self.latency = latency
        self.rtr = rtr

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "elapsed": self.elapsed,
            "latency": self.latency,
            "rtr": self.rtr,
        }


class VideoGenerationTestStatus(BaseTestStatus):
    """Test status for video generation models (Mochi, WAN, etc.)."""

    def __init__(
        self,
        status: bool,
        elapsed: float,
        num_inference_steps: int = 0,
        inference_steps_per_second: float = 0,
        job_id: Optional[str] = None,
        video_path: Optional[str] = None,
    ):
        super().__init__(status, elapsed)
        self.num_inference_steps = num_inference_steps
        self.inference_steps_per_second = inference_steps_per_second
        self.job_id = job_id
        self.video_path = video_path

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "elapsed": self.elapsed,
            "num_inference_steps": self.num_inference_steps,
            "inference_steps_per_second": self.inference_steps_per_second,
            "job_id": self.job_id,
            "video_path": self.video_path,
        }
