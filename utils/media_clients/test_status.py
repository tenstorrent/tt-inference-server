# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

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
