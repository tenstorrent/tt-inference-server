#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Standalone diarization / VAD entry point that runs inside the audio_venv
# (torch 2.7.x + whisperx + pyannote + silero_vad). It is invoked as a
# subprocess from `utils/audio_manager.py` which runs inside the main venv
# (torch 2.10 + vLLM 0.18.1). Splitting the two avoids the torch ABI clash
# between vLLM 0.18 (>=2.10) and whisperx/pyannote (~2.7.x).
#
# Communication contract with the parent process:
#   - audio is passed as a .npy file path
#   - results are written as JSON to an output path
#   - the schema is defined by `DiarizeResponse.to_dict()` below
#
# Usage:
#   python diarize.py --audio <path.npy> --output <path.json> --mode {diarize,vad}
#                     [--model-name <name>] [--hf-token <token>]

from __future__ import annotations

import argparse
import json
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"


class OperationMode(str, Enum):
    """Supported audio processing operations exposed by this script."""

    DIARIZE = "diarize"
    VAD = "vad"

    @classmethod
    def values(cls) -> list[str]:
        return [m.value for m in cls]


@dataclass
class AudioSegment:
    """A timed audio segment with an optional speaker label."""

    start: float
    end: float
    speaker: str | None = None

    def to_dict(self) -> dict:
        result: dict = {"start": self.start, "end": self.end}
        if self.speaker is not None:
            result["speaker"] = self.speaker
        return result


@dataclass
class DiarizeResponse:
    """Response payload written by the subprocess and parsed by the parent."""

    segments: list[AudioSegment] = field(default_factory=list)
    status: str = "success"
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "segments": [seg.to_dict() for seg in self.segments],
            "status": self.status,
            "error": self.error,
        }

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def error_response(cls, message: str) -> DiarizeResponse:
        return cls(status="error", error=message)


class TorchLoadPatcher:
    """Restore the pre-PyTorch-2.6 `torch.load` default (`weights_only=False`).

    PyTorch 2.6 flipped the default of `torch.load`'s `weights_only` from
    `False` to `True`, and `lightning>=2.6` propagated that into
    `lightning_fabric.utilities.cloud_io._load`. pyannote-audio<=3.4.0
    (transitively pinned by whisperx==3.4.3) does not forward
    `weights_only=False`, so pyannote checkpoints, which legitimately contain
    non-tensor objects, fail to unpickle.

    Audio worker only loads checkpoints from trusted sources (gated HF repos
    accessed via HF_TOKEN and bundled weights), so the strict default offers
    no real safety here. We treat both "kwarg missing" and "kwarg present but
    None" as "use legacy default": lightning's `pl_load` explicitly forwards
    `weights_only=None`, which a plain `dict.setdefault()` would miss.
    Callers passing `weights_only=True/False` explicitly are respected.
    """

    def __init__(self, torch_module):
        self._torch = torch_module
        self._original_load = torch_module.load

    def install(self) -> None:
        self._torch.load = self._patched_load

    def _patched_load(self, *args, **kwargs):
        if kwargs.get("weights_only") is None:
            kwargs["weights_only"] = False
        return self._original_load(*args, **kwargs)


class AudioProcessor(ABC):
    """Base class for an audio analysis pipeline run by this script."""

    @abstractmethod
    def process(self, audio_array: np.ndarray) -> list[AudioSegment]:
        """Process the audio array and return a list of segments."""


class DiarizationProcessor(AudioProcessor):
    """Speaker diarization via whisperx's pyannote-backed pipeline."""

    def __init__(
        self,
        model_name: str,
        hf_token: str | None = None,
        device: str = "cpu",
    ):
        self._model_name = model_name
        self._hf_token = hf_token
        self._device = device
        self._pipeline = None

    def _ensure_pipeline(self) -> None:
        if self._pipeline is not None:
            return

        import torch

        TorchLoadPatcher(torch).install()

        # whisperx is only installed in the audio_venv; the linter running in
        # the main venv will flag this import as unresolved which is expected.
        from whisperx.diarize import DiarizationPipeline  # type: ignore[import-not-found]

        self._pipeline = DiarizationPipeline(
            model_name=self._model_name,
            use_auth_token=self._hf_token,
            device=self._device,
        )

    def process(self, audio_array: np.ndarray) -> list[AudioSegment]:
        self._ensure_pipeline()
        result = self._pipeline(audio_array)

        segments: list[AudioSegment] = []
        for _, row in result.iterrows():
            segments.append(
                AudioSegment(
                    start=float(row.get("start", 0)),
                    end=float(row.get("end", 0)),
                    speaker=str(row.get("speaker", "SPEAKER_00")),
                )
            )
        return segments


class VADProcessor(AudioProcessor):
    """Voice Activity Detection via Silero VAD."""

    def __init__(self, threshold: float = 0.5):
        self._threshold = threshold
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        # silero_vad is only installed in the audio_venv; the linter running
        # in the main venv will flag this import as unresolved which is
        # expected.
        from silero_vad import load_silero_vad  # type: ignore[import-not-found]

        self._model = load_silero_vad()

    def process(self, audio_array: np.ndarray) -> list[AudioSegment]:
        import torch
        from silero_vad import get_speech_timestamps  # type: ignore[import-not-found]

        self._ensure_model()

        audio_tensor = torch.from_numpy(audio_array).float()
        vad_segments = get_speech_timestamps(
            audio_tensor,
            self._model,
            threshold=self._threshold,
            return_seconds=True,
        )

        return [
            AudioSegment(
                start=float(seg.get("start", 0)),
                end=float(seg.get("end", 0)),
            )
            for seg in vad_segments
        ]


class ProcessorFactory:
    """Builds the right `AudioProcessor` for a given `OperationMode`."""

    def __init__(self, model_name: str, hf_token: str | None):
        self._model_name = model_name
        self._hf_token = hf_token

    def create(self, mode: OperationMode) -> AudioProcessor:
        if mode is OperationMode.DIARIZE:
            return DiarizationProcessor(
                model_name=self._model_name,
                hf_token=self._hf_token,
            )
        if mode is OperationMode.VAD:
            return VADProcessor()
        # Defensive: OperationMode is a closed enum so this is unreachable
        # unless someone adds a value without updating the factory.
        raise ValueError(f"Unsupported operation mode: {mode}")


class DiarizeService:
    """Runs one diarize/VAD job: load audio, process, write JSON response."""

    def __init__(
        self,
        audio_file: Path,
        output_file: Path,
        mode: OperationMode,
        processor_factory: ProcessorFactory,
    ):
        self._audio_file = audio_file
        self._output_file = output_file
        self._mode = mode
        self._processor_factory = processor_factory

    def run(self) -> int:
        import numpy as np

        try:
            audio_array = np.load(self._audio_file)
            processor = self._processor_factory.create(self._mode)
            segments = processor.process(audio_array)
            response = DiarizeResponse(segments=segments)
        except Exception as e:
            response = DiarizeResponse.error_response(str(e))

        response.save(self._output_file)
        return 0 if response.status == "success" else 1


class DiarizeCLI:
    """Argparse-based CLI wrapper around `DiarizeService`.

    Encapsulating argv parsing here keeps `__main__` a one-liner and makes
    the entry point unit-testable (just instantiate with an argv list).
    """

    def __init__(self, argv: list[str] | None = None):
        self._argv = argv

    def _build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description=(
                "Run diarization or VAD on a numpy audio file and write a "
                "JSON response. Intended to be invoked as a subprocess from "
                "the main media-server venv."
            )
        )
        parser.add_argument(
            "--audio",
            required=True,
            type=Path,
            help="Path to a .npy file containing a 1-D float32 audio array.",
        )
        parser.add_argument(
            "--output",
            required=True,
            type=Path,
            help="Path where the JSON response will be written.",
        )
        parser.add_argument(
            "--mode",
            required=True,
            choices=OperationMode.values(),
            help="Which operation to run.",
        )
        parser.add_argument(
            "--model-name",
            default=DEFAULT_DIARIZATION_MODEL,
            help=(
                "HF model id for the diarization pipeline "
                f"(default: {DEFAULT_DIARIZATION_MODEL}). Ignored for VAD."
            ),
        )
        parser.add_argument(
            "--hf-token",
            default=None,
            help=(
                "Hugging Face token for gated pyannote models. Falls back to "
                "the HF_TOKEN environment variable when omitted."
            ),
        )
        return parser

    def build_service(self) -> DiarizeService:
        args = self._build_parser().parse_args(self._argv)
        hf_token = args.hf_token or os.getenv("HF_TOKEN")
        factory = ProcessorFactory(
            model_name=args.model_name,
            hf_token=hf_token,
        )
        return DiarizeService(
            audio_file=args.audio,
            output_file=args.output,
            mode=OperationMode(args.mode),
            processor_factory=factory,
        )

    def run(self) -> int:
        return self.build_service().run()


if __name__ == "__main__":
    sys.exit(DiarizeCLI().run())
