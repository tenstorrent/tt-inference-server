#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Standalone diarization / VAD entry point that runs inside the audio_venv
# (torch 2.7.x + whisperx + pyannote + silero_vad). It is spawned as a
# subprocess from `utils/audio_manager.py` which runs inside the main venv
# (torch 2.10 + vLLM 0.18.1). Splitting the two avoids the torch ABI clash
# between vLLM 0.18 (>=2.10) and whisperx/pyannote (~2.7.x).
#
# Two run modes are supported:
#
#   --serve    (production)
#       Long-lived worker. Emits a ready signal as soon as it starts reading
#       stdin, then processes line-delimited JSON requests forever. Models
#       (pyannote / silero_vad) are lazy-loaded on first use of each mode --
#       torch itself is not imported until the first request. This avoids
#       paying the model-load cost per request and is what `AudioVenvWorker`
#       in the main venv talks to.
#
#       The eager preload was intentionally dropped: `CpuWorkloadHandler`
#       already sends a warmup task per worker at server startup which flows
#       through `AudioManager` -> this server, exercising both modes and
#       triggering the lazy load exactly once before any real user traffic.
#
#       Ready signal:  {"status": "ready"}
#       Request:       {"id": "<opaque>", "mode": "diarize"|"vad",
#                       "audio_path": "/tmp/xxx.npy"}
#       Response:      {"id": "<opaque>", "status": "success"|"error",
#                       "segments": [...], "error": null | "..."}
#       Shutdown:      the parent closes stdin (EOF); the server exits cleanly.
#
#   one-shot   (debugging / ad-hoc)
#       Runs a single request from CLI flags and writes the JSON response to
#       a file. Useful for reproducing an issue without spinning up the
#       server. Kept for operator convenience.
#
# Usage:
#   python diarize.py --serve [--model-name <name>] [--hf-token <token>]
#   python diarize.py --audio <path.npy> --output <path.json>
#                     --mode {diarize,vad} [--model-name <name>] [--hf-token <token>]

from __future__ import annotations

import argparse
import json
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import IO, TYPE_CHECKING

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
    """Response payload written by the worker and parsed by the parent."""

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
        """Process the audio array and return a list of segments.

        Implementations should lazy-load their model on first call (via a
        `_ensure_*` helper). Deferring the load means:

        * the parent `AudioVenvWorker.start()` doesn't block on torch +
          model download;
        * the load runs during the CpuWorkloadHandler warmup task, which
          already exists and would exercise the pipeline anyway;
        * a mode that never receives traffic (e.g. VAD-only server that
          skips diarize) never pays the cost.
        """


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
    """One-shot mode: load a `.npy`, run one operation, write a JSON response
    and exit. Kept for ad-hoc / debugging use; production traffic goes via
    :class:`DiarizeServer` instead."""

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


class DiarizeServer:
    """Persistent audio worker.

    :meth:`serve` emits a single JSON ``{"status": "ready"}`` line as soon
    as it is ready to read requests, then loops on stdin reading one JSON
    request per line. Processors (and their heavy torch / pyannote /
    silero_vad dependencies) are constructed lazily on first use per mode
    and cached — subsequent requests hit the already-loaded model.

    The parent closes stdin (EOF) to trigger a clean shutdown.

    Wire protocol (all requests and responses are single JSON lines):

    * Request  ``{"id": <opaque>, "mode": "diarize"|"vad", "audio_path": "..."}``
    * Response ``{"id": <opaque>, "status": "success"|"error",
                  "segments": [...], "error": null | "..."}``
    """

    def __init__(
        self,
        processor_factory: ProcessorFactory,
        stdin: IO[str] | None = None,
        stdout: IO[str] | None = None,
        stderr: IO[str] | None = None,
    ):
        self._factory = processor_factory
        self._stdin = stdin if stdin is not None else sys.stdin
        self._stdout = stdout if stdout is not None else sys.stdout
        self._stderr = stderr if stderr is not None else sys.stderr
        self._processors: dict[OperationMode, AudioProcessor] = {}

    def _emit(self, payload: dict) -> None:
        self._stdout.write(json.dumps(payload) + "\n")
        self._stdout.flush()

    def _get_or_build_processor(self, mode: OperationMode) -> AudioProcessor:
        processor = self._processors.get(mode)
        if processor is None:
            processor = self._factory.create(mode)
            self._processors[mode] = processor
        return processor

    def _handle_request(self, req: dict) -> dict:
        req_id = req.get("id")

        try:
            mode = OperationMode(req["mode"])
        except (KeyError, ValueError):
            return {
                "id": req_id,
                "status": "error",
                "error": f"invalid mode: {req.get('mode')!r}",
                "segments": [],
            }

        audio_path = req.get("audio_path")
        if not audio_path:
            return {
                "id": req_id,
                "status": "error",
                "error": "missing audio_path",
                "segments": [],
            }

        try:
            import numpy as np

            audio_array = np.load(audio_path)
            processor = self._get_or_build_processor(mode)
            segments = processor.process(audio_array)
        except Exception as e:
            return {
                "id": req_id,
                "status": "error",
                "error": str(e),
                "segments": [],
            }

        response = DiarizeResponse(segments=segments)
        payload = response.to_dict()
        payload["id"] = req_id
        return payload

    def serve(self) -> int:
        """Blocking serve loop. Returns process exit code."""
        self._emit({"status": "ready"})

        for line in self._stdin:
            line = line.strip()
            if not line:
                continue

            try:
                req = json.loads(line)
            except json.JSONDecodeError as e:
                self._emit({"status": "error", "error": f"malformed request: {e}"})
                continue

            self._emit(self._handle_request(req))

        return 0


class DiarizeCLI:
    """Argparse-based CLI wrapper. Dispatches to either the one-shot
    :class:`DiarizeService` or the long-lived :class:`DiarizeServer`.

    Encapsulating argv parsing here keeps ``__main__`` a one-liner and makes
    both entry points unit-testable (just instantiate with an argv list)."""

    def __init__(self, argv: list[str] | None = None):
        self._argv = argv

    def _build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description=(
                "Run diarization or VAD as a persistent worker (--serve) or "
                "as a single one-shot invocation (--audio/--output/--mode). "
                "Invoked from the main media-server venv, runs inside the "
                "audio venv."
            )
        )
        parser.add_argument(
            "--serve",
            action="store_true",
            help=(
                "Run as a long-lived worker: preload models, then read "
                "line-delimited JSON requests from stdin and write responses "
                "to stdout until stdin is closed."
            ),
        )
        parser.add_argument(
            "--audio",
            type=Path,
            help="[one-shot] Path to a .npy file with the audio array.",
        )
        parser.add_argument(
            "--output",
            type=Path,
            help="[one-shot] Path where the JSON response will be written.",
        )
        parser.add_argument(
            "--mode",
            choices=OperationMode.values(),
            help="[one-shot] Which operation to run.",
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
                "Hugging Face token for gated pyannote models. Falls back "
                "to the HF_TOKEN environment variable when omitted."
            ),
        )
        return parser

    def _factory(self, args: argparse.Namespace) -> ProcessorFactory:
        hf_token = args.hf_token or os.getenv("HF_TOKEN")
        return ProcessorFactory(model_name=args.model_name, hf_token=hf_token)

    def build_serve(self) -> DiarizeServer:
        args = self._build_parser().parse_args(self._argv)
        if not args.serve:
            raise SystemExit(
                "build_serve called without --serve; use build_service for one-shot"
            )
        return DiarizeServer(processor_factory=self._factory(args))

    def build_service(self) -> DiarizeService:
        args = self._build_parser().parse_args(self._argv)
        if args.serve:
            raise SystemExit(
                "build_service called with --serve; use build_serve for server mode"
            )
        missing = [
            name
            for name, value in (
                ("--audio", args.audio),
                ("--output", args.output),
                ("--mode", args.mode),
            )
            if value is None
        ]
        if missing:
            raise SystemExit(
                f"one-shot mode requires {', '.join(missing)} "
                "(or use --serve for long-lived worker mode)"
            )
        return DiarizeService(
            audio_file=args.audio,
            output_file=args.output,
            mode=OperationMode(args.mode),
            processor_factory=self._factory(args),
        )

    def run(self) -> int:
        args = self._build_parser().parse_args(self._argv)
        if args.serve:
            return DiarizeServer(processor_factory=self._factory(args)).serve()
        # Re-use build_service for its argument validation
        return self.build_service().run()


if __name__ == "__main__":
    sys.exit(DiarizeCLI().run())
