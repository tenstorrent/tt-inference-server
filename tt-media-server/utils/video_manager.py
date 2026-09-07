# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from telemetry.telemetry_client import get_telemetry_client

from utils.decorators import log_execution_time
from utils.logger import TTLogger

_MIN_CRF = 0
_MAX_CRF = 51
_FFMPEG_ENCODE_TIMEOUT_S = 600
_FFMPEG_REMUX_TIMEOUT_S = 60
_VIDEO_OUTPUT_DIR = Path(os.environ.get("TT_VIDEO_OUTPUT_DIR", "/tmp/videos"))
_VALID_CHANNEL_COUNTS = (1, 3, 4)
_RGB_CHANNELS = 3
_MAX_PIXEL_VALUE = 255.0
_NORMALIZED_RANGE_MAX = 1.0
# PyAV reports container-level durations in microseconds (av.time_base).
_AV_TIME_BASE = 1_000_000


@dataclass(frozen=True)
class VideoAudioResult:
    """A runner's raw (un-encoded) output when the model emits a soundtrack.

    An audio model returns this in place of a bare frame array so the encoder
    muxes video + audio via ``export_to_mp4_with_audio``. Fields map 1:1 to that
    exporter's arguments, so any future audio runner can reuse this contract.
    """

    frames: NDArray
    audio: NDArray
    sampling_rate: int
    fps: int = 16


class VideoManager:
    """MP4 export via FFmpeg subprocess pipe (raw RGB → libx264)."""

    def __init__(self):
        self._logger = TTLogger()

    @log_execution_time("Exporting video to MP4")
    def export_to_mp4(self, frames: NDArray, fps: int = 16) -> str:
        """
        Export frames to MP4 (H.264 via ffmpeg).

        Frames are streamed to ffmpeg one at a time so encoding overlaps with
        Python-side dtype conversion, avoiding large temporary allocations.

        Env (optional):
            TT_VIDEO_EXPORT_CRF: 0–51, lower = better quality. Default 23.
            TT_VIDEO_EXPORT_PRESET: ultrafast … veryslow. Default medium.
        """
        if hasattr(frames, "frames"):
            frames = frames.frames

        _VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(_VIDEO_OUTPUT_DIR / f"{uuid.uuid4()}.mp4")

        crf = int(os.environ.get("TT_VIDEO_EXPORT_CRF", "23"))
        crf = max(_MIN_CRF, min(_MAX_CRF, crf))
        preset = os.environ.get("TT_VIDEO_EXPORT_PRESET", "ultrafast").strip()

        frames = _normalize_shape(frames)
        frames = _normalize_channels(frames)

        frame_count, height, width = frames.shape[0], frames.shape[1], frames.shape[2]
        started = time.monotonic()

        try:
            cmd = self._build_encode_cmd(frames, output_path, fps, crf, preset)
            self._stream_to_ffmpeg(cmd, frames)
            self._record_encode(started, frame_count, width, height, status=True)
            return output_path

        except Exception as e:
            self._record_encode(started, frame_count, width, height, status=False)
            self._logger.error(f"Video export failed: {e}")
            raise RuntimeError(f"Failed to export video: {e}") from e

    @log_execution_time("Processing frames for export")
    def _process_frames_for_export(self, frames: NDArray) -> NDArray[np.uint8]:
        """Normalize to contiguous uint8 (N, H, W, 3) for rawvideo rgb24."""
        frames = _normalize_shape(frames)
        frames = _normalize_channels(frames)
        if not frames.flags["C_CONTIGUOUS"]:
            frames = np.ascontiguousarray(frames)
        return _normalize_dtype(frames)

    @log_execution_time("Exporting video with audio to MP4")
    def export_to_mp4_with_audio(
        self,
        frames: NDArray,
        audio: NDArray,
        sampling_rate: int,
        fps: int = 16,
    ) -> str:
        """Export frames plus a soundtrack to a single muxed MP4.

        `audio` is `(channels, samples)` float in [-1, 1]. It goes to a temp WAV rather than a
        pipe because ffmpeg reads only one stream from stdin. `-shortest` trims the two streams
        to the common length; they round independently and differ slightly.
        """
        if hasattr(frames, "frames"):
            frames = frames.frames

        _VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = uuid.uuid4()
        output_path = str(_VIDEO_OUTPUT_DIR / f"{stem}.mp4")
        silent_path = str(_VIDEO_OUTPUT_DIR / f"{stem}_silent.mp4")
        wav_path = str(_VIDEO_OUTPUT_DIR / f"{stem}.wav")

        crf = int(os.environ.get("TT_VIDEO_EXPORT_CRF", "23"))
        crf = max(_MIN_CRF, min(_MAX_CRF, crf))
        preset = os.environ.get("TT_VIDEO_EXPORT_PRESET", "ultrafast").strip()

        try:
            processed = self._process_frames_for_export(frames)
            self._run_ffmpeg(
                self._build_encode_cmd(processed, silent_path, fps, crf, preset),
                stdin_data=memoryview(processed),
            )
            self._write_wav(audio, sampling_rate, wav_path)
            self._run_ffmpeg(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    silent_path,
                    "-i",
                    wav_path,
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-shortest",
                    "-movflags",
                    "+faststart",
                    output_path,
                ]
            )
            return output_path
        except Exception as e:
            self._logger.error(f"Video+audio export failed: {e}")
            raise RuntimeError(f"Failed to export video with audio: {e}") from e
        finally:
            for path in (silent_path, wav_path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    @staticmethod
    def _write_wav(audio: NDArray, sampling_rate: int, path: str) -> None:
        """Write `(channels, samples)` float audio in [-1, 1] as 16-bit interleaved PCM."""
        import wave

        samples = np.asarray(audio)
        if samples.ndim == 1:
            samples = samples[None, :]
        if samples.ndim == 3 and samples.shape[0] == 1:
            samples = samples[0]
        if samples.ndim != 2:
            raise ValueError(
                f"expected audio shaped (channels, samples), got {samples.shape}"
            )

        channels = samples.shape[0]
        # Clip before scaling; out-of-range would wrap to the opposite int16 rail.
        interleaved = np.clip(samples.T, -1.0, 1.0)
        pcm = (interleaved * 32767.0).astype("<i2")

        with wave.open(path, "wb") as handle:
            handle.setnchannels(channels)
            handle.setsampwidth(2)
            handle.setframerate(int(sampling_rate))
            handle.writeframes(pcm.tobytes())

    def _record_encode(
        self,
        started: float,
        num_frames: int,
        width: int,
        height: int,
        status: bool,
    ) -> None:
        """Report encode cost to Prometheus without ever failing the export.

        Encoding runs in whichever process owns the frames — a CPU
        postprocessing worker, an in-process runner, or (for SP_RUNNER) the
        external peer. Only the first two are in the server's process tree and
        therefore visible on /metrics; the peer's observations land nowhere,
        which is why this is best-effort and silent on error.
        """
        try:
            get_telemetry_client().record_video_encode(
                duration=time.monotonic() - started,
                num_frames=num_frames,
                width=width,
                height=height,
                status=status,
            )
        except Exception as e:  # pragma: no cover - telemetry must never raise
            self._logger.warning(f"Failed to record video encode telemetry: {e}")

    @staticmethod
    def _build_encode_cmd(
        frames: NDArray, output_path: str, fps: int, crf: int, preset: str
    ) -> list[str]:
        """Build the ffmpeg rawvideo → libx264 command list."""
        _, height, width, channels = frames.shape
        if channels != _RGB_CHANNELS:
            raise ValueError(
                f"Expected {_RGB_CHANNELS} RGB channels after processing, got {channels}"
            )

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(fps),
            "-i",
            "-",
        ]

        if crf == 0:
            cmd.extend(["-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p"])
        else:
            cmd.extend(
                [
                    "-c:v",
                    "libx264",
                    "-crf",
                    str(crf),
                    "-pix_fmt",
                    "yuv420p",
                    "-tune",
                    "film",
                    "-profile:v",
                    "high",
                    "-level",
                    "4.2",
                ]
            )

        if preset:
            cmd.extend(["-preset", preset])

        cmd.extend(["-movflags", "+faststart", output_path])
        return cmd

    @staticmethod
    def _ffmpeg_binary() -> str:
        """ffmpeg from PATH (the image installs it), else the `imageio_ffmpeg` wheel."""
        found = shutil.which("ffmpeg")
        if found:
            return found
        try:
            import imageio_ffmpeg

            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as e:
            raise RuntimeError(
                "ffmpeg not found on PATH and imageio_ffmpeg is not available; "
                "install ffmpeg or `pip install imageio-ffmpeg`"
            ) from e

    @classmethod
    def _run_ffmpeg(
        cls,
        cmd: list[str],
        stdin_data: bytes | None = None,
        timeout: int = _FFMPEG_ENCODE_TIMEOUT_S,
    ) -> None:
        """Execute an ffmpeg command, raising on failure or timeout."""
        if cmd and cmd[0] == "ffmpeg":
            cmd = [cls._ffmpeg_binary(), *cmd[1:]]
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if stdin_data else None,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        try:
            _, stderr = process.communicate(input=stdin_data, timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            raise RuntimeError("FFmpeg export timed out") from None

        if process.returncode != 0:
            error_msg = stderr.decode(errors="replace") if stderr else "Unknown error"
            raise RuntimeError(f"FFmpeg failed: {error_msg}")

    @staticmethod
    def _stream_to_ffmpeg(
        cmd: list[str],
        frames: NDArray,
        timeout: int = _FFMPEG_ENCODE_TIMEOUT_S,
    ) -> None:
        """Stream frames one-by-one to ffmpeg, converting dtype per-frame."""
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        try:
            for frame in frames:
                if frame.dtype != np.uint8:
                    frame = _normalize_dtype_single(frame)
                process.stdin.write(frame.tobytes())
            process.stdin.close()
            stderr = process.stderr.read()
            rc = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise RuntimeError("FFmpeg export timed out") from None
        except Exception:
            process.kill()
            process.wait()
            raise

        if rc != 0:
            error_msg = stderr.decode(errors="replace") if stderr else "Unknown error"
            raise RuntimeError(f"FFmpeg failed: {error_msg}")

    @classmethod
    def ensure_faststart(cls, input_path: str, output_path: str) -> None:
        """Rewrites the MP4 file with -movflags faststart using ffmpeg."""
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-c",
            "copy",
            "-movflags",
            "faststart",
            output_path,
        ]
        cls._run_ffmpeg(cmd, timeout=_FFMPEG_REMUX_TIMEOUT_S)


def _normalize_shape(frames: NDArray) -> NDArray:
    """Squeeze batch dim and validate 4D (N, H, W, C)."""
    if frames.ndim == 5:
        frames = frames[0]

    if frames.ndim != 4:
        raise ValueError(f"Unexpected frame dimensions: {frames.shape}")

    return frames


def _normalize_channels(frames: NDArray) -> NDArray:
    """Convert grayscale or RGBA to RGB."""
    _, _, _, channels = frames.shape

    if channels not in _VALID_CHANNEL_COUNTS:
        raise ValueError(f"Frames have {channels} channels, expected 1, 3, or 4")

    if channels == 1:
        return np.repeat(frames, _RGB_CHANNELS, axis=-1)
    if channels == 4:
        return frames[..., :_RGB_CHANNELS]

    return frames


def _normalize_dtype(frames: NDArray) -> NDArray[np.uint8]:
    """Convert a whole (N, H, W, C) stack to uint8, handling float [0,1] and [0,255].

    The batch counterpart to `_normalize_dtype_single`, and deliberately the same rule: the
    range is decided from the max over the *whole* stack, not per frame, so a dark frame in a
    [0,255] video cannot be mistaken for normalized data and brightened on its own.
    """
    if frames.dtype == np.uint8:
        return frames

    if frames.dtype in (np.float32, np.float64):
        max_val = float(np.max(frames)) if frames.size else 0.0
        if max_val <= _NORMALIZED_RANGE_MAX:
            return (frames * _MAX_PIXEL_VALUE).clip(0, 255).astype(np.uint8)
        return frames.clip(0, 255).astype(np.uint8)

    return frames.clip(0, 255).astype(np.uint8)


def _normalize_dtype_single(frame: NDArray) -> NDArray[np.uint8]:
    """Convert a single frame (H, W, C) to uint8."""
    if frame.dtype in (np.float32, np.float64):
        max_val = float(np.max(frame)) if frame.size else 0.0
        if max_val <= _NORMALIZED_RANGE_MAX:
            return (frame * _MAX_PIXEL_VALUE).clip(0, 255).astype(np.uint8)
        return frame.clip(0, 255).astype(np.uint8)

    return frame.clip(0, 255).astype(np.uint8)


@dataclass(frozen=True)
class VideoProbe:
    """Shape and length facts read back off a produced mp4.

    Every field is optional. The server side of a multihost video deployment
    (SP_RUNNER) receives only a file path from its peer — it never sees the
    frame tensors — so probing the container is the only way to learn what was
    actually generated. A probe that partly fails yields partly-populated
    stats rather than nothing, and callers skip the unknown fields.
    """

    size_bytes: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    num_frames: Optional[int] = None
    duration_seconds: Optional[float] = None


def probe_video(path: str) -> VideoProbe:
    """Read width/height/frames/duration from an mp4 without decoding it.

    Uses PyAV (already a dependency for the LTX audio-video muxing) so this
    costs one container-header read rather than an ffprobe subprocess. Never
    raises: a metrics helper must not be able to fail a request that already
    produced a valid video.
    """
    if not path or not isinstance(path, str):
        return VideoProbe()

    try:
        size_bytes = os.path.getsize(path)
    except OSError:
        return VideoProbe()

    try:
        import av
    except ImportError:
        return VideoProbe(size_bytes=size_bytes)

    try:
        with av.open(path) as container:
            if not container.streams.video:
                return VideoProbe(size_bytes=size_bytes)
            stream = container.streams.video[0]

            width = getattr(stream, "width", None) or None
            height = getattr(stream, "height", None) or None

            duration = None
            if stream.duration and stream.time_base:
                duration = float(stream.duration * stream.time_base)
            elif container.duration:
                duration = container.duration / _AV_TIME_BASE

            # ``stream.frames`` is authoritative for a well-formed mp4 but is 0
            # for streamed/fragmented containers; fall back to duration x fps.
            num_frames = stream.frames or None
            if num_frames is None and duration:
                rate = stream.average_rate or stream.guessed_rate
                if rate:
                    num_frames = round(duration * float(rate)) or None

            return VideoProbe(
                size_bytes=size_bytes,
                width=width,
                height=height,
                num_frames=num_frames,
                duration_seconds=duration,
            )
    except Exception:
        return VideoProbe(size_bytes=size_bytes)
