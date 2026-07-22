# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import uuid
import wave
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from utils.decorators import log_execution_time
from utils.logger import TTLogger

_MIN_CRF = 0
_MAX_CRF = 51
_FFMPEG_ENCODE_TIMEOUT_S = 600
_FFMPEG_REMUX_TIMEOUT_S = 60
_VIDEO_OUTPUT_DIR = Path("/tmp/videos")
_VALID_CHANNEL_COUNTS = (1, 3, 4)
_RGB_CHANNELS = 3
_MAX_PIXEL_VALUE = 255.0
_NORMALIZED_RANGE_MAX = 1.0


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

        try:
            cmd = self._build_encode_cmd(frames, output_path, fps, crf, preset)
            self._stream_to_ffmpeg(cmd, frames)
            return output_path

        except Exception as e:
            self._logger.error(f"Video export failed: {e}")
            raise RuntimeError(f"Failed to export video: {e}") from e

    @log_execution_time("Exporting RGB-planar video+audio to MP4")
    def export_rgb_planar_to_mp4_with_audio(
        self,
        frames,
        audio_waveform,
        sample_rate: int,
        fps: int = 24,
    ) -> str:
        """Export planar RGB frames + audio to an AV MP4 (H.264 yuv420p + AAC).

        ``frames`` is RGB planar, (B, 3, T, H, W) or (3, T, H, W) uint8 (planes R, G, B). It streams
        as ffmpeg ``gbrp`` (channels-first, so the device gather has a large innermost dim); libx264
        does RGB→yuv420p. Audio is muxed from a temp WAV.
        """
        arr = frames.frames if hasattr(frames, "frames") else frames
        arr = np.asarray(arr)
        if arr.ndim == 5:
            arr = arr[0]
        if arr.ndim != 4 or arr.shape[0] != 3:
            raise ValueError(f"Expected RGB planar (3, T, H, W), got {arr.shape}")
        _, t_frames, height, width = arr.shape

        _VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(_VIDEO_OUTPUT_DIR / f"{uuid.uuid4()}.mp4")
        crf = max(_MIN_CRF, min(_MAX_CRF, int(os.environ.get("TT_VIDEO_EXPORT_CRF", "23"))))
        preset = os.environ.get("TT_VIDEO_EXPORT_PRESET", "ultrafast").strip()

        wav_path = _write_temp_wav(audio_waveform, sample_rate)
        try:
            cmd = [
                "ffmpeg", "-y", "-nostats", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "gbrp",
                "-s", f"{width}x{height}", "-r", str(fps),
                "-i", "-",
                "-i", wav_path,
                "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p",
            ]
            if preset:
                cmd += ["-preset", preset]
            cmd += ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", output_path]
            # gbrp plane order is G, B, R; frames are R, G, B.
            self._stream_planar_to_ffmpeg(cmd, arr, t_frames, plane_order=(1, 2, 0))
            return output_path
        except Exception as e:
            self._logger.error(f"RGB-planar export failed: {e}")
            raise RuntimeError(f"Failed to export RGB-planar video+audio: {e}") from e
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass

    @staticmethod
    def _run_streaming_ffmpeg(cmd, write_stdin, timeout: int = _FFMPEG_ENCODE_TIMEOUT_S) -> None:
        """Run ffmpeg while ``write_stdin(pipe)`` feeds its stdin.

        stderr is drained on a separate thread throughout the write: for the
        largest payloads (1080p planar) ffmpeg's stderr can outrun a fixed pipe
        buffer, and reading it only after the full stdin write would let a full
        stderr pipe stall ffmpeg's stdin reads and deadlock the writer.
        """
        process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        stderr_box: list[bytes] = []
        drainer = threading.Thread(
            target=lambda: stderr_box.append(process.stderr.read()), daemon=True
        )
        drainer.start()
        try:
            write_stdin(process.stdin)
            process.stdin.close()
            rc = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            drainer.join(timeout=5)
            raise RuntimeError("FFmpeg export timed out") from None
        except Exception:
            process.kill()
            process.wait()
            drainer.join(timeout=5)
            raise
        drainer.join(timeout=5)
        if rc != 0:
            stderr = b"".join(c for c in stderr_box if c)
            error_msg = stderr.decode(errors="replace") if stderr else "Unknown error"
            raise RuntimeError(f"FFmpeg failed: {error_msg}")

    @classmethod
    def _stream_planar_to_ffmpeg(
        cls,
        cmd: list[str],
        arr: NDArray,
        t_frames: int,
        plane_order: tuple[int, int, int] = (0, 1, 2),
        timeout: int = _FFMPEG_ENCODE_TIMEOUT_S,
    ) -> None:
        """Stream planar frames (arr[c, t] = (H, W) per plane) to ffmpeg in ``plane_order``."""

        def write_stdin(stdin):
            for t in range(t_frames):
                for c in plane_order:
                    # arr[c, t] is already C-contiguous; write its buffer directly to skip a
                    # ~900MB/frame-set tobytes() copy on the export critical path.
                    stdin.write(np.ascontiguousarray(arr[c, t]))

        cls._run_streaming_ffmpeg(cmd, write_stdin, timeout)

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
    def _run_ffmpeg(
        cmd: list[str],
        stdin_data: bytes | None = None,
        timeout: int = _FFMPEG_ENCODE_TIMEOUT_S,
    ) -> None:
        """Execute an ffmpeg command, raising on failure or timeout."""
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

    @classmethod
    def _stream_to_ffmpeg(
        cls,
        cmd: list[str],
        frames: NDArray,
        timeout: int = _FFMPEG_ENCODE_TIMEOUT_S,
    ) -> None:
        """Stream frames one-by-one to ffmpeg, converting dtype per-frame."""

        def write_stdin(stdin):
            for frame in frames:
                if frame.dtype != np.uint8:
                    frame = _normalize_dtype_single(frame)
                stdin.write(frame.tobytes())

        cls._run_streaming_ffmpeg(cmd, write_stdin, timeout)

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


def _normalize_dtype_single(frame: NDArray) -> NDArray[np.uint8]:
    """Convert a single frame (H, W, C) to uint8."""
    if frame.dtype in (np.float32, np.float64):
        max_val = float(np.max(frame)) if frame.size else 0.0
        if max_val <= _NORMALIZED_RANGE_MAX:
            return (frame * _MAX_PIXEL_VALUE).clip(0, 255).astype(np.uint8)
        return frame.clip(0, 255).astype(np.uint8)

    return frame.clip(0, 255).astype(np.uint8)


def _write_temp_wav(waveform, sample_rate: int) -> str:
    """Stage an audio waveform to a temp 16-bit PCM WAV for ffmpeg muxing.

    Accepts torch or numpy: mono ``(N,)``, interleaved ``(N, C)``, or
    channels-first stereo ``(2, N)`` (the decoder's layout), float [-1, 1] or
    int16. Channels-first is only disambiguated for stereo.
    """
    arr = waveform.detach().cpu().numpy() if hasattr(waveform, "detach") else np.asarray(waveform)
    if arr.ndim == 1:
        arr = arr[:, None]
    # (2, N) → (N, 2): the decoder emits channels-first stereo.
    if arr.shape[1] != 2 and arr.shape[0] == 2:
        arr = arr.T
    if arr.dtype != np.int16:
        arr = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16)

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        with wave.open(path, "wb") as w:
            w.setnchannels(arr.shape[1])
            w.setsampwidth(2)
            w.setframerate(int(sample_rate))
            w.writeframes(np.ascontiguousarray(arr).tobytes())
    except Exception:
        try:
            os.remove(path)
        except OSError:
            pass
        raise
    return path
