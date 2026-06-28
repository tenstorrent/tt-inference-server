# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from __future__ import annotations

import os
import subprocess
import tempfile
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
                "ffmpeg", "-y",
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

    def export_yuv420p_to_mp4_with_audio(
        self,
        frames_planar,
        audio_waveform,
        sample_rate: int,
        width: int,
        height: int,
        fps: int = 24,
    ) -> str:
        """Export pre-converted YUV 4:2:0 planar frames + audio to an AV MP4 (H.264 + AAC).

        ``frames_planar`` is uint8 shape ``(T, height*width + 2*(height/2 * width/2))`` —
        per-frame ``[Y | Cb | Cr]`` row-major, the ffmpeg yuv420p layout the device YUV op
        produces. Feeding yuv420p directly skips libx264's RGB→YUV conversion and moves
        ~half the bytes of the RGB path.
        """
        arr = np.asarray(frames_planar)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"Expected YUV420p planar (T, planar_bytes), got {arr.shape}")
        expected = height * width + 2 * ((height // 2) * (width // 2))
        if arr.shape[1] != expected:
            raise ValueError(
                f"yuv420p planar row must be {expected} bytes for {width}x{height}, got {arr.shape[1]}"
            )
        arr = np.ascontiguousarray(arr, dtype=np.uint8)

        _VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(_VIDEO_OUTPUT_DIR / f"{uuid.uuid4()}.mp4")
        crf = max(_MIN_CRF, min(_MAX_CRF, int(os.environ.get("TT_VIDEO_EXPORT_CRF", "23"))))
        preset = os.environ.get("TT_VIDEO_EXPORT_PRESET", "ultrafast").strip()

        wav_path = _write_temp_wav(audio_waveform, sample_rate)
        try:
            cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-pix_fmt", "yuv420p",
                "-s", f"{width}x{height}", "-r", str(fps),
                "-i", "-",
                "-i", wav_path,
                "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p",
            ]
            if preset:
                cmd += ["-preset", preset]
            cmd += ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", output_path]
            self._stream_bytes_to_ffmpeg(cmd, arr)
            return output_path
        except Exception as e:
            self._logger.error(f"YUV420p export failed: {e}")
            raise RuntimeError(f"Failed to export YUV420p video+audio: {e}") from e
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass

    @staticmethod
    def _stream_bytes_to_ffmpeg(cmd, arr, timeout: int = _FFMPEG_ENCODE_TIMEOUT_S) -> None:
        """Stream a contiguous uint8 array to ffmpeg stdin in chunks (zero-copy via memoryview)."""
        process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        try:
            mv = memoryview(arr.reshape(-1))
            chunk = 8 << 20
            for i in range(0, len(mv), chunk):
                process.stdin.write(mv[i : i + chunk])
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

    @staticmethod
    def _stream_planar_to_ffmpeg(
        cmd: list[str],
        arr: NDArray,
        t_frames: int,
        plane_order: tuple[int, int, int] = (0, 1, 2),
        timeout: int = _FFMPEG_ENCODE_TIMEOUT_S,
    ) -> None:
        """Stream planar frames (arr[c, t] = (H, W) per plane) to ffmpeg in ``plane_order``."""
        process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        try:
            for t in range(t_frames):
                for c in plane_order:
                    # arr[c, t] is already C-contiguous; write its buffer directly to skip a
                    # ~900MB/frame-set tobytes() copy on the export critical path.
                    process.stdin.write(np.ascontiguousarray(arr[c, t]))
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

    Accepts torch or numpy, shape (N,), (C, N), or (N, C), float [-1, 1] or int16.
    """
    arr = waveform.detach().cpu().numpy() if hasattr(waveform, "detach") else np.asarray(waveform)
    if arr.ndim == 1:
        arr = arr[:, None]
    # (C, N) → (N, C): the decoder emits channels-first stereo.
    if arr.shape[1] != 2 and arr.shape[0] == 2:
        arr = arr.T
    if arr.dtype != np.int16:
        arr = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16)

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with wave.open(path, "wb") as w:
        w.setnchannels(arr.shape[1])
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(np.ascontiguousarray(arr).tobytes())
    return path
