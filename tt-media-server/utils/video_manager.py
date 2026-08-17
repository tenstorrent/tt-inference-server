# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from __future__ import annotations

import os
import subprocess
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
_FFMPEG_MUX_TIMEOUT_S = 120
_AUDIO_AAC_BITRATE = "192k"
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
    def export_to_mp4(
        self,
        frames: NDArray,
        fps: int = 16,
        audio: NDArray | None = None,
        sample_rate: int | None = None,
    ) -> str:
        """
        Export frames to MP4 (H.264 via ffmpeg), optionally muxing an audio track.

        Frames are streamed to ffmpeg one at a time so encoding overlaps with
        Python-side dtype conversion, avoiding large temporary allocations.

        Audio is optional and fully backward-compatible: silent models pass
        ``audio=None`` and get exactly the previous video-only output. When the
        runner returns an object exposing ``.frames`` and ``.audio`` (e.g. an
        audio-capable video model such as MiniMax-H3), the audio is pulled off
        that object automatically. When present, the encoded video is muxed with
        an AAC track via a second ffmpeg pass; the API contract is unchanged
        (still one ``video/mp4``, now with sound).

        Env (optional):
            TT_VIDEO_EXPORT_CRF: 0–51, lower = better quality. Default 23.
            TT_VIDEO_EXPORT_PRESET: ultrafast … veryslow. Default medium.
        """
        # Pull audio (and the model's frame rate) off a result object (e.g.
        # VideoWithAudio) before it is collapsed to a bare frame array below.
        # Explicit args take precedence.
        if audio is None and hasattr(frames, "audio"):
            audio = frames.audio
            if sample_rate is None:
                sample_rate = getattr(frames, "sample_rate", None)
        _obj_fps = getattr(frames, "fps", None)
        if _obj_fps:
            fps = int(_obj_fps)
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

            if audio is not None and sample_rate:
                output_path = self._mux_audio_into_mp4(
                    output_path, audio, int(sample_rate)
                )
            return output_path

        except Exception as e:
            self._logger.error(f"Video export failed: {e}")
            raise RuntimeError(f"Failed to export video: {e}") from e

    def _mux_audio_into_mp4(
        self, video_path: str, audio: NDArray, sample_rate: int
    ) -> str:
        """Mux a stereo/mono audio track into an already-encoded silent mp4.

        Writes the audio to a temp WAV, then remuxes with ``-c:v copy -c:a aac``
        (video is not re-encoded). Returns the path to the muxed file and removes
        the silent intermediate. On any failure the original silent video is
        kept and returned, so audio never breaks video delivery.
        """
        wav_path = str(_VIDEO_OUTPUT_DIR / f"{uuid.uuid4()}.wav")
        muxed_path = str(_VIDEO_OUTPUT_DIR / f"{uuid.uuid4()}.mp4")
        try:
            channels = _write_wav(audio, sample_rate, wav_path)
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-i",
                wav_path,
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                _AUDIO_AAC_BITRATE,
                "-ac",
                str(channels),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-shortest",
                "-movflags",
                "+faststart",
                muxed_path,
            ]
            self._run_ffmpeg(cmd, timeout=_FFMPEG_MUX_TIMEOUT_S)
        except Exception as e:
            self._logger.error(f"Audio mux failed, keeping silent video: {e}")
            _safe_unlink(muxed_path)
            return video_path
        finally:
            _safe_unlink(wav_path)

        _safe_unlink(video_path)
        return muxed_path

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


def _audio_to_int16_frames(audio: NDArray) -> tuple[NDArray, int]:
    """Normalize an audio array to interleaved int16 (num_samples, channels).

    Accepts mono ``(N,)``, or 2D in either layout — channels-first ``(C, N)``
    (the model's native stereo shape) or samples-first ``(N, C)``. Float input
    in [-1, 1] is scaled to int16; integer input is passed through as int16.
    """
    audio = np.asarray(audio)
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]  # (N, 1)
    elif audio.ndim == 2:
        rows, cols = audio.shape
        # Channels-first if the first axis is the small one (1 or 2) and clearly
        # shorter than the sample axis.
        if rows in (1, 2) and rows < cols:
            audio = audio.T  # (C, N) -> (N, C)
    else:
        raise ValueError(f"Unexpected audio dimensions: {audio.shape}")

    if np.issubdtype(audio.dtype, np.floating):
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767.0).round().astype(np.int16)
    else:
        audio = audio.astype(np.int16)

    channels = audio.shape[1]
    return np.ascontiguousarray(audio), channels


def _write_wav(audio: NDArray, sample_rate: int, path: str) -> int:
    """Write audio to a 16-bit PCM WAV file. Returns the channel count."""
    frames, channels = _audio_to_int16_frames(audio)
    with wave.open(path, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)  # int16
        wav.setframerate(int(sample_rate))
        wav.writeframes(frames.tobytes())
    return channels


def _safe_unlink(path: str) -> None:
    """Remove a file if it exists, ignoring errors (best-effort cleanup)."""
    try:
        os.remove(path)
    except OSError:
        pass
