# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""
Shared ffmpeg pipe-based conversion (same pattern as audio_manager).
Used by AudioManager (decode to WAV) and TTS post-process (encode WAV to MP3/OGG).
"""

import shutil
import subprocess
from typing import List


def run_ffmpeg_stdin_stdout(input_bytes: bytes, ffmpeg_args: List[str]) -> bytes:
    """
    Run ffmpeg with stdin/stdout pipes (same as audio_manager._decode_audio_file).
    Feeds input_bytes to stdin, returns stdout bytes. Raises on failure.
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg not found in PATH. Install ffmpeg and ensure it is available "
            "when the server starts."
        )
    process = subprocess.Popen(
        ["ffmpeg"] + ffmpeg_args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output_bytes, error_output = process.communicate(input=input_bytes)
    if process.returncode != 0:
        error_msg = (
            error_output.decode("utf-8") if error_output else "Unknown ffmpeg error"
        )
        raise subprocess.CalledProcessError(process.returncode, "ffmpeg", error_msg)
    return output_bytes


def decode_to_wav(audio_bytes: bytes, sample_rate: int = 16000) -> bytes:
    """
    Convert ffmpeg-supported input (MP3, OGG, etc.) to WAV bytes.
    Same args as audio_manager._decode_audio_file.
    """
    args = [
        "-i",
        "pipe:0",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-f",
        "wav",
        "-y",
        "pipe:1",
    ]
    return run_ffmpeg_stdin_stdout(audio_bytes, args)


def encode_wav_to(wav_bytes: bytes, output_format: str) -> bytes:
    """
    Convert WAV bytes to MP3 or OGG. Used by TTS post-process.
    """
    if output_format == "mp3":
        args = ["-i", "pipe:0", "-f", "mp3", "-y", "pipe:1"]
    elif output_format == "ogg":
        args = [
            "-i",
            "pipe:0",
            "-acodec",
            "libvorbis",
            "-f",
            "ogg",
            "-y",
            "pipe:1",
        ]
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    return run_ffmpeg_stdin_stdout(wav_bytes, args)


def encode_wav_bytes(
    wav_bytes: bytes,
    output_format: str,
    sample_rate: int = None,
    bit_rate: int = None,
) -> bytes:
    """
    One-shot WAV -> {wav, mp3, ogg, pcm} conversion, optionally resampling.
    Used for independently-encoding EACH streaming chunk (as opposed to
    encode_wav_to's one-shot whole-file conversion, or
    streaming_audio_encoder's continuous multi-chunk pipe) -- see
    open_ai_api/inworld_voice_stream.py.

    "pcm" means headerless raw 16-bit PCM (Inworld's LINEAR16/PCM encodings),
    as opposed to "wav" which keeps the WAV container.
    """
    if output_format == "wav" and not sample_rate:
        return wav_bytes
    if output_format == "mp3":
        out_args = ["-f", "mp3"]
        if bit_rate:
            out_args += ["-b:a", str(bit_rate)]
    elif output_format == "ogg":
        out_args = ["-acodec", "libvorbis", "-f", "ogg"]
    elif output_format == "wav":
        out_args = ["-f", "wav"]
    elif output_format == "pcm":
        out_args = ["-f", "s16le"]
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    if sample_rate:
        out_args += ["-ar", str(sample_rate)]
    args = ["-i", "pipe:0"] + out_args + ["-y", "pipe:1"]
    return run_ffmpeg_stdin_stdout(wav_bytes, args)
