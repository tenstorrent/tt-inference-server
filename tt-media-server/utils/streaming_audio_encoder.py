# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""
Turns a sequence of independent small per-chunk WAV files (what the Inworld
TTS runner's streaming path already produces, one per progressively-decoded
audio chunk) into ONE continuous encoded byte stream -- as opposed to
ffmpeg_utils.encode_wav_to, which one-shot-encodes a single complete WAV.

Needed for POST /v1/audio/speech's raw-bytes streaming contract: the OpenAI
SDK reads the whole chunked HTTP response body as a single audio file, so
naively concatenating N independent per-chunk WAV files (each with its own
RIFF header) would not be a valid file. This module strips each chunk's WAV
header down to raw PCM frames and either:
  - wraps the raw PCM in ONE streaming-style WAV header (no ffmpeg, fastest
    path) when the caller wants uncompressed WAV/PCM output at the source
    sample rate, or
  - feeds the raw PCM into ONE long-lived ffmpeg process (for mp3/ogg, or
    any sample-rate conversion) whose stdout is forwarded to the client as
    it's produced.
"""

import asyncio
import shutil
import struct
import wave
from io import BytesIO
from typing import AsyncIterator, Optional, Tuple

_FFMPEG_CODEC_ARGS = {
    "mp3": ["-f", "mp3"],
    "ogg": ["-acodec", "libvorbis", "-f", "ogg"],
    "wav": ["-f", "wav"],
}


def parse_wav_chunk(wav_bytes: bytes) -> Tuple[bytes, int, int, int]:
    """Returns (raw_pcm_frames, framerate, sample_width_bytes, channels) for
    one small self-contained WAV file (as produced by
    InworldTTSRunner._wav_base64)."""
    with wave.open(BytesIO(wav_bytes), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        return frames, wf.getframerate(), wf.getsampwidth(), wf.getnchannels()


def build_streaming_wav_header(framerate: int, sample_width: int, channels: int) -> bytes:
    """RIFF/WAV header for a stream of UNKNOWN total length -- written once,
    followed by raw PCM frames with no further per-chunk headers. Sizes use
    the standard 0xFFFFFFFF "streaming/unknown length" sentinel; most
    consumers (ffmpeg, browsers, media players) read until the connection
    closes rather than trusting these fields for a live stream.
    """
    if sample_width != 2:
        raise ValueError(f"only 16-bit PCM is supported, got sample_width={sample_width}")
    byte_rate = framerate * channels * sample_width
    block_align = channels * sample_width
    bits_per_sample = sample_width * 8
    fmt_chunk = struct.pack(
        "<IHHIIHH", 16, 1, channels, framerate, byte_rate, block_align, bits_per_sample
    )
    return (
        b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE"
        + b"fmt " + fmt_chunk
        + b"data" + struct.pack("<I", 0xFFFFFFFF)
    )


async def encode_pcm_stream(
    pcm_chunks: AsyncIterator[bytes],
    framerate: int,
    channels: int,
    output_format: str,
    output_sample_rate: Optional[int] = None,
    bit_rate: Optional[int] = None,
) -> AsyncIterator[bytes]:
    """Feeds headerless 16-bit-PCM chunks into ONE long-lived ffmpeg process
    (input rate/channels fixed, output format/rate/bitrate as requested) and
    yields encoded bytes as ffmpeg produces them. Used whenever resampling is
    needed, or output_format is mp3/ogg.
    """
    if output_format not in _FFMPEG_CODEC_ARGS:
        raise ValueError(f"unsupported output_format for ffmpeg streaming: {output_format}")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found in PATH")

    out_args = list(_FFMPEG_CODEC_ARGS[output_format])
    if output_sample_rate is not None:
        out_args += ["-ar", str(output_sample_rate)]
    if output_format == "mp3" and bit_rate:
        out_args += ["-b:a", str(bit_rate)]

    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-f", "s16le", "-ar", str(framerate), "-ac", str(channels), "-i", "pipe:0",
        *out_args,
        "-y", "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    async def _writer():
        try:
            async for pcm in pcm_chunks:
                proc.stdin.write(pcm)
                await proc.stdin.drain()
        finally:
            proc.stdin.close()

    writer_task = asyncio.create_task(_writer())
    try:
        while True:
            chunk = await proc.stdout.read(65536)
            if not chunk:
                break
            yield chunk
    finally:
        await writer_task
        returncode = await proc.wait()
        if returncode != 0:
            stderr = await proc.stderr.read()
            raise RuntimeError(
                f"ffmpeg streaming encode failed (exit {returncode}): "
                f"{stderr.decode(errors='replace')}"
            )
