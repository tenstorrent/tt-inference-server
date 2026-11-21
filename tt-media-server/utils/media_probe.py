# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import io
from dataclasses import dataclass

from PIL import Image, ImageOps


@dataclass(frozen=True)
class ReferenceMediaProbe:
    """Container metadata of one video or audio clip.

    ``size`` (width, height), ``fps`` and ``video_codec`` are those of the first video stream,
    ``audio_channels`` and ``audio_codec`` those of the first audio stream; each is None when the
    stream is absent. ``container`` is the FFmpeg demuxer's name.
    """

    duration_s: float
    size: tuple[int, int] | None
    fps: float | None
    audio_channels: int | None
    container: str | None = None
    video_codec: str | None = None
    audio_codec: str | None = None


# The FFmpeg demuxer that reads both MP4 and MOV.
_MP4_OR_MOV = "mov,mp4,m4a,3gp,3g2,mj2"


def media_format_error(kind: str, probe: ReferenceMediaProbe) -> str | None:
    """Why a clip of ``kind`` ("video" or "audio") is not in a format the API accepts: MP4 or MOV
    with H.264 or H.265 video and an optional AAC or MP3 track, or WAV or MP3 audio."""
    if kind == "video":
        if probe.container != _MP4_OR_MOV:
            return f"is {probe.container}; must be MP4 or MOV"
        if probe.video_codec not in (None, "h264", "hevc"):
            return f"has {probe.video_codec} video; must be H.264 or H.265"
        if probe.audio_codec not in (None, "aac", "mp3"):
            return f"has a {probe.audio_codec} audio track; must be AAC or MP3"
        return None
    if probe.container not in ("wav", "mp3"):
        return f"is {probe.container}; must be WAV or MP3"
    return None


# PIL reads a JPEG with several pictures, e.g. a phone photo with a depth or
# gain map, as MPO.
_IMAGE_FORMATS = ("PNG", "JPEG", "MPO", "WEBP")


def decode_image(data: bytes, *, rgb: bool = True) -> Image.Image:
    """The image in ``data``, upright as a camera's EXIF orientation says.

    It is in RGB unless ``rgb`` is false. Then it keeps its own mode, e.g. RGBA
    with its alpha channel, for a pipeline that converts it itself.

    Raises ``ValueError`` if it is not PNG, JPEG or WebP or has more than
    ``Image.MAX_IMAGE_PIXELS`` pixels, and PIL's ``UnidentifiedImageError`` or
    ``OSError`` if it is not a decodable image.
    """
    too_large = ValueError(f"image has more than {Image.MAX_IMAGE_PIXELS} pixels")
    try:
        image = Image.open(io.BytesIO(data))
    except Image.DecompressionBombError:
        raise too_large
    if image.format not in _IMAGE_FORMATS:
        raise ValueError(f"is a {image.format} image; must be PNG, JPEG or WebP")
    # Checked before decoding: PIL only refuses twice the limit.
    if Image.MAX_IMAGE_PIXELS and image.width * image.height > Image.MAX_IMAGE_PIXELS:
        raise too_large
    image.load()
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB") if rgb else image


def probe_reference_media(raw: bytes) -> ReferenceMediaProbe:
    """Probe encoded audio/video bytes without decoding them.

    Uses PyAV's container duration (microseconds) and the frame rate a decoder reads (average,
    else guessed). Raises ``ValueError`` when the bytes are not a readable audio/video container or
    have no duration.
    """
    import av

    try:
        with av.open(io.BytesIO(raw)) as container:
            if container.duration is None or container.duration <= 0:
                raise ValueError("media container reports no duration")
            size = fps = audio_channels = video_codec = audio_codec = None
            if container.streams.video:
                stream = container.streams.video[0]
                size = (stream.codec_context.width, stream.codec_context.height)
                rate = stream.average_rate or stream.guessed_rate
                fps = float(rate) if rate else None
                video_codec = stream.codec_context.name
            if container.streams.audio:
                stream = container.streams.audio[0]
                audio_channels = len(stream.layout.channels)
                # FFmpeg names MP3 by its decoder.
                audio_codec = {"mp3float": "mp3"}.get(
                    stream.codec_context.name, stream.codec_context.name
                )
            return ReferenceMediaProbe(
                duration_s=container.duration / 1_000_000,
                size=size,
                fps=fps,
                audio_channels=audio_channels,
                container=container.format.name,
                video_codec=video_codec,
                audio_codec=audio_codec,
            )
    except av.FFmpegError as exc:  # av.AVError was removed in PyAV 14
        raise ValueError("not a readable video or audio file") from exc
