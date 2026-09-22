# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""MiniMax-H3 serving policy for this deployment.

The served envelope -- aspect ratios, durations, step count -- and its validators are owned by the
model's own policy module (``models.tt_dit.pipelines.minimax_h3.policy``) and re-exported here, so
what the media-server accepts cannot drift from what the pipeline is warmed for. Only the
request-transport concerns that pull in ``av`` -- media probing and reference-clip windows -- live
here.

Ref2VA reference *counts* match ``packing_ref2va`` (9 / 3 / 3). Per-clip and combined duration
windows are the product contract and are enforced here; the pipeline truncates instead of refusing.

The **input media card** (``MINIMAX_H3_*_MAX_BYTES``, formats, pixel and aspect windows, frame-rate
window, the 64 MB request-body cap) mirrors the published MiniMax API limits one to one and is
enforced at admission by ``check_h3_image`` / ``check_h3_reference_video`` /
``check_h3_reference_audio``. A source that passes admission is one the pipeline can use; a source
that fails is refused with a 4xx naming the limit, instead of a 202 followed by a failed job.
"""

import base64
import binascii
import io
from dataclasses import dataclass
from typing import Optional

# The serving envelope and its validators are owned by the model's policy module and re-exported
# lazily: pulling them at import time would drag ``ttnn`` into every consumer, and the server's unit
# tests mock the whole ``models.tt_dit`` tree. ``__getattr__`` defers the metal import to first
# access, so this module stays importable (and testable) without metal present. A name the metal
# module does not define yet falls back to ``_LEGACY_REEXPORTS`` below.
_METAL_REEXPORTS = frozenset(
    {
        "MINIMAX_H3_ASPECT_RATIOS",
        "MINIMAX_H3_DEFAULT_ASPECT_RATIO",
        "MINIMAX_H3_DEFAULT_DURATION_S",
        "MINIMAX_H3_DURATIONS_S",
        "MINIMAX_H3_NUM_INFERENCE_STEPS",
        "minimax_h3_parse_aspect_ratio",
        "minimax_h3_frames_are_aligned",
    }
)


# Pre-move server definitions of the re-exported items. A metal python tree that predates the
# move (e.g. 34260b25483, the pinned OM Quad3 tree) ships ``policy.py`` with the aspect/duration
# constants but without ``MINIMAX_H3_NUM_INFERENCE_STEPS``, ``minimax_h3_parse_aspect_ratio`` and
# ``minimax_h3_frames_are_aligned``; a name the metal module lacks resolves here, byte-for-byte
# what metal defines today, so the server runs against either generation. Metal wins when it has
# the name.
_LEGACY_ASPECT_RATIOS = ((21, 9), (16, 9), (4, 3), (1, 1), (3, 4), (9, 16))
_LEGACY_DEFAULT_ASPECT_RATIO = (16, 9)
_LEGACY_DURATIONS_S = tuple(range(4, 16))
_LEGACY_DEFAULT_DURATION_S = 5
_LEGACY_NUM_INFERENCE_STEPS = 50


def _metal_policy_module():
    try:
        from models.tt_dit.pipelines.minimax_h3 import policy as metal_policy
    except ImportError:
        return None
    return metal_policy


def _legacy_parse_aspect_ratio(value: str) -> tuple[int, int]:
    """``"16:9"`` -> ``(16, 9)``, restricted to the published set (pre-move server copy)."""
    ratios = __getattr__("MINIMAX_H3_ASPECT_RATIOS")
    text = str(value).strip().replace("x", ":").replace("/", ":")
    parts = text.split(":")
    if len(parts) != 2 or not all(part.strip().isdigit() for part in parts):
        raise ValueError(
            f"aspect_ratio must look like 'W:H' (got {value!r}); supported: "
            + ", ".join(f"{w}:{h}" for w, h in ratios)
        )
    pair = (int(parts[0]), int(parts[1]))
    if pair not in ratios:
        raise ValueError(
            f"aspect_ratio {pair[0]}:{pair[1]} is not served; supported: "
            + ", ".join(f"{w}:{h}" for w, h in ratios)
        )
    return pair


def _legacy_frames_are_aligned(num_frames: int) -> bool:
    """``num_frames`` must be ``17n + 5`` (pre-move server copy; modulus read from packing)."""
    from models.tt_dit.pipelines.minimax_h3.packing import (
        MINIMAX_H3_FRAMES_PER_CHUNK,
        MINIMAX_H3_LATENTS_PER_CHUNK,
    )

    return (
        num_frames >= MINIMAX_H3_LATENTS_PER_CHUNK
        and num_frames % MINIMAX_H3_FRAMES_PER_CHUNK == MINIMAX_H3_LATENTS_PER_CHUNK
    )


_LEGACY_REEXPORTS = {
    "MINIMAX_H3_ASPECT_RATIOS": lambda: _LEGACY_ASPECT_RATIOS,
    "MINIMAX_H3_DEFAULT_ASPECT_RATIO": lambda: _LEGACY_DEFAULT_ASPECT_RATIO,
    "MINIMAX_H3_DEFAULT_DURATION_S": lambda: _LEGACY_DEFAULT_DURATION_S,
    "MINIMAX_H3_DURATIONS_S": lambda: _LEGACY_DURATIONS_S,
    "MINIMAX_H3_NUM_INFERENCE_STEPS": lambda: _LEGACY_NUM_INFERENCE_STEPS,
    "minimax_h3_parse_aspect_ratio": lambda: _legacy_parse_aspect_ratio,
    "minimax_h3_frames_are_aligned": lambda: _legacy_frames_are_aligned,
}


def __getattr__(name: str):
    if name in _METAL_REEXPORTS:
        metal_policy = _metal_policy_module()
        if metal_policy is not None and hasattr(metal_policy, name):
            return getattr(metal_policy, name)
        return _LEGACY_REEXPORTS[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MINIMAX_H3_ASPECT_RATIOS",
    "MINIMAX_H3_DEFAULT_ASPECT_RATIO",
    "MINIMAX_H3_DEFAULT_DURATION_S",
    "MINIMAX_H3_DURATIONS_S",
    "MINIMAX_H3_NUM_INFERENCE_STEPS",
    "MINIMAX_H3_MAX_REFERENCE_IMAGES",
    "MINIMAX_H3_MAX_REFERENCE_VIDEOS",
    "MINIMAX_H3_MAX_REFERENCE_AUDIOS",
    "MINIMAX_H3_REF_CLIP_MIN_S",
    "MINIMAX_H3_REF_CLIP_MAX_S",
    "MINIMAX_H3_REF_COMBINED_MAX_S",
    "MINIMAX_H3_FL2VA_FRAME_POS",
    "MINIMAX_H3_MAX_REFERENCES_TOTAL",
    "MINIMAX_H3_MAX_REQUEST_BODY_BYTES",
    "MINIMAX_H3_IMAGE_MAX_BYTES",
    "MINIMAX_H3_IMAGE_FORMATS",
    "MINIMAX_H3_MEDIA_MIN_SIDE_PX",
    "MINIMAX_H3_MEDIA_MAX_SIDE_PX",
    "MINIMAX_H3_MEDIA_MIN_ASPECT",
    "MINIMAX_H3_MEDIA_MAX_ASPECT",
    "MINIMAX_H3_VIDEO_MAX_BYTES",
    "MINIMAX_H3_VIDEO_CONTAINERS",
    "MINIMAX_H3_VIDEO_CODECS",
    "MINIMAX_H3_VIDEO_AUDIO_CODECS",
    "MINIMAX_H3_VIDEO_MIN_FPS",
    "MINIMAX_H3_VIDEO_MAX_FPS",
    "MINIMAX_H3_AUDIO_MAX_BYTES",
    "MINIMAX_H3_AUDIO_CONTAINERS",
    "MEDIA_B64_FIELD_HEADROOM",
    "MediaTooLargeError",
    "MediaProbe",
    "base64_len_for_bytes",
    "decode_base64_media",
    "check_h3_image",
    "probe_media",
    "check_h3_reference_video",
    "check_h3_reference_audio",
    "minimax_h3_parse_aspect_ratio",
    "minimax_h3_frames_are_aligned",
    "probe_media_duration_seconds",
    "check_reference_clip_durations",
]

# Ref2VA omni-reference limits. Counts match packing_ref2va; duration windows
# are the product card (the pipeline does not enforce them).
MINIMAX_H3_MAX_REFERENCE_IMAGES = 9
MINIMAX_H3_MAX_REFERENCE_VIDEOS = 3
MINIMAX_H3_MAX_REFERENCE_AUDIOS = 3
MINIMAX_H3_REF_CLIP_MIN_S = 2.0
MINIMAX_H3_REF_CLIP_MAX_S = 15.0
MINIMAX_H3_REF_COMBINED_MAX_S = 15.0

# FL2VA keyframe sentinels on ``image_prompts[].frame_pos``.
MINIMAX_H3_FL2VA_FRAME_POS = frozenset({0, -1})

# The pipeline (``packing_ref2va.check_references``) refuses more than this many references of
# all kinds together, so 9 images + 3 videos + 3 audios is within every per-modality count and
# still unservable. Refused at admission too, or the client pays for a job that cannot run.
MINIMAX_H3_MAX_REFERENCES_TOTAL = 12

# ---------------------------------------------------------------------------------------------
# Input media card -- the published MiniMax API limits, mirrored one to one.
#
#   total request body <= 64 MB (large files go by URL, not inline base64)
#   image  (first frame, last frame, reference): JPG/JPEG/PNG/WEBP/HEIC/HEIF, <= 30 MB,
#          width and height in [256, 5760] px, aspect ratio w/h in [0.4, 2.5]
#   video  (reference only): MP4/MOV, H.264 or H.265 video, AAC or MP3 audio, <= 50 MB,
#          [256, 5760] px, w/h in [0.4, 2.5], 23.976-60 fps, 2-15 s per clip, <= 15 s combined
#   audio  (reference only): WAV/MP3, <= 15 MB, 2-15 s per clip, <= 15 s combined
#
# "MB" is read as MiB (1024^2): a client that sizes its file against the card in either unit
# is admitted. Counts (1 first, 1 last, 9 / 3 / 3 references) are the constants above.
# ---------------------------------------------------------------------------------------------
MINIMAX_H3_MAX_REQUEST_BODY_BYTES = 64 * 1024 * 1024

MINIMAX_H3_IMAGE_MAX_BYTES = 30 * 1024 * 1024
# PIL ``Image.format`` names. HEIC and HEIF both decode as "HEIF" once pillow-heif has
# registered its opener (utils.image_manager does that at import when the wheel is present).
# "MPO" is a JPEG with a multi-picture APP2 segment -- what iPhones and many Samsung, Sony
# and Fujifilm cameras write for every .jpg -- and Pillow reports it under that name.
MINIMAX_H3_IMAGE_FORMATS = frozenset({"JPEG", "MPO", "PNG", "WEBP", "HEIF"})
MINIMAX_H3_IMAGE_FORMATS_TEXT = "JPG, JPEG, PNG, WEBP, HEIC, HEIF"
# Base64 fields carry the encoded bytes plus, optionally, a "data:<mime>;base64," prefix that
# decode_base64_media strips. The pydantic max_length on those fields is the encoded size of
# the byte cap plus this headroom, so a file exactly at the cap is admitted with the prefix
# too; the byte cap itself is enforced on the decoded bytes.
MEDIA_B64_FIELD_HEADROOM = 128
MINIMAX_H3_MEDIA_MIN_SIDE_PX = 256
MINIMAX_H3_MEDIA_MAX_SIDE_PX = 5760
MINIMAX_H3_MEDIA_MIN_ASPECT = 0.4
MINIMAX_H3_MEDIA_MAX_ASPECT = 2.5

MINIMAX_H3_VIDEO_MAX_BYTES = 50 * 1024 * 1024
# libavformat names one demuxer for the whole QuickTime family ("mov,mp4,m4a,3gp,3g2,mj2"), so
# the container check is on the comma-separated name list.
MINIMAX_H3_VIDEO_CONTAINERS = frozenset({"mp4", "mov"})
MINIMAX_H3_VIDEO_CODECS = frozenset({"h264", "hevc"})
MINIMAX_H3_VIDEO_AUDIO_CODECS = frozenset({"aac", "mp3"})
MINIMAX_H3_VIDEO_MIN_FPS = 23.976
MINIMAX_H3_VIDEO_MAX_FPS = 60.0

MINIMAX_H3_AUDIO_MAX_BYTES = 15 * 1024 * 1024
MINIMAX_H3_AUDIO_CONTAINERS = frozenset({"wav", "mp3"})

_HEIF_BRANDS = frozenset(
    {
        b"heic",
        b"heix",
        b"hevc",
        b"hevx",
        b"heim",
        b"heis",
        b"hevm",
        b"hevs",
        b"mif1",
        b"msf1",
    }
)


class MediaTooLargeError(ValueError):
    """A media source is over its byte cap.

    A ``ValueError`` so pydantic validators report it as a 422 like every other field error,
    and a distinct class so the endpoint can answer 413 where it handles the bytes itself.
    """


def base64_len_for_bytes(num_bytes: int) -> int:
    """Length of the padded base64 text that encodes ``num_bytes`` bytes."""
    return 4 * ((num_bytes + 2) // 3)


def decode_base64_media(value: str) -> bytes:
    """Inline media field -> bytes: ``data:`` URL prefix tolerated, stripped padding restored.

    Mirrors ``ImageManager.base64_to_pil_image`` so the two never disagree on what decodes.
    """
    if value.startswith("data:"):
        value = value.split(",", 1)[-1]
    value += "=" * (-len(value) % 4)
    try:
        return base64.b64decode(value)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("media is not valid base64") from exc


def _looks_like_heif(raw: bytes) -> bool:
    return len(raw) >= 12 and raw[4:8] == b"ftyp" and raw[8:12] in _HEIF_BRANDS


def _mb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):g} MB"


def check_h3_image(raw: bytes, *, label: str = "image"):
    """Admit one image source against the card; return the (lazily decoded) PIL image.

    Size, format, pixel window and aspect window, in that order, each refused with a message
    naming the limit. Only the header is parsed (``Image.open`` is lazy), so a 30 MB source
    costs milliseconds here; the pixels are decoded once, on the worker.
    """
    if len(raw) > MINIMAX_H3_IMAGE_MAX_BYTES:
        raise MediaTooLargeError(
            f"{label} is {len(raw)} bytes; an image must be at most "
            f"{MINIMAX_H3_IMAGE_MAX_BYTES} bytes ({_mb(MINIMAX_H3_IMAGE_MAX_BYTES)})"
        )
    from PIL import Image

    try:
        image = Image.open(io.BytesIO(raw))
    except (
        Exception
    ) as exc:  # UnidentifiedImageError, DecompressionBombError, OSError, ...
        if _looks_like_heif(raw):
            raise ValueError(
                f"{label} is HEIC/HEIF but this server cannot decode it "
                "(pillow-heif is not installed); send JPG, PNG or WEBP"
            ) from exc
        raise ValueError(
            f"{label} could not be decoded; supported formats: {MINIMAX_H3_IMAGE_FORMATS_TEXT}"
        ) from exc
    fmt = (image.format or "").upper()
    if fmt not in MINIMAX_H3_IMAGE_FORMATS:
        raise ValueError(
            f"{label} format {fmt or 'unknown'} is not supported; "
            f"supported formats: {MINIMAX_H3_IMAGE_FORMATS_TEXT}"
        )
    _check_pixel_window(image.size[0], image.size[1], label=label)
    return image


def _check_pixel_window(width: int, height: int, *, label: str) -> None:
    lo, hi = MINIMAX_H3_MEDIA_MIN_SIDE_PX, MINIMAX_H3_MEDIA_MAX_SIDE_PX
    if not (lo <= width <= hi and lo <= height <= hi):
        raise ValueError(
            f"{label} is {width}x{height} px; width and height must each be within [{lo}, {hi}] px"
        )
    ratio = width / height
    if not MINIMAX_H3_MEDIA_MIN_ASPECT <= ratio <= MINIMAX_H3_MEDIA_MAX_ASPECT:
        raise ValueError(
            f"{label} aspect ratio (width/height) is {ratio:.3f}; must be within "
            f"[{MINIMAX_H3_MEDIA_MIN_ASPECT:g}, {MINIMAX_H3_MEDIA_MAX_ASPECT:g}]"
        )


@dataclass(frozen=True)
class MediaProbe:
    """What libavformat sees in a reference clip, normalized for the card checks."""

    containers: frozenset  # demuxer name list, e.g. {"mov", "mp4", ...} or {"wav"}
    duration_s: Optional[float]
    video_codec: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    audio_codec: Optional[str] = None


def _stream_codec_name(stream) -> str:
    """Normalized codec name of a stream; ``"unknown"`` when libavcodec has no decoder for it.

    PyAV leaves ``stream.codec_context`` as ``None`` for a codec id it cannot decode (an
    unregistered fourcc, MPEG-5 EVC, LCEVC ...). That is an unsupported codec, not a crash.
    """
    context = getattr(stream, "codec_context", None)
    name = getattr(context, "name", None) if context is not None else None
    return _normalize_codec(name) or "unknown"


def _normalize_codec(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    name = name.lower()
    if name.startswith("mp3"):  # "mp3", "mp3float", "mp3adu", ...
        return "mp3"
    if name.startswith("aac"):  # "aac", "aac_latm", "aac_fixed"
        return "aac"
    return name


def probe_media(raw: bytes) -> MediaProbe:
    """Container, streams, geometry, frame rate and duration of encoded bytes, via PyAV."""
    import av

    try:
        with av.open(io.BytesIO(raw)) as container:
            names = frozenset(part.strip() for part in container.format.name.split(","))
            duration = (
                container.duration / 1_000_000
                if container.duration is not None and container.duration > 0
                else None
            )
            video = next((s for s in container.streams if s.type == "video"), None)
            audio = next((s for s in container.streams if s.type == "audio"), None)
            fps = None
            width = height = None
            video_codec = None
            if video is not None:
                video_codec = _stream_codec_name(video)
                width, height = video.width, video.height
                rate = video.average_rate or video.guessed_rate or video.base_rate
                fps = float(rate) if rate else None
                if duration is None and video.duration is not None and video.time_base:
                    duration = float(video.duration * video.time_base)
            audio_codec = _stream_codec_name(audio) if audio is not None else None
            if (
                duration is None
                and audio is not None
                and audio.duration is not None
                and audio.time_base
            ):
                duration = float(audio.duration * audio.time_base)
    except Exception as exc:
        # av.FFmpegError for an unreadable file (av.AVError was removed in PyAV 14), and
        # anything else PyAV raises on an odd container: the caller's 422, never a 500.
        raise ValueError(
            "media could not be probed (not a readable audio/video file)"
        ) from exc
    return MediaProbe(
        containers=names,
        duration_s=duration,
        video_codec=video_codec,
        width=width,
        height=height,
        fps=fps,
        audio_codec=audio_codec,
    )


def check_h3_reference_video(raw: bytes, *, label: str = "video") -> float:
    """Admit one reference video against the card; return its duration in seconds.

    The 2-15 s per-clip and <= 15 s combined windows are ``check_reference_clip_durations``,
    which needs every clip's duration at once.
    """
    if len(raw) > MINIMAX_H3_VIDEO_MAX_BYTES:
        raise MediaTooLargeError(
            f"{label} is {len(raw)} bytes; a reference video must be at most "
            f"{MINIMAX_H3_VIDEO_MAX_BYTES} bytes ({_mb(MINIMAX_H3_VIDEO_MAX_BYTES)})"
        )
    probe = probe_media(raw)
    if not probe.containers & MINIMAX_H3_VIDEO_CONTAINERS:
        raise ValueError(
            f"{label} container {'/'.join(sorted(probe.containers))!r} is not supported; "
            "a reference video must be MP4 (.mp4) or MOV (.mov)"
        )
    if probe.video_codec is None:
        raise ValueError(f"{label} has no video stream")
    if probe.video_codec not in MINIMAX_H3_VIDEO_CODECS:
        raise ValueError(
            f"{label} video codec {probe.video_codec!r} is not supported; "
            "must be H.264/AVC or H.265/HEVC"
        )
    if (
        probe.audio_codec is not None
        and probe.audio_codec not in MINIMAX_H3_VIDEO_AUDIO_CODECS
    ):
        raise ValueError(
            f"{label} audio codec {probe.audio_codec!r} is not supported; must be AAC or MP3"
        )
    _check_pixel_window(probe.width or 0, probe.height or 0, label=label)
    if probe.fps is not None and not (
        MINIMAX_H3_VIDEO_MIN_FPS - 1e-3 <= probe.fps <= MINIMAX_H3_VIDEO_MAX_FPS + 1e-3
    ):
        raise ValueError(
            f"{label} frame rate is {probe.fps:.3f} fps; must be within "
            f"[{MINIMAX_H3_VIDEO_MIN_FPS:g}, {MINIMAX_H3_VIDEO_MAX_FPS:g}] fps"
        )
    if probe.duration_s is None:
        raise ValueError(f"{label} reports no duration")
    return probe.duration_s


def check_h3_reference_audio(raw: bytes, *, label: str = "audio") -> float:
    """Admit one reference audio clip against the card; return its duration in seconds."""
    if len(raw) > MINIMAX_H3_AUDIO_MAX_BYTES:
        raise MediaTooLargeError(
            f"{label} is {len(raw)} bytes; a reference audio clip must be at most "
            f"{MINIMAX_H3_AUDIO_MAX_BYTES} bytes ({_mb(MINIMAX_H3_AUDIO_MAX_BYTES)})"
        )
    probe = probe_media(raw)
    if not probe.containers & MINIMAX_H3_AUDIO_CONTAINERS:
        raise ValueError(
            f"{label} format {'/'.join(sorted(probe.containers))!r} is not supported; "
            "a reference audio clip must be WAV or MP3"
        )
    if probe.audio_codec is None:
        raise ValueError(f"{label} has no audio stream")
    if probe.duration_s is None:
        raise ValueError(f"{label} reports no duration")
    return probe.duration_s


def probe_media_duration_seconds(raw: bytes) -> float:
    """Container duration in seconds, from encoded bytes.

    Uses PyAV's container duration (microseconds). Raises ``ValueError`` when
    the bytes are not a readable audio/video container or have no duration.
    """
    import av

    try:
        with av.open(io.BytesIO(raw)) as container:
            if container.duration is None or container.duration <= 0:
                raise ValueError("media container reports no duration")
            return container.duration / 1_000_000
    except av.FFmpegError as exc:  # av.AVError was removed in PyAV 14
        raise ValueError("media could not be probed for duration") from exc


def check_reference_clip_durations(
    *,
    video_durations: list[float],
    audio_durations: list[float],
) -> None:
    """Refuse reference clips outside the 2–15 s / combined ≤ 15 s window."""

    def _check(kind: str, durations: list[float]) -> None:
        for index, seconds in enumerate(durations):
            if not MINIMAX_H3_REF_CLIP_MIN_S <= seconds <= MINIMAX_H3_REF_CLIP_MAX_S:
                raise ValueError(
                    f"{kind}[{index}] is {seconds:g} s; each {kind} clip must be "
                    f"{MINIMAX_H3_REF_CLIP_MIN_S:g}–{MINIMAX_H3_REF_CLIP_MAX_S:g} s"
                )
        combined = sum(durations)
        if combined > MINIMAX_H3_REF_COMBINED_MAX_S:
            raise ValueError(
                f"combined {kind} duration is {combined:g} s; must be "
                f"≤ {MINIMAX_H3_REF_COMBINED_MAX_S:g} s"
            )

    _check("videos", video_durations)
    _check("audios", audio_durations)
