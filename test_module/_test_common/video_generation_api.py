# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The generation API's video models with sound, for the spec tests.

LTX-2.3 22B distilled (text to video) and MiniMax-H3, which a deployment
serves as FL2VA (text, optionally pinned to a first and/or last frame) or as
Ref2VA (text with reference images, videos and audio clips), each also with
lightx2v's Turbo adapter. Paths, fields, limits and output rules follow each
model's request schema (``tt-media-server/domain/ltx23_distilled_request.py``
and ``minimax_h3_request.py``) and the common part of the design
(``generation_api``). Wan2.2 I2V, which makes no sound and has its own GPU
references, keeps its own module (``wan_i2v_generation_api``).

A deployment serves one model; the spec tests pick it by the deployment's
Hugging Face repo (``model_for``) and hold it to what is written here.
"""

from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .generation_api import SEED_MAX, is_int, output_url_problem

VIDEO_CONTENT_TYPE = "video/mp4"
# The duration is a float the server computes; allow for its rounding.
DURATION_TOLERANCE_SECONDS = 0.05

# LTX-2.3 distilled serves one shape: 1920x1088 (1080 rounded up to a multiple
# of 64), 6 s at 25 fps, which is 153 frames (the frame count must be 8k + 1).
LTX_SHAPE = (1920, 1088)
LTX_FPS = 25
LTX_FRAMES_STEP = 8

# MiniMax-H3: 24 fps; the VAE encodes 17n + 5 frames. The canvas has a short
# edge of 768, its area capped at 768x1344, sides multiples of 32.
H3_FPS = 24
H3_FRAMES_PER_CHUNK = 17
H3_LATENTS_PER_CHUNK = 5
H3_SHORT_EDGE = 768
H3_MAX_PIXELS = 768 * 1344
H3_CANVAS_MULTIPLE = 32
H3_RATIOS = ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16")
# A keyframe's sides, and every image's and clip's aspect ratio (1:4 to 4:1).
H3_KEYFRAME_MIN_SIDE = 256
H3_KEYFRAME_MAX_SIDE = 5760
H3_MAX_RATIO = 4
# Reference clips: 2-15 s each, at most 3 audio references.
H3_CLIP_MIN_SECONDS = 2
H3_CLIP_MAX_SECONDS = 15


def ltx_num_frames(duration: int, fps: int = LTX_FPS) -> int:
    """The frames of ``duration`` seconds: the nearest 8k + 1 count."""
    return round((duration * fps - 1) / LTX_FRAMES_STEP) * LTX_FRAMES_STEP + 1


def h3_num_frames(duration: int) -> int:
    """The frames of ``duration`` seconds: the 24 fps count, up to the next 17n + 5."""
    frames = round(duration * H3_FPS)
    while frames % H3_FRAMES_PER_CHUNK != H3_LATENTS_PER_CHUNK:
        frames += 1
    return frames


def h3_canvas(width: float, height: float) -> tuple[int, int]:
    """The output ``(width, height)`` for a shape, by the size tier's rule."""
    ratio = width / height
    if ratio >= 1:
        width, height = H3_SHORT_EDGE * ratio, float(H3_SHORT_EDGE)
    else:
        width, height = float(H3_SHORT_EDGE), H3_SHORT_EDGE / ratio
    if width * height > H3_MAX_PIXELS:
        scale = (H3_MAX_PIXELS / (width * height)) ** 0.5
        width, height = width * scale, height * scale
    multiple = H3_CANVAS_MULTIPLE
    return (
        max(multiple, round(width / multiple) * multiple),
        max(multiple, round(height / multiple) * multiple),
    )


@dataclass(frozen=True)
class VideoModel:
    """One model's request schema and output rules."""

    model_id: str
    # Hugging Face repos a deployment of this model is spec'd under.
    hf_repos: tuple[str, ...]
    # Another model's route, which this deployment answers 404 (NotServed).
    unserved_model_id: str
    # What a request is made of besides its prompt: nothing (``text``), a
    # first and/or last frame (``keyframes``), or reference media (``references``).
    kind: str
    request_fields: frozenset[str]
    fps: int
    aspect_ratios: tuple[str, ...]
    default_aspect_ratio: str
    size: str
    # The shortest and longest ``duration`` (seconds) and the default.
    min_duration: int
    max_duration: int
    default_duration: int
    # Every output has an AAC track.
    audio: bool = True
    # Reference limits (``references`` only).
    max_images: int = 0
    max_videos: int = 0
    max_audios: int = 0
    # A fixed fps the request may state (LTX-2.3), else not a field.
    fps_is_a_field: bool = False

    @property
    def submit_path(self) -> str:
        return f"/v1/generations/{self.model_id}"

    @property
    def unserved_submit_path(self) -> str:
        return f"/v1/generations/{self.unserved_model_id}"

    @property
    def is_h3(self) -> bool:
        return self.kind in ("keyframes", "references")

    def num_frames(self, duration: int | None = None) -> int:
        duration = self.default_duration if duration is None else duration
        return h3_num_frames(duration) if self.is_h3 else ltx_num_frames(duration)


_LTX_FIELDS = frozenset({"prompt", "aspect_ratio", "size", "duration", "fps", "seed"})
_H3_FIELDS = frozenset({"prompt", "aspect_ratio", "size", "duration", "seed"})
_FL2VA_FIELDS = _H3_FIELDS | {"first_frame", "last_frame"}
_REF2VA_FIELDS = _H3_FIELDS | {"images", "videos", "audios"}


def _h3(
    model_id: str, hf_repos: tuple[str, ...], kind: str, unserved: str
) -> VideoModel:
    references = kind == "references"
    return VideoModel(
        model_id=model_id,
        hf_repos=hf_repos,
        unserved_model_id=unserved,
        kind=kind,
        request_fields=_REF2VA_FIELDS if references else _FL2VA_FIELDS,
        fps=H3_FPS,
        aspect_ratios=H3_RATIOS,
        default_aspect_ratio="16:9",
        size="768p",
        min_duration=4,
        max_duration=15,
        default_duration=5,
        max_images=9 if references else 0,
        max_videos=3 if references else 0,
        max_audios=3 if references else 0,
    )


MODELS = {
    model.model_id: model
    for model in (
        VideoModel(
            model_id="lightricks/ltx-2.3-22b-distilled",
            hf_repos=("Lightricks/LTX-2.3-distilled", "Lightricks/LTX-2.3"),
            unserved_model_id="wan-ai/wan2.2-i2v-a14b",
            kind="text",
            request_fields=_LTX_FIELDS,
            fps=LTX_FPS,
            aspect_ratios=("16:9",),
            default_aspect_ratio="16:9",
            size="1080p",
            min_duration=6,
            max_duration=6,
            default_duration=6,
            fps_is_a_field=True,
        ),
        _h3(
            "minimaxai/minimax-h3-fl2va",
            ("MiniMaxAI/MiniMax-H3-FL2VA",),
            "keyframes",
            "minimaxai/minimax-h3-ref2va",
        ),
        _h3(
            "tenstorrent/minimax-h3-fl2va-lightx2v",
            ("MiniMaxAI/MiniMax-H3-FL2VA-LightX2V",),
            "keyframes",
            "minimaxai/minimax-h3-fl2va",
        ),
        _h3(
            "minimaxai/minimax-h3-ref2va",
            ("MiniMaxAI/MiniMax-H3-Ref2VA",),
            "references",
            "minimaxai/minimax-h3-fl2va",
        ),
        _h3(
            "tenstorrent/minimax-h3-ref2va-lightx2v",
            ("MiniMaxAI/MiniMax-H3-Ref2VA-LightX2V",),
            "references",
            "minimaxai/minimax-h3-ref2va",
        ),
    )
}


def model_for(name: str) -> VideoModel:
    """The model a deployment serves, by its API id or Hugging Face repo."""
    key = name.lower()
    for model in MODELS.values():
        if key == model.model_id or key in (repo.lower() for repo in model.hf_repos):
            return model
    raise KeyError(
        f"{name!r} is not a video model of the generation API here: {sorted(MODELS)}"
    )


def is_video_generation_api_model(name: str) -> bool:
    try:
        model_for(name)
    except KeyError:
        return False
    return True


def ratio_shape(aspect_ratio: str) -> tuple[int, int]:
    width, height = aspect_ratio.split(":")
    return int(width), int(height)


def requested_shape(
    model: VideoModel,
    request: dict[str, Any],
    frame_size: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """The output ``(width, height)`` a valid request asks for, defaults applied.

    ``frame_size`` is the size of the request's first frame, or of its last
    frame when it has no first: a keyframes request follows it (``adaptive``).
    """
    if not model.is_h3:
        return LTX_SHAPE
    ratio = request.get("aspect_ratio")
    has_frame = model.kind == "keyframes" and (
        request.get("first_frame") is not None or request.get("last_frame") is not None
    )
    if has_frame and ratio in (None, "adaptive"):
        if frame_size is None:
            raise ValueError("an adaptive request needs its frame's size")
        return h3_canvas(*frame_size)
    return h3_canvas(*ratio_shape(ratio or model.default_aspect_ratio))


def video_output_problems(
    model: VideoModel,
    job: dict[str, Any],
    request: dict[str, Any],
    frame_size: tuple[int, int] | None = None,
) -> list[str]:
    """Where a completed job's one VideoOutput disagrees with its request.

    The seed is the requested one, or any valid seed when the request had none
    (the output reports the one chosen). The URL is the job's ``content/0``
    path, or an absolute URL if outputs ever move to storage. ``audio``
    describes the AAC track every output of these models has.
    """
    outputs = job.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 1:
        return [f"expected one output, got {outputs!r:.200}"]
    output = outputs[0]
    if not isinstance(output, dict):
        return [f"output is not an object: {output!r:.200}"]
    width, height = requested_shape(model, request, frame_size)
    num_frames = model.num_frames(request.get("duration"))
    expected = {
        "type": "video",
        "content_type": VIDEO_CONTENT_TYPE,
        "prompt": request["prompt"],
        "width": width,
        "height": height,
        "fps": model.fps,
        "num_frames": num_frames,
    }
    problems = [
        f"{key} is {output.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if output.get(key) != value
    ]

    if problem := output_url_problem(output.get("url"), job.get("id")):
        problems.append(problem)

    seed = output.get("seed")
    if "seed" in request:
        if seed != request["seed"]:
            problems.append(f"seed is {seed!r}, expected {request['seed']!r}")
    elif not (is_int(seed) and 0 <= seed <= SEED_MAX):
        problems.append(f"seed is {seed!r}, expected the one chosen")

    duration = output.get("duration")
    expected_duration = num_frames / model.fps
    if not (
        isinstance(duration, (int, float))
        and not isinstance(duration, bool)
        and math.isclose(
            duration, expected_duration, abs_tol=DURATION_TOLERANCE_SECONDS
        )
    ):
        problems.append(
            f"duration is {duration!r}, expected ~{expected_duration:.2f} s"
        )

    audio = output.get("audio")
    if not model.audio:
        if audio is not None:
            problems.append(f"audio is {audio!r}, expected null")
    elif not (
        isinstance(audio, dict)
        and audio.get("codec") == "aac"
        and is_int(audio.get("sample_rate"))
        and audio["sample_rate"] > 0
        and audio.get("channels") in (1, 2)
    ):
        problems.append(
            f"audio is {audio!r}, expected an aac track's codec, sample_rate and channels"
        )
    return problems


def clip_problems(
    video_path: str | Path,
    model: VideoModel,
    shape: tuple[int, int],
    num_frames: int,
) -> list[str]:
    """Why a downloaded clip is not the video its output describes.

    The clip decodes, is not black, flat or frozen, has ``num_frames`` frames
    at ``shape``, and is an H.264 High, yuv420p, faststart MP4 with an audio
    track when the model makes sound.
    """
    from .video_output_check import check_mp4_container
    from .video_quality_metrics import (
        MissingVideoQualityDependency,
        probe_video,
        sample_video_frames,
        structural_frame_metrics,
    )

    path = Path(video_path)
    try:
        structural = structural_frame_metrics(sample_video_frames(path))
    except (ValueError, MissingVideoQualityDependency) as e:
        return [f"clip: could not be checked: {e}"]
    problems = [
        f"clip: {kind} ({structural[f'{kind}_frame_fraction']:.0%} of sampled frames)"
        for kind in ("black", "flat")
        if structural[f"is_{kind}"]
    ]
    if structural["is_frozen"]:
        problems.append(
            f"clip: frozen ({structural['frozen_pair_fraction']:.0%} of sampled "
            "frame pairs unchanged)"
        )
    decoded = structural["total_decoded_frames"]
    if decoded != num_frames:
        problems.append(f"clip: {decoded} frames, expected {num_frames}")
    width, height = shape
    try:
        probe = probe_video(
            path,
            expected_duration=num_frames / model.fps,
            expected_ratio=f"{width}:{height}",
        )
    except (ValueError, MissingVideoQualityDependency) as e:
        problems.append(f"clip: could not be probed: {e}")
    else:
        if (probe["width"], probe["height"]) != (width, height):
            problems.append(
                f"clip: {probe['width']}x{probe['height']}, expected {width}x{height}"
            )
    container = check_mp4_container(path, audio=model.audio)
    if container:
        problems.append(f"container: {container}")
    return problems


def solid_png(width: int, height: int, rgb: tuple[int, int, int] = (72, 118, 170)):
    """A one-colour ``width`` x ``height`` PNG, a few hundred bytes at any size.

    For requests whose image only has to have a given shape.
    """

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    row = b"\x00" + bytes(rgb) * width
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(row * height, 9))
        + chunk(b"IEND", b"")
    )


def tone_wav(seconds: float = 3.0, sample_rate: int = 16000) -> bytes:
    """A mono 440 Hz tone as a WAV file: a reference audio clip of ``seconds``."""
    import io
    import wave

    samples = int(seconds * sample_rate)
    frames = b"".join(
        struct.pack(
            "<h", int(12000 * math.sin(2 * math.pi * 440 * index / sample_rate))
        )
        for index in range(samples)
    )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(frames)
    return buffer.getvalue()


def media_data_uri(data: bytes, content_type: str) -> str:
    """``data`` as the data URI the API takes inline, of the given media type."""
    import base64

    return f"data:{content_type};base64,{base64.b64encode(data).decode('ascii')}"


__all__ = [
    "H3_CLIP_MAX_SECONDS",
    "H3_CLIP_MIN_SECONDS",
    "H3_KEYFRAME_MAX_SIDE",
    "H3_KEYFRAME_MIN_SIDE",
    "H3_RATIOS",
    "LTX_SHAPE",
    "MODELS",
    "VIDEO_CONTENT_TYPE",
    "VideoModel",
    "clip_problems",
    "h3_canvas",
    "h3_num_frames",
    "is_video_generation_api_model",
    "ltx_num_frames",
    "media_data_uri",
    "model_for",
    "requested_shape",
    "solid_png",
    "tone_wav",
    "video_output_problems",
]
