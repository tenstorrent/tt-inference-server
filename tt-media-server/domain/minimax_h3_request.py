# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""The MiniMax-H3 requests: text, optionally with first/last keyframes (FL2VA),
or text with reference images, videos and audio (Ref2VA), to a video with its
soundtrack. A deployment serves FL2VA or Ref2VA, which load different
transformers, so each is its own model.

The served envelope mirrors tt-metal's models/tt_dit/pipelines/minimax_h3
(packing.py and policy.py), which also rejects anything outside it.
"""

from typing import Annotated, Literal, Optional

from domain.generation_request import (
    EXAMPLE_AUDIO_URL,
    EXAMPLE_IMAGE_URL,
    EXAMPLE_VIDEO_URL,
    AudioMedia,
    GenerationModel,
    GenerationRequest,
    Media,
    Seed,
    VideoMedia,
)
from pydantic import BeforeValidator, Field, field_validator, model_validator


def _normalized_ratio(value):
    """``16x9``, ``16/9`` and `` 16 : 9 `` as ``16:9``, as the earlier H3 API
    took them."""
    if not isinstance(value, str):
        return value
    parts = value.strip().replace("x", ":").replace("/", ":").split(":")
    if len(parts) != 2 or not all(part.strip().isdigit() for part in parts):
        return value
    return f"{int(parts[0])}:{int(parts[1])}"


H3Ratio = Annotated[
    Literal["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"],
    BeforeValidator(_normalized_ratio),
]
# `adaptive` follows the keyframe's shape.
H3KeyframesRatio = Annotated[
    Literal["adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"],
    BeforeValidator(_normalized_ratio),
]
H3_FPS = 24
H3_SHORT_EDGE = 768
H3_MAX_PIXELS = 768 * 1344
H3_CANVAS_MULTIPLE = 32
# The VAE encodes 17n + 5 frames.
H3_FRAMES_PER_CHUNK = 17
H3_LATENTS_PER_CHUNK = 5
H3_KEYFRAME_MIN_SIDE = 256
H3_KEYFRAME_MAX_SIDE = 5760
# Ref2VA reference clips: the product's limits; the pipeline truncates instead.
H3_CLIP_MIN_S = 2.0
H3_CLIP_MAX_S = 15.0
H3_CLIPS_MAX_S = 15.0
# The audio clips plus the reference videos with a soundtrack.
H3_AUDIO_REFERENCES_MAX = 3


def h3_num_frames(duration: int) -> int:
    """The frames generated for ``duration`` seconds: the 24 fps count, rounded
    up to the next 17n + 5."""
    frames = round(duration * H3_FPS)
    while frames % H3_FRAMES_PER_CHUNK != H3_LATENTS_PER_CHUNK:
        frames += 1
    return frames


def h3_canvas(width: float, height: float) -> tuple[int, int]:
    """(width, height) of the output for a shape: the short side scaled to 768,
    both scaled down if the area is then above 768 × 1344, each rounded to a
    multiple of 32."""
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


def _within_1_4_and_4_1(width: int, height: int) -> bool:
    return width <= 4 * height and height <= 4 * width


_H3_SIZE_TABLE = (
    "| `aspect_ratio` | `768p`    |\n"
    "| -------------- | --------- |\n"
    "| `21:9`         | 1536×672  |\n"
    "| `16:9`         | 1344×768  |\n"
    "| `4:3`          | 1024×768  |\n"
    "| `1:1`          | 768×768   |\n"
    "| `3:4`          | 768×1024  |\n"
    "| `9:16`         | 768×1344  |"
)


class _MiniMaxH3Request(GenerationRequest):
    OUTPUT_TYPE = "video"
    # Reference images can each decode to hundreds of MB, too much to send to
    # every rank of a multi-host mesh.
    RUNNER_DECODES_IMAGES = True

    prompt: str = Field(min_length=1)
    aspect_ratio: H3Ratio = Field(
        default="16:9",
        description="Output shape. Each output reports its exact size.",
    )
    size: Literal["768p"] = Field(
        default="768p",
        description=(
            "Size tier: a short edge of 768, the area capped at 768×1344, sides "
            f"multiples of 32:\n\n{_H3_SIZE_TABLE}\n\nOnly `768p` for now; kept so "
            "other tiers can be added without a schema change."
        ),
    )
    duration: int = Field(
        default=5,
        ge=4,
        le=15,
        description=(
            "Seconds. The video runs at 24 fps and its length is rounded up to the "
            "frame counts the model generates (17k + 5): 5 s gives 124 frames "
            "(5.17 s). Each output reports its exact `duration` and `num_frames`."
        ),
    )
    seed: Optional[Seed] = None

    @field_validator("prompt")
    @classmethod
    def _prompt_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be empty")
        return value

    @property
    def fps(self) -> int:
        return H3_FPS

    @property
    def frames(self) -> int:
        return h3_num_frames(self.duration)

    def output_size(self) -> tuple[int, int]:
        return h3_canvas(*map(int, self.aspect_ratio.split(":")))

    @classmethod
    def warmup_sizes(cls) -> list[tuple[int, int]]:
        # The pipeline warms itself up.
        return []


class _MiniMaxH3KeyframesRequest(_MiniMaxH3Request):
    MEDIA_FIELDS = ("first_frame", "last_frame")

    first_frame: Optional[Media] = Field(
        default=None,
        description=(
            "The frame the video starts on. 256–5760 pixels per side, aspect "
            "ratio from 1:4 to 4:1."
        ),
    )
    last_frame: Optional[Media] = Field(
        default=None,
        description=(
            "The frame the video ends on. Allowed with or without `first_frame`. "
            "Same limits as `first_frame`."
        ),
    )
    aspect_ratio: Optional[H3KeyframesRatio] = Field(
        default=None,
        description=(
            "Output shape.\n\n"
            "- With a frame: only `adaptive` (the default), meaning the output "
            "follows the first frame's shape, or the last frame's when there is no "
            "first frame. Any other value is rejected with 422.\n"
            "- Without frames: one of the listed ratios, default `16:9`; "
            "`adaptive` is rejected with 422.\n\n"
            "Each output reports its exact size."
        ),
    )
    size: Literal["768p"] = Field(
        default="768p",
        description=(
            "Size tier: a short edge of 768, the area capped at 768×1344, sides "
            "multiples of 32. For the listed ratios:\n\n"
            f"{_H3_SIZE_TABLE}\n\n"
            "`adaptive` applies the same rule to the frame's own shape. Only "
            "`768p` for now; kept so other tiers can be added without a schema "
            "change."
        ),
    )
    duration: int = Field(
        default=5,
        ge=4,
        le=15,
        description=(
            "Seconds. The video runs at 24 fps and its length is rounded up to the "
            "frame counts the model generates (17k + 5), so it is slightly longer: "
            "5 s gives 124 frames (5.17 s), 15 s gives 362 frames (15.08 s). Each "
            "output reports its exact `duration` and `num_frames`."
        ),
    )

    @model_validator(mode="after")
    def _adaptive_with_frames(self):
        has_frame = self.first_frame is not None or self.last_frame is not None
        if self.aspect_ratio is None:
            self.aspect_ratio = "adaptive" if has_frame else "16:9"
            # A default, not the caller's choice (model_fields_set).
            self.__pydantic_fields_set__.discard("aspect_ratio")
        elif has_frame and self.aspect_ratio != "adaptive":
            raise ValueError(
                "with a first_frame or last_frame, aspect_ratio must be adaptive: "
                "the output follows the frame's shape"
            )
        elif not has_frame and self.aspect_ratio == "adaptive":
            raise ValueError("aspect_ratio adaptive needs a first_frame or last_frame")
        return self

    def media_errors(self) -> list[tuple[str, int, str]]:
        errors = []
        for name, images in self._images.items():
            width, height = images[0].size
            if not (
                H3_KEYFRAME_MIN_SIDE <= min(width, height)
                and max(width, height) <= H3_KEYFRAME_MAX_SIDE
            ):
                errors.append(
                    (
                        name,
                        0,
                        f"is {width}×{height}; each side must be from "
                        f"{H3_KEYFRAME_MIN_SIDE} to {H3_KEYFRAME_MAX_SIDE} px",
                    )
                )
            elif not _within_1_4_and_4_1(width, height):
                errors.append(
                    (name, 0, f"is {width}×{height}; the ratio must be from 1:4 to 4:1")
                )
        return errors

    def output_size(self) -> tuple[int, int]:
        if self.aspect_ratio == "adaptive":
            keyframe = self._images.get("first_frame") or self._images["last_frame"]
            return h3_canvas(*keyframe[0].size)
        return super().output_size()


class MiniMaxH3FL2VARequest(_MiniMaxH3KeyframesRequest):
    """Shared by every MiniMax-H3 FL2VA deployment, whatever its sampling
    (e.g. `tenstorrent/minimax-h3-fl2va-lightx2v`)."""


_FL2VA_EXAMPLE = {
    "prompt": (
        "The plumeria flower sways in a warm breeze as soft morning light moves "
        "across its petals."
    ),
    "first_frame": EXAMPLE_IMAGE_URL,
    "duration": 5,
}
MINIMAX_H3_FL2VA = GenerationModel(
    id="minimaxai/minimax-h3-fl2va",
    request=MiniMaxH3FL2VARequest,
    title="MiniMax-H3 FL2VA",
    summary="Submit a MiniMax-H3 FL2VA job",
    description=(
        "Video with sound from text, optionally pinned to a first and/or last "
        "frame. Runs the `FL2VA` checkpoint of `MiniMaxAI/MiniMax-H3`; without "
        "frames it is text-to-video."
    ),
    example=_FL2VA_EXAMPLE,
)
MINIMAX_H3_FL2VA_LIGHTX2V = GenerationModel(
    id="tenstorrent/minimax-h3-fl2va-lightx2v",
    request=MiniMaxH3FL2VARequest,
    title="MiniMax-H3 FL2VA LightX2V",
    summary="Submit a MiniMax-H3 FL2VA LightX2V job",
    description=(
        "MiniMax-H3 FL2VA with lightx2v's Turbo distillation adapter: the same "
        "request as `minimaxai/minimax-h3-fl2va`, generated in far fewer "
        "denoising steps, so faster, with somewhat different output."
    ),
    example=_FL2VA_EXAMPLE,
)


class _MiniMaxH3ReferencesRequest(_MiniMaxH3Request):
    MEDIA_FIELDS = ("images", "videos", "audios")
    MEDIA_KINDS = {"videos": "video", "audios": "audio"}

    prompt: str = Field(
        min_length=1,
        description=(
            "Refer to references by kind and position, numbered per kind in the "
            "order given: `Picture 1` (the first of `images`), `Video 1`, `Audio "
            "1`. The model sees each reference under that label. A video's own "
            "sound track counts as an audio reference too."
        ),
    )
    images: list[Media] = Field(
        default_factory=list,
        max_length=9,
        description="Reference images, aspect ratio from 1:4 to 4:1.",
    )
    videos: list[VideoMedia] = Field(
        default_factory=list,
        max_length=3,
        description=(
            "Reference video clips, 2–15 s each, at most 15 s combined. A clip's "
            "soundtrack, if it has one, is also audio conditioning and counts "
            "toward the audio limits (see `audios`)."
        ),
    )
    audios: list[AudioMedia] = Field(
        default_factory=list,
        max_length=3,
        description=(
            "Reference audio clips, 2–15 s each. The audio limits cover these "
            "clips plus the soundtrack of every reference video that has one: at "
            "most 3 audio references and at most 15 s of audio combined, or 422."
        ),
    )
    aspect_ratio: H3Ratio = Field(
        default="16:9",
        description=(
            "Output shape, as for text-to-video: the references don't set it. "
            "Each output reports its exact size."
        ),
    )

    @model_validator(mode="after")
    def _references(self):
        if not (self.images or self.videos):
            raise ValueError("needs at least one reference image or video")
        if len(self.images) + len(self.videos) + len(self.audios) > 12:
            raise ValueError("at most 12 references in total")
        return self

    def media_errors(self) -> list[tuple[str, int, str]]:
        errors = []
        for index, image in enumerate(self._images.get("images", [])):
            if not _within_1_4_and_4_1(*image.size):
                errors.append(
                    (
                        "images",
                        index,
                        f"is {image.size[0]}×{image.size[1]}; the "
                        "ratio must be from 1:4 to 4:1",
                    )
                )
        for name in ("videos", "audios"):
            probes = self._probes.get(name, [])
            for index, probe in enumerate(probes):
                if not H3_CLIP_MIN_S <= probe.duration_s <= H3_CLIP_MAX_S:
                    errors.append(
                        (
                            name,
                            index,
                            f"is {probe.duration_s:g} s; each clip must "
                            f"be {H3_CLIP_MIN_S:g} to {H3_CLIP_MAX_S:g} s",
                        )
                    )
                if name == "videos":
                    if probe.size is None:
                        errors.append((name, index, "has no video stream"))
                    elif not _within_1_4_and_4_1(*probe.size):
                        errors.append(
                            (
                                name,
                                index,
                                f"is {probe.size[0]}×{probe.size[1]}; "
                                "the ratio must be from 1:4 to 4:1",
                            )
                        )
                    if not probe.fps:
                        errors.append((name, index, "reports no frame rate"))
                elif probe.audio_channels is None:
                    errors.append((name, index, "has no audio stream"))
                if probe.audio_channels not in (None, 1, 2):
                    errors.append(
                        (
                            name,
                            index,
                            f"has {probe.audio_channels} audio channels; "
                            "must be mono or stereo",
                        )
                    )
        videos = self._probes.get("videos", [])
        total = sum(probe.duration_s for probe in videos)
        if total > H3_CLIPS_MAX_S:
            errors.append(
                (
                    "videos",
                    len(videos) - 1,
                    f"the clips are {total:g} s together; at most {H3_CLIPS_MAX_S:g} s",
                )
            )
        # The soundtracks of the videos alone stay within the audio limits, or
        # the videos are over theirs; going over takes an audio clip, and the
        # error points at the last one.
        audios = self._probes.get("audios", [])
        if audios:
            sounds = audios + [probe for probe in videos if probe.audio_channels]
            if len(sounds) > H3_AUDIO_REFERENCES_MAX:
                errors.append(
                    (
                        "audios",
                        len(audios) - 1,
                        f"with the videos' soundtracks, "
                        f"{len(sounds)} audio references; at most "
                        f"{H3_AUDIO_REFERENCES_MAX}",
                    )
                )
            total = sum(probe.duration_s for probe in sounds)
            if total > H3_CLIPS_MAX_S:
                errors.append(
                    (
                        "audios",
                        len(audios) - 1,
                        f"with the videos' soundtracks, "
                        f"{total:g} s of audio together; at most {H3_CLIPS_MAX_S:g} s",
                    )
                )
        return errors


class MiniMaxH3Ref2VARequest(_MiniMaxH3ReferencesRequest):
    """Shared by every MiniMax-H3 Ref2VA deployment, whatever its sampling
    (e.g. `tenstorrent/minimax-h3-ref2va-lightx2v`).

    References: at most 12 in total, and at least one image or video (audio
    alone is rejected with 422). Video and audio clips are 2–15 s each and
    at most 15 s combined per kind.

    A reference video's soundtrack is audio conditioning: it counts toward
    the audio limits (at most 3 audio references, at most 15 s of audio
    combined) as well as the video ones. A video with sound counts once
    toward the 12-reference total."""


_REF2VA_EXAMPLE = {
    "prompt": (
        "The plumeria from Picture 1 sways in the garden from Video 1 while the "
        "wren from Audio 1 sings nearby."
    ),
    "images": [EXAMPLE_IMAGE_URL],
    "videos": [EXAMPLE_VIDEO_URL],
    "audios": [EXAMPLE_AUDIO_URL],
    "duration": 5,
}
MINIMAX_H3_REF2VA = GenerationModel(
    id="minimaxai/minimax-h3-ref2va",
    request=MiniMaxH3Ref2VARequest,
    title="MiniMax-H3 Ref2VA",
    summary="Submit a MiniMax-H3 Ref2VA job",
    description=(
        "Video with sound from text plus reference images, videos and audio "
        "clips. Runs the `Ref2VA` checkpoint of `MiniMaxAI/MiniMax-H3`."
    ),
    example=_REF2VA_EXAMPLE,
)
MINIMAX_H3_REF2VA_LIGHTX2V = GenerationModel(
    id="tenstorrent/minimax-h3-ref2va-lightx2v",
    request=MiniMaxH3Ref2VARequest,
    title="MiniMax-H3 Ref2VA LightX2V",
    summary="Submit a MiniMax-H3 Ref2VA LightX2V job",
    description=(
        "MiniMax-H3 Ref2VA with lightx2v's Turbo distillation adapter: the same "
        "request as `minimaxai/minimax-h3-ref2va`, generated in far fewer "
        "denoising steps, so faster, with somewhat different output."
    ),
    example=_REF2VA_EXAMPLE,
)
