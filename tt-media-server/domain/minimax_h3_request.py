# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""The MiniMax-H3 requests: text, optionally with first/last keyframes (FL2VA),
or text with reference images, videos and audio (Ref2VA), to a video with its
soundtrack. A deployment serves FL2VA or Ref2VA, which load different
transformers, so each is its own model.

The served values and the checks over them come from tt-metal's
models/tt_dit/pipelines/minimax_h3/policy.py; this module defines only the
API's wiring and the product rules tt-metal does not have.
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
from models.tt_dit.pipelines.minimax_h3.policy import (
    MINIMAX_H3_ASPECT_RATIOS,
    MINIMAX_H3_CANVAS_MULTIPLE,
    MINIMAX_H3_DEFAULT_ASPECT_RATIO,
    MINIMAX_H3_DEFAULT_DURATION_S,
    MINIMAX_H3_DURATIONS_S,
    MINIMAX_H3_FPS,
    MINIMAX_H3_FRAMES_PER_CHUNK,
    MINIMAX_H3_KEYFRAME_MAX_SIDE,
    MINIMAX_H3_KEYFRAME_MIN_SIDE,
    MINIMAX_H3_LATENTS_PER_CHUNK,
    MINIMAX_H3_MAX_ASPECT_RATIO,
    MINIMAX_H3_MAX_PIXELS,
    MINIMAX_H3_MAX_REFERENCE_AUDIOS,
    MINIMAX_H3_MAX_REFERENCE_IMAGES,
    MINIMAX_H3_MAX_REFERENCE_VIDEOS,
    MINIMAX_H3_MAX_REFERENCES,
    MINIMAX_H3_MIN_ASPECT_RATIO,
    MINIMAX_H3_SHORT_EDGE,
    MiniMaxH3AdmissionError,
    MiniMaxH3ReferenceInfo,
    admit_request,
    check_reference_kinds,
    get_num_frames,
    minimax_h3_parse_aspect_ratio,
    resolve_canvas_size,
)
from pydantic import BeforeValidator, Field, field_validator, model_validator


def _ratio_text(ratio: tuple[int, int]) -> str:
    return "{}:{}".format(*ratio)


def _normalized_ratio(value):
    """``16x9``, ``16/9`` and `` 16 : 9 `` as ``16:9``, as the earlier H3 API
    took them. An unparseable or unserved ratio gets tt-metal's error."""
    if not isinstance(value, str) or value == "adaptive":
        return value
    return _ratio_text(minimax_h3_parse_aspect_ratio(value))


H3_RATIOS = tuple(_ratio_text(ratio) for ratio in MINIMAX_H3_ASPECT_RATIOS)
H3_DEFAULT_RATIO = _ratio_text(MINIMAX_H3_DEFAULT_ASPECT_RATIO)
H3Ratio = Annotated[Literal[H3_RATIOS], BeforeValidator(_normalized_ratio)]
# `adaptive` follows the keyframe's shape.
H3KeyframesRatio = Annotated[
    Literal[("adaptive", *H3_RATIOS)], BeforeValidator(_normalized_ratio)
]
_SIZE = f"{MINIMAX_H3_SHORT_EDGE}p"
H3Size = Literal[_SIZE]
# Ref2VA reference clips: the product's limits; the pipeline truncates instead.
H3_CLIP_MIN_S = 2.0
H3_CLIP_MAX_S = 15.0
H3_CLIPS_MAX_S = 15.0


def h3_canvas(width: float, height: float) -> tuple[int, int]:
    """(width, height) of the output for a shape."""
    canvas_height, canvas_width = resolve_canvas_size(width, height)
    return canvas_width, canvas_height


_RATIO_RANGE = (
    f"1:{round(1 / MINIMAX_H3_MIN_ASPECT_RATIO)} to "
    f"{round(MINIMAX_H3_MAX_ASPECT_RATIO)}:1"
)
_SIZE_RULE = (
    f"a short edge of {MINIMAX_H3_SHORT_EDGE}, the area capped at "
    f"{MINIMAX_H3_SHORT_EDGE}×{MINIMAX_H3_MAX_PIXELS // MINIMAX_H3_SHORT_EDGE}, "
    f"sides multiples of {MINIMAX_H3_CANVAS_MULTIPLE}"
)
_FRAME_RULE = f"{MINIMAX_H3_FRAMES_PER_CHUNK}k + {MINIMAX_H3_LATENTS_PER_CHUNK}"
_CLIP_RANGE = f"{H3_CLIP_MIN_S:g}–{H3_CLIP_MAX_S:g} s"


def _frames_for(duration: int) -> str:
    frames = get_num_frames(duration)
    return f"{duration} s gives {frames} frames ({frames / MINIMAX_H3_FPS:.2f} s)"


def _size_table() -> str:
    def line(ratio: str, size: str) -> str:
        return f"| {ratio:<14} | {size:<9} |"

    rows = [line("`aspect_ratio`", f"`{_SIZE}`"), line("-" * 14, "-" * 9)]
    for ratio in MINIMAX_H3_ASPECT_RATIOS:
        width, height = h3_canvas(*ratio)
        rows.append(line(f"`{_ratio_text(ratio)}`", f"{width}×{height}"))
    return "\n".join(rows)


_H3_SIZE_TABLE = _size_table()


class _MiniMaxH3Request(GenerationRequest):
    OUTPUT_TYPE = "video"
    # Reference images can each decode to hundreds of MB, too much to send to
    # every rank of a multi-host mesh.
    RUNNER_DECODES_IMAGES = True

    prompt: str = Field(min_length=1)
    aspect_ratio: H3Ratio = Field(
        default=H3_DEFAULT_RATIO,
        description="Output shape. Each output reports its exact size.",
    )
    size: H3Size = Field(
        default=_SIZE,
        description=(
            f"Size tier: {_SIZE_RULE}:\n\n{_H3_SIZE_TABLE}\n\nOnly `{_SIZE}` for "
            "now; kept so other tiers can be added without a schema change."
        ),
    )
    duration: int = Field(
        default=MINIMAX_H3_DEFAULT_DURATION_S,
        ge=MINIMAX_H3_DURATIONS_S[0],
        le=MINIMAX_H3_DURATIONS_S[-1],
        description=(
            f"Seconds. The video runs at {MINIMAX_H3_FPS} fps and its length is "
            "rounded up to the frame counts the model generates "
            f"({_FRAME_RULE}): {_frames_for(MINIMAX_H3_DEFAULT_DURATION_S)}. Each "
            "output reports its exact `duration` and `num_frames`."
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
        return MINIMAX_H3_FPS

    @property
    def frames(self) -> int:
        return get_num_frames(self.duration)

    def output_size(self) -> tuple[int, int]:
        return h3_canvas(*map(int, self.aspect_ratio.split(":")))

    @classmethod
    def warmup_sizes(cls) -> list[tuple[int, int]]:
        # The pipeline warms itself up.
        return []

    def _admission_errors(
        self,
        request_field: str,
        locs: dict[str, tuple[str, int]],
        **media,
    ) -> list[tuple[str, int | None, str]]:
        """tt-metal's admission check, each error at its input's ``locs``
        entry, or at ``request_field`` when it is about the request as a
        whole."""
        try:
            admit_request(
                aspect_ratio=None
                if self.aspect_ratio == "adaptive"
                else self.aspect_ratio,
                duration_s=self.duration,
                height=None,
                width=None,
                **media,
            )
        except MiniMaxH3AdmissionError as e:
            return [
                (*locs[name], message)
                if name is not None
                else (request_field, None, message)
                for name, message in e.errors
            ]
        return []


class _MiniMaxH3KeyframesRequest(_MiniMaxH3Request):
    MEDIA_FIELDS = ("first_frame", "last_frame")

    first_frame: Optional[Media] = Field(
        default=None,
        description=(
            f"The frame the video starts on. {MINIMAX_H3_KEYFRAME_MIN_SIDE}–"
            f"{MINIMAX_H3_KEYFRAME_MAX_SIDE} pixels per side, aspect ratio from "
            f"{_RATIO_RANGE}."
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
            f"- Without frames: one of the listed ratios, default `{H3_DEFAULT_RATIO}`; "
            "`adaptive` is rejected with 422.\n\n"
            "Each output reports its exact size."
        ),
    )
    size: H3Size = Field(
        default=_SIZE,
        description=(
            f"Size tier: {_SIZE_RULE}. For the listed ratios:\n\n"
            f"{_H3_SIZE_TABLE}\n\n"
            "`adaptive` applies the same rule to the frame's own shape. Only "
            f"`{_SIZE}` for now; kept so other tiers can be added without a schema "
            "change."
        ),
    )
    duration: int = Field(
        default=MINIMAX_H3_DEFAULT_DURATION_S,
        ge=MINIMAX_H3_DURATIONS_S[0],
        le=MINIMAX_H3_DURATIONS_S[-1],
        description=(
            f"Seconds. The video runs at {MINIMAX_H3_FPS} fps and its length is "
            "rounded up to the frame counts the model generates "
            f"({_FRAME_RULE}), so it is slightly longer: "
            f"{_frames_for(MINIMAX_H3_DEFAULT_DURATION_S)}, "
            f"{_frames_for(MINIMAX_H3_DURATIONS_S[-1])}. Each output reports its "
            "exact `duration` and `num_frames`."
        ),
    )

    @model_validator(mode="after")
    def _adaptive_with_frames(self):
        has_frame = self.first_frame is not None or self.last_frame is not None
        if self.aspect_ratio is None:
            self.aspect_ratio = "adaptive" if has_frame else H3_DEFAULT_RATIO
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

    def media_errors(self) -> list[tuple[str, int | None, str]]:
        errors = []
        keyframe_sizes = {}
        for name, images in self._images.items():
            width, height = keyframe_sizes[name] = images[0].size
            # tt-metal leaves the side limits to the server.
            if not (
                MINIMAX_H3_KEYFRAME_MIN_SIDE <= min(width, height)
                and max(width, height) <= MINIMAX_H3_KEYFRAME_MAX_SIDE
            ):
                errors.append(
                    (
                        name,
                        0,
                        f"is {width}×{height}; each side must be from "
                        f"{MINIMAX_H3_KEYFRAME_MIN_SIDE} to "
                        f"{MINIMAX_H3_KEYFRAME_MAX_SIDE} px",
                    )
                )
        return errors + self._admission_errors(
            "first_frame",
            {name: (name, 0) for name in keyframe_sizes},
            keyframe_sizes=keyframe_sizes,
        )

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
        max_length=MINIMAX_H3_MAX_REFERENCE_IMAGES,
        description=f"Reference images, aspect ratio from {_RATIO_RANGE}.",
    )
    videos: list[VideoMedia] = Field(
        default_factory=list,
        max_length=MINIMAX_H3_MAX_REFERENCE_VIDEOS,
        description=(
            f"Reference video clips, {_CLIP_RANGE} each, at most "
            f"{H3_CLIPS_MAX_S:g} s combined. A clip's soundtrack, if it has one, "
            "is also audio conditioning and counts toward the audio limits (see "
            "`audios`)."
        ),
    )
    audios: list[AudioMedia] = Field(
        default_factory=list,
        max_length=MINIMAX_H3_MAX_REFERENCE_AUDIOS,
        description=(
            f"Reference audio clips, {_CLIP_RANGE} each. The audio limits cover "
            "these clips plus the soundtrack of every reference video that has "
            f"one: at most {MINIMAX_H3_MAX_REFERENCE_AUDIOS} audio references and "
            f"at most {H3_CLIPS_MAX_S:g} s of audio combined, or 422."
        ),
    )
    aspect_ratio: H3Ratio = Field(
        default=H3_DEFAULT_RATIO,
        description=(
            "Output shape, as for text-to-video: the references don't set it. "
            "Each output reports its exact size."
        ),
    )

    @model_validator(mode="after")
    def _references(self):
        # At parse time, before any input is fetched; admit_request checks the
        # same again at submit.
        check_reference_kinds(
            ["image"] * len(self.images)
            + ["video"] * len(self.videos)
            + ["audio"] * len(self.audios)
        )
        return self

    def media_errors(self) -> list[tuple[str, int | None, str]]:
        errors = []
        references = []
        locs = {}

        def add(name: str, index: int, **info) -> None:
            kind = self.media_kind(name)
            reference = MiniMaxH3ReferenceInfo(
                name=f"{name}[{index}]", kind=kind, **info
            )
            references.append(reference)
            locs[reference.name] = (name, index)

        for index, image in enumerate(self._images.get("images", [])):
            add("images", index, size=image.size)
        for name in ("videos", "audios"):
            for index, probe in enumerate(self._probes.get(name, [])):
                add(
                    name,
                    index,
                    size=probe.size,
                    fps=probe.fps,
                    audio_channels=probe.audio_channels,
                )
                if not H3_CLIP_MIN_S <= probe.duration_s <= H3_CLIP_MAX_S:
                    errors.append(
                        (
                            name,
                            index,
                            f"is {probe.duration_s:g} s; each clip must "
                            f"be {H3_CLIP_MIN_S:g} to {H3_CLIP_MAX_S:g} s",
                        )
                    )
                # admit_request checks a reference's shape only when it has one.
                if name == "videos" and probe.size is None:
                    errors.append((name, index, "has no video stream"))
        errors += self._admission_errors("references", locs, references=references)
        videos = self._probes.get("videos", [])
        total = sum(probe.duration_s for probe in videos)
        if total > H3_CLIPS_MAX_S:
            errors.append(
                (
                    "videos",
                    None,
                    f"the clips are {total:g} s together; at most {H3_CLIPS_MAX_S:g} s",
                )
            )
        # tt-metal counts references by kind; a video's soundtrack also counts
        # toward the audio limits here. The soundtracks of the videos alone stay
        # within them, or the videos are over theirs.
        audios = self._probes.get("audios", [])
        if audios:
            sounds = audios + [probe for probe in videos if probe.audio_channels]
            if len(sounds) > MINIMAX_H3_MAX_REFERENCE_AUDIOS:
                errors.append(
                    (
                        "audios",
                        None,
                        f"with the videos' soundtracks, "
                        f"{len(sounds)} audio references; at most "
                        f"{MINIMAX_H3_MAX_REFERENCE_AUDIOS}",
                    )
                )
            total = sum(probe.duration_s for probe in sounds)
            if total > H3_CLIPS_MAX_S:
                errors.append(
                    (
                        "audios",
                        None,
                        f"with the videos' soundtracks, "
                        f"{total:g} s of audio together; at most {H3_CLIPS_MAX_S:g} s",
                    )
                )
        return errors


class MiniMaxH3Ref2VARequest(_MiniMaxH3ReferencesRequest):
    __doc__ = f"""Shared by every MiniMax-H3 Ref2VA deployment, whatever its sampling
    (e.g. `tenstorrent/minimax-h3-ref2va-lightx2v`).

    References: at most {MINIMAX_H3_MAX_REFERENCES} in total, and at least one image or video (audio
    alone is rejected with 422). Video and audio clips are {_CLIP_RANGE} each and
    at most {H3_CLIPS_MAX_S:g} s combined per kind.

    A reference video's soundtrack is audio conditioning: it counts toward
    the audio limits (at most {MINIMAX_H3_MAX_REFERENCE_AUDIOS} audio references, at most {H3_CLIPS_MAX_S:g} s of audio
    combined) as well as the video ones. A video with sound counts once
    toward the {MINIMAX_H3_MAX_REFERENCES}-reference total."""


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
