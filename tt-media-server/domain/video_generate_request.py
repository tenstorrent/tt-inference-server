# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Text-to-video request schema.

LTX shape fields (``height``/``width``/``fps``, and the frame count as
``duration`` or ``num_frames``) are validated against the shape baked into the
running pipeline's traces, not honoured as free variables; ``None`` means "use
the served config". MiniMax-H3 selects its shape with ``aspect_ratio`` +
``duration_seconds``. Each model refuses the other's shape fields.
"""

from typing import Optional

from config.constants import (
    DEFAULT_VIDEO_INFERENCE_STEPS,
    LTX_NUM_INFERENCE_STEPS,
    MAX_VIDEO_INFERENCE_STEPS,
    MIN_VIDEO_INFERENCE_STEPS,
    ModelRunners,
    ltx_served_shape,
    snap_num_frames,
)
from config.settings import get_settings
from domain.base_request import BaseRequest
from pydantic import Field, field_validator, model_validator

# Shape fields only LTX reads; MiniMax-H3 treats them as unknown.
_LTX_SHAPE_FIELDS = frozenset({"height", "width", "fps", "duration", "num_frames"})


class VideoGenerateRequest(BaseRequest):
    # Required fields
    prompt: str

    # Optional fields
    negative_prompt: Optional[str] = None
    # None = the model's default; resolved in _validate_shape.
    num_inference_steps: Optional[int] = Field(
        default=None,
        ge=MIN_VIDEO_INFERENCE_STEPS,
        le=MAX_VIDEO_INFERENCE_STEPS,
    )
    seed: Optional[int] = None

    # TODO: Make generic for all video models, and remove model specific logic
    # Output shape. Both are model-specific and both are optional: a model that serves one fixed
    # shape ignores them, and MiniMax-H3 t2va resolves them against its published working points
    # (see `minimax_h3_parse_aspect_ratio` / `MINIMAX_H3_DURATIONS_S`). Left as free-form here
    # rather than an enum so a per-model validator can reject with a message naming what it does
    # serve -- a 422 from pydantic on a shared field cannot say that.
    aspect_ratio: Optional[str] = Field(default=None, examples=["16:9", "9:16", "1:1"])
    duration_seconds: Optional[int] = Field(
        default=None, ge=1, le=60, examples=[5, 10, 15]
    )

    # TODO: Make generic for all video models, and remove model specific logic
    # Unknown fields are refused for MiniMax-H3 rather than ignored. Pydantic's default is to
    # drop them silently, which meant `{"resolution": "1080P", "model": "NotMiniMax", "duration": 9}`
    # came back 202 with none of it applied -- the caller believes it asked for something it did
    # not get, which is worse than an error. Scoped to H3 by runner rather than set as
    # `extra="forbid"` on the class, because Wan clients share this model and may send fields this
    # deployment does not read.
    @model_validator(mode="before")
    @classmethod
    def _reject_unknown_fields(cls, data):
        if not isinstance(data, dict) or not _is_minimax_h3():
            return data
        readable = set(cls.model_fields) - _LTX_SHAPE_FIELDS
        unknown = sorted(set(data) - readable)
        if unknown:
            known = ", ".join(sorted(readable))
            raise ValueError(
                f"unknown field(s) for MiniMax-H3: {', '.join(unknown)}. "
                f"This deployment reads: {known}. Note `duration` is not one of them -- the field "
                "is `duration_seconds` -- and resolution is selected with `aspect_ratio`."
            )
        return data

    # TODO: Make generic for all video models, and remove model specific logic
    # Admission-time validation. The device worker validates too (it owns the shape it warmed),
    # but that happens after the request is queued, so the client would get a 202 and a failed job
    # instead of a straight refusal. These run at parse time and surface as a 422 naming what is
    # served. Guarded on the runner so nothing here changes Wan's behaviour.
    @field_validator("aspect_ratio")
    @classmethod
    def _validate_aspect_ratio(cls, value):
        if value is None or not _is_minimax_h3():
            return value
        from tt_model_runners.minimax_h3_policy import minimax_h3_parse_aspect_ratio

        minimax_h3_parse_aspect_ratio(value)  # raises with the supported list
        return value

    # TODO: Make generic for all video models, and remove model specific logic
    @field_validator("duration_seconds")
    @classmethod
    def _validate_duration_seconds(cls, value):
        if value is None or not _is_minimax_h3():
            return value
        from tt_model_runners.minimax_h3_policy import MINIMAX_H3_DURATIONS_S

        if value not in MINIMAX_H3_DURATIONS_S:
            raise ValueError(
                f"duration_seconds must be an integer from {min(MINIMAX_H3_DURATIONS_S)} to "
                f"{max(MINIMAX_H3_DURATIONS_S)}; got {value}"
            )
        return value

    # Shape. None = use the served config; see _validate_shape.
    height: Optional[int] = Field(default=None, gt=0)
    width: Optional[int] = Field(default=None, gt=0)
    fps: Optional[float] = Field(default=None, gt=0)
    duration: Optional[float] = Field(default=None, gt=0)
    num_frames: Optional[int] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_shape(self):
        """Resolve and validate shape + step count against the served config.

        Resolved values are written back so the job record echoes what was
        actually generated.
        """
        if get_settings().model_runner != ModelRunners.TT_LTX_2_3_DISTILLED.value:
            # Other models resolve their shape elsewhere; only default the steps.
            if self.num_inference_steps is None:
                self.num_inference_steps = DEFAULT_VIDEO_INFERENCE_STEPS
            return self

        # Refuse MiniMax-H3's shape selectors rather than silently dropping them.
        for field, hint in (
            ("aspect_ratio", "height/width"),
            ("duration_seconds", "duration or num_frames"),
        ):
            if getattr(self, field) is not None:
                raise ValueError(
                    f"{field} is not supported by LTX-2.3; use {hint} instead."
                )

        served = ltx_served_shape()
        served_duration = served.num_frames / served.fps
        detail = (
            f"this deployment serves {served.num_frames} frames at "
            f"{served.height}x{served.width}, {served.fps:g} fps "
            f"({served_duration:.2f}s)"
        )

        if self.fps is not None and float(self.fps) != served.fps:
            raise ValueError(
                f"fps={self.fps:g} is not served; {detail}. FPS is fixed at "
                "pipeline construction (it sets the audio latent length and the "
                "A/V cross-PE, both baked into the captured traces)."
            )
        if self.height is not None and self.height != served.height:
            raise ValueError(f"height={self.height} is not served; {detail}.")
        if self.width is not None and self.width != served.width:
            raise ValueError(f"width={self.width} is not served; {detail}.")

        # Frame count, from duration (snapped to the nearest legal 8k+1 value)
        # or given directly. Both is allowed only if they agree.
        frames = self.num_frames
        if self.duration is not None:
            from_duration = snap_num_frames(round(self.duration * served.fps))
            if frames is not None and frames != from_duration:
                raise ValueError(
                    f"duration={self.duration:g}s implies num_frames="
                    f"{from_duration} at {served.fps:g} fps, which contradicts "
                    f"the requested num_frames={frames}; pass one or the other."
                )
            frames = from_duration
        if frames is None:
            frames = served.num_frames
        if frames != served.num_frames:
            raise ValueError(
                f"num_frames={frames} is not served; {detail}. Frame counts must "
                f"satisfy (num_frames - 1) % 8 == 0, so a duration in seconds is "
                f"snapped to the nearest legal value before this check."
            )

        if self.num_inference_steps is None:
            self.num_inference_steps = LTX_NUM_INFERENCE_STEPS
        elif self.num_inference_steps != LTX_NUM_INFERENCE_STEPS:
            raise ValueError(
                f"num_inference_steps={self.num_inference_steps} is not "
                f"supported; this model runs a fixed distilled schedule of "
                f"{LTX_NUM_INFERENCE_STEPS} steps and takes no step count."
            )

        self.height = served.height
        self.width = served.width
        self.fps = served.fps
        self.num_frames = served.num_frames
        self.duration = served_duration
        return self


# TODO: Remove model specific logic
def _is_minimax_h3() -> bool:
    from config.constants import ModelRunners

    try:
        return get_settings().model_runner in {
            ModelRunners.TT_MINIMAX_H3_T2VA.value,
            ModelRunners.TT_MINIMAX_H3_FL2VA.value,
            ModelRunners.TT_MINIMAX_H3_REF2VA.value,
        }
    except Exception:  # noqa: BLE001 - settings unavailable (tests, tooling): do not gate on it
        return False
