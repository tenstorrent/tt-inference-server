# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Text-to-video request schema.

Shape parameters (``height``/``width``/``fps`` and the frame count, expressed
either as ``duration`` seconds or ``num_frames``) are accepted but *not*
honoured as free variables: for LTX the shape is baked into the captured traces
at ``create_pipeline()`` time, so a request may only ask for the shape the
running process already serves. ``_validate_shape`` below rejects anything else
rather than letting a mismatch reach the pipeline, where nothing would catch it
-- tt-metal guards ``fps`` (``LTXPipeline._resolve_fps``) but not the frame
count or resolution.

All shape fields default to ``None``, meaning "use the served config", so a
prompt-only request behaves exactly as it did before they existed.
"""

from typing import Optional

from config.constants import (
    LTX_NUM_INFERENCE_STEPS,
    ModelRunners,
    ltx_served_shape,
    snap_num_frames,
)
from config.settings import get_settings
from domain.base_request import BaseRequest
from pydantic import Field, model_validator

# Step range for the video models that take a client-supplied step count. LTX
# does not (its distilled sigma schedules are fixed), so it is excluded below.
_DEFAULT_NUM_INFERENCE_STEPS = 20
_MIN_NUM_INFERENCE_STEPS = 12
_MAX_NUM_INFERENCE_STEPS = 50


class VideoGenerateRequest(BaseRequest):
    # Required fields
    prompt: str

    # Optional fields
    negative_prompt: Optional[str] = None
    # Bound is deliberately wide: the real per-model range is applied in
    # _validate_shape, because LTX's truthful value (11) sits below the 12-step
    # floor the other video models enforce. None means "use the model's default".
    num_inference_steps: Optional[int] = Field(default=None, ge=1, le=50)
    seed: Optional[int] = None

    # Shape. None = use the served config; see _validate_shape.
    height: Optional[int] = Field(default=None, gt=0)
    width: Optional[int] = Field(default=None, gt=0)
    fps: Optional[float] = Field(default=None, gt=0)
    duration: Optional[float] = Field(default=None, gt=0)
    num_frames: Optional[int] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_shape(self):
        """Resolve and validate shape + step count against the served config.

        ``mode="after"`` because the checks are cross-field: ``duration``,
        ``fps`` and ``num_frames`` are three views of the same two numbers.

        Resolved values are written back onto the model so that (a) downstream
        ``getattr(request, "height", DEFAULT)`` sees real ints and (b) the job
        record echoes what was actually generated rather than what was asked
        for.
        """
        if get_settings().model_runner != ModelRunners.TT_LTX_2_3_DISTILLED.value:
            # Every other video model resolves its shape from settings/mesh
            # (see wan22_target_resolution), so leave the shape fields alone and
            # only apply the conventional step range.
            if self.num_inference_steps is None:
                self.num_inference_steps = _DEFAULT_NUM_INFERENCE_STEPS
            elif not (
                _MIN_NUM_INFERENCE_STEPS
                <= self.num_inference_steps
                <= _MAX_NUM_INFERENCE_STEPS
            ):
                raise ValueError(
                    f"num_inference_steps must be between "
                    f"{_MIN_NUM_INFERENCE_STEPS} and {_MAX_NUM_INFERENCE_STEPS}, "
                    f"got {self.num_inference_steps}"
                )
            return self

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
