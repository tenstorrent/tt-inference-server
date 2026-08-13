# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from typing import Optional

from config.settings import get_settings
from domain.base_request import BaseRequest
from pydantic import Field, field_validator


class VideoGenerateRequest(BaseRequest):
    # Required fields
    prompt: str

    # Optional fields
    negative_prompt: Optional[str] = None
    num_inference_steps: Optional[int] = Field(default=20, ge=12, le=50)
    seed: Optional[int] = None

    # Output shape. Both are model-specific and both are optional: a model that serves one fixed
    # shape ignores them, and MiniMax-H3 t2va resolves them against its published working points
    # (see `minimax_h3_parse_aspect_ratio` / `MINIMAX_H3_DURATIONS_S`). Left as free-form here
    # rather than an enum so a per-model validator can reject with a message naming what it does
    # serve -- a 422 from pydantic on a shared field cannot say that.
    aspect_ratio: Optional[str] = Field(default=None, examples=["16:9", "9:16", "1:1"])
    duration_seconds: Optional[int] = Field(default=None, ge=1, le=60, examples=[5, 10, 15])

    # Admission-time validation. The device worker validates too (it owns the shape it warmed),
    # but that happens after the request is queued, so the client would get a 202 and a failed job
    # instead of a straight refusal. These run at parse time and surface as a 422 naming what is
    # served. Guarded on the runner so nothing here changes Wan's behaviour.
    @field_validator("aspect_ratio")
    @classmethod
    def _validate_aspect_ratio(cls, value):
        if value is None or not _is_minimax_h3():
            return value
        from config.constants import minimax_h3_parse_aspect_ratio

        minimax_h3_parse_aspect_ratio(value)  # raises with the supported list
        return value

    @field_validator("duration_seconds")
    @classmethod
    def _validate_duration_seconds(cls, value):
        if value is None or not _is_minimax_h3():
            return value
        from config.constants import MINIMAX_H3_DURATIONS_S

        if value not in MINIMAX_H3_DURATIONS_S:
            raise ValueError(
                "duration_seconds must be one of "
                f"{', '.join(str(d) for d in MINIMAX_H3_DURATIONS_S)}; got {value}"
            )
        return value

    @field_validator("num_inference_steps")
    @classmethod
    def _reject_steps_for_minimax_h3(cls, value):
        # Only fires when the caller actually sent the field: pydantic does not run validators for
        # defaults, so an omitted `num_inference_steps` still reaches the runner as 20 and is
        # served at the deployment's 50.
        if value is None or not _is_minimax_h3():
            return value
        from config.constants import MINIMAX_H3_NUM_INFERENCE_STEPS

        raise ValueError(
            "num_inference_steps is not accepted for MiniMax-H3 t2va; omit it. This deployment "
            f"always runs {MINIMAX_H3_NUM_INFERENCE_STEPS} steps."
        )


def _is_minimax_h3() -> bool:
    from config.constants import ModelRunners

    try:
        return get_settings().model_runner == ModelRunners.TT_MINIMAX_H3_T2VA.value
    except Exception:  # noqa: BLE001 - settings unavailable (tests, tooling): do not gate on it
        return False
