# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Ref2VA request schema: multimodal references (images, videos, audio).

Unlike ``ImagePromptEntry``, these assets do not pin an output frame. They are
an ordered bag of reference media, grouped by modality. Pack order is images,
then videos, then audios.
"""

from typing import List, Optional

from domain.video_generate_request import VideoGenerateRequest
from pydantic import BaseModel, Field, field_validator, model_validator
from utils.image_manager import ImageManager
from utils.media_downloader import is_media_url

# Larger than the I2V image cap: a reference clip will not fit in 10M chars.
MAX_BASE64_MEDIA_LEN = 80_000_000


class MediaSource(BaseModel):
    """Exactly one of inline base64 or a remote http(s) URL."""

    b64: Optional[str] = Field(
        default=None, min_length=1, max_length=MAX_BASE64_MEDIA_LEN
    )
    url: Optional[str] = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _exactly_one_source(self):
        has_b64 = self.b64 is not None
        has_url = self.url is not None
        if has_b64 == has_url:
            raise ValueError("provide exactly one of b64 or url")
        if has_url and not is_media_url(self.url):
            raise ValueError("url must be an http(s) URL")
        return self


class MultimodalReferences(BaseModel):
    """Reference images, videos, and audio for omni-reference generation.

    Counts match MiniMax-H3 ref2va (9 / 3 / 3). Audio cannot stand alone.
    ``frame_pos`` is not a field here: these are not output-frame pins.
    """

    images: List[MediaSource] = Field(default_factory=list)
    videos: List[MediaSource] = Field(default_factory=list)
    audios: List[MediaSource] = Field(default_factory=list)

    @field_validator("images")
    @classmethod
    def _images_are_decodable(cls, value: List[MediaSource]) -> List[MediaSource]:
        from tt_model_runners.minimax_h3_policy import MINIMAX_H3_MAX_REFERENCE_IMAGES

        if len(value) > MINIMAX_H3_MAX_REFERENCE_IMAGES:
            raise ValueError(
                f"at most {MINIMAX_H3_MAX_REFERENCE_IMAGES} reference images, "
                f"got {len(value)}"
            )
        for index, source in enumerate(value):
            if source.b64 is None:
                continue
            try:
                img = ImageManager().base64_to_pil_image(source.b64)
            except Exception as exc:
                raise ValueError(
                    f"images[{index}] could not be decoded to a valid PIL image"
                ) from exc
            if img.size[0] < 1 or img.size[1] < 1:
                raise ValueError(
                    f"images[{index}] has invalid dimensions "
                    "(width and height must be >= 1)"
                )
        return value

    @field_validator("videos")
    @classmethod
    def _video_count(cls, value: List[MediaSource]) -> List[MediaSource]:
        from tt_model_runners.minimax_h3_policy import MINIMAX_H3_MAX_REFERENCE_VIDEOS

        if len(value) > MINIMAX_H3_MAX_REFERENCE_VIDEOS:
            raise ValueError(
                f"at most {MINIMAX_H3_MAX_REFERENCE_VIDEOS} reference videos, "
                f"got {len(value)}"
            )
        return value

    @field_validator("audios")
    @classmethod
    def _audio_count(cls, value: List[MediaSource]) -> List[MediaSource]:
        from tt_model_runners.minimax_h3_policy import MINIMAX_H3_MAX_REFERENCE_AUDIOS

        if len(value) > MINIMAX_H3_MAX_REFERENCE_AUDIOS:
            raise ValueError(
                f"at most {MINIMAX_H3_MAX_REFERENCE_AUDIOS} reference audios, "
                f"got {len(value)}"
            )
        return value

    @model_validator(mode="after")
    def _not_empty_and_audio_not_alone(self):
        if not self.images and not self.videos and not self.audios:
            raise ValueError(
                "ref2va needs at least one reference image, video, or audio"
            )
        if self.audios and not self.images and not self.videos:
            raise ValueError(
                "an audio reference must be paired with at least one image or video"
            )
        return self


class VideoRef2VAGenerateRequest(VideoGenerateRequest):
    """Video generation request with multimodal references (Ref2VA)."""

    references: MultimodalReferences
