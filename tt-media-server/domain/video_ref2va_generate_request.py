# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Ref2VA request schema: multimodal references (images, videos, audio).

Unlike ``ImagePromptEntry``, these assets do not pin an output frame. They are
an ordered bag of reference media, grouped by modality. Pack order is images,
then videos, then audios.
"""

import math
from typing import List, Optional

from domain.video_generate_request import VideoGenerateRequest
from domain.video_i2v_generate_request import MAX_BASE64_IMAGE_LEN
from pydantic import BaseModel, Field, field_validator, model_validator
from utils.media_downloader import is_media_url

# MiniMax video_generation_input per-file caps, as base64 lengths.
MAX_VIDEO_BYTES = 50_000_000
MAX_AUDIO_BYTES = 15_000_000
MAX_BASE64_MEDIA_LEN = 4 * math.ceil(MAX_VIDEO_BYTES / 3)
MAX_BASE64_AUDIO_LEN = 4 * math.ceil(MAX_AUDIO_BYTES / 3)


def _check_b64_len(group: str, sources: List["MediaSource"], cap: int) -> None:
    for index, source in enumerate(sources):
        if source.b64 is not None and len(source.b64) > cap:
            raise ValueError(
                f"{group}[{index}] base64 length {len(source.b64)} exceeds "
                f"the {cap}-char cap"
            )


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

    Counts, decoding and media limits are checked at submit (open_ai_api/video.py).
    ``frame_pos`` is not a field here: these are not output-frame pins.
    """

    images: List[MediaSource] = Field(default_factory=list)
    videos: List[MediaSource] = Field(default_factory=list)
    audios: List[MediaSource] = Field(default_factory=list)

    @field_validator("images")
    @classmethod
    def _image_b64_len(cls, value: List[MediaSource]) -> List[MediaSource]:
        _check_b64_len("images", value, MAX_BASE64_IMAGE_LEN)
        return value

    @field_validator("audios")
    @classmethod
    def _audio_b64_len(cls, value: List[MediaSource]) -> List[MediaSource]:
        _check_b64_len("audios", value, MAX_BASE64_AUDIO_LEN)
        return value


class VideoRef2VAGenerateRequest(VideoGenerateRequest):
    """Video generation request with multimodal references (Ref2VA)."""

    references: MultimodalReferences
