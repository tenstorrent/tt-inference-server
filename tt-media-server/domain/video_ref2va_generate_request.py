# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Ref2VA request schema: multimodal references (images, videos, audio).

Unlike ``ImagePromptEntry``, these assets do not pin an output frame. They are
an ordered bag of reference media, grouped by modality. Pack order is images,
then videos, then audios.

Admission follows the MiniMax input media card (``tt_model_runners.minimax_h3_policy``):
counts 9 / 3 / 3 and 12 in total, inline images checked here (<= 30 MB,
JPG/PNG/WEBP/HEIC/HEIF, [256, 5760] px, aspect 0.4-2.5); videos and audio are
probed at the endpoint once URL sources are downloaded (``_enforce_ref2va_media_limits``).
"""

from typing import List, Optional

from domain.video_generate_request import VideoGenerateRequest
from pydantic import BaseModel, Field, field_validator, model_validator
from tt_model_runners.minimax_h3_policy import (
    MEDIA_B64_FIELD_HEADROOM,
    MINIMAX_H3_MAX_REFERENCE_AUDIOS,
    MINIMAX_H3_MAX_REFERENCE_IMAGES,
    MINIMAX_H3_MAX_REFERENCE_VIDEOS,
    MINIMAX_H3_MAX_REFERENCES_TOTAL,
    MINIMAX_H3_VIDEO_MAX_BYTES,
    base64_len_for_bytes,
    check_h3_image,
    decode_base64_media,
)
from utils.media_downloader import is_media_url

# One MediaSource field serves images, videos and audio, so its text cap is the
# largest file the card admits -- a 50 MB reference video -- as base64. The
# per-modality byte caps (30 / 50 / 15 MB) are enforced on the decoded bytes.
MAX_BASE64_MEDIA_LEN = (
    base64_len_for_bytes(MINIMAX_H3_VIDEO_MAX_BYTES) + MEDIA_B64_FIELD_HEADROOM
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

    Counts match MiniMax-H3 ref2va (9 / 3 / 3). Audio cannot stand alone.
    ``frame_pos`` is not a field here: these are not output-frame pins.
    """

    images: List[MediaSource] = Field(default_factory=list)
    videos: List[MediaSource] = Field(default_factory=list)
    audios: List[MediaSource] = Field(default_factory=list)

    @field_validator("images")
    @classmethod
    def _images_are_admissible(cls, value: List[MediaSource]) -> List[MediaSource]:
        if len(value) > MINIMAX_H3_MAX_REFERENCE_IMAGES:
            raise ValueError(
                f"at most {MINIMAX_H3_MAX_REFERENCE_IMAGES} reference images, "
                f"got {len(value)}"
            )
        for index, source in enumerate(value):
            if source.b64 is None:
                # URL source: downloaded and checked against the same card at the
                # API layer before enqueue (open_ai_api/video.py).
                continue
            label = f"images[{index}]"
            try:
                raw = decode_base64_media(source.b64)
            except ValueError as exc:
                raise ValueError(f"{label} is not valid base64") from exc
            check_h3_image(raw, label=label)
        return value

    @field_validator("videos")
    @classmethod
    def _video_count(cls, value: List[MediaSource]) -> List[MediaSource]:
        if len(value) > MINIMAX_H3_MAX_REFERENCE_VIDEOS:
            raise ValueError(
                f"at most {MINIMAX_H3_MAX_REFERENCE_VIDEOS} reference videos, "
                f"got {len(value)}"
            )
        return value

    @field_validator("audios")
    @classmethod
    def _audio_count(cls, value: List[MediaSource]) -> List[MediaSource]:
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
        total = len(self.images) + len(self.videos) + len(self.audios)
        if total > MINIMAX_H3_MAX_REFERENCES_TOTAL:
            # The pipeline refuses this too (packing_ref2va.check_references); saying
            # it here spares the client a 202 for a job that could never run.
            raise ValueError(
                f"H3 accepts at most {MINIMAX_H3_MAX_REFERENCES_TOTAL} references in "
                f"total (images + videos + audios), got {total}"
            )
        return self


class VideoRef2VAGenerateRequest(VideoGenerateRequest):
    """Video generation request with multimodal references (Ref2VA)."""

    references: MultimodalReferences
