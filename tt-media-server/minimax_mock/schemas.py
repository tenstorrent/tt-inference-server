"""Request and response schemas for the MiniMax video-generation mock."""

from __future__ import annotations

import re
from enum import Enum
from typing import Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)


class ContentType(str, Enum):
    TEXT = "text"
    IMAGE_URL = "image_url"
    VIDEO_URL = "video_url"
    AUDIO_URL = "audio_url"


class ContentRole(str, Enum):
    FIRST_FRAME = "first_frame"
    LAST_FRAME = "last_frame"
    REFERENCE_IMAGE = "reference_image"
    REFERENCE_VIDEO = "reference_video"
    REFERENCE_AUDIO = "reference_audio"


class Resolution(str, Enum):
    P768 = "768P"
    P2K = "2K"


class AspectRatio(str, Enum):
    ADAPTIVE = "adaptive"
    RATIO_21_9 = "21:9"
    RATIO_16_9 = "16:9"
    RATIO_4_3 = "4:3"
    RATIO_1_1 = "1:1"
    RATIO_3_4 = "3:4"
    RATIO_9_16 = "9:16"


class MediaLocation(BaseModel):
    """A public URL, MiniMax file reference, or supported data URI."""

    model_config = ConfigDict(extra="forbid")

    url: StrictStr = Field(min_length=1)


class ContentItem(BaseModel):
    """One item from the MiniMax multimodal ``content`` array."""

    model_config = ConfigDict(extra="forbid")

    type: ContentType
    text: StrictStr | None = Field(default=None, max_length=7000)
    image_url: MediaLocation | None = None
    video_url: MediaLocation | None = None
    audio_url: MediaLocation | None = None
    role: ContentRole | None = None

    @model_validator(mode="after")
    def validate_item_shape(self) -> ContentItem:
        media_fields = {
            ContentType.IMAGE_URL: self.image_url,
            ContentType.VIDEO_URL: self.video_url,
            ContentType.AUDIO_URL: self.audio_url,
        }

        if self.type is ContentType.TEXT:
            if self.text is None or not self.text.strip():
                raise ValueError("text content must be non-empty")
            if any(value is not None for value in media_fields.values()):
                raise ValueError("text content cannot include a media URL")
            if self.role is not None:
                raise ValueError("text content cannot have a role")
            return self

        expected_media = media_fields[self.type]
        if expected_media is None:
            raise ValueError(
                f"{self.type.value} content requires a {self.type.value} field"
            )
        if self.text is not None:
            raise ValueError(f"{self.type.value} content cannot include text")
        if any(
            value is not None
            for content_type, value in media_fields.items()
            if content_type is not self.type
        ):
            raise ValueError(
                f"{self.type.value} content cannot include another media URL type"
            )

        if self.type is ContentType.IMAGE_URL:
            allowed_roles = {
                None,
                ContentRole.FIRST_FRAME,
                ContentRole.LAST_FRAME,
                ContentRole.REFERENCE_IMAGE,
            }
            if self.role not in allowed_roles:
                raise ValueError("image_url has an incompatible role")
            _validate_media_location(expected_media.url, "image")
        elif self.type is ContentType.VIDEO_URL:
            if self.role is not ContentRole.REFERENCE_VIDEO:
                raise ValueError("video_url requires role=reference_video")
            _validate_media_location(expected_media.url, "video")
        else:
            if self.role is not ContentRole.REFERENCE_AUDIO:
                raise ValueError("audio_url requires role=reference_audio")
            _validate_media_location(expected_media.url, "audio")

        return self


class VideoGenerationRequest(BaseModel):
    """Published request contract for ``POST /v2/video_generation``."""

    model_config = ConfigDict(extra="forbid")

    model: Literal["MiniMax-H3"]
    content: list[ContentItem] = Field(min_length=1, max_length=16)
    resolution: Resolution
    duration: StrictInt = Field(ge=4, le=15)
    ratio: AspectRatio | None = None
    callback_url: StrictStr | None = None

    @field_validator("callback_url")
    @classmethod
    def validate_callback_url(cls, value: str | None) -> str | None:
        if value is None:
            return value
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("callback_url must be an absolute HTTP or HTTPS URL")
        return value

    @model_validator(mode="after")
    def validate_content_combination(self) -> VideoGenerationRequest:
        text_items = [item for item in self.content if item.type is ContentType.TEXT]
        if len(text_items) != 1:
            raise ValueError("content must include exactly one non-empty text item")

        image_items = [
            item for item in self.content if item.type is ContentType.IMAGE_URL
        ]
        video_items = [
            item for item in self.content if item.type is ContentType.VIDEO_URL
        ]
        audio_items = [
            item for item in self.content if item.type is ContentType.AUDIO_URL
        ]

        unassigned_images = [item for item in image_items if item.role is None]
        if unassigned_images and (len(image_items) != 1 or video_items or audio_items):
            raise ValueError(
                "an image without a role is only valid as a single first-frame image"
            )

        first_frame_count = sum(
            item.role is ContentRole.FIRST_FRAME for item in image_items
        )
        if unassigned_images:
            first_frame_count += 1
        last_frame_count = sum(
            item.role is ContentRole.LAST_FRAME for item in image_items
        )
        reference_image_count = sum(
            item.role is ContentRole.REFERENCE_IMAGE for item in image_items
        )

        if first_frame_count > 1:
            raise ValueError("content supports at most one first-frame image")
        if last_frame_count > 1:
            raise ValueError("content supports at most one last-frame image")
        if reference_image_count > 9:
            raise ValueError("content supports at most nine reference images")
        if len(video_items) > 3:
            raise ValueError("content supports at most three reference videos")
        if len(audio_items) > 3:
            raise ValueError("content supports at most three reference audio items")

        has_frame_input = bool(first_frame_count or last_frame_count)
        has_reference_input = bool(reference_image_count or video_items or audio_items)
        if has_frame_input and has_reference_input:
            raise ValueError(
                "first/last-frame inputs cannot be mixed with reference inputs"
            )

        has_non_text_input = bool(image_items or video_items or audio_items)
        if not has_non_text_input and (
            self.ratio is None or self.ratio is AspectRatio.ADAPTIVE
        ):
            raise ValueError(
                "ratio is required for text-to-video and cannot be adaptive"
            )

        return self


class CreateTaskResponse(BaseModel):
    task_id: str


_BASE64_PAYLOAD_RE = re.compile(r"[A-Za-z0-9+/]*={0,2}")
_DATA_URI_FORMATS = {
    "image": {"jpg", "jpeg", "png", "webp", "heic", "heif"},
    "video": {"mp4"},
    "audio": {"wav", "mp3"},
}


def _validate_media_location(url: str, media_type: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return

    if url.startswith("mm_file://"):
        file_id = url.removeprefix("mm_file://")
        if file_id and not file_id.isspace():
            return
        raise ValueError("mm_file URL must include a file_id")

    metadata, separator, payload = url.partition(",")
    prefix = f"data:{media_type}/"
    if (
        not separator
        or not metadata.startswith(prefix)
        or not metadata.endswith(";base64")
    ):
        raise ValueError(
            f"{media_type} URL must be public HTTP(S), mm_file, or a data URI"
        )

    media_format = metadata[len(prefix) : -len(";base64")]
    if media_format not in _DATA_URI_FORMATS[media_type]:
        raise ValueError(f"unsupported {media_type} data URI format")
    if not payload or _BASE64_PAYLOAD_RE.fullmatch(payload) is None:
        raise ValueError(f"invalid {media_type} base64 payload")
