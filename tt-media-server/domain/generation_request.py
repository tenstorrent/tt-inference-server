# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Annotated, ClassVar, Literal, Optional

from domain.base_request import BaseRequest
from PIL import Image
from pydantic import AfterValidator, ConfigDict, Field, PrivateAttr

MAX_SEED = 2**31 - 1

MediaKind = Literal["image", "video", "audio"]
# Per file, inline or fetched.
MEDIA_MAX_BYTES: dict[str, int] = {
    "image": 30_000_000,
    "video": 50_000_000,
    "audio": 15_000_000,
}


def _media_checker(kind: MediaKind, example: str):
    data_uri = re.compile(rf"^data:{kind}/[a-z0-9.+-]+;base64,", re.IGNORECASE)

    def check(value: str) -> str:
        # An http:// URL passes here: the fetch policy refuses it with a 400.
        if value.startswith(("https://", "http://")) or data_uri.match(value):
            return value
        raise ValueError(f"must be an https:// URL or a base64 data URI ({example})")

    return AfterValidator(check)


class SchemaComponent:
    """Makes an annotated type a named schema of its own, which the fields
    using it refer to, so a field's description doesn't replace the type's."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def __get_pydantic_core_schema__(self, source, handler):
        schema = handler(source)
        schema["ref"] = self.name
        return schema

    def __get_pydantic_json_schema__(self, schema, handler):
        json_schema = handler(schema)
        handler.resolve_ref_schema(json_schema)["description"] = self.description
        return json_schema


Media = Annotated[
    str,
    _media_checker("image", "data:image/png;base64,..."),
    SchemaComponent(
        "Media",
        "An `https://` URL, which the server fetches, or a data URI "
        "(`data:image/png;base64,…`). Inline images must be data URIs: bare "
        "base64 without the `data:` prefix is rejected with 422.\n\n"
        "PNG, JPEG or WebP, at most 30 MB per image. URLs are fetched when the "
        "job is submitted, so a URL only has to stay valid until then (a "
        "presigned URL is fine). Which hosts may be fetched is set per "
        "deployment: an allowlist of domains, or any public host; private, "
        "loopback and cloud-metadata addresses never are, including after a "
        "redirect. Submission fails with 400 if a URL is not allowed, 413 if an "
        "image is too large, and 422 if a URL can't be fetched (expired, network "
        "error, timeout) or doesn't hold a supported image.",
    ),
]
VideoMedia = Annotated[
    str,
    _media_checker("video", "data:video/mp4;base64,..."),
    SchemaComponent(
        "VideoMedia",
        "A video clip: an `https://` URL or a data URI "
        "(`data:video/mp4;base64,…`), fetched and checked as for `Media`.\n\n"
        "MP4 or MOV, H.264 or H.265 video, optionally with an AAC or MP3 audio "
        "track; at most 50 MB. Each model states the durations it accepts.",
    ),
]
AudioMedia = Annotated[
    str,
    _media_checker("audio", "data:audio/wav;base64,..."),
    SchemaComponent(
        "AudioMedia",
        "An audio clip: an `https://` URL or a data URI "
        "(`data:audio/wav;base64,…`), fetched and checked as for `Media`.\n\n"
        "WAV or MP3, mono or stereo; at most 15 MB. Each model states the "
        "durations it accepts.",
    ),
]
Seed = Annotated[
    int,
    Field(ge=0, le=MAX_SEED),
    SchemaComponent(
        "Seed", "Omitted = random. Each output reports the seed it was made with."
    ),
]
ImageFormat = Annotated[
    Literal["png", "jpeg", "webp"],
    SchemaComponent("ImageFormat", "Output image format."),
]
Count = Annotated[
    int,
    Field(
        ge=1,
        le=1,
        description=(
            "Number of images. Only 1 for now; kept so more can be allowed later "
            "without a schema change."
        ),
    ),
]


def closest_ratio(width: int, height: int, ratios) -> str:
    """The ratio in ``ratios`` ("W:H" strings) closest to ``width`` x
    ``height`` by factor. The distances are exact, so a tie, such as 4:3
    between 16:9 and 1:1, goes to the earlier ratio in ``ratios``."""
    target = Fraction(width, height)

    def distance(ratio: str) -> Fraction:
        value = Fraction(*map(int, ratio.split(":")))
        return max(value / target, target / value)

    return min(ratios, key=distance)


# The request of one model's submit endpoint. The service fills in the private
# attributes before queueing it; the model's runner then consumes it directly.
class GenerationRequest(BaseRequest):
    model_config = ConfigDict(extra="forbid")

    OUTPUT_TYPE: ClassVar[Literal["image", "video"]]
    # Fields holding media, alone or in a list.
    MEDIA_FIELDS: ClassVar[tuple[str, ...]] = ()
    # The kind of each media field that does not hold images.
    MEDIA_KINDS: ClassVar[dict[str, MediaKind]] = {}
    # Whether the runner decodes the input images from their files. The
    # request then carries files rather than pixels to the device worker and
    # to the other ranks of a multi-host mesh: a decoded image can be a
    # hundred times its file.
    RUNNER_DECODES_IMAGES: ClassVar[bool] = False

    _images: dict[str, list[Image.Image]] = PrivateAttr(default_factory=dict)
    # The input files, and the metadata probed from videos and audio clips.
    _files: dict[str, list[bytes]] = PrivateAttr(default_factory=dict)
    _probes: dict[str, list[object]] = PrivateAttr(default_factory=dict)
    _width: int = PrivateAttr(default=0)
    _height: int = PrivateAttr(default=0)
    _seed: int = PrivateAttr(default=0)
    _prompt: str = PrivateAttr(default="")
    # Shared with the device worker: a set cancel event keeps a worker from
    # claiming the request, and the assignment tells the job manager whether
    # one already has.
    _cancel_event: object = PrivateAttr(default=None)
    _worker_assignment: object = PrivateAttr(default=None)

    def media(self) -> dict[str, list[str]]:
        media = {}
        for name in self.MEDIA_FIELDS:
            value = getattr(self, name)
            if value is not None:
                media[name] = value if isinstance(value, list) else [value]
        return media

    @classmethod
    def media_kind(cls, name: str) -> MediaKind:
        return cls.MEDIA_KINDS.get(name, "image")

    def media_errors(self) -> list[tuple[str, int, str]]:
        """(field, index, message) for each loaded input the model cannot
        take, from what the schema cannot see: image sizes, clip lengths."""
        return []

    def output_size(self) -> tuple[int, int]:
        """(width, height) of the outputs, from the fields and the decoded input
        images."""
        raise NotImplementedError

    @classmethod
    def warmup_sizes(cls) -> list[tuple[int, int]]:
        """Every (width, height) a request can produce, as far as they can be
        listed, so the runner can compile and trace them before serving."""
        raise NotImplementedError

    def model_prompt(self) -> str:
        """The prompt as the model gets it."""
        return getattr(self, "prompt")


# Inputs for the example requests: a sailboat at dawn, 256×256, inline; and
# public files, which a deployment fetches only if it allows public hosts.
EXAMPLE_PNG = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAQAAAAEABAMAAACuXLVVAAAAFVBMVEX0smbw"
    "jFr61oz/7KooTm4cOFQeHiinjRUIAAAB7klEQVR42u2d222DQBBF90EBSyqIqGDTQ8pOC2khogOoAPIR"
    "RVEk46xmZ3dMOPfLNmZ9dOeFANt+dLYKDgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AIQQQgghhBBCCCGEEEIIIYQQQsbyVXvH7Jxz71YAXx9fieAn0W6v7s359PN8EQMM4j1/fWN63FfhMkEp"
    "duJQigHSPUN6AOS/iFoDRKV6UryFI3UE8Ldi4DsCpFsxkFkQ5CWoY0HQjHbqBOAP6kBkQU0VZI3aCar1"
    "lroA+MNeJIlBVSPKRiE4bsd9AJJqO66bBdnCAa87Es9+R2W2BogGAOmuBelqOVAfg2oHsjWAuQPR3IFs"
    "DRCvngO1MVAAiJ0B1oqtjXIgWydhNK+C3BVgF29s50BNDIZJg+Bl/n70dMVOWLeKkgPP1gDBGsB+HItj"
    "ID5b/vFgDpwXoCQEU+GL29zIgdJ1Z5EDBRcaSs86jK1yoHDG7u2SsOxAbz1pFRRd8Ss57bAt53QgqBXY"
    "3BJgU3nLqWfB3CoCpQBbqwicZxzPjSLwAJ1Q55fbd/EyxceEi3EO/ONjwuXqDgAAAAAADPyPCTnANAQA"
    "AADohExDcoBpCAAAANAJmYbkANMQAAAAoBMyDckBpiEAAABweYBPtZUsv0RpNtoAAAAASUVORK5CYII="
)
EXAMPLE_IMAGE_URL = (
    "https://interactive-examples.mdn.mozilla.net/media/examples/plumeria.jpg"
)
EXAMPLE_VIDEO_URL = (
    "https://interactive-examples.mdn.mozilla.net/media/cc0-videos/flower.mp4"
)
EXAMPLE_AUDIO_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/c/ce/"
    "Carolina_Wren_%28Thryothorus_ludovicianus%29_-_New_Jersey_2023-02-23.mp3"
)


@dataclass(frozen=True)
class GenerationModel:
    """What a deployment serves: the model, the request it takes, and its
    submit route's docs. Deployments that differ only in sampling take the same
    request."""

    # The `{org}/{model}` of the submit path, the upstream model's lowercase
    # Hugging Face ID.
    id: str
    request: type[GenerationRequest]
    # The submit route's tag, summary and description, and an example request.
    title: str
    summary: str
    example: dict
    description: Optional[str] = None
