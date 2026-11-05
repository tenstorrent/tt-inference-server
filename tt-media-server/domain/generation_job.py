# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, model_serializer


class JobError(BaseModel):
    code: Union[
        Literal[
            "processing_error", "stale_job", "server_restart", "upsampler_unavailable"
        ],
        str,
    ] = Field(
        description=(
            "Why the job failed. More codes may be added; treat an unknown one "
            "like `processing_error`.\n\n"
            "- `processing_error`: the pipeline raised an error. Resubmitting "
            "the same request is unlikely to help.\n"
            "- `stale_job`: the job made no progress for too long and was "
            "force-cancelled. An infrastructure failure: safe to resubmit, "
            "preferably to another runner.\n"
            "- `server_restart`: the server restarted while the job was queued "
            "or running. Safe to resubmit.\n"
            "- `upsampler_unavailable`: the prompt upsampler, which rewrites the "
            "prompt before the image is generated, could not be reached or was "
            "overloaded; nothing was generated. Safe to resubmit."
        )
    )
    message: str = Field(description="Human-readable.")


class _OutputBase(BaseModel):
    url: str = Field(
        json_schema_extra={"format": "uri-reference"},
        examples=["/v1/jobs/j_8f2c1d/content/0"],
        description=(
            "Where to download the file. Today it is a path relative to this "
            "server's base URL, `/v1/jobs/{job_id}/content/{index}`, fetched with "
            "the same API key; the runner does not know the host the console "
            "reaches it by, so it never returns one. If outputs later move to "
            "shared storage, it becomes an absolute URL. Clients handle both: an "
            "absolute URL is used as given, a relative one is resolved against the "
            "runner's base URL (RFC 3986). Otherwise opaque.\n\n"
            "Internal only. The console's public API returns its own signed "
            "storage URLs and never exposes, proxies or rewrites this one, so "
            "customers never come to depend on a runner path."
        ),
    )
    content_type: str = Field(examples=["image/png", "video/mp4"])
    seed: int = Field(
        description=(
            "The job's seed (the one chosen at random if the request had none). "
            "The same for every output of a job, since a batch is generated from "
            "one seed: resubmitting with this seed and the same request, `count` "
            "included, reproduces all of them."
        )
    )
    prompt: str = Field(
        description=(
            "The prompt as the model received it. For Ideogram 4, the "
            "`structured_prompt` caption as compact JSON, in the model's key order."
        )
    )
    width: int
    height: int


class ImageOutput(_OutputBase):
    type: Literal["image"]


class AudioTrack(BaseModel):
    codec: Literal["aac"]
    sample_rate: int = Field(description="Hz.", examples=[32000])
    channels: int = Field(examples=[2])


class VideoOutput(_OutputBase):
    """An MP4 file: H.264 (High profile, yuv420p), faststart. When the model
    generates sound, an AAC track described by `audio`; otherwise no audio
    track and `audio` is null."""

    type: Literal["video"]
    fps: int
    num_frames: int
    duration: float = Field(description="Seconds.")
    audio: Optional[AudioTrack] = None


Output = Annotated[Union[ImageOutput, VideoOutput], Field(discriminator="type")]


class Job(BaseModel):
    id: str
    model: str = Field(
        description="The `{org}/{model}` it was submitted to.",
        examples=["wan-ai/wan2.2-i2v-a14b", "qwen/qwen-image-edit-2511"],
    )
    status: Literal[
        "queued", "in_progress", "cancelling", "completed", "failed", "cancelled"
    ] = Field(description="Final states are `completed`, `failed` and `cancelled`.")
    created_at: int = Field(description="Unix seconds.")
    completed_at: Optional[int] = Field(
        default=None, description="Unix seconds, once the status is final."
    )
    expires_at: Optional[int] = Field(
        default=None,
        description=(
            "Unix seconds when the job and its output files are deleted, once the "
            "status is final: `completed_at` plus the server's retention period, "
            "24 hours by default. Applies to completed, failed and cancelled jobs "
            "alike. Download outputs before then; after it, the job returns 404."
        ),
    )
    error: Optional[JobError] = Field(default=None, description="Set when `failed`.")
    outputs: Optional[list[Output]] = Field(
        default=None,
        min_length=1,
        description="Set when `completed`: one per image (`count` of them) or one video.",
    )

    @model_serializer(mode="wrap")
    def _without_unset(self, handler):
        # These are left out until they apply, rather than sent as null.
        data = handler(self)
        for name in ("completed_at", "expires_at", "error", "outputs"):
            if data.get(name) is None:
                data.pop(name, None)
        return data

    @classmethod
    def from_metadata(
        cls,
        metadata: dict,
        retention_seconds: int,
        outputs: Optional[list[dict]] = None,
    ) -> "Job":
        """From the job manager's public job dict."""
        completed_at = metadata.get("completed_at")
        return cls(
            id=metadata["id"],
            model=metadata["model"],
            status=metadata["status"],
            created_at=metadata["created_at"],
            completed_at=completed_at,
            expires_at=completed_at + retention_seconds if completed_at else None,
            error=metadata.get("error"),
            outputs=outputs,
        )


class Deleted(BaseModel):
    id: str
    deleted: Literal[True]
