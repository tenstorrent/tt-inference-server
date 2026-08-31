# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Contract checks for MiniMax-H3 Ref2VA on ``POST /v1/videos/generations/ref2va``.

Issue #5044 item 4. Runs against a deployment booted with
``MODEL_RUNNER=tt-minimax-h3-ref2va``.

The endpoint takes a ``references`` object -- ``{"images": [...], "videos":
[...], "audios": [...]}`` -- where every element is exactly one of ``{"b64":
...}`` or ``{"url": ...}``. What this file pins:

* counts: at most 9 images, 3 videos, 3 audios,
* source shape: exactly one of ``b64``/``url``, and a ``url`` must be http(s),
* audio cannot stand alone, and an empty ``references`` is not a request this
  endpoint serves -- a ref2va call with no reference is a t2va call, and the
  server must say so instead of silently degrading,
* clip durations: each video/audio clip 2-15 s, combined <= 15 s per modality,
* cross-deployment: ``references`` sent to a deployment that does not serve
  ref2va is refused at admission, not accepted and later failed.

Everything is observed over HTTP. Nothing here imports tt-media-server, patches
anything, or builds a pydantic model.

Two profiles:

``validation``
    Refusals only. No case here can start a generation. Any 202 that does come
    back (because the server accepted something it should have refused) is
    cancelled immediately so a contract failure does not leave a job on device.
``smoke``
    The validation cases plus the acceptance cases, each of which creates a
    real job and then cancels it.

Rejection cases additionally require the error body to carry a FastAPI
``detail``, and -- where a wrong-reason 422 is plausible -- that the detail
names the offending part of the request. A refusal that does not say what was
wrong is not a usable refusal.

Reference clips cannot be synthesised here: a duration boundary can only be
expressed with a real container that actually has that duration, and encoding
one in this file would be faking the fixture rather than testing it. Those
cases are gated on ``targets.reference_clips`` (see ``REQUIRED_CLIP_FIXTURES``)
and are reported as skipped-with-reason when the fixture is absent. They are
never quietly counted as passes.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import aiohttp  # pyright: ignore[reportMissingImports]

from test_module._test_common import BaseTest, HardwareRequirement, TestConfig
from test_module._test_common.minimax_h3_client import (
    CANCEL_PATH,
    CREATE_PATH,
    RESPONSE_EXCERPT_LENGTH,
    resolve_server_api_key,
)

if TYPE_CHECKING:
    from report_module.schema import Block
    from test_module.context import MediaContext

logger = logging.getLogger(__name__)

Profile = Literal["validation", "smoke"]
Deployment = Literal["ref2va", "non_ref2va"]

REF2VA_PATH = f"{CREATE_PATH}/ref2va"

DEFAULT_PROFILE: Profile = "validation"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0
DEFAULT_TEST_TIMEOUT_SECONDS = 600
_PROFILES = frozenset({"validation", "smoke"})

# A 1x1 PNG. Small enough that nine of them fit in a request without the body
# dominating the check, and a real image so a count/shape rejection is the only
# thing a refusal can be about.
_MINIMAL_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGA"
    "hKmMIQAAAABJRU5ErkJggg=="
)

# Syntactically valid base64 that is not any media container. Used only where
# the request is refused before its bytes are ever read (a count cap, or the
# audio-cannot-stand-alone rule, both of which are body-parse concerns), and in
# the one case that deliberately asks what happens to unreadable clip bytes.
_NOT_MEDIA_B64 = "bm90LWEtbWVkaWEtY29udGFpbmVy"

# Substituted for a clip whose fixture was not supplied. It is never sent: the
# runner skips such a case before the request is built into a POST.
_MISSING_CLIP_B64 = "fixture-not-supplied"

_PROMPT = (
    "A grey cat walks along a sunlit windowsill and settles down, with quiet "
    "room tone and birdsong outside."
)


@dataclass(frozen=True)
class _ClipFixture:
    """A reference clip this suite cannot synthesise and will not fake."""

    name: str
    modality: Literal["videos", "audios"]
    duration: str
    why: str

    @property
    def requirement(self) -> str:
        container = "video" if self.modality == "videos" else "audio"
        return f"a real {container} container of {self.duration} ({self.why})"


# Supply these through ``targets.reference_clips`` as ``{name: path}`` (or
# ``--reference-clip name=path`` on the CLI). Durations are the container
# duration the server probes, so a clip that is "about 2 s" is not a boundary
# fixture -- it has to be the duration named here.
REQUIRED_CLIP_FIXTURES: tuple[_ClipFixture, ...] = (
    _ClipFixture(
        "video_2s",
        "videos",
        "exactly 2.0 s",
        "the inclusive floor of the per-clip window",
    ),
    _ClipFixture(
        "video_15s",
        "videos",
        "exactly 15.0 s",
        "the inclusive ceiling of the per-clip window",
    ),
    _ClipFixture(
        "video_1s",
        "videos",
        "1.0 s",
        "below the 2 s floor",
    ),
    _ClipFixture(
        "video_16s",
        "videos",
        "16.0 s",
        "above the 15 s ceiling",
    ),
    _ClipFixture(
        "video_6s",
        "videos",
        "6.0 s",
        "three of these combine to 18 s, over the 15 s combined cap",
    ),
    _ClipFixture(
        "audio_2s",
        "audios",
        "exactly 2.0 s",
        "the inclusive floor of the per-clip window",
    ),
    _ClipFixture(
        "audio_1s",
        "audios",
        "1.0 s",
        "below the 2 s floor",
    ),
    _ClipFixture(
        "audio_6s",
        "audios",
        "6.0 s",
        "three of these combine to 18 s, over the 15 s combined cap",
    ),
)

_CLIP_SPECS: dict[str, _ClipFixture] = {
    fixture.name: fixture for fixture in REQUIRED_CLIP_FIXTURES
}


@dataclass(frozen=True)
class _Ref2vaCase:
    """One request, the status it must come back with, and why."""

    name: str
    contract: str
    payload: dict[str, Any]
    expected_status: int
    path: str = REF2VA_PATH
    deployment: Deployment = "ref2va"
    auth_mode: Literal["valid", "missing", "invalid"] = "valid"
    # Lowercase fragments the refusal detail must contain, so a 422 raised for
    # an unrelated reason cannot be read as this case passing.
    detail_contains: tuple[str, ...] = ()
    # Names from REQUIRED_CLIP_FIXTURES this payload needs to be meaningful.
    requires_clips: tuple[str, ...] = ()
    # Accepted requests must hand back a job id; the runner then cancels it.
    requires_job_id: bool = False


def _base_payload() -> dict[str, Any]:
    return {
        "prompt": _PROMPT,
        "aspect_ratio": "16:9",
        "duration_seconds": 5,
        "seed": 0,
    }


def _payload(references: Any, **overrides: Any) -> dict[str, Any]:
    payload = _base_payload()
    payload["references"] = references
    payload.update(overrides)
    return payload


def _images(count: int) -> list:
    return [{"b64": _MINIMAL_PNG_B64} for _ in range(count)]


def _clips(loaded: dict[str, str], name: str, count: int) -> list:
    return [{"b64": loaded.get(name, _MISSING_CLIP_B64)} for _ in range(count)]


def _refusal_cases(loaded: dict[str, str]) -> list:
    """Every case that must not create a job."""

    return [
        # --- authentication on the new route -------------------------------
        _Ref2vaCase(
            "ref2va_endpoint_requires_bearer_authentication",
            "The ref2va route is a distinct route and carries its own auth.",
            _payload({"images": _images(1)}),
            401,
            auth_mode="missing",
        ),
        _Ref2vaCase(
            "ref2va_endpoint_rejects_an_invalid_bearer_token",
            "A wrong key is refused before the references are looked at.",
            _payload({"images": _images(1)}),
            401,
            auth_mode="invalid",
        ),
        # --- reference counts (9 / 3 / 3) ----------------------------------
        _Ref2vaCase(
            "ten_reference_images_exceed_the_nine_image_cap",
            "MiniMax-H3 ref2va packs at most 9 reference images.",
            _payload({"images": _images(10)}),
            422,
            detail_contains=("images",),
        ),
        _Ref2vaCase(
            "four_reference_videos_exceed_the_three_video_cap",
            (
                "At most 3 reference videos. The clip bytes are placeholders "
                "on purpose: the cap is a body-parse concern, so the request "
                "must be refused before any byte is probed, and the detail "
                "must name videos rather than the unreadable bytes."
            ),
            _payload({"videos": [{"b64": _NOT_MEDIA_B64} for _ in range(4)]}),
            422,
            detail_contains=("videos",),
        ),
        _Ref2vaCase(
            "four_reference_audios_exceed_the_three_audio_cap",
            (
                "At most 3 reference audios. An image rides along so the "
                "refusal cannot be the audio-cannot-stand-alone rule."
            ),
            _payload(
                {
                    "images": _images(1),
                    "audios": [{"b64": _NOT_MEDIA_B64} for _ in range(4)],
                }
            ),
            422,
            detail_contains=("audios",),
        ),
        # --- media source shape --------------------------------------------
        _Ref2vaCase(
            "a_reference_source_with_both_b64_and_url_is_rejected",
            (
                "Exactly one of b64/url. Accepting both leaves the server to "
                "pick, and the caller cannot know which reference was used."
            ),
            _payload(
                {
                    "images": [
                        {
                            "b64": _MINIMAL_PNG_B64,
                            "url": "https://example.com/reference.png",
                        }
                    ]
                }
            ),
            422,
            detail_contains=("images",),
        ),
        _Ref2vaCase(
            "a_reference_source_with_neither_b64_nor_url_is_rejected",
            "An empty source object carries no reference at all.",
            _payload({"images": [{}]}),
            422,
            detail_contains=("images",),
        ),
        _Ref2vaCase(
            "a_reference_source_with_a_non_http_url_is_rejected",
            (
                "A url must be http(s). A file:// reference would ask the "
                "server to read its own filesystem on a caller's behalf."
            ),
            _payload({"images": [{"url": "file:///etc/passwd"}]}),
            422,
            detail_contains=("images",),
        ),
        _Ref2vaCase(
            "a_reference_source_carrying_frame_pos_is_rejected",
            (
                "References are not output-frame pins -- frame_pos belongs to "
                "fl2va image_prompts. Ignoring it silently hands back a video "
                "the caller believes they pinned a frame of."
            ),
            _payload({"images": [{"b64": _MINIMAL_PNG_B64, "frame_pos": 0}]}),
            422,
            detail_contains=("frame_pos",),
        ),
        _Ref2vaCase(
            "a_reference_image_that_is_not_decodable_is_rejected",
            (
                "An undecodable image is refused at admission, not carried "
                "into the job and failed on device."
            ),
            _payload({"images": [{"b64": _NOT_MEDIA_B64}]}),
            422,
            detail_contains=("images",),
        ),
        _Ref2vaCase(
            "image_prompts_alongside_references_is_rejected",
            (
                "fl2va keyframes and ref2va references are different "
                "mechanisms; a request carrying both is asking for two "
                "conditionings and must be told so."
            ),
            _payload(
                {"images": _images(1)},
                image_prompts=[{"image": _MINIMAL_PNG_B64, "frame_pos": 0}],
            ),
            422,
            detail_contains=("image_prompts",),
        ),
        # --- empty references ----------------------------------------------
        _Ref2vaCase(
            "references_with_all_three_lists_empty_is_rejected",
            (
                "A ref2va request with no reference is a t2va request. The "
                "server must say so rather than silently degrading to t2va."
            ),
            _payload({"images": [], "videos": [], "audios": []}),
            422,
            detail_contains=("references",),
        ),
        _Ref2vaCase(
            "a_references_object_with_no_lists_at_all_is_rejected",
            "Omitted lists default to empty, which is still no reference.",
            _payload({}),
            422,
            detail_contains=("references",),
        ),
        _Ref2vaCase(
            "a_request_without_a_references_field_is_rejected",
            "references is required on this route; it has no t2va fallback.",
            _base_payload(),
            422,
            detail_contains=("references",),
        ),
        # --- audio cannot stand alone ---------------------------------------
        _Ref2vaCase(
            "audio_only_references_are_rejected",
            (
                "An audio reference must be paired with an image or a video: "
                "there is nothing for the audio to be a reference *of*."
            ),
            _payload({"audios": [{"b64": _NOT_MEDIA_B64}]}),
            422,
            detail_contains=("audio",),
        ),
        # --- clip durations --------------------------------------------------
        _Ref2vaCase(
            "a_reference_video_whose_bytes_are_not_a_media_container_is_rejected",
            (
                "A clip whose duration cannot be probed is refused at "
                "admission, not accepted and failed later by the worker."
            ),
            _payload({"videos": [{"b64": _NOT_MEDIA_B64}]}),
            422,
        ),
        _Ref2vaCase(
            "a_reference_video_clip_below_two_seconds_is_rejected",
            "Each reference video clip must be at least 2 s.",
            _payload({"videos": _clips(loaded, "video_1s", 1)}),
            422,
            requires_clips=("video_1s",),
            detail_contains=("videos",),
        ),
        _Ref2vaCase(
            "a_reference_video_clip_above_fifteen_seconds_is_rejected",
            "Each reference video clip must be at most 15 s.",
            _payload({"videos": _clips(loaded, "video_16s", 1)}),
            422,
            requires_clips=("video_16s",),
            detail_contains=("videos",),
        ),
        _Ref2vaCase(
            "three_six_second_reference_videos_exceed_the_combined_cap",
            (
                "Combined video duration must be <= 15 s even when every "
                "clip is individually inside the 2-15 s window."
            ),
            _payload({"videos": _clips(loaded, "video_6s", 3)}),
            422,
            requires_clips=("video_6s",),
            detail_contains=("videos",),
        ),
        _Ref2vaCase(
            "a_reference_audio_clip_below_two_seconds_is_rejected",
            "Each reference audio clip must be at least 2 s.",
            _payload(
                {
                    "images": _images(1),
                    "audios": _clips(loaded, "audio_1s", 1),
                }
            ),
            422,
            requires_clips=("audio_1s",),
            detail_contains=("audios",),
        ),
        _Ref2vaCase(
            "three_six_second_reference_audios_exceed_the_combined_cap",
            "Combined audio duration must be <= 15 s.",
            _payload(
                {
                    "images": _images(1),
                    "audios": _clips(loaded, "audio_6s", 3),
                }
            ),
            422,
            requires_clips=("audio_6s",),
            detail_contains=("audios",),
        ),
        # --- cross-deployment -------------------------------------------------
        _Ref2vaCase(
            "a_text_only_request_on_a_ref2va_deployment_is_rejected",
            (
                "The complement of the empty-references rule: this deployment "
                "loaded transformer_ref/ and must point a reference-less "
                "caller at the endpoint it wants instead of generating."
            ),
            _base_payload(),
            422,
            path=CREATE_PATH,
            detail_contains=("ref2va",),
        ),
        _Ref2vaCase(
            "the_ref2va_route_on_a_non_ref2va_deployment_is_rejected",
            (
                "Needs a second, non-ref2va deployment "
                "(targets.non_ref2va_base_url): a t2va or fl2va server must "
                "refuse /generations/ref2va at admission rather than accept "
                "references it cannot pack."
            ),
            _payload({"images": _images(1)}),
            422,
            deployment="non_ref2va",
            detail_contains=("ref2va",),
        ),
        _Ref2vaCase(
            "references_in_a_t2va_body_on_a_non_ref2va_deployment_is_rejected",
            (
                "Needs a second, non-ref2va deployment "
                "(targets.non_ref2va_base_url): references smuggled into a "
                "plain /generations body must be refused at admission, not "
                "dropped on the floor while the job is accepted."
            ),
            _payload({"images": _images(1)}),
            422,
            path=CREATE_PATH,
            deployment="non_ref2va",
            detail_contains=("references",),
        ),
    ]


def _acceptance_cases(loaded: dict[str, str]) -> list:
    """Cases that create a real job. Smoke profile only; each is cancelled."""

    return [
        _Ref2vaCase(
            "a_single_image_reference_request_is_accepted",
            "Control: the minimal valid ref2va request is served.",
            _payload({"images": _images(1)}),
            202,
            requires_job_id=True,
        ),
        _Ref2vaCase(
            "nine_reference_images_are_accepted",
            "9 images is the documented cap, not one past it.",
            _payload({"images": _images(9)}),
            202,
            requires_job_id=True,
        ),
        _Ref2vaCase(
            "three_reference_videos_are_accepted",
            "3 videos is the documented cap; 3 x 2 s stays inside 15 s.",
            _payload({"videos": _clips(loaded, "video_2s", 3)}),
            202,
            requires_clips=("video_2s",),
            requires_job_id=True,
        ),
        _Ref2vaCase(
            "three_reference_audios_with_a_visual_reference_are_accepted",
            "3 audios is the cap when a visual reference is present.",
            _payload(
                {
                    "images": _images(1),
                    "audios": _clips(loaded, "audio_2s", 3),
                }
            ),
            202,
            requires_clips=("audio_2s",),
            requires_job_id=True,
        ),
        _Ref2vaCase(
            "an_audio_reference_paired_with_an_image_is_accepted",
            "The pairing rule accepts, it does not merely fail to reject.",
            _payload(
                {
                    "images": _images(1),
                    "audios": _clips(loaded, "audio_2s", 1),
                }
            ),
            202,
            requires_clips=("audio_2s",),
            requires_job_id=True,
        ),
        _Ref2vaCase(
            "a_two_second_reference_video_clip_is_accepted",
            "2 s is inside the per-clip window, being its inclusive floor.",
            _payload({"videos": _clips(loaded, "video_2s", 1)}),
            202,
            requires_clips=("video_2s",),
            requires_job_id=True,
        ),
        _Ref2vaCase(
            "a_fifteen_second_reference_video_clip_is_accepted",
            "15 s is inside the window, being its inclusive ceiling.",
            _payload({"videos": _clips(loaded, "video_15s", 1)}),
            202,
            requires_clips=("video_15s",),
            requires_job_id=True,
        ),
    ]


def _cases(profile: Profile, loaded: dict[str, str]) -> list:
    cases = _refusal_cases(loaded)
    if profile == "smoke":
        cases.extend(_acceptance_cases(loaded))
    return cases


def load_reference_clips(
    paths: dict[str, str] | None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Read clip fixtures into base64. Returns (loaded, unusable-with-reason)."""

    loaded: dict[str, str] = {}
    unusable: dict[str, str] = {}
    for name, raw_path in dict(paths or {}).items():
        if name not in _CLIP_SPECS:
            logger.warning(
                "ignoring unknown reference clip %r; known clips: %s",
                name,
                ", ".join(sorted(_CLIP_SPECS)),
            )
            continue
        try:
            data = Path(str(raw_path)).read_bytes()
        except OSError as exc:
            unusable[name] = f"{raw_path}: {type(exc).__name__}: {exc}"
            continue
        if not data:
            unusable[name] = f"{raw_path}: file is empty"
            continue
        loaded[name] = base64.b64encode(data).decode("ascii")
    return loaded, unusable


def _skip_reason(
    case: _Ref2vaCase,
    *,
    loaded: dict[str, str],
    unusable: dict[str, str],
    non_ref2va_base_url: str | None,
) -> str:
    if case.deployment == "non_ref2va" and not non_ref2va_base_url:
        return (
            "requires a second deployment that does not serve ref2va; set "
            "targets.non_ref2va_base_url (or --non-ref2va-base-url)"
        )
    for name in case.requires_clips:
        if name in loaded:
            continue
        spec = _CLIP_SPECS[name]
        detail = unusable.get(name)
        problem = f"unreadable ({detail})" if detail else "not supplied"
        return (
            f"requires reference clip fixture {name!r} ({problem}); it must be "
            f"{spec.requirement}. Supply it via targets.reference_clips"
        )
    return ""


def _headers(api_key: str, auth_mode: str) -> dict[str, str]:
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if auth_mode == "valid":
        headers["Authorization"] = f"Bearer {api_key}"
    elif auth_mode == "invalid":
        headers["Authorization"] = "Bearer definitely-invalid-key"
    return headers


async def _cancel_created_job(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    api_key: str,
    task_id: str,
) -> dict[str, Any] | None:
    url = f"{base_url.rstrip('/')}{CANCEL_PATH.format(job_id=task_id)}"
    try:
        async with session.post(url, headers=_headers(api_key, "valid")) as response:
            if response.status != 200:
                return None
            data = await response.json()
            return data if isinstance(data, dict) else None
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
        return None


def _missing_detail_fragments(data: Any, case: _Ref2vaCase) -> tuple[str, ...]:
    if not isinstance(data, dict) or "detail" not in data:
        return case.detail_contains
    rendered = json.dumps(data["detail"], default=str).lower()
    return tuple(
        fragment
        for fragment in case.detail_contains
        if fragment.lower() not in rendered
    )


async def _run_case(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    non_ref2va_base_url: str | None,
    api_key: str,
    case: _Ref2vaCase,
    loaded: dict[str, str],
    unusable: dict[str, str],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "check": case.name,
        "contract": case.contract,
        "expected_status": case.expected_status,
        "deployment": case.deployment,
        "endpoint": case.path,
    }

    skip_reason = _skip_reason(
        case,
        loaded=loaded,
        unusable=unusable,
        non_ref2va_base_url=non_ref2va_base_url,
    )
    if skip_reason:
        result.update(
            {
                "passed": None,
                "skipped": True,
                "actual_status": "skipped",
                "message": skip_reason,
            }
        )
        return result

    target_base_url = (
        base_url if case.deployment == "ref2va" else str(non_ref2va_base_url)
    )
    endpoint_url = f"{target_base_url.rstrip('/')}{case.path}"
    result["endpoint_url"] = endpoint_url

    try:
        async with session.post(
            endpoint_url,
            headers=_headers(api_key, case.auth_mode),
            json=case.payload,
        ) as response:
            response_text = await response.text()
            try:
                data = json.loads(response_text) if response_text else None
            except json.JSONDecodeError:
                data = None

            task_id = data.get("id") if isinstance(data, dict) else None
            if not isinstance(task_id, str) or not task_id:
                task_id = None

            # Cancel whatever was created, including a job the server should
            # have refused: a contract failure must not leave work on device.
            cancellation = None
            if response.status == 202 and task_id is not None:
                cancellation = await _cancel_created_job(
                    session,
                    base_url=target_base_url,
                    api_key=api_key,
                    task_id=task_id,
                )

            passed = response.status == case.expected_status
            message = ""
            if not passed and response.status == 202:
                message = (
                    "server accepted a request it must refuse; the created job "
                    + ("was cancelled" if cancellation else "could not be cancelled")
                )
            elif case.requires_job_id:
                if task_id is None:
                    passed = False
                    message = "accepted response did not include a non-empty id"
                elif cancellation is None:
                    passed = False
                    message = "accepted job could not be cancelled"
            elif response.status >= 400:
                if not isinstance(data, dict) or "detail" not in data:
                    passed = False
                    message = "error response did not include FastAPI detail"
                else:
                    missing = _missing_detail_fragments(data, case)
                    if missing:
                        passed = False
                        message = (
                            "refusal detail does not name "
                            f"{', '.join(repr(item) for item in missing)}"
                        )

            result.update(
                {
                    "passed": passed,
                    "skipped": False,
                    "actual_status": response.status,
                    "task_id": task_id,
                    "cancellation": cancellation,
                    "message": message,
                    "response_excerpt": response_text.replace("\n", " ")[
                        :RESPONSE_EXCERPT_LENGTH
                    ],
                }
            )
            return result
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
        result.update(
            {
                "passed": False,
                "skipped": False,
                "actual_status": "request_error",
                "message": f"{type(exc).__name__}: {exc}",
            }
        )
        return result


async def run_ref2va_contract(
    *,
    base_url: str,
    api_key: str,
    profile: Profile = DEFAULT_PROFILE,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    non_ref2va_base_url: str | None = None,
    reference_clips: dict[str, str] | None = None,
) -> dict[str, Any]:
    normalized_profile = str(profile).lower()
    if normalized_profile not in _PROFILES:
        raise ValueError(f"profile must be one of {sorted(_PROFILES)}, got {profile!r}")

    loaded, unusable = load_reference_clips(reference_clips)
    endpoint_url = f"{base_url.rstrip('/')}{REF2VA_PATH}"
    timeout = aiohttp.ClientTimeout(total=request_timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = [
            await _run_case(
                session,
                base_url=base_url,
                non_ref2va_base_url=non_ref2va_base_url,
                api_key=api_key,
                case=case,
                loaded=loaded,
                unusable=unusable,
            )
            for case in _cases(normalized_profile, loaded)  # type: ignore[arg-type]
        ]

    executed = [result for result in results if not result.get("skipped")]
    passed = sum(1 for result in executed if result.get("passed"))
    skipped = [result["check"] for result in results if result.get("skipped")]
    summary = f"{passed}/{len(executed)} checks passed"
    if skipped:
        summary += f", {len(skipped)} skipped"

    return {
        "endpoint_url": endpoint_url,
        "task_name": "minimax_h3_ref2va_contract",
        "profile": normalized_profile,
        "summary": summary,
        "detailed_test_results": results,
        "skipped_checks": skipped,
        "reference_clips_loaded": sorted(loaded),
        "reference_clips_unusable": unusable,
        "non_ref2va_base_url": non_ref2va_base_url,
        "success": bool(executed) and passed == len(executed),
    }


class MiniMaxH3Ref2vaContractTest(BaseTest):
    """Workflow-compatible wrapper around the Ref2VA request contract suite."""

    KIND = "minimax_h3_ref2va_contract"
    TASK_TYPE = "functional"
    HARDWARE_REQUIREMENT = HardwareRequirement.ANY_CHIP

    async def _run_specific_test_async(self) -> dict[str, Any]:
        non_ref2va_base_url = self.targets.get("non_ref2va_base_url")
        reference_clips = self.targets.get("reference_clips") or {}
        return await run_ref2va_contract(
            base_url=self.base_url,
            api_key=resolve_server_api_key(),
            profile=str(self.targets.get("profile", DEFAULT_PROFILE)),  # type: ignore[arg-type]
            request_timeout=float(
                self.targets.get(
                    "request_timeout",
                    DEFAULT_REQUEST_TIMEOUT_SECONDS,
                )
            ),
            non_ref2va_base_url=(
                str(non_ref2va_base_url) if non_ref2va_base_url else None
            ),
            reference_clips=dict(reference_clips),
        )


def run_minimax_h3_ref2va_contract(
    ctx: MediaContext,
    targets: dict[str, Any] | None = None,
) -> Block:
    return MiniMaxH3Ref2vaContractTest(
        TestConfig(
            {
                "timeout": DEFAULT_TEST_TIMEOUT_SECONDS,
                "retry_attempts": 0,
                "retry_delay": 0,
                "break_on_failure": False,
            }
        ),
        targets or {},
        ctx=ctx,
    ).run_tests()


def _parse_reference_clips(values: list[str] | None) -> dict[str, str]:
    clips: dict[str, str] = {}
    for item in values or []:
        name, separator, path = str(item).partition("=")
        if not separator or not name or not path:
            raise argparse.ArgumentTypeError(
                f"--reference-clip expects NAME=PATH, got {item!r}"
            )
        clips[name] = path
    return clips


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    known = "\n".join(
        f"  {fixture.name}: {fixture.requirement}" for fixture in REQUIRED_CLIP_FIXTURES
    )
    parser = argparse.ArgumentParser(
        description=(
            "Check MiniMax-H3 Ref2VA on POST /v1/videos/generations/ref2va. "
            "Requires MODEL_RUNNER=tt-minimax-h3-ref2va."
        ),
        epilog=f"reference clip fixtures (--reference-clip NAME=PATH):\n{known}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--profile", choices=sorted(_PROFILES), default=DEFAULT_PROFILE)
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--non-ref2va-base-url",
        default=None,
        help=(
            "Base URL of a t2va/fl2va deployment; without it the "
            "cross-deployment checks are skipped rather than passed."
        ),
    )
    parser.add_argument(
        "--reference-clip",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "Real reference clip fixture. Repeatable. Cases needing a clip "
            "that is not supplied are skipped with a reason."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = asyncio.run(
            run_ref2va_contract(
                base_url=args.base_url,
                api_key=resolve_server_api_key(),
                profile=args.profile,
                request_timeout=args.request_timeout,
                non_ref2va_base_url=args.non_ref2va_base_url,
                reference_clips=_parse_reference_clips(args.reference_clip),
            )
        )
    except Exception as exc:  # noqa: BLE001 - CLI emits a structured failure
        logger.exception("MiniMax-H3 Ref2VA contract could not run")
        result = {
            "task_name": "minimax_h3_ref2va_contract",
            "success": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("success") else 1


__all__ = [
    "REF2VA_PATH",
    "REQUIRED_CLIP_FIXTURES",
    "MiniMaxH3Ref2vaContractTest",
    "load_reference_clips",
    "run_minimax_h3_ref2va_contract",
    "run_ref2va_contract",
]


if __name__ == "__main__":
    sys.exit(main())
