# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Keyframe contract checks for MiniMax-H3 FL2VA on the video V1 API.

Issue #5044 item 3. These run against a deployment booted with
``MODEL_RUNNER=tt-minimax-h3-fl2va`` — the first/last-keyframe variant, where
the keyframes ride the ordinary ``image_prompts`` list and are told apart by
position sentinels: ``frame_pos=0`` is the first keyframe, ``frame_pos=-1`` is
the last, and nothing else is a keyframe position on this deployment.

Two transports carry the same contract and are both checked here:
``POST /v1/videos/generations/i2v`` (JSON, base64 or URL keyframes) and
``POST /v1/videos/generations/i2v/upload`` (multipart file parts).

Everything is observed over HTTP — request in, status and JSON body out. The
checks are a data table (:func:`_cases`); a new requirement is one more entry
in that table, not another function.

The run begins by asking the endpoint which MiniMax-H3 variant it is
(:func:`_probe_deployment`, which cannot enqueue anything). Every case here
requires FL2VA: against the t2va or Ref2VA sibling these same payloads are
either legal input or a different refusal, so an unconfirmed endpoint leaves
every case skipped with a reason rather than scored.

Profiles:

* ``validation`` — refusals only, nothing is enqueued.
* ``smoke`` — adds the cases whose contract *is* acceptance (202). Those
  create real jobs, so each one is cancelled as soon as its id comes back.
  The ~10 MB regression case lives here, so run this profile if you want that
  one graded rather than skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import struct
import sys
import zlib
from dataclasses import dataclass, field
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
Deployment = Literal["t2va", "fl2va", "ref2va"]
DEFAULT_PROFILE: Profile = "validation"
DEFAULT_DEPLOYMENT: Deployment = "fl2va"
# A 10 MB keyframe is a ~13.3 MB request body; 60 s is not enough headroom on
# a loaded box, and the point of that case is to reach the app, not to time out.
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0
DEFAULT_TEST_TIMEOUT_SECONDS = 900
_PROFILES = frozenset({"validation", "smoke"})
_DEPLOYMENTS = frozenset({"t2va", "fl2va", "ref2va"})
# What the run targets when the endpoint could not be identified as the
# deployment this file was written for. No case requires it, so every case
# skips with a reason.
UNIDENTIFIED_DEPLOYMENT = "unidentified"

I2V_CREATE_PATH = f"{CREATE_PATH}/i2v"
I2V_UPLOAD_PATH = f"{I2V_CREATE_PATH}/upload"

HTTP_ACCEPTED = 202
HTTP_UNPROCESSABLE = 422
# Some refusals are legitimately several different 4xx codes depending on where
# in the request path they fire (SSRF policy vs. download failure vs. content
# rule). What the contract pins there is "client error, not server error".
CLIENT_ERROR = "4xx"

# The per-keyframe content rules #5044 asks FL2VA to enforce, whatever
# transport carried the image. Named so each case reads as the rule it pins;
# the server owns the actual numbers.
MIN_SHORT_SIDE_PX = 256
MAX_LONG_SIDE_PX = 5760
MIN_ASPECT_RATIO = 0.4
MAX_ASPECT_RATIO = 2.5
MAX_CONTENT_BYTES = 30_000_000
# The transport-level cap the server puts on one base64 image field. It exists
# to bound HTTP body size and is NOT the content rule above — conflating the
# two is the regression pinned by ``in_spec_ten_megabyte_keyframe_is_accepted``.
TRANSPORT_BASE64_CAP = 10_000_000

PROMPT = (
    "A paper boat drifts from the near bank of a still canal to the far bank "
    "as the afternoon light moves across the water, smooth continuous motion "
    "between the first and last frames."
)
ASPECT_RATIO = "16:9"
DURATION_SECONDS = 5

# An https URL that cannot resolve anywhere: ``.invalid`` is reserved by
# RFC 2606 precisely so it never becomes a real name.
UNREACHABLE_KEYFRAME_HOST = "fl2va-keyframe.invalid"
UNREACHABLE_KEYFRAME_URL = f"https://{UNREACHABLE_KEYFRAME_HOST}/first.png"


# --------------------------------------------------------------------------
# Image fixtures
#
# These are API-conformance tests, so the bytes on the wire have to be real
# image bytes — but Pillow is not guaranteed to be installed in the venv that
# runs this file, and importing it at module scope would make the whole suite
# un-collectable where it is missing. So nothing here uses an image library:
# every fixture is assembled by hand from the format's own byte layout using
# only ``struct`` and ``zlib`` (CRC-32 and DEFLATE), which is all a PNG needs.
#
# That also buys exact dimensions, which literal base64 blobs could not: the
# content cases each need an image that violates exactly one rule (short side,
# long side, aspect ratio) and satisfies the others, so the server's refusal
# can only be about the rule under test.
#
# The ~10 MB fixture uses the same assembler at DEFLATE level 0. Level 0 emits
# stored (uncompressed) blocks, so the encoded PNG comes out the size of its
# raw scanlines — a payload of an exact, chosen length without needing a real
# encoder or 10 MB of entropy. It is a genuine, decodable image, which matters:
# that case expects 202, so a server that decodes it must succeed.
# --------------------------------------------------------------------------


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    """One length-tag-payload-CRC32 PNG chunk."""
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
    )


def _png_b64(width: int, height: int, *, deflate_level: int = 6) -> str:
    """A valid 8-bit greyscale PNG of exactly ``width`` x ``height``, base64.

    Every scanline is a zero filter byte followed by zero pixels — a black
    image, which compresses to a few hundred bytes at the default level and to
    its raw size at level 0.
    """
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    compressor = zlib.compressobj(deflate_level)
    scanline = b"\x00" * (width + 1)
    body = []
    for _ in range(height):
        chunk = compressor.compress(scanline)
        if chunk:
            body.append(chunk)
    body.append(compressor.flush())
    raw = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", header),
            _png_chunk(b"IDAT", b"".join(body)),
            _png_chunk(b"IEND", b""),
        ]
    )
    return base64.b64encode(raw).decode("ascii")


def _gif_b64(width: int, height: int) -> str:
    """A valid GIF89a whose logical screen is ``width`` x ``height``, base64.

    The canvas is the size a decoder reports; the single image block inside it
    stays 1x1 (legal GIF, and it keeps the fixture 35 bytes) with the canonical
    "one pixel of colour 0" LZW payload. Sized in spec on purpose so the only
    thing wrong with it is the format.
    """
    raw = (
        b"GIF89a"
        + struct.pack("<HH", width, height)
        + b"\x80\x00\x00"  # global colour table of 2, background 0, no aspect
        + b"\x00\x00\x00\xff\xff\xff"  # black, white
        + b","  # image separator
        + struct.pack("<HHHH", 0, 0, 1, 1)  # left, top, width, height
        + b"\x00"  # no local colour table, not interlaced
        + b"\x02\x02\x4c\x01\x00"  # LZW min code size 2, one data sub-block
        + b";"  # trailer
    )
    return base64.b64encode(raw).decode("ascii")


def _in_spec_keyframe() -> str:
    """512x512: short side >= 256, long side <= 5760, aspect 1.0. Breaks nothing."""
    return _png_b64(512, 512)


def _short_side_keyframe() -> str:
    """255x255: one pixel under the short-side floor, in spec on every other rule."""
    return _png_b64(MIN_SHORT_SIDE_PX - 1, MIN_SHORT_SIDE_PX - 1)


def _long_side_keyframe() -> str:
    """6000x3000: over the long-side ceiling, aspect 2.0 and short side both fine."""
    return _png_b64(MAX_LONG_SIDE_PX + 240, 3000)


def _wide_keyframe() -> str:
    """900x300: aspect 3.0, above the ceiling; both sides inside the size band."""
    return _png_b64(900, 300)


def _tall_keyframe() -> str:
    """300x900: aspect 0.33, below the floor; both sides inside the size band."""
    return _png_b64(300, 900)


def _ten_megabyte_keyframe() -> str:
    """3200x3125 stored-DEFLATE PNG: ~10.0 MB of content, ~13.3 M base64 chars.

    In spec on every content rule (aspect 1.024, sides inside 256..5760, well
    under the 30 MB content ceiling) and deliberately over the
    ``TRANSPORT_BASE64_CAP`` of 10,000,000 base64 characters.
    """
    return _png_b64(3200, 3125, deflate_level=0)


# --------------------------------------------------------------------------
# Case table
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Part:
    """One multipart part: a plain form field, or a file when ``filename`` is set.

    ``value`` is the field text, or base64 image bytes for a file part (decoded
    just before the request so the table stays declarative).
    """

    name: str
    value: str
    filename: str = ""
    content_type: str = ""


@dataclass(frozen=True)
class _KeyframeCase:
    """One HTTP request and the single contract statement it pins.

    ``why`` is the case's rationale and is emitted with the result so a failure
    explains itself in the report. ``requires`` names the deployment on which
    the case can be evaluated at all; the runner skips every other one with a
    reason instead of scoring it.
    """

    name: str
    why: str
    expected_status: Any
    path: str = I2V_CREATE_PATH
    transport: Literal["json", "multipart"] = "json"
    payload: dict[str, Any] = field(default_factory=dict)
    parts: tuple[_Part, ...] = ()
    # Substrings, any one of which the error body must contain. Used only where
    # the refusal could legitimately come from several places and the case is
    # about *which* thing was refused.
    body_keywords: tuple[str, ...] = ()
    requires: Deployment = DEFAULT_DEPLOYMENT


def _payload(*keyframes: tuple[str, int]) -> dict[str, Any]:
    """An FL2VA request body carrying ``(image, frame_pos)`` keyframes."""
    return {
        "prompt": PROMPT,
        "aspect_ratio": ASPECT_RATIO,
        "duration_seconds": DURATION_SECONDS,
        "seed": 0,
        "image_prompts": [
            {"image": image, "frame_pos": frame_pos} for image, frame_pos in keyframes
        ],
    }


def _normalize_profile(profile: str) -> str:
    normalized = str(profile).lower()
    if normalized not in _PROFILES:
        raise ValueError(f"profile must be one of {sorted(_PROFILES)}, got {profile!r}")
    return normalized


def _cases(profile: Profile) -> list[_KeyframeCase]:
    """The whole case table.

    ``profile`` is validated here but never removes a case: the report should
    list every contract this file pins, including the ones the running profile
    refuses to send. The runner turns the profile into a skip-with-reason for
    the cases that would enqueue a generation job (see :func:`_skip_reason`).
    """
    _normalize_profile(profile)
    in_spec = _in_spec_keyframe()

    return [
        # -- 1. the two sentinels, alone and as a pair ----------------------
        _KeyframeCase(
            name="first_keyframe_alone_is_accepted",
            why="frame_pos=0 on its own is a complete FL2VA request",
            expected_status=HTTP_ACCEPTED,
            payload=_payload((in_spec, 0)),
        ),
        _KeyframeCase(
            name="last_keyframe_alone_is_accepted",
            why="frame_pos=-1 on its own is a complete FL2VA request",
            expected_status=HTTP_ACCEPTED,
            payload=_payload((in_spec, -1)),
        ),
        _KeyframeCase(
            name="first_and_last_keyframe_pair_is_accepted",
            why="the {0, -1} pair is the headline FL2VA request",
            expected_status=HTTP_ACCEPTED,
            payload=_payload((in_spec, 0), (in_spec, -1)),
        ),
        # -- 2. every other position is not a keyframe position -------------
        _KeyframeCase(
            name="frame_pos_one_is_rejected",
            why="frame_pos=1 is not a keyframe sentinel on FL2VA",
            expected_status=HTTP_UNPROCESSABLE,
            payload=_payload((in_spec, 1)),
        ),
        _KeyframeCase(
            name="frame_pos_five_is_rejected",
            why="an interior frame index is not a keyframe sentinel on FL2VA",
            expected_status=HTTP_UNPROCESSABLE,
            payload=_payload((in_spec, 5)),
        ),
        _KeyframeCase(
            name="frame_pos_negative_two_is_rejected",
            why="-1 is the only negative sentinel; -2 is not second-to-last",
            expected_status=HTTP_UNPROCESSABLE,
            payload=_payload((in_spec, -2)),
        ),
        # A third entry can only ever carry a position outside {0, -1}, so the
        # server may cite either the count or the position; what is pinned is
        # that three keyframes are refused rather than silently truncated.
        _KeyframeCase(
            name="three_keyframes_are_rejected",
            why="FL2VA takes at most two keyframes",
            expected_status=HTTP_UNPROCESSABLE,
            payload=_payload((in_spec, 0), (in_spec, -1), (in_spec, 1)),
        ),
        # -- 3. one image per position --------------------------------------
        _KeyframeCase(
            name="duplicate_frame_pos_is_rejected",
            why="two images cannot claim the same keyframe position",
            expected_status=HTTP_UNPROCESSABLE,
            payload=_payload((in_spec, 0), (in_spec, 0)),
        ),
        # -- 4. per-keyframe content rules ----------------------------------
        _KeyframeCase(
            name="gif_keyframe_is_rejected",
            why="GIF is not a served keyframe format",
            expected_status=HTTP_UNPROCESSABLE,
            payload=_payload((_gif_b64(512, 512), 0)),
        ),
        _KeyframeCase(
            name="keyframe_short_side_below_256px_is_rejected",
            why=f"a keyframe's short side must be >= {MIN_SHORT_SIDE_PX} px",
            expected_status=HTTP_UNPROCESSABLE,
            payload=_payload((_short_side_keyframe(), 0)),
        ),
        _KeyframeCase(
            name="keyframe_long_side_above_5760px_is_rejected",
            why=f"a keyframe's long side must be <= {MAX_LONG_SIDE_PX} px",
            expected_status=HTTP_UNPROCESSABLE,
            payload=_payload((_long_side_keyframe(), 0)),
        ),
        _KeyframeCase(
            name="keyframe_aspect_ratio_above_the_band_is_rejected",
            why=f"a keyframe's width/height must be <= {MAX_ASPECT_RATIO}",
            expected_status=HTTP_UNPROCESSABLE,
            payload=_payload((_wide_keyframe(), 0)),
        ),
        _KeyframeCase(
            name="keyframe_aspect_ratio_below_the_band_is_rejected",
            why=f"a keyframe's width/height must be >= {MIN_ASPECT_RATIO}",
            expected_status=HTTP_UNPROCESSABLE,
            payload=_payload((_tall_keyframe(), 0)),
        ),
        # The rules are per keyframe, not per request: a server that validates
        # image_prompts[0] and trusts the rest would pass every case above.
        _KeyframeCase(
            name="content_rules_apply_to_the_last_keyframe_too",
            why="every image_prompts entry is content-checked, not just the first",
            expected_status=HTTP_UNPROCESSABLE,
            payload=_payload((in_spec, 0), (_gif_b64(512, 512), -1)),
        ),
        # -- 5. the regression ----------------------------------------------
        #
        # THE case in this file. The content rule for a keyframe is 30 MB;
        # MAX_BASE64_IMAGE_LEN (10,000,000 base64 chars, ~7.5 MB decoded) is a
        # transport cap on one JSON string field, put there to bound HTTP body
        # size. Before #5044 the transport cap was the only size check in the
        # path, so it silently became the content limit and an ordinary in-spec
        # ~10 MB photo — the most obvious thing a user uploads as a keyframe —
        # came back 422 "String should have at most 10000000 characters".
        #
        # This fixture is in spec on every content rule and over the transport
        # cap by design. It must be accepted. If it 422s, the caps are still
        # conflated; if it 413s, check the ingress body limit before the app.
        _KeyframeCase(
            name="in_spec_ten_megabyte_keyframe_is_accepted",
            why=(
                f"a keyframe under the {MAX_CONTENT_BYTES}-byte content rule is "
                f"served even though it exceeds the {TRANSPORT_BASE64_CAP}-char "
                "base64 transport cap"
            ),
            expected_status=HTTP_ACCEPTED,
            payload=_payload((_ten_megabyte_keyframe(), 0)),
        ),
        # -- 6. URL-sourced keyframes ---------------------------------------
        #
        # A URL keyframe is downloaded at the API layer and then held to the
        # same rules as an inline one. Only half of that is checkable from an
        # offline suite: proving the *content* rules run after the download
        # would need a reachable, allowlisted host serving an out-of-spec
        # image, which no self-contained test can stand up. What is pinned
        # here is the other half — a keyframe URL that cannot be fetched is the
        # caller's problem (4xx naming the URL), never an unhandled 500. On a
        # deployment with no media_url_allowed_domains the same assertion holds
        # one step earlier, at the SSRF policy refusal.
        #
        # The keywords are phrases only an answer about the fetch can contain.
        # A bare "url" would also match the docs link pydantic staples onto its
        # own validation errors, which would let an unrelated 422 pass this.
        _KeyframeCase(
            name="unfetchable_url_keyframe_is_a_client_error_naming_the_url",
            why="a keyframe URL that cannot be fetched is a 4xx about the URL",
            expected_status=CLIENT_ERROR,
            payload=_payload((UNREACHABLE_KEYFRAME_URL, 0)),
            body_keywords=(
                "media url",
                "keyframe url",
                "image url",
                "download",
                UNREACHABLE_KEYFRAME_HOST,
            ),
        ),
        # -- 7. the multipart transport -------------------------------------
        #
        # The multipart endpoint is the same contract with a different wire
        # format, so it must be able to carry both keyframes. Sent as repeated
        # ``image`` file parts each followed by its ``frame_pos`` — the minimal
        # extension of the field names the endpoint already uses.
        _KeyframeCase(
            name="multipart_upload_accepts_a_first_and_last_keyframe_pair",
            why="the multipart transport can carry both keyframes, not just one",
            expected_status=HTTP_ACCEPTED,
            path=I2V_UPLOAD_PATH,
            transport="multipart",
            parts=(
                _Part("prompt", PROMPT),
                _Part("image", in_spec, "first.png", "image/png"),
                _Part("frame_pos", "0"),
                _Part("image", in_spec, "last.png", "image/png"),
                _Part("frame_pos", "-1"),
            ),
        ),
        # Guard for the case above: an endpoint that binds a single file and
        # drops the extra parts would answer 202 to it and look correct. Two
        # parts at the same position must trip the duplicate rule, which is
        # only reachable if both files were actually read.
        _KeyframeCase(
            name="multipart_upload_reads_every_uploaded_keyframe",
            why="extra multipart image parts are consumed, not silently dropped",
            expected_status=HTTP_UNPROCESSABLE,
            path=I2V_UPLOAD_PATH,
            transport="multipart",
            parts=(
                _Part("prompt", PROMPT),
                _Part("image", in_spec, "first.png", "image/png"),
                _Part("frame_pos", "0"),
                _Part("image", in_spec, "also-first.png", "image/png"),
                _Part("frame_pos", "0"),
            ),
        ),
        # -- 8. wrong endpoint for this deployment --------------------------
        _KeyframeCase(
            name="text_only_create_is_refused_on_an_fl2va_deployment",
            why="FL2VA weights need a keyframe; /generations must refuse, not 500",
            expected_status=HTTP_UNPROCESSABLE,
            path=CREATE_PATH,
            payload={
                "prompt": PROMPT,
                "aspect_ratio": ASPECT_RATIO,
                "duration_seconds": DURATION_SECONDS,
                "seed": 0,
            },
        ),
    ]


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------


def _headers(api_key: str, *, json_body: bool = True) -> dict[str, str]:
    headers = {"Accept": "application/json", "Authorization": f"Bearer {api_key}"}
    if json_body:
        headers["Content-Type"] = "application/json"
    return headers


def _status_matches(expected: Any, actual: int) -> bool:
    if expected == CLIENT_ERROR:
        return 400 <= actual < 500
    return actual == expected


def _decode(response_text: str) -> Any:
    if not response_text:
        return None
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return None


def _body_text(data: Any, response_text: str) -> str:
    if data is None:
        return response_text.lower()
    return json.dumps(data).lower()


def _response_excerpt(response_text: str) -> str:
    """Truncate a response body before it reaches the report.

    Pydantic echoes the offending input back in its validation detail, so a
    refused 10 MB keyframe would otherwise paste 13 million characters of
    base64 into the run report.
    """
    return response_text.replace("\n", " ")[:RESPONSE_EXCERPT_LENGTH]


def _skip_reason(case: _KeyframeCase, *, profile: str, deployment: str) -> str | None:
    """Why this case cannot be graded here, or ``None`` if it can be."""
    if case.requires != deployment:
        return (
            f"case needs a {case.requires} deployment; this run is against "
            f"{deployment} (see deployment_probe). On another MiniMax-H3 "
            "variant these payloads are legal input, so grading them would "
            "enqueue real generations instead of observing a refusal"
        )
    if case.expected_status == HTTP_ACCEPTED and profile != "smoke":
        return (
            "the contract here is acceptance, so grading it enqueues a "
            "generation job; re-run with --profile smoke"
        )
    return None


def _skipped_result(case: _KeyframeCase, *, reason: str) -> dict[str, Any]:
    return {
        "check": case.name,
        "passed": False,
        "skipped": True,
        "expected_status": case.expected_status,
        "actual_status": "skipped",
        "endpoint": case.path,
        "transport": case.transport,
        "message": reason,
        "why": case.why,
        "requires": case.requires,
    }


def _build_form(case: _KeyframeCase) -> aiohttp.FormData:
    form = aiohttp.FormData()
    for part in case.parts:
        if not part.filename:
            form.add_field(part.name, part.value)
            continue
        form.add_field(
            part.name,
            base64.b64decode(part.value),
            filename=part.filename,
            content_type=part.content_type or "application/octet-stream",
        )
    return form


async def _cancel_created_job(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    api_key: str,
    task_id: str,
) -> dict[str, Any] | None:
    """Best-effort cleanup so an accepted case does not leave work running."""
    url = f"{base_url.rstrip('/')}{CANCEL_PATH.format(job_id=task_id)}"
    try:
        async with session.post(url, headers=_headers(api_key)) as response:
            if response.status != 200:
                return None
            data = await response.json()
            return data if isinstance(data, dict) else None
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
        return None


async def _run_case(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    api_key: str,
    case: _KeyframeCase,
    profile: str,
    deployment: str,
) -> dict[str, Any]:
    reason = _skip_reason(case, profile=profile, deployment=deployment)
    if reason is not None:
        return _skipped_result(case, reason=reason)

    url = f"{base_url.rstrip('/')}{case.path}"
    try:
        if case.transport == "multipart":
            request = session.post(
                url,
                headers=_headers(api_key, json_body=False),
                data=_build_form(case),
            )
        else:
            request = session.post(
                url,
                headers=_headers(api_key),
                json=case.payload,
            )

        async with request as response:
            response_text = await response.text()
            data = _decode(response_text)

            passed = _status_matches(case.expected_status, response.status)
            messages: list[str] = []
            task_id: str | None = None
            cancellation: dict[str, Any] | None = None

            if response.status == HTTP_ACCEPTED:
                task_id = data.get("id") if isinstance(data, dict) else None
                if not isinstance(task_id, str) or not task_id:
                    task_id = None
                    passed = False
                    messages.append("accepted response did not include a non-empty id")
                else:
                    # Cancel whatever was created, including an acceptance the
                    # case did not want — the job is running either way.
                    cancellation = await _cancel_created_job(
                        session,
                        base_url=base_url,
                        api_key=api_key,
                        task_id=task_id,
                    )
                    if cancellation is None:
                        messages.append(f"created job {task_id} could not be cancelled")
            elif response.status >= 400:
                if not isinstance(data, dict) or "detail" not in data:
                    passed = False
                    messages.append("error response did not include FastAPI detail")
                if case.body_keywords:
                    body = _body_text(data, response_text)
                    if not any(word in body for word in case.body_keywords):
                        passed = False
                        messages.append(
                            "error body names none of "
                            f"{list(case.body_keywords)}: {body[:200]}"
                        )

            if response.status == 413 and case.expected_status == HTTP_ACCEPTED:
                messages.append(
                    "413 can come from an ingress body cap rather than the app; "
                    "check the proxy's client_max_body_size before reading this "
                    "as an app-level refusal"
                )

            return {
                "check": case.name,
                "passed": passed,
                "skipped": False,
                "expected_status": case.expected_status,
                "actual_status": response.status,
                "endpoint": case.path,
                "transport": case.transport,
                "task_id": task_id,
                "cancellation": cancellation,
                "message": "; ".join(messages),
                "response": _response_excerpt(response_text),
                "why": case.why,
                "requires": case.requires,
            }
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
        return {
            "check": case.name,
            "passed": False,
            "skipped": False,
            "expected_status": case.expected_status,
            "actual_status": "request_error",
            "endpoint": case.path,
            "transport": case.transport,
            "message": f"{type(exc).__name__}: {exc}",
            "why": case.why,
            "requires": case.requires,
        }


def _probe_payload() -> dict[str, Any]:
    """A request that no deployment can accept and only FL2VA answers uniquely.

    ``duration_seconds`` is far outside the shared field bound, so this is
    refused everywhere before anything is enqueued — the probe cannot start a
    generation by accident, which is the whole point of running it first.

    Pydantic reports every field's error, not just the first, so on FL2VA the
    detail also carries its keyframe-sentinel refusal of ``frame_pos=5``. That
    sentence names the variant, and no other deployment produces it: t2va does
    not police frame positions, and Ref2VA refuses the route outright.
    """
    return {
        "prompt": PROMPT,
        "duration_seconds": 999,
        "image_prompts": [{"image": _in_spec_keyframe(), "frame_pos": 5}],
    }


async def _probe_deployment(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    api_key: str,
) -> dict[str, Any]:
    """Ask the endpoint which MiniMax-H3 variant it is, without enqueuing work.

    Worth doing because the three variants ship from one branch and answer on
    the same paths: a suite pointed at the t2va sibling would find ``frame_pos``
    5 perfectly legal there and enqueue a real generation for every case this
    file expects to be refused. Identification is positive — anything this
    cannot confirm as FL2VA leaves the run skipped rather than scored.
    """
    url = f"{base_url.rstrip('/')}{I2V_CREATE_PATH}"
    try:
        async with session.post(
            url,
            headers=_headers(api_key),
            json=_probe_payload(),
        ) as response:
            response_text = await response.text()
            body = _body_text(_decode(response_text), response_text)
            if "ref2va" in body:
                identified = "ref2va"
            elif "fl2va" in body:
                identified = "fl2va"
            else:
                identified = UNIDENTIFIED_DEPLOYMENT
            return {
                "status": response.status,
                "identified_as": identified,
                "body_excerpt": body[:RESPONSE_EXCERPT_LENGTH],
            }
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
        return {
            "status": "request_error",
            "identified_as": UNIDENTIFIED_DEPLOYMENT,
            "body_excerpt": f"{type(exc).__name__}: {exc}",
        }


async def run_fl2va_contract(
    *,
    base_url: str,
    api_key: str,
    profile: Profile = DEFAULT_PROFILE,
    deployment: Deployment = DEFAULT_DEPLOYMENT,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    normalized_profile = _normalize_profile(profile)
    normalized_deployment = str(deployment).lower()
    if normalized_deployment not in _DEPLOYMENTS:
        raise ValueError(
            f"deployment must be one of {sorted(_DEPLOYMENTS)}, got {deployment!r}"
        )

    endpoint_url = f"{base_url.rstrip('/')}{I2V_CREATE_PATH}"
    timeout = aiohttp.ClientTimeout(total=request_timeout)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        probe = await _probe_deployment(session, base_url=base_url, api_key=api_key)
        # ``--deployment`` says what this run is aimed at; the probe says what
        # actually answered. Cases are graded only where the two agree, so a
        # suite pointed at a sibling deployment skips rather than enqueuing work
        # against weights these cases were not written for.
        effective_deployment = (
            probe["identified_as"]
            if probe["identified_as"] == normalized_deployment
            else UNIDENTIFIED_DEPLOYMENT
        )

        results = [
            await _run_case(
                session,
                base_url=base_url,
                api_key=api_key,
                case=case,
                profile=normalized_profile,
                deployment=effective_deployment,
            )
            for case in _cases(normalized_profile)  # type: ignore[arg-type]
        ]

    evaluated = [result for result in results if not result["skipped"]]
    skipped = len(results) - len(evaluated)
    passed = sum(1 for result in evaluated if result["passed"])

    report: dict[str, Any] = {
        "endpoint_url": endpoint_url,
        "task_name": "minimax_h3_fl2va_contract",
        "profile": normalized_profile,
        "deployment": effective_deployment,
        "deployment_probe": probe,
        "summary": f"{passed}/{len(evaluated)} checks passed, {skipped} skipped",
        "detailed_test_results": results,
        "success": bool(evaluated) and passed == len(evaluated),
    }
    if not evaluated:
        # Nothing was graded: say so rather than reporting a green run.
        report["status"] = "skip"
        report["reason"] = (
            f"no FL2VA keyframe case could be graded against {endpoint_url}: "
            f"aimed at {normalized_deployment}, endpoint identified itself as "
            f"{probe['identified_as']} (profile={normalized_profile})"
        )
    return report


class MiniMaxH3Fl2vaContractTest(BaseTest):
    """Workflow-compatible wrapper around the FL2VA keyframe contract suite."""

    KIND = "minimax_h3_fl2va_contract"
    TASK_TYPE = "functional"
    HARDWARE_REQUIREMENT = HardwareRequirement.ANY_CHIP

    async def _run_specific_test_async(self) -> dict[str, Any]:
        return await run_fl2va_contract(
            base_url=self.base_url,
            api_key=resolve_server_api_key(),
            profile=str(self.targets.get("profile", DEFAULT_PROFILE)),  # type: ignore[arg-type]
            deployment=str(  # type: ignore[arg-type]
                self.targets.get("deployment", DEFAULT_DEPLOYMENT)
            ),
            request_timeout=float(
                self.targets.get(
                    "request_timeout",
                    DEFAULT_REQUEST_TIMEOUT_SECONDS,
                )
            ),
        )


def run_minimax_h3_fl2va_contract(
    ctx: MediaContext,
    targets: dict[str, Any] | None = None,
) -> Block:
    return MiniMaxH3Fl2vaContractTest(
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check the MiniMax-H3 FL2VA keyframe contract on POST "
            "/v1/videos/generations/i2v and /i2v/upload."
        )
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--profile", choices=sorted(_PROFILES), default=DEFAULT_PROFILE)
    parser.add_argument(
        "--deployment",
        choices=sorted(_DEPLOYMENTS),
        default=DEFAULT_DEPLOYMENT,
        help=(
            "Deployment this run is aimed at. Cases are scored only when this "
            "matches what the endpoint identifies itself as; anything else "
            "leaves every case skipped with a reason."
        ),
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = asyncio.run(
            run_fl2va_contract(
                base_url=args.base_url,
                api_key=resolve_server_api_key(),
                profile=args.profile,
                deployment=args.deployment,
                request_timeout=args.request_timeout,
            )
        )
    except Exception as exc:  # noqa: BLE001 - CLI emits a structured failure
        logger.exception("MiniMax-H3 FL2VA keyframe contract could not run")
        result = {
            "task_name": "minimax_h3_fl2va_contract",
            "success": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("success") else 1


__all__ = [
    "MiniMaxH3Fl2vaContractTest",
    "run_fl2va_contract",
    "run_minimax_h3_fl2va_contract",
]


if __name__ == "__main__":
    sys.exit(main())
