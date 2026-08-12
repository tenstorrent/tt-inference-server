# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Contract checks for MiniMax-H3 video-generation task creation.

The cases in this module are derived from MiniMax's published
``POST /v2/video_generation`` contract:

https://platform.minimax.io/docs/api-reference/video-generation-v2-create

The HTTP suite is deliberately independent of the fixture-backed MiniMax mock.
It can be run directly against an already-running endpoint during development,
then invoked through :class:`BaseTest` by the workflow engine later.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping, cast

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import aiohttp

from report_module.schema import Block
from test_module._test_common import BaseTest, HardwareRequirement, TestConfig

if TYPE_CHECKING:
    from test_module.context import MediaContext

logger = logging.getLogger(__name__)

CREATE_PATH = "/v2/video_generation"
MODEL_NAME = "MiniMax-H3"
DEFAULT_PROFILE = "smoke"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0
DEFAULT_TEST_TIMEOUT_SECONDS = 300
MAX_RESPONSE_EXCERPT = 500

HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 401

Profile = Literal["validation", "smoke", "full"]
AuthMode = Literal["valid", "missing", "invalid"]

_PROFILES = frozenset({"validation", "smoke", "full"})

# These media URLs are the examples published in the Create Video Generation
# Task documentation. Keeping them here makes the full profile exercise the
# documented request shapes rather than mock-specific fixture conventions.
_DOCUMENTED_FIRST_FRAME_URL = (
    "https://cdn.hailuoai.com/prod/hailuo_demo/testsets/H3_AA_I2VA/"
    "gallery/sr_v17_variants_seed42_43_20260724/inputs/"
    "4a3a90bf9100_KDmcbkhzYo5sjjxr9FqcVmWVnzb.png"
)
_DOCUMENTED_REFERENCE_VIDEO_URL = (
    "https://cdn.hailuoai.com/prod/hailuo_demo/testsets/"
    "h3_promo_eval_ref2va/gallery/sr_v2p26_trio_seed42_20260724/inputs/"
    "297573323635_00_%E8%A7%86%E9%A2%911_"
    "YnyRbxEwio_video_20260525_163755_1927e9d3.mp4"
)
_DOCUMENTED_REFERENCE_AUDIO_URL = (
    "https://cdn.hailuoai.com/prod/hailuo_demo/testsets/"
    "h3_promo_eval_ref2va/gallery/sr_v2p26_trio_seed42_20260724/inputs/"
    "f463d523c5ce_01_%E9%9F%B3%E9%A2%911_"
    "RSLcbpzJPo_6%E6%9C%885%E6%97%A5(1).mp3"
)


@dataclass(frozen=True)
class _RequestCase:
    """One independently reported request/response contract check."""

    name: str
    expected_status: int
    json_payload: Mapping[str, Any] | None = None
    raw_payload: str | None = None
    auth_mode: AuthMode = "valid"
    content_type: str = "application/json"
    expected_error_type: str | None = None
    require_task_id: bool = False


def _text_to_video_payload() -> dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "content": [
            {
                "type": "text",
                "text": (
                    "Epic space-opera theatrical teaser: a female captain "
                    "stands alone before a massive observation window as the "
                    "last fleet gathers and jumps away in a blinding flash, "
                    "the bridge shaking, leaving her behind."
                ),
            }
        ],
        "resolution": "2K",
        "duration": 5,
        "ratio": "16:9",
    }


def _image_to_video_payload() -> dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "content": [
            {
                "type": "text",
                "text": (
                    "Pull focus to the people in the background and add more "
                    "steam to the ramen bowl."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": _DOCUMENTED_FIRST_FRAME_URL},
                "role": "first_frame",
            },
        ],
        "resolution": "2K",
        "duration": 5,
        "ratio": "adaptive",
    }


def _reference_to_video_payload() -> dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "content": [
            {
                "type": "text",
                "text": (
                    "Character speaks: Follow the wind, live free. Leave "
                    "worries behind, enjoy the moment. Voice timbre follows "
                    "reference audio 1."
                ),
            },
            {
                "type": "video_url",
                "video_url": {"url": _DOCUMENTED_REFERENCE_VIDEO_URL},
                "role": "reference_video",
            },
            {
                "type": "audio_url",
                "audio_url": {"url": _DOCUMENTED_REFERENCE_AUDIO_URL},
                "role": "reference_audio",
            },
        ],
        "resolution": "2K",
        "duration": 5,
        "ratio": "adaptive",
    }


def _copy_with(
    payload: Mapping[str, Any],
    *,
    updates: Mapping[str, Any] | None = None,
    remove: Iterable[str] = (),
) -> dict[str, Any]:
    result = deepcopy(dict(payload))
    for field in remove:
        result.pop(field, None)
    if updates:
        result.update(deepcopy(dict(updates)))
    return result


def _validation_cases() -> list[_RequestCase]:
    base = _text_to_video_payload()

    no_text = _copy_with(
        base,
        updates={
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": _DOCUMENTED_FIRST_FRAME_URL},
                    "role": "first_frame",
                }
            ]
        },
    )
    empty_text = deepcopy(base)
    empty_text["content"][0]["text"] = "   "
    oversized_text = deepcopy(base)
    oversized_text["content"][0]["text"] = "x" * 7001

    mixed_frame_and_reference = deepcopy(base)
    mixed_frame_and_reference["content"].extend(
        [
            {
                "type": "image_url",
                "image_url": {"url": _DOCUMENTED_FIRST_FRAME_URL},
                "role": "first_frame",
            },
            {
                "type": "audio_url",
                "audio_url": {"url": _DOCUMENTED_REFERENCE_AUDIO_URL},
                "role": "reference_audio",
            },
        ]
    )

    video_without_role = deepcopy(base)
    video_without_role["content"].append(
        {
            "type": "video_url",
            "video_url": {"url": _DOCUMENTED_REFERENCE_VIDEO_URL},
        }
    )
    audio_without_role = deepcopy(base)
    audio_without_role["content"].append(
        {
            "type": "audio_url",
            "audio_url": {"url": _DOCUMENTED_REFERENCE_AUDIO_URL},
        }
    )

    duplicate_first_frame = deepcopy(base)
    duplicate_first_frame["content"].extend(
        [
            {
                "type": "image_url",
                "image_url": {"url": _DOCUMENTED_FIRST_FRAME_URL},
                "role": "first_frame",
            },
            {
                "type": "image_url",
                "image_url": {"url": _DOCUMENTED_FIRST_FRAME_URL},
                "role": "first_frame",
            },
        ]
    )
    duplicate_last_frame = deepcopy(base)
    duplicate_last_frame["content"].extend(
        [
            {
                "type": "image_url",
                "image_url": {"url": _DOCUMENTED_FIRST_FRAME_URL},
                "role": "last_frame",
            },
            {
                "type": "image_url",
                "image_url": {"url": _DOCUMENTED_FIRST_FRAME_URL},
                "role": "last_frame",
            },
        ]
    )

    too_many_reference_images = deepcopy(base)
    too_many_reference_images["content"].extend(
        [
            {
                "type": "image_url",
                "image_url": {"url": f"https://example.com/reference-{index}.png"},
                "role": "reference_image",
            }
            for index in range(10)
        ]
    )
    too_many_reference_videos = deepcopy(base)
    too_many_reference_videos["content"].extend(
        [
            {
                "type": "video_url",
                "video_url": {"url": f"https://example.com/reference-{index}.mp4"},
                "role": "reference_video",
            }
            for index in range(4)
        ]
    )
    too_many_reference_audio = deepcopy(base)
    too_many_reference_audio["content"].extend(
        [
            {
                "type": "audio_url",
                "audio_url": {"url": f"https://example.com/reference-{index}.mp3"},
                "role": "reference_audio",
            }
            for index in range(4)
        ]
    )

    invalid_media_location = deepcopy(base)
    invalid_media_location["content"].append(
        {
            "type": "image_url",
            "image_url": {"url": "file:///tmp/frame.png"},
            "role": "first_frame",
        }
    )
    invalid_video_data_format = deepcopy(base)
    invalid_video_data_format["content"].append(
        {
            "type": "video_url",
            "video_url": {"url": "data:video/mov;base64,aGVsbG8="},
            "role": "reference_video",
        }
    )
    invalid_audio_data_format = deepcopy(base)
    invalid_audio_data_format["content"].append(
        {
            "type": "audio_url",
            "audio_url": {"url": "data:audio/ogg;base64,aGVsbG8="},
            "role": "reference_audio",
        }
    )

    cases = [
        _RequestCase(
            name="missing_bearer_authentication",
            expected_status=HTTP_UNAUTHORIZED,
            expected_error_type="authorized_error",
            json_payload=base,
            auth_mode="missing",
        ),
        _RequestCase(
            name="invalid_bearer_authentication",
            expected_status=HTTP_UNAUTHORIZED,
            expected_error_type="authorized_error",
            json_payload=base,
            auth_mode="invalid",
        ),
        _RequestCase(
            name="invalid_content_type",
            expected_status=HTTP_BAD_REQUEST,
            expected_error_type="bad_request_error",
            raw_payload=json.dumps(base),
            content_type="text/plain",
        ),
        _RequestCase(
            name="malformed_json",
            expected_status=HTTP_BAD_REQUEST,
            expected_error_type="bad_request_error",
            raw_payload="{",
        ),
    ]

    for field in ("model", "content", "resolution", "duration"):
        cases.append(
            _RequestCase(
                name=f"missing_required_{field}",
                expected_status=HTTP_BAD_REQUEST,
                expected_error_type="bad_request_error",
                json_payload=_copy_with(base, remove=(field,)),
            )
        )

    invalid_payloads = [
        ("wrong_model", _copy_with(base, updates={"model": "Wan2.2"})),
        ("missing_text_content", no_text),
        ("empty_text_content", empty_text),
        ("text_over_7000_characters", oversized_text),
        ("invalid_resolution", _copy_with(base, updates={"resolution": "4K"})),
        ("duration_below_minimum", _copy_with(base, updates={"duration": 3})),
        ("duration_above_maximum", _copy_with(base, updates={"duration": 16})),
        ("duration_not_integer", _copy_with(base, updates={"duration": 5.5})),
        ("missing_t2v_ratio", _copy_with(base, remove=("ratio",))),
        ("adaptive_t2v_ratio", _copy_with(base, updates={"ratio": "adaptive"})),
        ("invalid_ratio", _copy_with(base, updates={"ratio": "2:1"})),
        ("mixed_frame_and_reference_inputs", mixed_frame_and_reference),
        ("reference_video_without_role", video_without_role),
        ("reference_audio_without_role", audio_without_role),
        ("more_than_one_first_frame", duplicate_first_frame),
        ("more_than_one_last_frame", duplicate_last_frame),
        ("more_than_nine_reference_images", too_many_reference_images),
        ("more_than_three_reference_videos", too_many_reference_videos),
        ("more_than_three_reference_audio_items", too_many_reference_audio),
        ("unsupported_media_location", invalid_media_location),
        ("unsupported_video_data_uri_format", invalid_video_data_format),
        ("unsupported_audio_data_uri_format", invalid_audio_data_format),
    ]
    cases.extend(
        _RequestCase(
            name=name,
            expected_status=HTTP_BAD_REQUEST,
            expected_error_type="bad_request_error",
            json_payload=payload,
        )
        for name, payload in invalid_payloads
    )
    return cases


def _success_cases(profile: Profile) -> list[_RequestCase]:
    if profile == "validation":
        return []

    cases = [
        _RequestCase(
            name="documented_text_to_video",
            expected_status=HTTP_OK,
            json_payload=_text_to_video_payload(),
            require_task_id=True,
        )
    ]
    if profile == "full":
        cases.extend(
            [
                _RequestCase(
                    name="documented_image_to_video",
                    expected_status=HTTP_OK,
                    json_payload=_image_to_video_payload(),
                    require_task_id=True,
                ),
                _RequestCase(
                    name="documented_reference_to_video",
                    expected_status=HTTP_OK,
                    json_payload=_reference_to_video_payload(),
                    require_task_id=True,
                ),
            ]
        )
    return cases


def _resolve_api_key() -> str:
    for env_name in ("MINIMAX_API_KEY", "MINIMAX_MOCK_API_KEY"):
        value = os.getenv(env_name)
        if value:
            return value
    raise RuntimeError(
        "Set MINIMAX_API_KEY (real API) or MINIMAX_MOCK_API_KEY (mock API)"
    )


def _normalize_profile(value: Any) -> Profile:
    profile = str(value or DEFAULT_PROFILE).lower()
    if profile not in _PROFILES:
        raise ValueError(f"profile must be one of {sorted(_PROFILES)}, got {value!r}")
    return cast(Profile, profile)


def _headers(api_key: str, case: _RequestCase) -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "Content-Type": case.content_type,
    }
    if case.auth_mode == "valid":
        headers["Authorization"] = f"Bearer {api_key}"
    elif case.auth_mode == "invalid":
        headers["Authorization"] = "Bearer definitely-invalid-minimax-key"
    return headers


def _decode_json(response_text: str) -> Any:
    if not response_text:
        return None
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return None


def _is_json_content_type(content_type: str) -> bool:
    return content_type.lower().split(";", 1)[0].strip() == "application/json"


def _validate_success_response(
    *,
    response_data: Any,
    content_type: str,
) -> tuple[bool, str, str | None]:
    if not _is_json_content_type(content_type):
        return False, f"response Content-Type is not JSON: {content_type!r}", None
    if not isinstance(response_data, dict):
        return False, "response body is not a JSON object", None

    task_id = response_data.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        return False, "response is missing a non-empty string task_id", None
    return True, "", task_id


def _validate_error_response(
    *,
    response_data: Any,
    content_type: str,
    expected_status: int,
    expected_error_type: str,
) -> tuple[bool, str]:
    if not _is_json_content_type(content_type):
        return False, f"error response Content-Type is not JSON: {content_type!r}"
    if not isinstance(response_data, dict):
        return False, "error response body is not a JSON object"
    if response_data.get("type") != "error":
        return False, "error response type is not 'error'"

    error = response_data.get("error")
    if not isinstance(error, dict):
        return False, "error response is missing the error object"
    if error.get("type") != expected_error_type:
        return (
            False,
            f"expected error.type={expected_error_type!r}, got {error.get('type')!r}",
        )
    if error.get("http_code") != str(expected_status):
        return (
            False,
            (
                f"expected error.http_code={str(expected_status)!r}, "
                f"got {error.get('http_code')!r}"
            ),
        )
    if not isinstance(error.get("message"), str) or not error["message"].strip():
        return False, "error.message is missing or empty"

    request_id = response_data.get("request_id")
    if not isinstance(request_id, str) or not request_id.strip():
        return False, "error response is missing a non-empty request_id"
    return True, ""


async def _post_case(
    session: aiohttp.ClientSession,
    endpoint_url: str,
    api_key: str,
    case: _RequestCase,
) -> dict[str, Any]:
    request_args: dict[str, Any]
    if case.json_payload is not None:
        request_args = {"json": deepcopy(dict(case.json_payload))}
    else:
        request_args = {"data": case.raw_payload or ""}

    try:
        async with session.post(
            endpoint_url,
            headers=_headers(api_key, case),
            **request_args,
        ) as response:
            response_text = await response.text()
            response_data = _decode_json(response_text)
            content_type = response.headers.get("Content-Type", "")
            passed = response.status == case.expected_status
            message = ""
            task_id: str | None = None

            if not passed:
                excerpt = response_text.replace("\n", " ")[:MAX_RESPONSE_EXCERPT]
                message = (
                    f"expected HTTP {case.expected_status}, got {response.status}; "
                    f"response={excerpt!r}"
                )
            elif case.require_task_id:
                passed, message, task_id = _validate_success_response(
                    response_data=response_data,
                    content_type=content_type,
                )
            elif case.expected_error_type:
                passed, message = _validate_error_response(
                    response_data=response_data,
                    content_type=content_type,
                    expected_status=case.expected_status,
                    expected_error_type=case.expected_error_type,
                )

            result: dict[str, Any] = {
                "check": case.name,
                "expected_status": case.expected_status,
                "actual_status": response.status,
                "status": "PASS" if passed else "FAIL",
                "passed": passed,
                "message": message,
            }
            if task_id is not None:
                result["task_id"] = task_id
            return result
    except Exception as exc:
        logger.exception("MiniMax create contract case %s failed", case.name)
        return {
            "check": case.name,
            "expected_status": case.expected_status,
            "actual_status": "request_error",
            "status": "FAIL",
            "passed": False,
            "message": f"{type(exc).__name__}: {exc}",
        }


async def run_create_contract(
    *,
    base_url: str,
    api_key: str,
    profile: Profile = DEFAULT_PROFILE,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run the documented create-endpoint cases and return structured results."""

    normalized_profile = _normalize_profile(profile)
    endpoint_url = f"{base_url.rstrip('/')}{CREATE_PATH}"
    cases = _validation_cases() + _success_cases(normalized_profile)
    timeout = aiohttp.ClientTimeout(total=request_timeout)
    results: list[dict[str, Any]] = []

    logger.info(
        "Running %d MiniMax-H3 create contract cases at %s (profile=%s)",
        len(cases),
        endpoint_url,
        normalized_profile,
    )
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for case in cases:
            results.append(await _post_case(session, endpoint_url, api_key, case))

    passed = sum(1 for result in results if result["passed"])
    successful_tasks = [
        {"check": result["check"], "task_id": result["task_id"]}
        for result in results
        if result.get("task_id")
    ]
    return {
        "endpoint_url": endpoint_url,
        "task_name": "minimax_h3_create_contract",
        "profile": normalized_profile,
        "summary": f"{passed}/{len(results)} checks passed",
        "successful_task_ids": successful_tasks,
        "detailed_test_results": results,
        "success": passed == len(results),
    }


class MiniMaxH3CreateContractTest(BaseTest):
    """Workflow-compatible wrapper around the standalone contract suite."""

    KIND = "minimax_h3_create_contract"
    TASK_TYPE = "functional"
    HARDWARE_REQUIREMENT = HardwareRequirement.ANY_CHIP

    async def _run_specific_test_async(self) -> dict[str, Any]:
        return await run_create_contract(
            base_url=self.base_url,
            api_key=_resolve_api_key(),
            profile=_normalize_profile(self.targets.get("profile")),
            request_timeout=float(
                self.targets.get(
                    "request_timeout",
                    DEFAULT_REQUEST_TIMEOUT_SECONDS,
                )
            ),
        )


def run_minimax_h3_create_contract(
    ctx: MediaContext,
    targets: dict[str, Any] | None = None,
) -> Block:
    """Run the MiniMax-H3 create contract under a workflow ``MediaContext``."""

    test_config = TestConfig(
        {
            "timeout": DEFAULT_TEST_TIMEOUT_SECONDS,
            # Retrying successful create calls would submit additional billable
            # tasks to the real API, so this contract suite never retries.
            "retry_attempts": 0,
            "retry_delay": 0,
            "break_on_failure": False,
        }
    )
    return MiniMaxH3CreateContractTest(
        test_config,
        targets or {},
        ctx=ctx,
    ).run_tests()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run MiniMax-H3 POST /v2/video_generation contract checks "
            "against an already-running endpoint."
        )
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="Server origin, for example http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(_PROFILES),
        default=DEFAULT_PROFILE,
        help=(
            "validation: invalid requests only; smoke: validation plus T2V; "
            "full: validation plus all three documented examples"
        ),
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help="Per-request timeout in seconds",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the JSON report; stdout is always populated",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = asyncio.run(
            run_create_contract(
                base_url=args.base_url,
                api_key=_resolve_api_key(),
                profile=_normalize_profile(args.profile),
                request_timeout=args.request_timeout,
            )
        )
    except Exception as exc:
        logger.exception("MiniMax-H3 create contract could not run")
        result = {
            "task_name": "minimax_h3_create_contract",
            "success": False,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }

    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
