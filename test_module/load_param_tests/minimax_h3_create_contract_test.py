# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Contract checks for MiniMax-H3 on the inference-server video V1 API.

The checks follow the task the deployment serves (``resolve_h3_task``): they are sent to
that task's create route with a valid body of its request shape (``request_task_for``), and
every route the deployment cannot serve must refuse an empty body with a 422 route refusal.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import uuid
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit, urlunsplit

import aiohttp  # pyright: ignore[reportMissingImports]

from test_module._test_common import BaseTest, HardwareRequirement, TestConfig
from test_module._test_common.minimax_h3_client import (
    CREATE_PATHS,
    DEFAULT_TASK,
    H3_TASKS,
    MiniMaxClientError,
    MiniMaxH3Client,
    build_create_payload,
    request_task_for,
    resolve_h3_task,
    resolve_server_api_key,
)

if TYPE_CHECKING:
    from report_module.schema import Block
    from test_module.context import MediaContext

logger = logging.getLogger(__name__)

Profile = Literal["validation", "smoke"]
DEFAULT_PROFILE: Profile = "validation"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0
DEFAULT_TEST_TIMEOUT_SECONDS = 300
_PROFILES = frozenset({"validation", "smoke"})
# Routes each deployment refuses with a 422 naming the deployment (tt-media-server
# open_ai_api/video.py). FL2VA serves text-only /generations, so it refuses only Ref2VA.
REFUSED_TASKS = {
    "t2va": ("fl2va", "ref2va"),
    "fl2va": ("ref2va",),
    "ref2va": ("t2va", "fl2va"),
}
PROMPT = (
    "A red fox steps through wet grass at dawn while birds sing in "
    "the background and the camera tracks alongside."
)


@dataclass(frozen=True)
class _RequestCase:
    name: str
    payload: dict[str, Any]
    expected_status: int
    auth_mode: Literal["valid", "missing", "invalid"] = "valid"
    requires_job_id: bool = False
    # Request shape whose route the case is posted to (CREATE_PATHS).
    route_task: str = DEFAULT_TASK
    # A route refusal: 422 with a string detail, not a list of field errors.
    expects_refusal: bool = False


def _valid_payload(task: str = DEFAULT_TASK) -> dict[str, Any]:
    return build_create_payload(
        task, prompt=PROMPT, aspect_ratio="16:9", duration_seconds=5
    )


def _validation_cases(task: str = DEFAULT_TASK) -> list[_RequestCase]:
    """The field checks on ``task``'s own route, then the routes it refuses."""

    route_task = request_task_for(task)
    valid = _valid_payload(route_task)
    field_cases = [
        _RequestCase("missing_bearer_authentication", valid, 401, "missing"),
        _RequestCase("invalid_bearer_authentication", valid, 401, "invalid"),
        _RequestCase(
            "missing_prompt",
            {key: value for key, value in valid.items() if key != "prompt"},
            422,
        ),
        _RequestCase(
            "provider_model_field_is_rejected",
            {**valid, "model": "MiniMax-H3"},
            422,
        ),
        _RequestCase(
            "provider_resolution_field_is_rejected",
            {**valid, "resolution": "768P"},
            422,
        ),
        _RequestCase(
            "provider_ratio_field_is_rejected",
            {**valid, "ratio": "16:9"},
            422,
        ),
        _RequestCase(
            "provider_duration_field_is_rejected",
            {**valid, "duration": 5},
            422,
        ),
        _RequestCase(
            "unsupported_aspect_ratio",
            {**valid, "aspect_ratio": "2:1"},
            422,
        ),
        _RequestCase(
            "duration_below_minimum",
            {**valid, "duration_seconds": 3},
            422,
        ),
        _RequestCase(
            "duration_above_maximum",
            {**valid, "duration_seconds": 16},
            422,
        ),
        _RequestCase(
            "explicit_num_inference_steps_is_rejected",
            {**valid, "num_inference_steps": 50},
            422,
        ),
    ]
    # The route gate runs before body validation, so an empty body tells a refused route
    # (string detail) from a served one (list of field errors) without ever creating a job.
    return [replace(case, route_task=route_task) for case in field_cases] + [
        _RequestCase(
            f"{refused}_route_is_refused",
            {},
            422,
            route_task=refused,
            expects_refusal=True,
        )
        for refused in REFUSED_TASKS[task]
    ]


def _cases(profile: Profile, task: str = DEFAULT_TASK) -> list[_RequestCase]:
    cases = _validation_cases(task)
    if profile == "smoke":
        route_task = request_task_for(task)
        cases.append(
            _RequestCase(
                f"valid_{route_task}_job",
                _valid_payload(route_task),
                202,
                requires_job_id=True,
                route_task=route_task,
            )
        )
    return cases


def _headers(api_key: str, auth_mode: str) -> dict[str, str]:
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if auth_mode == "valid":
        headers["Authorization"] = f"Bearer {api_key}"
    elif auth_mode == "invalid":
        headers["Authorization"] = "Bearer definitely-invalid-key"
    return headers


def _service_root(base_url: str) -> str:
    """The server under test as ``scheme://host[:port][/prefix]``; only http(s) qualifies.

    Rebuilt from the parsed parts, so every request URL is a validated origin plus one of the
    module's route constants rather than raw configuration text."""
    parts = urlsplit(str(base_url).strip())
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError(f"base_url must be an http(s) URL, got {base_url!r}")
    return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))


def _job_uuid(task_id: Any) -> str | None:
    """The server mints job ids with uuid4 (domain/base_request.py); anything else is not an
    id this suite places in a URL. Re-serialising through uuid.UUID keeps response text out."""
    try:
        return str(uuid.UUID(str(task_id)))
    except (ValueError, AttributeError, TypeError):
        return None


async def _cancel_created_job(
    *,
    base_url: str,
    api_key: str,
    task_id: str,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> dict[str, Any] | None:
    """Cancel a job the suite created through the shared client; None when it could not be."""
    job_id = _job_uuid(task_id)
    if job_id is None:
        return None
    try:
        async with MiniMaxH3Client(
            base_url=_service_root(base_url),
            api_key=api_key,
            request_timeout=request_timeout,
        ) as client:
            return await client.cancel_task(job_id)
    except (MiniMaxClientError, RuntimeError, ValueError, asyncio.TimeoutError):
        return None


async def _run_case(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    api_key: str,
    case: _RequestCase,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    endpoint_url = f"{base_url}{CREATE_PATHS[case.route_task]}"
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

            passed = response.status == case.expected_status
            message = ""
            task_id: str | None = None
            job_id: str | None = None
            cancellation: dict[str, Any] | None = None
            if case.requires_job_id or 200 <= response.status < 300:
                task_id = data.get("id") if isinstance(data, dict) else None
                job_id = _job_uuid(task_id) if isinstance(task_id, str) else None
                if job_id:
                    # Also a job a negative check was wrongly accepted with, so it cannot
                    # hold the queue for the tests that run after this suite.
                    cancellation = await _cancel_created_job(
                        base_url=base_url,
                        api_key=api_key,
                        task_id=job_id,
                        request_timeout=request_timeout,
                    )
            if case.requires_job_id:
                passed = passed and job_id is not None and cancellation is not None
                if not task_id:
                    message = "accepted response did not include a non-empty id"
                elif job_id is None:
                    message = "accepted response id is not a UUID"
                elif cancellation is None:
                    message = "accepted smoke job could not be cancelled"
            elif response.status >= 400:
                passed = passed and isinstance(data, dict) and "detail" in data
                if not isinstance(data, dict) or "detail" not in data:
                    message = "error response did not include FastAPI detail"
                elif case.expects_refusal and not isinstance(data["detail"], str):
                    passed = False
                    message = "422 was field validation, not a route refusal"
            elif job_id:
                message = "request was accepted; " + (
                    "its job was cancelled"
                    if cancellation is not None
                    else "its job could not be cancelled"
                )

            return {
                "check": case.name,
                "endpoint_url": endpoint_url,
                "passed": passed,
                "expected_status": case.expected_status,
                "actual_status": response.status,
                "task_id": task_id,
                "cancellation": cancellation,
                "message": message,
                "response": data,
            }
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
        return {
            "check": case.name,
            "endpoint_url": endpoint_url,
            "passed": False,
            "expected_status": case.expected_status,
            "actual_status": "request_error",
            "message": f"{type(exc).__name__}: {exc}",
        }


async def run_create_contract(
    *,
    base_url: str,
    api_key: str,
    profile: Profile = DEFAULT_PROFILE,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    task: str = DEFAULT_TASK,
) -> dict[str, Any]:
    normalized_profile = str(profile).lower()
    if normalized_profile not in _PROFILES:
        raise ValueError(f"profile must be one of {sorted(_PROFILES)}, got {profile!r}")
    if task not in REFUSED_TASKS:
        raise ValueError(f"task must be one of {list(H3_TASKS)}, got {task!r}")

    root = _service_root(base_url)
    endpoint_url = f"{root}{CREATE_PATHS[request_task_for(task)]}"
    timeout = aiohttp.ClientTimeout(total=request_timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = [
            await _run_case(
                session,
                base_url=root,
                api_key=api_key,
                case=case,
                request_timeout=request_timeout,
            )
            for case in _cases(normalized_profile, task)  # type: ignore[arg-type]
        ]

    passed = sum(bool(result["passed"]) for result in results)
    return {
        "endpoint_url": endpoint_url,
        "task_name": "minimax_h3_create_contract",
        "deployment_task": task,
        "profile": normalized_profile,
        "summary": f"{passed}/{len(results)} checks passed",
        "detailed_test_results": results,
        "success": passed == len(results),
    }


class MiniMaxH3CreateContractTest(BaseTest):
    """Workflow-compatible wrapper around the V1 contract suite."""

    KIND = "minimax_h3_create_contract"
    TASK_TYPE = "functional"
    HARDWARE_REQUIREMENT = HardwareRequirement.ANY_CHIP

    async def _run_specific_test_async(self) -> dict[str, Any]:
        return await run_create_contract(
            base_url=self.base_url,
            api_key=resolve_server_api_key(),
            profile=str(self.targets.get("profile", DEFAULT_PROFILE)),  # type: ignore[arg-type]
            request_timeout=float(
                self.targets.get(
                    "request_timeout",
                    DEFAULT_REQUEST_TIMEOUT_SECONDS,
                )
            ),
            task=resolve_h3_task(self.ctx),
        )


def run_minimax_h3_create_contract(
    ctx: MediaContext,
    targets: dict[str, Any] | None = None,
) -> Block:
    return MiniMaxH3CreateContractTest(
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
        description="Check the MiniMax-H3 V1 create contract of one deployment."
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--profile", choices=sorted(_PROFILES), default=DEFAULT_PROFILE)
    parser.add_argument(
        "--task",
        choices=H3_TASKS,
        help="task the deployment serves (default: from MODEL_RUNNER, else t2va)",
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
            run_create_contract(
                base_url=args.base_url,
                api_key=resolve_server_api_key(),
                profile=args.profile,
                request_timeout=args.request_timeout,
                task=args.task or resolve_h3_task(),
            )
        )
    except Exception as exc:  # noqa: BLE001 - CLI emits a structured failure
        logger.exception("MiniMax-H3 V1 create contract could not run")
        result = {
            "task_name": "minimax_h3_create_contract",
            "success": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("success") else 1


__all__ = [
    "MiniMaxH3CreateContractTest",
    "run_create_contract",
    "run_minimax_h3_create_contract",
]


if __name__ == "__main__":
    sys.exit(main())
