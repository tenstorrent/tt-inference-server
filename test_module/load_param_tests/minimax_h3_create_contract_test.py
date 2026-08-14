# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Contract checks for MiniMax-H3 on the inference-server video V1 API."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import aiohttp  # pyright: ignore[reportMissingImports]

from test_module._test_common import BaseTest, HardwareRequirement, TestConfig
from test_module._test_common.minimax_h3_client import (
    CANCEL_PATH,
    CREATE_PATH,
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


@dataclass(frozen=True)
class _RequestCase:
    name: str
    payload: dict[str, Any]
    expected_status: int
    auth_mode: Literal["valid", "missing", "invalid"] = "valid"
    requires_job_id: bool = False


def _valid_payload() -> dict[str, Any]:
    return {
        "prompt": (
            "A red fox steps through wet grass at dawn while birds sing in "
            "the background and the camera tracks alongside."
        ),
        "aspect_ratio": "16:9",
        "duration_seconds": 5,
        "seed": 0,
    }


def _validation_cases() -> list[_RequestCase]:
    valid = _valid_payload()
    return [
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


def _cases(profile: Profile) -> list[_RequestCase]:
    cases = _validation_cases()
    if profile == "smoke":
        cases.append(
            _RequestCase(
                "valid_text_to_video_job",
                _valid_payload(),
                202,
                requires_job_id=True,
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


async def _cancel_created_job(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    api_key: str,
    task_id: str,
) -> dict[str, Any] | None:
    url = f"{base_url.rstrip('/')}{CANCEL_PATH.format(job_id=task_id)}"
    try:
        async with session.post(
            url,
            headers=_headers(api_key, "valid"),
        ) as response:
            if response.status != 200:
                return None
            data = await response.json()
            return data if isinstance(data, dict) else None
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
        return None


async def _run_case(
    session: aiohttp.ClientSession,
    *,
    endpoint_url: str,
    base_url: str,
    api_key: str,
    case: _RequestCase,
) -> dict[str, Any]:
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
            cancellation: dict[str, Any] | None = None
            if case.requires_job_id:
                task_id = data.get("id") if isinstance(data, dict) else None
                passed = passed and isinstance(task_id, str) and bool(task_id)
                if task_id:
                    cancellation = await _cancel_created_job(
                        session,
                        base_url=base_url,
                        api_key=api_key,
                        task_id=task_id,
                    )
                    passed = passed and cancellation is not None
                if not task_id:
                    message = "accepted response did not include a non-empty id"
                elif cancellation is None:
                    message = "accepted smoke job could not be cancelled"
            elif response.status >= 400:
                passed = passed and isinstance(data, dict) and "detail" in data
                if not isinstance(data, dict) or "detail" not in data:
                    message = "error response did not include FastAPI detail"

            return {
                "check": case.name,
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
) -> dict[str, Any]:
    normalized_profile = str(profile).lower()
    if normalized_profile not in _PROFILES:
        raise ValueError(f"profile must be one of {sorted(_PROFILES)}, got {profile!r}")

    endpoint_url = f"{base_url.rstrip('/')}{CREATE_PATH}"
    timeout = aiohttp.ClientTimeout(total=request_timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = [
            await _run_case(
                session,
                endpoint_url=endpoint_url,
                base_url=base_url,
                api_key=api_key,
                case=case,
            )
            for case in _cases(normalized_profile)  # type: ignore[arg-type]
        ]

    passed = sum(bool(result["passed"]) for result in results)
    return {
        "endpoint_url": endpoint_url,
        "task_name": "minimax_h3_create_contract",
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
        description="Check MiniMax-H3 on POST /v1/videos/generations."
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--profile", choices=sorted(_PROFILES), default=DEFAULT_PROFILE)
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
