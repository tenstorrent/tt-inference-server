# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Alternate MiniMax-H3 V1 lifecycle that ends in cancellation."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import TYPE_CHECKING, Any

from test_module._test_common import BaseTest, HardwareRequirement, TestConfig
from test_module._test_common.minimax_h3_client import (
    MiniMaxClientError,
    MiniMaxH3Client,
    resolve_server_api_key,
)

if TYPE_CHECKING:
    from report_module.schema import Block
    from test_module.context import MediaContext

logger = logging.getLogger(__name__)

PROMPT = (
    "A bright red kite flies across a clear blue daytime sky above a sunlit "
    "green field, with smooth visible motion and wind in the soundtrack."
)
ASPECT_RATIO = "16:9"
DURATION_SECONDS = 5
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0
DEFAULT_POLL_INTERVAL_SECONDS = 1.0
DEFAULT_POLL_TIMEOUT_SECONDS = 300.0
DEFAULT_TEST_TIMEOUT_SECONDS = 600


def _create_payload() -> dict[str, Any]:
    return {
        "prompt": PROMPT,
        "aspect_ratio": ASPECT_RATIO,
        "duration_seconds": DURATION_SECONDS,
        "seed": 0,
    }


def _validate_cancelled_task(task: dict[str, Any], *, task_id: str) -> None:
    if task.get("id") != task_id:
        raise MiniMaxClientError(
            f"cancelled query returned id={task.get('id')!r}, expected {task_id!r}",
            task_id=task_id,
        )
    if task.get("status") != "cancelled":
        raise MiniMaxClientError(
            f"expected cancelled terminal status, got {task.get('status')!r}",
            task_id=task_id,
            response_body=json.dumps(task.get("error")),
        )
    if task.get("job_type") != "video":
        raise MiniMaxClientError(
            f"cancelled task has unexpected job_type={task.get('job_type')!r}",
            task_id=task_id,
        )


def _find_listed_task(
    tasks: list[dict[str, Any]],
    *,
    task_id: str,
) -> dict[str, Any]:
    matches = [task for task in tasks if task.get("id") == task_id]
    if len(matches) != 1:
        raise MiniMaxClientError(
            f"expected cancelled task {task_id!r} exactly once in job list; "
            f"found {len(matches)} matches",
            task_id=task_id,
        )
    return matches[0]


async def run_cancel_lifecycle(
    *,
    base_url: str,
    api_key: str,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
    poll_timeout: float = DEFAULT_POLL_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Create a T2V job, cancel it immediately, then verify query/list state."""

    async with MiniMaxH3Client(
        base_url=base_url,
        api_key=api_key,
        request_timeout=request_timeout,
        poll_interval=poll_interval,
        poll_timeout=poll_timeout,
    ) as client:
        task_id = await client.create_video(_create_payload())
        cancellation = await client.cancel_task(task_id)
        terminal = await client.wait_for_terminal(task_id)
        _validate_cancelled_task(terminal.task, task_id=task_id)

        listed_jobs = await client.list_tasks()
        listed_task = _find_listed_task(listed_jobs, task_id=task_id)
        _validate_cancelled_task(listed_task, task_id=task_id)

    return {
        "task_name": "minimax_h3_cancel_lifecycle",
        "base_url": base_url.rstrip("/"),
        "task_id": task_id,
        "cancellation_response_status": cancellation["status"],
        "observed_statuses": list(terminal.observed_statuses),
        "query_status": terminal.task["status"],
        "listed_status": listed_task["status"],
        "success": True,
    }


class MiniMaxH3CancelLifecycleTest(BaseTest):
    KIND = "minimax_h3_cancel_lifecycle"
    TASK_TYPE = "functional"
    HARDWARE_REQUIREMENT = HardwareRequirement.ANY_CHIP

    async def _run_specific_test_async(self) -> dict[str, Any]:
        return await run_cancel_lifecycle(
            base_url=self.base_url,
            api_key=resolve_server_api_key(),
            request_timeout=float(
                self.targets.get(
                    "request_timeout",
                    DEFAULT_REQUEST_TIMEOUT_SECONDS,
                )
            ),
            poll_interval=float(
                self.targets.get(
                    "poll_interval",
                    DEFAULT_POLL_INTERVAL_SECONDS,
                )
            ),
            poll_timeout=float(
                self.targets.get(
                    "poll_timeout",
                    DEFAULT_POLL_TIMEOUT_SECONDS,
                )
            ),
        )


def run_minimax_h3_cancel_lifecycle(
    ctx: MediaContext,
    targets: dict[str, Any] | None = None,
) -> Block:
    return MiniMaxH3CancelLifecycleTest(
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
            "Create and cancel a MiniMax-H3 job through the V1 video API, "
            "then verify query and list state."
        )
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL_SECONDS,
    )
    parser.add_argument(
        "--poll-timeout",
        type=float,
        default=DEFAULT_POLL_TIMEOUT_SECONDS,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = asyncio.run(
            run_cancel_lifecycle(
                base_url=args.base_url,
                api_key=resolve_server_api_key(),
                request_timeout=args.request_timeout,
                poll_interval=args.poll_interval,
                poll_timeout=args.poll_timeout,
            )
        )
    except Exception as exc:  # noqa: BLE001 - CLI emits a structured failure
        logger.exception("MiniMax-H3 cancellation lifecycle could not run")
        result = {
            "task_name": "minimax_h3_cancel_lifecycle",
            "success": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("success") else 1


__all__ = [
    "MiniMaxH3CancelLifecycleTest",
    "run_cancel_lifecycle",
    "run_minimax_h3_cancel_lifecycle",
]


if __name__ == "__main__":
    sys.exit(main())
