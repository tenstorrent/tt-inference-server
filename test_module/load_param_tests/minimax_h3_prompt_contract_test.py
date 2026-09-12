# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Prompt-field contract checks for MiniMax-H3 on the video V1 API.

Issue #5044 item 2. Every check here is decided by the response to a single
``POST /v1/videos/generations``: the prompt is the only instruction a
text-to-video deployment gets, so the server has to separate "no instruction
at all" (reject) from "a short instruction" (accept) at the request boundary,
before a mesh is reserved.

Nothing in this file runs a generation. Cases that expect acceptance still
create a real job in order to observe the status, and every accepted job --
including one the server should never have created -- is cancelled the moment
its id is read back.
"""

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
DEFAULT_DEPLOYMENT: Deployment = "t2va"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0
DEFAULT_TEST_TIMEOUT_SECONDS = 300
_PROFILES = frozenset({"validation", "smoke"})
_DEPLOYMENTS = frozenset({"t2va", "fl2va", "ref2va"})

# The status POST /v1/videos/generations returns for an accepted job, pinned by
# minimax_h3_create_contract_test.MiniMaxH3CreateContractTest.
ACCEPTED_STATUS = 202
REJECTED_STATUS = 422

# Four kinds of blank. "" and "   " are the obvious ones; "\n\t" catches a
# `prompt.strip(" ")` fix, and the unicode line catches a fix that only knows
# about ASCII space. Every character below is whitespace to str.isspace().
EMPTY_PROMPT = ""
SPACES_PROMPT = "   "
NEWLINE_TAB_PROMPT = "\n\t"
UNICODE_WHITESPACE_PROMPT = "  　 "

SINGLE_CHARACTER_PROMPT = "a"
INTERNAL_WHITESPACE_PROMPT = "a  fox   runs"
ORDINARY_PROMPT = (
    "A red fox steps through wet grass at dawn while birds sing in the "
    "background and the camera tracks alongside."
)

# ~150 short words lands near ~150-170 BPE tokens: ordinary long-form usage,
# and still inside the compiled warm bucket described in _LONG_PROMPT_WHY.
LONG_PROMPT_WORD_COUNT = 150
# No documented maximum exists yet, so only the absurd end is pinned here.
ABSURD_PROMPT_CHARACTERS = 100_000

_PROMPT_SENTENCE = (
    "a red fox steps through wet grass at dawn while birds sing and the "
    "camera tracks slowly alongside over a low stone wall toward a line of "
    "bare trees with soft morning light behind them"
)


class _Omitted:
    """Sentinel: the ``prompt`` key is absent from the payload entirely."""

    def __repr__(self) -> str:
        return "<prompt key omitted>"


OMITTED = _Omitted()


def _prompt_of_words(word_count: int) -> str:
    """Build a prompt of ``word_count`` ordinary English words."""

    vocabulary = _PROMPT_SENTENCE.split()
    return " ".join(vocabulary[index % len(vocabulary)] for index in range(word_count))


def _prompt_of_characters(character_count: int) -> str:
    """Build a prompt of exactly ``character_count`` characters."""

    filler = "a fox runs through the wet grass at dawn "
    repeats = character_count // len(filler) + 1
    return (filler * repeats)[:character_count]


_BLANK_PROMPT_WHY = (
    "The prompt is the only instruction the deployment has, so a blank one "
    "carries no work order -- but it still reserves a max_concurrency-1 mesh "
    "for a full generation (69.5 s at 16:9/5 s, 325 s at 15 s). It has to be "
    "refused at the request boundary, not accepted and rendered."
)
_LONG_PROMPT_WHY = (
    "A ~150-word prompt is ordinary long-form usage, not abuse. Prompt length "
    "is a warm-bucket key: at 16:9/5 s the 256-row alignment absorbs about "
    "178 tokens for t2va (about 202 with a keyframe), so this stays inside a "
    "compiled bucket and must be accepted."
)
_ABSURD_PROMPT_WHY = (
    "Past some maximum a prompt must be refused rather than silently landing "
    "in an uncompiled bucket and paying a compile on the request path. No "
    "maximum is documented yet, so this pins only the absurd end at "
    f"{ABSURD_PROMPT_CHARACTERS} characters; the exact bound is an open "
    "question in #5039, and this case's threshold moves once it is answered."
)


@dataclass(frozen=True)
class _PromptCase:
    """One prompt value posted to create, and the status the contract requires.

    ``prompt`` is either a string or :data:`OMITTED`, which drops the key from
    the payload. ``why`` is the case's rationale and is emitted with the
    result so a failure explains itself in the report. ``requires`` names the
    deployment on which the case can be evaluated at all; the runner skips
    every other one with a reason instead of scoring it.
    """

    name: str
    prompt: Any
    expected_status: int
    why: str
    requires: Deployment = DEFAULT_DEPLOYMENT


def _validation_cases() -> list[_PromptCase]:
    return [
        _PromptCase(
            "empty_string_prompt_is_rejected",
            EMPTY_PROMPT,
            REJECTED_STATUS,
            _BLANK_PROMPT_WHY,
        ),
        _PromptCase(
            "spaces_only_prompt_is_rejected",
            SPACES_PROMPT,
            REJECTED_STATUS,
            _BLANK_PROMPT_WHY,
        ),
        _PromptCase(
            "newline_and_tab_only_prompt_is_rejected",
            NEWLINE_TAB_PROMPT,
            REJECTED_STATUS,
            _BLANK_PROMPT_WHY,
        ),
        _PromptCase(
            "unicode_whitespace_only_prompt_is_rejected",
            UNICODE_WHITESPACE_PROMPT,
            REJECTED_STATUS,
            _BLANK_PROMPT_WHY,
        ),
        # Deliberate overlap: minimax_h3_create_contract_test._validation_cases
        # also posts a payload with no prompt key, as part of pinning the
        # *field set* of the create request. This copy pins the *prompt field's*
        # own contract -- an absent key is a validation error, not a
        # default-to-empty back door around the blank-prompt cases above. If
        # the duplication is ever collapsed, delete this one and keep the
        # create-contract case, never the other way round.
        _PromptCase(
            "missing_prompt_key_is_rejected",
            OMITTED,
            REJECTED_STATUS,
            "An absent prompt key must be a validation error, not a silent "
            "default to the empty string -- otherwise it becomes a way to ask "
            "for a blank-prompt generation that the blank-prompt cases above "
            "would have refused. Overlaps missing_prompt in "
            "minimax_h3_create_contract_test on purpose; see the comment "
            "above this case.",
        ),
        # The control that stops the blank-prompt fix from being implemented as
        # a minimum-length rule. It must keep passing.
        _PromptCase(
            "single_character_prompt_is_accepted",
            SINGLE_CHARACTER_PROMPT,
            ACCEPTED_STATUS,
            "The rule the fix must implement is 'not blank', not 'not short'. "
            "A one-character prompt is a real instruction and this case fails "
            "the moment a minimum-length check is introduced instead.",
        ),
        _PromptCase(
            "long_but_reasonable_prompt_is_accepted",
            _prompt_of_words(LONG_PROMPT_WORD_COUNT),
            ACCEPTED_STATUS,
            _LONG_PROMPT_WHY,
        ),
        _PromptCase(
            "absurdly_long_prompt_is_rejected",
            _prompt_of_characters(ABSURD_PROMPT_CHARACTERS),
            REJECTED_STATUS,
            _ABSURD_PROMPT_WHY,
        ),
        _PromptCase(
            "ordinary_prompt_is_accepted",
            ORDINARY_PROMPT,
            ACCEPTED_STATUS,
            "Control: the ordinary create path must keep returning "
            f"{ACCEPTED_STATUS} while blank prompts start being rejected.",
        ),
        _PromptCase(
            "internal_whitespace_in_prompt_is_accepted",
            INTERNAL_WHITESPACE_PROMPT,
            ACCEPTED_STATUS,
            "Control: only leading/trailing emptiness is the defect. Repeated "
            "spaces inside a prompt are legal text and must not be swept up "
            "by an over-eager normalising fix.",
        ),
    ]


def _cases(profile: Profile) -> list[_PromptCase]:
    """Return the cases for ``profile``.

    Both profiles return the same list. Every check in this file is decided by
    the create response alone, and the accepted jobs are cancelled on sight,
    so "smoke" has no generating case to add here. The parameter is kept so
    this runner's signature matches its siblings in this package.
    """

    del profile
    return _validation_cases()


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }


def _base_payload() -> dict[str, Any]:
    """Everything except the prompt.

    16:9 / 5 s is the compiled warm bucket the case rationales refer to;
    holding it fixed leaves the prompt as the only variable under test.
    """

    return {"aspect_ratio": "16:9", "duration_seconds": 5, "seed": 0}


def _payload_for(case: _PromptCase) -> dict[str, Any]:
    payload = _base_payload()
    if case.prompt is not OMITTED:
        payload["prompt"] = case.prompt
    return payload


def _response_excerpt(response_text: str) -> str:
    """Truncate a response body before it reaches the report.

    Pydantic echoes the offending input back in its validation detail, so the
    100k-character case would otherwise paste the whole prompt into the run
    report.
    """

    return response_text.replace("\n", " ")[:RESPONSE_EXCERPT_LENGTH]


def _prompt_excerpt(prompt: Any) -> str:
    if prompt is OMITTED:
        return repr(prompt)
    return repr(prompt[:64])


def _skipped_result(case: _PromptCase, *, deployment: str) -> dict[str, Any]:
    return {
        "check": case.name,
        "passed": False,
        "skipped": True,
        "expected_status": case.expected_status,
        "actual_status": "skipped",
        "message": (
            f"case needs a {case.requires} deployment; this run targets "
            f"{deployment}, where a text-only payload is rejected for a "
            "missing keyframe or reference image -- the expected status would "
            "be observed for the wrong reason"
        ),
        "why": case.why,
        "requires": case.requires,
    }


async def _cancel_created_job(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    api_key: str,
    task_id: str,
) -> dict[str, Any] | None:
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
    endpoint_url: str,
    base_url: str,
    api_key: str,
    case: _PromptCase,
    deployment: str,
) -> dict[str, Any]:
    if case.requires != deployment:
        return _skipped_result(case, deployment=deployment)

    try:
        async with session.post(
            endpoint_url,
            headers=_headers(api_key),
            json=_payload_for(case),
        ) as response:
            response_text = await response.text()
            try:
                data = json.loads(response_text) if response_text else None
            except json.JSONDecodeError:
                data = None

            task_id = data.get("id") if isinstance(data, dict) else None
            if not isinstance(task_id, str) or not task_id:
                task_id = None

            cancellation: dict[str, Any] | None = None
            if response.status == ACCEPTED_STATUS and task_id is not None:
                # No case in this file may run a generation. Cancel every job
                # the server accepted, including one a blank prompt should
                # never have created -- otherwise a failing case leaves a
                # 69.5 s render holding the mesh.
                cancellation = await _cancel_created_job(
                    session,
                    base_url=base_url,
                    api_key=api_key,
                    task_id=task_id,
                )

            passed = response.status == case.expected_status
            message = ""
            if case.expected_status == ACCEPTED_STATUS:
                if response.status == ACCEPTED_STATUS and task_id is None:
                    passed = False
                    message = "accepted response did not include a non-empty id"
                elif task_id is not None and cancellation is None:
                    # Reported, not scored. A cancel can legitimately 404 because the
                    # job already reached a terminal state on the device before the
                    # POST landed, and these four acceptance cases are the controls
                    # that tell a correct blank-prompt fix from an over-reaching one:
                    # failing them on cleanup would indict the server for the one
                    # behaviour they exist to confirm. Same policy as the admission
                    # suite, which keeps cleanup out of `success` explicitly.
                    message = (
                        f"cleanup: accepted job {task_id} could not be cancelled; "
                        "it may still be generating"
                    )
            elif response.status >= 400:
                if not isinstance(data, dict) or "detail" not in data:
                    passed = False
                    message = "error response did not include FastAPI detail"
            elif task_id is not None and cancellation is None:
                message = (
                    f"job {task_id} was created by a request that should have "
                    "been rejected, and could not be cancelled"
                )

            return {
                "check": case.name,
                "passed": passed,
                "skipped": False,
                "expected_status": case.expected_status,
                "actual_status": response.status,
                "prompt_characters": (
                    None if case.prompt is OMITTED else len(case.prompt)
                ),
                "prompt_excerpt": _prompt_excerpt(case.prompt),
                "task_id": task_id,
                "cancellation": cancellation,
                "message": message,
                "why": case.why,
                "requires": case.requires,
                "response_excerpt": _response_excerpt(response_text),
            }
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
        return {
            "check": case.name,
            "passed": False,
            "skipped": False,
            "expected_status": case.expected_status,
            "actual_status": "request_error",
            "prompt_characters": None if case.prompt is OMITTED else len(case.prompt),
            "prompt_excerpt": _prompt_excerpt(case.prompt),
            "message": f"{type(exc).__name__}: {exc}",
            "why": case.why,
            "requires": case.requires,
        }


async def run_prompt_contract(
    *,
    base_url: str,
    api_key: str,
    profile: Profile = DEFAULT_PROFILE,
    deployment: Deployment = DEFAULT_DEPLOYMENT,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    normalized_profile = str(profile).lower()
    if normalized_profile not in _PROFILES:
        raise ValueError(f"profile must be one of {sorted(_PROFILES)}, got {profile!r}")
    normalized_deployment = str(deployment).lower()
    if normalized_deployment not in _DEPLOYMENTS:
        raise ValueError(
            f"deployment must be one of {sorted(_DEPLOYMENTS)}, got {deployment!r}"
        )

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
                deployment=normalized_deployment,
            )
            for case in _cases(normalized_profile)  # type: ignore[arg-type]
        ]

    evaluated = [result for result in results if not result.get("skipped")]
    skipped = len(results) - len(evaluated)
    passed = sum(bool(result["passed"]) for result in evaluated)

    report: dict[str, Any] = {
        "endpoint_url": endpoint_url,
        "task_name": "minimax_h3_prompt_contract",
        "profile": normalized_profile,
        "deployment": normalized_deployment,
        "summary": f"{passed}/{len(evaluated)} checks passed, {skipped} skipped",
        "detailed_test_results": results,
        "success": bool(evaluated) and passed == len(evaluated),
    }
    if not evaluated:
        # Nothing was graded. Report SKIP so the suite says "not run here"
        # rather than banking a vacuous pass.
        report["status"] = "skip"
        report["reason"] = (
            f"every prompt case needs a t2va deployment; this run targets "
            f"{normalized_deployment}"
        )
    return report


class MiniMaxH3PromptContractTest(BaseTest):
    """Workflow-compatible wrapper around the prompt-field contract suite."""

    KIND = "minimax_h3_prompt_contract"
    TASK_TYPE = "functional"
    HARDWARE_REQUIREMENT = HardwareRequirement.ANY_CHIP

    async def _run_specific_test_async(self) -> dict[str, Any]:
        return await run_prompt_contract(
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


def run_minimax_h3_prompt_contract(
    ctx: MediaContext,
    targets: dict[str, Any] | None = None,
) -> Block:
    return MiniMaxH3PromptContractTest(
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
            "Check the MiniMax-H3 prompt field on "
            "POST /v1/videos/generations (issue #5044 item 2)."
        )
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--profile", choices=sorted(_PROFILES), default=DEFAULT_PROFILE)
    parser.add_argument(
        "--deployment",
        choices=sorted(_DEPLOYMENTS),
        default=DEFAULT_DEPLOYMENT,
        help=(
            "Deployment under test. Cases that cannot be evaluated on it are "
            "skipped with a reason instead of being scored."
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
            run_prompt_contract(
                base_url=args.base_url,
                api_key=resolve_server_api_key(),
                profile=args.profile,
                deployment=args.deployment,
                request_timeout=args.request_timeout,
            )
        )
    except Exception as exc:  # noqa: BLE001 - CLI emits a structured failure
        logger.exception("MiniMax-H3 V1 prompt contract could not run")
        result = {
            "task_name": "minimax_h3_prompt_contract",
            "success": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("success") else 1


__all__ = [
    "MiniMaxH3PromptContractTest",
    "run_minimax_h3_prompt_contract",
    "run_prompt_contract",
]


if __name__ == "__main__":
    sys.exit(main())
