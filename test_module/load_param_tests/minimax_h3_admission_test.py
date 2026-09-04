# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Admission-control contract for MiniMax-H3 on the inference-server video V1 API.

Issue #5044 item 1. The owner decision encoded here is:

    "MAX_QUEUE_SIZE, we need to default this value to whatever we think makes
    sense. The value we set however does inform the value we would return in
    the Retry-After."

Two obligations follow, and both are graded below. A deployment that cannot
start the work it is offered must (a) refuse the overflow as 503 Service
Unavailable carrying a Retry-After the client can act on, and (b) size the
admission queue for the mesh it actually runs on -- MiniMax-H3 renders one
video at a time on a single 4x8 Blackhole Galaxy, so the shared 5000-deep
media-server default is not a number this API may advertise.

The queue is saturated exactly once per run: :func:`_saturate` submits real
jobs until one is refused, and every case in :func:`_cases` grades that single
observation. Adding a check is one more row in the list, not another round of
traffic against the deployment.

Every job this test creates is cancelled in a ``finally`` block. This is the
only MiniMax-H3 test that deliberately fills a real deployment's queue, so it
is also the only one obliged to hand the deployment back empty.

Profiles: this test creates real jobs, so it runs only under "smoke". Under
"validation" it reports SKIP with a reason instead of pretending to pass.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal

import aiohttp  # pyright: ignore[reportMissingImports]

from test_module._test_common import BaseTest, HardwareRequirement, TestConfig
from test_module._test_common.minimax_h3_client import (
    CANCEL_PATH,
    CREATE_PATH,
    LIST_PATH,
    QUERY_PATH,
    TERMINAL_STATUSES,
    resolve_server_api_key,
)

if TYPE_CHECKING:
    from report_module.schema import Block
    from test_module.context import MediaContext

logger = logging.getLogger(__name__)

Profile = Literal["validation", "smoke"]
DEFAULT_PROFILE: Profile = "validation"
_PROFILES = frozenset({"validation", "smoke"})
TASK_NAME = "minimax_h3_admission"

DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0
DEFAULT_TEST_TIMEOUT_SECONDS = 900

# Upper bound on how many jobs this test will offer the deployment. It has to
# clear MAX_ADMITTED_JOBS with margin (otherwise a queue at the bound would
# look "never refused"), and it has to stay small so a server still carrying
# the shared 5000-deep default is probed, not flooded.
DEFAULT_MAX_SUBMISSIONS = 12

# The admission-capacity bound asserted by
# ``admission_capacity_is_small_enough_for_a_one_at_a_time_mesh``.
MAX_ADMITTED_JOBS = 8

# Retry-After proportionality band, expressed per queued job. See
# ``retry_after_scales_with_the_observed_queue_depth``.
MIN_SECONDS_PER_QUEUED_JOB = 30
MAX_SECONDS_PER_QUEUED_JOB = 900

HTTP_OK = 200
HTTP_ACCEPTED = 202
HTTP_TOO_MANY_REQUESTS = 429
HTTP_SERVICE_UNAVAILABLE = 503

# Observation preconditions. A case whose requirements the run did not meet is
# skipped with the reason below rather than graded on data this deployment
# never produced.
REQ_REFUSAL = "an_admission_refusal_was_observed"
REQ_ACCEPTED = "this_run_had_a_job_admitted"
REQ_REFUSAL_DEPTH = "the_queue_depth_at_the_refusal_was_countable"
REQ_BOUNDED_CAPACITY = "the_admission_capacity_was_bounded"

PROMPT = (
    "A bright red kite flies across a clear blue daytime sky above a sunlit "
    "green field, with smooth visible motion and wind in the soundtrack."
)
ASPECT_RATIO = "16:9"
DURATION_SECONDS = 5


def _create_payload() -> dict[str, Any]:
    return {
        "prompt": PROMPT,
        "aspect_ratio": ASPECT_RATIO,
        "duration_seconds": DURATION_SECONDS,
        "seed": 0,
    }


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }


def _parse_retry_after(raw: str | None) -> int | None:
    """Parse a Retry-After header as a positive whole number of seconds.

    RFC 9110 also permits an HTTP-date. This API is asked for delta-seconds,
    because a queue depth converts to a wait, not to a wall-clock instant, and
    because a date form cannot be compared against the observed depth. A
    date-valued header therefore parses to ``None`` and fails the case that
    pins the header format.
    """

    if raw is None:
        return None
    text = raw.strip()
    if not text.isdigit():
        return None
    seconds = int(text)
    return seconds if seconds > 0 else None


@dataclass(frozen=True)
class _Saturation:
    """Everything one saturation run observed through the public API."""

    submissions_attempted: int
    accepted_ids: tuple[str, ...]
    pre_flight_list_status: int | None
    pre_existing_active_jobs: int | None
    refusal_status: int | None
    refusal_retry_after: str | None
    refusal_body: Any
    query_status_while_full: int | None
    list_status_while_full: int | None
    submit_error: str | None

    @property
    def retry_after_seconds(self) -> int | None:
        return _parse_retry_after(self.refusal_retry_after)

    @property
    def outstanding_at_stop(self) -> int | None:
        """Outstanding jobs the deployment was holding when this run stopped.

        Counted as (jobs already outstanding at pre-flight) + (jobs this run
        got admitted), so a deployment that is already half busy is not
        credited with a smaller queue than it really has.
        """

        if self.pre_existing_active_jobs is None:
            return None
        return self.pre_existing_active_jobs + len(self.accepted_ids)

    @property
    def refusal_depth(self) -> int | None:
        """Outstanding jobs held at the moment the deployment refused.

        A refusal at depth ``d`` says the admission capacity is at most ``d``
        -- which is what both the capacity bound and the Retry-After
        proportionality rule are graded against. ``None`` when no refusal was
        seen, or when the pre-flight count that anchors the depth is missing.
        """

        if self.refusal_status is None:
            return None
        return self.outstanding_at_stop

    def to_dict(self) -> dict[str, Any]:
        return {
            "submissions_attempted": self.submissions_attempted,
            "accepted_job_count": len(self.accepted_ids),
            "accepted_job_ids": list(self.accepted_ids),
            "pre_flight_list_status": self.pre_flight_list_status,
            "pre_existing_active_jobs": self.pre_existing_active_jobs,
            "outstanding_at_stop": self.outstanding_at_stop,
            "refusal_depth": self.refusal_depth,
            "refusal_status": self.refusal_status,
            "refusal_retry_after": self.refusal_retry_after,
            "retry_after_seconds": self.retry_after_seconds,
            "refusal_body": self.refusal_body,
            "query_status_while_full": self.query_status_while_full,
            "list_status_while_full": self.list_status_while_full,
            "submit_error": self.submit_error,
        }


@dataclass(frozen=True)
class _AdmissionCase:
    """One contract statement graded against a single saturation run."""

    name: str
    doc: str
    requires: tuple[str, ...]
    expected: Any
    check: Callable[[_Saturation], tuple[bool, Any, str]]
    expected_status: int | None = None


def _check_refusal_is_service_unavailable(
    saturation: _Saturation,
) -> tuple[bool, Any, str]:
    actual = saturation.refusal_status
    if actual == HTTP_SERVICE_UNAVAILABLE:
        return True, actual, ""
    if actual == HTTP_TOO_MANY_REQUESTS:
        return (
            False,
            actual,
            (
                "admission refusal returned 429 Too Many Requests; a full "
                "admission queue is a server capacity condition (503), not a "
                "per-client rate-limit verdict"
            ),
        )
    return (
        False,
        actual,
        (
            f"admission refusal returned HTTP {actual}, expected "
            f"{HTTP_SERVICE_UNAVAILABLE}"
        ),
    )


def _check_retry_after_is_a_positive_integer(
    saturation: _Saturation,
) -> tuple[bool, Any, str]:
    raw = saturation.refusal_retry_after
    if raw is None:
        return False, None, "admission refusal carried no Retry-After header"
    seconds = saturation.retry_after_seconds
    if seconds is None:
        return (
            False,
            raw,
            f"Retry-After {raw!r} is not a positive integer number of seconds",
        )
    return True, raw, ""


def _check_retry_after_tracks_queue_depth(
    saturation: _Saturation,
) -> tuple[bool, Any, str]:
    seconds = saturation.retry_after_seconds
    depth = saturation.refusal_depth
    actual: dict[str, Any] = {
        "retry_after_seconds": seconds,
        "refusal_queue_depth": depth,
    }
    if seconds is None:
        return (
            False,
            actual,
            "no positive-integer Retry-After to grade against the queue depth",
        )
    if not depth:
        return False, actual, "observed queue depth is not usable as a divisor"

    per_job = seconds / depth
    actual["seconds_per_queued_job"] = per_job
    if seconds % depth != 0:
        return (
            False,
            actual,
            (
                f"Retry-After {seconds}s is not a whole multiple of the observed "
                f"queue depth {depth}, so it is not derived from MAX_QUEUE_SIZE"
            ),
        )
    if not MIN_SECONDS_PER_QUEUED_JOB <= per_job <= MAX_SECONDS_PER_QUEUED_JOB:
        return (
            False,
            actual,
            (
                f"Retry-After {seconds}s over depth {depth} implies "
                f"{per_job:.1f}s per queued job, outside the plausible band "
                f"[{MIN_SECONDS_PER_QUEUED_JOB}, {MAX_SECONDS_PER_QUEUED_JOB}]"
            ),
        )
    return True, actual, ""


def _check_reads_survive_a_full_queue(
    saturation: _Saturation,
) -> tuple[bool, Any, str]:
    actual = {
        "query_status": saturation.query_status_while_full,
        "list_status": saturation.list_status_while_full,
    }
    failures = [
        f"{name} returned HTTP {status}"
        for name, status in (
            (QUERY_PATH, saturation.query_status_while_full),
            (LIST_PATH, saturation.list_status_while_full),
        )
        if status != HTTP_OK
    ]
    if failures:
        return (
            False,
            actual,
            "admission refusal made accepted work unreadable: " + "; ".join(failures),
        )
    return True, actual, ""


def _check_admission_capacity_is_small(
    saturation: _Saturation,
) -> tuple[bool, Any, str]:
    refused_at = saturation.refusal_depth
    depth = saturation.outstanding_at_stop
    actual = {
        "refusal_depth": refused_at,
        "outstanding_at_stop": depth,
        "pre_existing_active_jobs": saturation.pre_existing_active_jobs,
        "admitted_by_this_run": len(saturation.accepted_ids),
    }
    if refused_at is not None:
        if refused_at > MAX_ADMITTED_JOBS:
            return (
                False,
                actual,
                (
                    f"deployment was holding {refused_at} outstanding jobs "
                    f"when it refused, so it admits more than "
                    f"{MAX_ADMITTED_JOBS}; a one-request-at-a-time mesh must "
                    "not advertise that much queue"
                ),
            )
        return True, actual, ""
    if depth is not None and depth > MAX_ADMITTED_JOBS:
        return (
            False,
            actual,
            (
                f"deployment took at least {depth} outstanding jobs and had "
                f"still not refused; a one-request-at-a-time mesh must admit "
                f"at most {MAX_ADMITTED_JOBS}"
            ),
        )
    return False, actual, "admission capacity was never bounded by this run"


def _cases(profile: Profile) -> list[_AdmissionCase]:
    """Admission cases for ``profile``; only "smoke" may create real jobs."""

    if profile != "smoke":
        return []
    return [
        _AdmissionCase(
            name="admission_refusal_is_503_service_unavailable_not_429",
            doc=(
                "Submitting until the deployment refuses must yield 503 "
                "Service Unavailable. A full admission queue means this "
                "server cannot start more work right now -- a capacity "
                "condition every client shares -- not that this caller "
                "exceeded a rate limit, which is what 429 tells the caller "
                "and its retry middleware. Today's server raises "
                "HTTP_429_TOO_MANY_REQUESTS from the job manager's "
                "admission check, so this case fails until #5044 lands."
            ),
            requires=(REQ_REFUSAL,),
            expected=HTTP_SERVICE_UNAVAILABLE,
            expected_status=HTTP_SERVICE_UNAVAILABLE,
            check=_check_refusal_is_service_unavailable,
        ),
        _AdmissionCase(
            name="admission_refusal_carries_a_positive_integer_retry_after",
            doc=(
                "The refusal must carry a Retry-After header that parses as "
                "a positive whole number of seconds. Delta-seconds, not an "
                "HTTP-date: the server is quoting how long its queue needs "
                "to drain, and a client that cannot read a number back has "
                "no basis for a retry other than a busy-loop."
            ),
            requires=(REQ_REFUSAL,),
            expected="a Retry-After header holding an integer > 0",
            expected_status=HTTP_SERVICE_UNAVAILABLE,
            check=_check_retry_after_is_a_positive_integer,
        ),
        _AdmissionCase(
            name="retry_after_scales_with_the_observed_queue_depth",
            doc=(
                "Retry-After must be derived from the queue size, not from a "
                "constant baked into the handler -- that is the whole of the "
                "owner decision that the MAX_QUEUE_SIZE we pick 'informs the "
                "value we would return in the Retry-After'. "
                "Rule, with depth = the outstanding-job count the deployment "
                "was holding when it refused (jobs already active at "
                "pre-flight plus the jobs this run got admitted): Retry-After "
                "must be an exact whole multiple of depth, and "
                "Retry-After / depth must land "
                f"in [{MIN_SECONDS_PER_QUEUED_JOB}, "
                f"{MAX_SECONDS_PER_QUEUED_JOB}] seconds per queued job -- "
                "the band a single MiniMax-H3 clip on one 4x8 Blackhole "
                "Galaxy mesh actually occupies. An implementation that "
                "computes depth * per_job_estimate satisfies both by "
                "construction; a hardcoded constant satisfies them only for "
                "whatever depth it was written against, and breaks as soon "
                "as MAX_QUEUE_SIZE is retuned. No value is hardcoded here. "
                "One refusal is one data point, so this cannot by itself "
                "prove non-constancy -- it pins the relationship the header "
                "must hold to, and the deployment's own depth supplies the "
                "number."
            ),
            requires=(REQ_REFUSAL, REQ_REFUSAL_DEPTH),
            expected=(
                f"retry_after % depth == 0 and {MIN_SECONDS_PER_QUEUED_JOB} "
                f"<= retry_after / depth <= {MAX_SECONDS_PER_QUEUED_JOB}"
            ),
            check=_check_retry_after_tracks_queue_depth,
        ),
        _AdmissionCase(
            name="a_full_queue_still_serves_reads_of_accepted_jobs",
            doc=(
                "Refusing new work must not cost a client the work already "
                "accepted. With the queue saturated, GET "
                f"{QUERY_PATH} for a job this run created and GET "
                f"{LIST_PATH} must both answer 200. A deployment that sheds "
                "reads under admission pressure strands every job it already "
                "took, and the caller cannot even learn whether to wait or "
                "resubmit."
            ),
            requires=(REQ_REFUSAL, REQ_ACCEPTED),
            expected=HTTP_OK,
            expected_status=HTTP_OK,
            check=_check_reads_survive_a_full_queue,
        ),
        _AdmissionCase(
            name="admission_capacity_is_small_enough_for_a_one_at_a_time_mesh",
            doc=(
                "The deployment must hold at most "
                f"{MAX_ADMITTED_JOBS} outstanding jobs before refusing -- a "
                "refusal at depth d proves the admission capacity is at most "
                "d, whether those jobs came from this run or were already "
                "in flight. "
                "MiniMax-H3 runs at max_concurrency 1 on a single 4x8 "
                "Blackhole Galaxy and a 5-second clip takes minutes, so a "
                "queue deeper than a handful of jobs advertises capacity the "
                "mesh does not have: the tail requests wait hours, time out "
                "client-side, and there is no honest Retry-After to quote "
                "for them. The shared media-server default (max_queue_size "
                "= 5000) is a text-shaped number; the video model spec "
                "already narrows it per deployment. This case pins the "
                "upper bound the API may advertise, not the exact value -- "
                "the owner decision is that we choose a sensible default, "
                "and any choice above this bound is not sensible here. "
                "It deliberately does not require a refusal: a deployment "
                "that swallowed every submission this run offered without "
                "refusing has already exceeded the bound, and that is the "
                "verdict this test reports rather than skipping."
            ),
            requires=(REQ_BOUNDED_CAPACITY,),
            expected=f"at most {MAX_ADMITTED_JOBS} outstanding jobs admitted",
            check=_check_admission_capacity_is_small,
        ),
    ]


def _unmet_requirements(saturation: _Saturation) -> dict[str, str]:
    """Map each unmet observation precondition to a human-readable reason."""

    unmet: dict[str, str] = {}
    if saturation.refusal_status is None:
        detail = saturation.submit_error or (
            f"all {len(saturation.accepted_ids)} submission(s) were accepted, "
            f"so this deployment's admission queue is deeper than "
            f"{saturation.submissions_attempted} jobs and the refusal contract "
            "cannot be exercised through the API; the capacity case grades "
            "that queue depth and fails on it"
        )
        unmet[REQ_REFUSAL] = (
            f"no admission refusal was observed after "
            f"{saturation.submissions_attempted} submission(s): {detail}"
        )
    if not saturation.accepted_ids:
        unmet[REQ_ACCEPTED] = (
            "this run had no job admitted, so it neither measured the "
            "deployment's own admission capacity nor owns a job to read back; "
            "the queue was already full when the run started"
        )

    outstanding = saturation.outstanding_at_stop
    if outstanding is None:
        uncountable = (
            f"the pre-flight GET {LIST_PATH} returned "
            f"{saturation.pre_flight_list_status}, so the jobs already "
            "outstanding could not be counted and any queue depth read off "
            "this run would be an undercount"
        )
        unmet[REQ_REFUSAL_DEPTH] = uncountable
        unmet[REQ_BOUNDED_CAPACITY] = uncountable
        return unmet

    depth = saturation.refusal_depth
    if depth is None:
        unmet[REQ_REFUSAL_DEPTH] = (
            f"the run stopped at its own submission cap with {outstanding} "
            "job(s) outstanding and no refusal, so there is no refusal depth "
            "to grade the Retry-After against"
        )
        if outstanding <= MAX_ADMITTED_JOBS:
            unmet[REQ_BOUNDED_CAPACITY] = (
                f"the run stopped early with only {outstanding} job(s) "
                "outstanding and no refusal, which neither proves nor "
                f"disproves a capacity of at most {MAX_ADMITTED_JOBS}"
            )
    elif depth < 1:
        unmet[REQ_REFUSAL_DEPTH] = (
            "the deployment refused with nothing outstanding, so there is no "
            "queue depth for the Retry-After to be proportional to"
        )
    return unmet


async def _get_status(
    session: aiohttp.ClientSession,
    *,
    url: str,
    api_key: str,
) -> tuple[int | None, Any]:
    """GET ``url`` and return (status, decoded body); (None, error) on failure."""

    try:
        async with session.get(url, headers=_headers(api_key)) as response:
            text = await response.text()
            try:
                body = json.loads(text) if text else None
            except json.JSONDecodeError:
                body = None
            return response.status, body
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _count_active_jobs(body: Any) -> int | None:
    if not isinstance(body, list):
        return None
    return sum(
        1
        for job in body
        if isinstance(job, dict) and job.get("status") not in TERMINAL_STATUSES
    )


async def _saturate(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    api_key: str,
    max_submissions: int,
    created: list[str],
) -> _Saturation:
    """Submit real jobs until one is refused; observe the refusal and reads.

    ``created`` is appended to as jobs are accepted so the caller's ``finally``
    can cancel them even if this coroutine is interrupted part way through.
    """

    root = base_url.rstrip("/")
    create_url = f"{root}{CREATE_PATH}"
    list_url = f"{root}{LIST_PATH}"

    pre_flight_status, pre_flight_body = await _get_status(
        session, url=list_url, api_key=api_key
    )
    pre_existing = (
        _count_active_jobs(pre_flight_body) if pre_flight_status == HTTP_OK else None
    )

    attempted = 0
    refusal_status: int | None = None
    refusal_retry_after: str | None = None
    refusal_body: Any = None
    submit_error: str | None = None

    for _ in range(max_submissions):
        attempted += 1
        try:
            async with session.post(
                create_url,
                headers=_headers(api_key),
                json=_create_payload(),
            ) as response:
                text = await response.text()
                try:
                    body = json.loads(text) if text else None
                except json.JSONDecodeError:
                    body = None

                if response.status == HTTP_ACCEPTED:
                    job_id = body.get("id") if isinstance(body, dict) else None
                    if isinstance(job_id, str) and job_id:
                        created.append(job_id)
                        continue
                    submit_error = "accepted submission returned no job id"
                    break

                refusal_status = response.status
                refusal_retry_after = response.headers.get("Retry-After")
                refusal_body = body
                break
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            submit_error = f"{type(exc).__name__}: {exc}"
            break

    query_status: int | None = None
    list_status: int | None = None
    if refusal_status is not None and created:
        query_url = f"{root}{QUERY_PATH.format(job_id=created[0])}"
        query_status, _ = await _get_status(session, url=query_url, api_key=api_key)
        list_status, _ = await _get_status(session, url=list_url, api_key=api_key)

    return _Saturation(
        submissions_attempted=attempted,
        accepted_ids=tuple(created),
        pre_flight_list_status=pre_flight_status,
        pre_existing_active_jobs=pre_existing,
        refusal_status=refusal_status,
        refusal_retry_after=refusal_retry_after,
        refusal_body=refusal_body,
        query_status_while_full=query_status,
        list_status_while_full=list_status,
        submit_error=submit_error,
    )


async def _cancel_all(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    api_key: str,
    job_ids: list[str],
) -> dict[str, Any]:
    """Cancel every job this test created. Never raises."""

    root = base_url.rstrip("/")
    outcomes: dict[str, Any] = {}
    for job_id in job_ids:
        url = f"{root}{CANCEL_PATH.format(job_id=job_id)}"
        try:
            async with session.post(url, headers=_headers(api_key)) as response:
                outcomes[job_id] = response.status
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            outcomes[job_id] = f"{type(exc).__name__}: {exc}"
    cancelled = sum(1 for status in outcomes.values() if status == HTTP_OK)
    return {
        "attempted": len(job_ids),
        "cancelled": cancelled,
        "complete": cancelled == len(job_ids),
        "outcomes": outcomes,
    }


async def _run_case(
    case: _AdmissionCase,
    *,
    saturation: _Saturation,
    unmet: dict[str, str],
) -> dict[str, Any]:
    """Grade one case, or skip it with the reason the run could not grade it."""

    blocking = [unmet[key] for key in case.requires if key in unmet]
    if blocking:
        passed: bool | None = None
        actual: Any = None
        message = "; ".join(blocking)
    else:
        passed, actual, message = case.check(saturation)

    return {
        "check": case.name,
        "passed": passed,
        "skipped": bool(blocking),
        "contract": case.doc,
        "requires": list(case.requires),
        "expected": case.expected,
        "expected_status": case.expected_status,
        "actual": actual,
        "actual_status": saturation.refusal_status,
        "message": message,
    }


async def run_admission(
    *,
    base_url: str,
    api_key: str,
    profile: Profile = DEFAULT_PROFILE,
    max_submissions: int = DEFAULT_MAX_SUBMISSIONS,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Saturate the admission queue once and grade the refusal it produced."""

    normalized_profile = str(profile).lower()
    if normalized_profile not in _PROFILES:
        raise ValueError(f"profile must be one of {sorted(_PROFILES)}, got {profile!r}")
    if max_submissions < 1:
        raise ValueError(f"max_submissions must be >= 1, got {max_submissions!r}")

    endpoint_url = f"{base_url.rstrip('/')}{CREATE_PATH}"
    if normalized_profile != "smoke":
        reason = (
            "admission checks saturate a real deployment's queue with real "
            "jobs; they run only under the smoke profile"
        )
        return {
            "endpoint_url": endpoint_url,
            "task_name": TASK_NAME,
            "profile": normalized_profile,
            "summary": f"skipped: {reason}",
            "detailed_test_results": [],
            "success": False,
            "status": "skip",
            "skipped": True,
            "reason": reason,
        }

    created: list[str] = []
    timeout = aiohttp.ClientTimeout(total=request_timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            saturation = await _saturate(
                session,
                base_url=base_url,
                api_key=api_key,
                max_submissions=max_submissions,
                created=created,
            )
            unmet = _unmet_requirements(saturation)
            results = [
                await _run_case(case, saturation=saturation, unmet=unmet)
                for case in _cases(normalized_profile)  # type: ignore[arg-type]
            ]
        finally:
            cleanup = await _cancel_all(
                session,
                base_url=base_url,
                api_key=api_key,
                job_ids=created,
            )

    if not cleanup["complete"]:
        logger.warning(
            "MiniMax-H3 admission test left %d job(s) uncancelled: %s",
            cleanup["attempted"] - cleanup["cancelled"],
            cleanup["outcomes"],
        )

    graded = [result for result in results if not result["skipped"]]
    passed = sum(1 for result in graded if result["passed"])
    skipped = len(results) - len(graded)
    summary = (
        f"{passed}/{len(graded)} graded checks passed, {skipped} skipped; "
        f"cleanup cancelled {cleanup['cancelled']}/{cleanup['attempted']} job(s)"
    )

    # Cleanup completeness is reported and logged but deliberately kept out of
    # `success`: a flaky cancel must not be mistaken for an admission-contract
    # verdict.
    result: dict[str, Any] = {
        "endpoint_url": endpoint_url,
        "task_name": TASK_NAME,
        "profile": normalized_profile,
        "summary": summary,
        "detailed_test_results": results,
        "observation": saturation.to_dict(),
        "cleanup": cleanup,
        "success": bool(graded) and passed == len(graded),
    }
    if not graded:
        reason = "; ".join(sorted(set(unmet.values()))) or "no cases were gradable"
        result["reason"] = reason
        if saturation.submit_error is not None:
            # Nothing was graded because submission itself broke -- either the
            # deployment never answered, or it returned 202 without a job id. Neither
            # is a missing observation, and `status: "skip"` is non-blocking, so
            # leaving it grey means a dead or still-warming pod reports the same as a
            # queue that simply never filled. By then the bring-up is already spent.
            result["success"] = False
            result["summary"] = f"failed: {reason}"
        else:
            result["status"] = "skip"
            result["skipped"] = True
            result["summary"] = f"skipped: {reason}"
    return result


class MiniMaxH3AdmissionTest(BaseTest):
    """Workflow-compatible wrapper around the admission-control suite."""

    KIND = "minimax_h3_admission"
    TASK_TYPE = "functional"
    HARDWARE_REQUIREMENT = HardwareRequirement.ANY_CHIP

    async def _run_specific_test_async(self) -> dict[str, Any]:
        return await run_admission(
            base_url=self.base_url,
            api_key=resolve_server_api_key(),
            profile=str(self.targets.get("profile", DEFAULT_PROFILE)),  # type: ignore[arg-type]
            max_submissions=int(
                self.targets.get("max_submissions", DEFAULT_MAX_SUBMISSIONS)
            ),
            request_timeout=float(
                self.targets.get(
                    "request_timeout",
                    DEFAULT_REQUEST_TIMEOUT_SECONDS,
                )
            ),
        )


def run_minimax_h3_admission(
    ctx: MediaContext,
    targets: dict[str, Any] | None = None,
) -> Block:
    return MiniMaxH3AdmissionTest(
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
            "Saturate the MiniMax-H3 admission queue on "
            "POST /v1/videos/generations and check the refusal contract."
        )
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--profile", choices=sorted(_PROFILES), default=DEFAULT_PROFILE)
    parser.add_argument(
        "--max-submissions",
        type=int,
        default=DEFAULT_MAX_SUBMISSIONS,
        help="upper bound on jobs offered to the deployment before giving up",
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
            run_admission(
                base_url=args.base_url,
                api_key=resolve_server_api_key(),
                profile=args.profile,
                max_submissions=args.max_submissions,
                request_timeout=args.request_timeout,
            )
        )
    except Exception as exc:  # noqa: BLE001 - CLI emits a structured failure
        logger.exception("MiniMax-H3 admission-control checks could not run")
        result = {
            "task_name": TASK_NAME,
            "success": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }

    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("skipped"):
        return 0
    return 0 if result.get("success") else 1


__all__ = [
    "MiniMaxH3AdmissionTest",
    "run_admission",
    "run_minimax_h3_admission",
]


if __name__ == "__main__":
    sys.exit(main())
