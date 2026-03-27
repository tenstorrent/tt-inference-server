#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from string import hexdigits
from typing import Dict, List, Optional, Set, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

GITHUB_API = "https://api.github.com"
GITHUB_GRAPHQL_API = "https://api.github.com/graphql"
GITHUB_API_VERSION = "2022-11-28"
TT_SHIELD_OWNER = "tenstorrent"
TT_SHIELD_REPO = "tt-shield"
RELEASE_WORKFLOW_FILE = "release.yml"
TT_SHIELD_WORKFLOW_REF = "tstesco/disable-ai-summary-test"
TT_METAL_OWNER = "tenstorrent"
TT_METAL_REPO = "tt-metal"
VLLM_OWNER = "tenstorrent"
VLLM_REPO = "vllm"
WORKFLOW_RUN_LOOKUP_PAGE_SIZE = 10
WORKFLOW_RUN_LOOKUP_RETRIES = 5
WORKFLOW_RUN_LOOKUP_SLEEP_SECONDS = 2
WORKFLOW_RUN_CREATED_AT_SKEW_SECONDS = 1


@dataclass(frozen=True)
class WorkflowRunSummary:
    """Minimal workflow run fields used to correlate a dispatch to its run."""

    run_id: int
    html_url: Optional[str]
    created_at: Optional[datetime]
    status: Optional[str]


def normalize_dispatch_ref(base_ref: str) -> str:
    """Normalize a git ref into a workflow_dispatch-compatible branch or tag."""
    normalized_ref = base_ref.strip()
    if normalized_ref.startswith("refs/heads/"):
        normalized_ref = normalized_ref[len("refs/heads/") :]
    elif normalized_ref.startswith("refs/tags/"):
        normalized_ref = normalized_ref[len("refs/tags/") :]
    elif normalized_ref.startswith("origin/"):
        normalized_ref = normalized_ref[len("origin/") :]

    if not normalized_ref:
        raise ValueError("Dispatch ref cannot be empty.")

    is_hex_sha = all(char in "0123456789abcdefABCDEF" for char in normalized_ref)
    if is_hex_sha and 7 <= len(normalized_ref) <= 40:
        raise ValueError(
            "GitHub workflow_dispatch requires a branch or tag ref; "
            f"commit SHAs like {base_ref!r} are not supported."
        )

    return normalized_ref


def get_github_token() -> str:
    """Read the GitHub PAT used for workflow dispatch."""
    token = os.getenv("GH_PAT", "").strip()
    if not token:
        raise RuntimeError(
            "GH_PAT is required to dispatch the release workflow. "
            "Set it with: export GH_PAT='your_github_token_here'"
        )
    return token


def normalize_repo_ref(ref: str) -> str:
    """Normalize a repo ref so GitHub can resolve it to a commit."""
    normalized_ref = ref.strip()
    if normalized_ref.startswith("refs/heads/"):
        normalized_ref = normalized_ref[len("refs/heads/") :]
    elif normalized_ref.startswith("refs/tags/"):
        normalized_ref = normalized_ref[len("refs/tags/") :]
    elif normalized_ref.startswith("origin/"):
        normalized_ref = normalized_ref[len("origin/") :]

    if not normalized_ref:
        raise ValueError("Repository ref cannot be empty.")
    return normalized_ref


def _is_full_commit_sha(value: str) -> bool:
    """Return True when the value is a 40-character hexadecimal commit SHA."""
    return len(value) == 40 and all(char in hexdigits for char in value)


def _is_short_commit_sha(value: str) -> bool:
    """Return True when the value looks like an abbreviated hexadecimal SHA."""
    return 7 <= len(value) < 40 and all(char in hexdigits for char in value)


def _build_github_headers(token: str) -> Dict[str, str]:
    """Build standard GitHub REST API headers."""
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "pre-release/1.0",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
    }


def _github_api_request(
    url: str,
    token: str,
    method: str = "GET",
    payload: Optional[dict] = None,
) -> Tuple[int, bytes]:
    """Run one GitHub REST request and return status plus response bytes."""
    body = None
    headers = _build_github_headers(token)
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=60) as response:
            return response.getcode(), response.read()
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"GitHub API request failed ({error.code}) for {url}: {details}"
        ) from error
    except URLError as error:
        raise RuntimeError(f"GitHub API request failed for {url}: {error}") from error


def _parse_github_datetime(value: object) -> Optional[datetime]:
    """Parse a GitHub timestamp string into an aware UTC datetime."""
    if not isinstance(value, str) or not value:
        return None
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _list_recent_release_workflow_runs(
    dispatch_ref: str, token: str
) -> List[WorkflowRunSummary]:
    """List recent workflow_dispatch runs for release.yml on the dispatch ref."""
    encoded_workflow = quote(RELEASE_WORKFLOW_FILE, safe="")
    encoded_branch = quote(dispatch_ref, safe="")
    url = (
        f"{GITHUB_API}/repos/{TT_SHIELD_OWNER}/{TT_SHIELD_REPO}/actions/workflows/"
        f"{encoded_workflow}/runs?per_page={WORKFLOW_RUN_LOOKUP_PAGE_SIZE}"
        f"&event=workflow_dispatch&branch={encoded_branch}"
    )
    _, response_body = _github_api_request(url, token)
    data = json.loads(response_body.decode("utf-8")) if response_body else {}
    workflow_runs = data.get("workflow_runs", [])
    if not isinstance(workflow_runs, list):
        return []

    run_summaries = []
    for workflow_run in workflow_runs:
        if not isinstance(workflow_run, dict):
            continue
        run_id = workflow_run.get("id")
        if not isinstance(run_id, int):
            continue
        html_url = workflow_run.get("html_url")
        if not isinstance(html_url, str):
            html_url = None
        status = workflow_run.get("status")
        if not isinstance(status, str):
            status = None
        run_summaries.append(
            WorkflowRunSummary(
                run_id=run_id,
                html_url=html_url,
                created_at=_parse_github_datetime(workflow_run.get("created_at")),
                status=status,
            )
        )
    return run_summaries


def _select_new_release_workflow_run(
    recent_runs: List[WorkflowRunSummary],
    baseline_run_ids: Set[int],
    dispatch_started_at: datetime,
) -> Optional[WorkflowRunSummary]:
    """Return the first newly visible run created by this dispatch window."""
    created_at_threshold = dispatch_started_at - timedelta(
        seconds=WORKFLOW_RUN_CREATED_AT_SKEW_SECONDS
    )
    for workflow_run in recent_runs:
        if workflow_run.run_id in baseline_run_ids:
            continue
        if (
            workflow_run.created_at is not None
            and workflow_run.created_at < created_at_threshold
        ):
            continue
        return workflow_run
    return None


def _github_graphql_request(query: str, variables: Dict[str, str], token: str) -> dict:
    """Run one GitHub GraphQL request and return the decoded JSON object."""
    _, response_body = _github_api_request(
        GITHUB_GRAPHQL_API,
        token,
        method="POST",
        payload={"query": query, "variables": variables},
    )
    data = json.loads(response_body.decode("utf-8")) if response_body else {}
    if not isinstance(data, dict):
        raise RuntimeError("GitHub GraphQL response was not a JSON object.")
    errors = data.get("errors")
    if errors:
        raise RuntimeError(f"GitHub GraphQL request failed: {errors}")
    return data


def _resolve_commit_sha_from_graphql(
    owner: str, repo: str, ref: str, token: str
) -> str:
    """Resolve an abbreviated SHA via a repo-scoped GitHub GraphQL lookup."""
    query = """
    query ResolveCommitOid($owner: String!, $repo: String!, $expression: String!) {
      repository(owner: $owner, name: $repo) {
        object(expression: $expression) {
          __typename
          ... on Commit {
            oid
          }
        }
      }
    }
    """
    data = _github_graphql_request(
        query,
        {"owner": owner, "repo": repo, "expression": ref},
        token,
    )
    repository = data.get("data", {}).get("repository")
    if not isinstance(repository, dict):
        raise RuntimeError(
            f"GitHub GraphQL did not return repository data for {owner}/{repo}."
        )
    git_object = repository.get("object")
    if not isinstance(git_object, dict) or git_object.get("__typename") != "Commit":
        raise RuntimeError(
            "GitHub GraphQL did not resolve a commit object for "
            f"{owner}/{repo} ref {ref!r}."
        )
    resolved_sha = git_object.get("oid")
    if not isinstance(resolved_sha, str) or not _is_full_commit_sha(resolved_sha):
        raise RuntimeError(
            "GitHub GraphQL returned an invalid commit SHA for "
            f"{owner}/{repo} ref {ref!r}: {resolved_sha!r}"
        )
    return resolved_sha


def _resolve_commit_sha(owner: str, repo: str, ref: str, token: str) -> str:
    """Resolve a branch, tag, or abbreviated commit to a full commit SHA."""
    normalized_ref = normalize_repo_ref(ref)
    if _is_full_commit_sha(normalized_ref):
        return normalized_ref

    url = f"{GITHUB_API}/repos/{owner}/{repo}/commits/{quote(normalized_ref, safe='')}"
    try:
        _, response_body = _github_api_request(url, token)
    except RuntimeError:
        if _is_short_commit_sha(normalized_ref):
            return _resolve_commit_sha_from_graphql(owner, repo, normalized_ref, token)
        raise
    data = json.loads(response_body.decode("utf-8")) if response_body else {}
    resolved_sha = data.get("sha")
    if not isinstance(resolved_sha, str):
        raise RuntimeError(
            f"GitHub did not return a commit SHA for {owner}/{repo} ref {normalized_ref!r}."
        )
    if not _is_full_commit_sha(resolved_sha):
        raise RuntimeError(
            "GitHub returned an invalid commit SHA for "
            f"{owner}/{repo} ref {normalized_ref!r}: {resolved_sha!r}"
        )
    return resolved_sha


def dispatch_release_workflow(
    *,
    release_branch: str,
    tt_metal_ref: str,
    vllm_ref: str,
    run_ai_summary: Optional[bool] = None,
) -> Optional[str]:
    """Dispatch tt-shield release.yml and return a run URL when available."""
    dispatch_ref = TT_SHIELD_WORKFLOW_REF
    token = get_github_token()
    tt_metal_sha = _resolve_commit_sha(
        TT_METAL_OWNER, TT_METAL_REPO, tt_metal_ref, token
    )
    vllm_sha = _resolve_commit_sha(VLLM_OWNER, VLLM_REPO, vllm_ref, token)
    baseline_run_ids = {
        workflow_run.run_id
        for workflow_run in _list_recent_release_workflow_runs(dispatch_ref, token)
    }
    payload = {
        "ref": dispatch_ref,
        "inputs": {
            "tt-metal-commit": tt_metal_sha,
            "tt-inference-server-commit": release_branch,
            "vllm-commit": vllm_sha,
            "workflow": "release",
        },
    }
    if run_ai_summary is not None:
        payload["inputs"]["run-ai-summary"] = run_ai_summary
    dispatch_url = (
        f"{GITHUB_API}/repos/{TT_SHIELD_OWNER}/{TT_SHIELD_REPO}/actions/workflows/"
        f"{quote(RELEASE_WORKFLOW_FILE, safe='')}/dispatches"
    )
    dispatch_started_at = datetime.now(timezone.utc)
    status_code, response_body = _github_api_request(
        dispatch_url, token, method="POST", payload=payload
    )
    if response_body:
        response_data = json.loads(response_body.decode("utf-8"))
        run_url = response_data.get("html_url") or response_data.get("run_url")
        if run_url:
            return run_url
    if status_code not in (200, 201, 204):
        raise RuntimeError(
            f"Unexpected status code from release workflow dispatch: {status_code}"
        )

    print(
        "Release workflow dispatch accepted successfully; "
        "polling for the newly started workflow run ID..."
    )
    for _ in range(WORKFLOW_RUN_LOOKUP_RETRIES):
        recent_runs = _list_recent_release_workflow_runs(dispatch_ref, token)
        workflow_run = _select_new_release_workflow_run(
            recent_runs, baseline_run_ids, dispatch_started_at
        )
        if workflow_run and workflow_run.html_url:
            return workflow_run.html_url
        time.sleep(WORKFLOW_RUN_LOOKUP_SLEEP_SECONDS)
    return None
