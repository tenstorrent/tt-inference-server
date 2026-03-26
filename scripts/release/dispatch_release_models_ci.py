#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

GITHUB_API = "https://api.github.com"
GITHUB_API_VERSION = "2022-11-28"
TT_SHIELD_OWNER = "tenstorrent"
TT_SHIELD_REPO = "tt-shield"
RELEASE_WORKFLOW_FILE = "release.yml"


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


def _iter_release_model_specs(node: object) -> Iterable[dict]:
    """Yield serialized model spec dicts from the nested release_model_spec JSON."""
    if isinstance(node, dict):
        if "model_id" in node:
            yield node
            return
        for value in node.values():
            yield from _iter_release_model_specs(value)


def _require_single_unique_value(values: Iterable[str], field_name: str) -> str:
    """Require exactly one non-empty unique value for one release field."""
    unique_values = sorted(
        {value.strip() for value in values if value and value.strip()}
    )
    if not unique_values:
        raise ValueError(
            f"release_model_spec.json does not contain any non-empty {field_name} values."
        )
    if len(unique_values) > 1:
        formatted_values = ", ".join(unique_values)
        raise ValueError(
            f"release_model_spec.json contains multiple {field_name} values: "
            f"{formatted_values}"
        )
    return unique_values[0]


def resolve_release_workflow_refs(release_model_spec_path: Path) -> Tuple[str, str]:
    """Resolve the single tt-metal and vLLM refs required by release.yml."""
    data = json.loads(release_model_spec_path.read_text())
    serialized_specs = list(_iter_release_model_specs(data.get("model_specs", {})))
    tt_metal_ref = _require_single_unique_value(
        (spec.get("tt_metal_commit", "") for spec in serialized_specs),
        "tt_metal_commit",
    )
    vllm_ref = _require_single_unique_value(
        (spec.get("vllm_commit", "") for spec in serialized_specs),
        "vllm_commit",
    )
    return tt_metal_ref, vllm_ref


def _find_recent_release_workflow_run_url(
    dispatch_ref: str, token: str
) -> Optional[str]:
    """Look up the most recent workflow_dispatch run for release.yml."""
    encoded_workflow = quote(RELEASE_WORKFLOW_FILE, safe="")
    encoded_branch = quote(dispatch_ref, safe="")
    url = (
        f"{GITHUB_API}/repos/{TT_SHIELD_OWNER}/{TT_SHIELD_REPO}/actions/workflows/"
        f"{encoded_workflow}/runs?per_page=10&event=workflow_dispatch&branch={encoded_branch}"
    )
    _, response_body = _github_api_request(url, token)
    data = json.loads(response_body.decode("utf-8")) if response_body else {}
    workflow_runs = data.get("workflow_runs", [])
    if not workflow_runs:
        return None
    return workflow_runs[0].get("html_url")


def dispatch_release_workflow(
    *,
    base_ref: str,
    release_branch: str,
    tt_metal_ref: str,
    vllm_ref: str,
) -> Optional[str]:
    """Dispatch tt-shield release.yml and return a run URL when available."""
    dispatch_ref = normalize_dispatch_ref(base_ref)
    token = get_github_token()
    payload = {
        "ref": dispatch_ref,
        "inputs": {
            "tt-metal-commit": tt_metal_ref,
            "tt-inference-server-commit": release_branch,
            "vllm-commit": vllm_ref,
            "workflow": "release",
        },
    }
    dispatch_url = (
        f"{GITHUB_API}/repos/{TT_SHIELD_OWNER}/{TT_SHIELD_REPO}/actions/workflows/"
        f"{quote(RELEASE_WORKFLOW_FILE, safe='')}/dispatches"
    )
    status_code, response_body = _github_api_request(
        dispatch_url, token, method="POST", payload=payload
    )
    if response_body:
        response_data = json.loads(response_body.decode("utf-8"))
        return response_data.get("html_url") or response_data.get("run_url")
    if status_code not in (200, 201, 204):
        raise RuntimeError(
            f"Unexpected status code from release workflow dispatch: {status_code}"
        )

    for _ in range(3):
        run_url = _find_recent_release_workflow_run_url(dispatch_ref, token)
        if run_url:
            return run_url
        time.sleep(2)
    return None
