import json

import pytest

import scripts.release.dispatch_release_models_ci as dispatch_ci


def _workflow_run(
    run_id: int,
    created_at: str,
    html_url: str = None,
    status: str = "queued",
):
    return {
        "id": run_id,
        "created_at": created_at,
        "html_url": html_url or f"https://github.com/run/{run_id}",
        "status": status,
    }


def test_normalize_dispatch_ref_rejects_commit_sha():
    with pytest.raises(ValueError, match="workflow_dispatch requires a branch or tag"):
        dispatch_ci.normalize_dispatch_ref("0123456789abcdef")


def test_dispatch_release_workflow_posts_expected_payload(monkeypatch):
    captured = {"requests": []}
    tt_metal_sha = "a" * 40
    vllm_sha = "b" * 40

    def fake_github_api_request(url, token, method="GET", payload=None):
        captured["requests"].append(
            {"url": url, "token": token, "method": method, "payload": payload}
        )
        if method == "GET":
            if "/repos/tenstorrent/tt-metal/commits/" in url:
                return 200, json.dumps({"sha": tt_metal_sha}).encode("utf-8")
            if "/repos/tenstorrent/vllm/commits/" in url:
                return 200, json.dumps({"sha": vllm_sha}).encode("utf-8")
            if "/actions/workflows/release.yml/runs" in url:
                return 200, json.dumps({"workflow_runs": []}).encode("utf-8")
        return 200, json.dumps({"html_url": "https://github.com/run/123"}).encode(
            "utf-8"
        )

    monkeypatch.setattr(dispatch_ci, "get_github_token", lambda: "token-123")
    monkeypatch.setattr(dispatch_ci, "_github_api_request", fake_github_api_request)

    run_url = dispatch_ci.dispatch_release_workflow(
        release_branch="stable",
        tt_metal_ref="metal-sha",
        vllm_ref="vllm-sha",
    )

    assert run_url == "https://github.com/run/123"
    assert len(captured["requests"]) == 4
    assert captured["requests"][0]["token"] == "token-123"
    assert captured["requests"][0]["method"] == "GET"
    assert (
        "/repos/tenstorrent/tt-metal/commits/metal-sha"
        in captured["requests"][0]["url"]
    )
    assert captured["requests"][1]["token"] == "token-123"
    assert captured["requests"][1]["method"] == "GET"
    assert "/repos/tenstorrent/vllm/commits/vllm-sha" in captured["requests"][1]["url"]
    assert captured["requests"][2]["method"] == "GET"
    assert "/actions/workflows/release.yml/runs" in captured["requests"][2]["url"]
    assert captured["requests"][3]["method"] == "POST"
    assert captured["requests"][3]["url"].endswith(
        "/repos/tenstorrent/tt-shield/actions/workflows/release.yml/dispatches"
    )
    assert captured["requests"][3]["payload"] == {
        "ref": dispatch_ci.TT_SHIELD_WORKFLOW_REF,
        "inputs": {
            "tt-metal-commit": tt_metal_sha,
            "tt-inference-server-commit": "stable",
            "vllm-commit": vllm_sha,
            "workflow": "release",
        },
    }


def test_dispatch_release_workflow_includes_run_ai_summary_input(monkeypatch, capsys):
    captured = {"requests": []}
    tt_metal_sha = "a" * 40
    vllm_sha = "b" * 40
    workflow_runs_responses = [
        [],
        [_workflow_run(456, "2999-01-01T00:00:00Z", status="in_progress")],
    ]

    def fake_github_api_request(url, token, method="GET", payload=None):
        captured["requests"].append(
            {"url": url, "token": token, "method": method, "payload": payload}
        )
        if method == "GET":
            if "/repos/tenstorrent/tt-metal/commits/" in url:
                return 200, json.dumps({"sha": tt_metal_sha}).encode("utf-8")
            if "/repos/tenstorrent/vllm/commits/" in url:
                return 200, json.dumps({"sha": vllm_sha}).encode("utf-8")
            if "/actions/workflows/release.yml/runs" in url:
                return 200, json.dumps(
                    {"workflow_runs": workflow_runs_responses.pop(0)}
                ).encode("utf-8")
        return 200, b"{}"

    monkeypatch.setattr(dispatch_ci, "get_github_token", lambda: "token-123")
    monkeypatch.setattr(dispatch_ci, "_github_api_request", fake_github_api_request)
    monkeypatch.setattr(dispatch_ci.time, "sleep", lambda _: None)

    run_url = dispatch_ci.dispatch_release_workflow(
        release_branch="stable",
        tt_metal_ref="metal-sha",
        vllm_ref="vllm-sha",
        run_ai_summary=False,
    )
    captured_output = capsys.readouterr()

    assert run_url == "https://github.com/run/456"
    assert "Release workflow dispatch accepted successfully" in captured_output.out
    assert "polling for the newly started workflow run ID" in captured_output.out
    assert captured["requests"][3]["payload"] == {
        "ref": dispatch_ci.TT_SHIELD_WORKFLOW_REF,
        "inputs": {
            "tt-metal-commit": tt_metal_sha,
            "tt-inference-server-commit": "stable",
            "vllm-commit": vllm_sha,
            "workflow": "release",
            "run-ai-summary": False,
        },
    }


def test_dispatch_release_workflow_returns_new_run_not_previous_visible_run(
    monkeypatch,
):
    tt_metal_sha = "a" * 40
    vllm_sha = "b" * 40
    previous_run = _workflow_run(111, "2000-01-01T00:00:00Z", status="in_progress")
    new_run = _workflow_run(222, "2999-01-01T00:00:00Z", status="queued")
    workflow_runs_responses = [
        [previous_run],
        [previous_run],
        [new_run, previous_run],
    ]

    def fake_github_api_request(url, token, method="GET", payload=None):
        if method == "GET":
            if "/repos/tenstorrent/tt-metal/commits/" in url:
                return 200, json.dumps({"sha": tt_metal_sha}).encode("utf-8")
            if "/repos/tenstorrent/vllm/commits/" in url:
                return 200, json.dumps({"sha": vllm_sha}).encode("utf-8")
            if "/actions/workflows/release.yml/runs" in url:
                return 200, json.dumps(
                    {"workflow_runs": workflow_runs_responses.pop(0)}
                ).encode("utf-8")
        return 204, b""

    monkeypatch.setattr(dispatch_ci, "get_github_token", lambda: "token-123")
    monkeypatch.setattr(dispatch_ci, "_github_api_request", fake_github_api_request)
    monkeypatch.setattr(dispatch_ci.time, "sleep", lambda _: None)

    run_url = dispatch_ci.dispatch_release_workflow(
        release_branch="stable",
        tt_metal_ref="metal-sha",
        vllm_ref="vllm-sha",
    )

    assert run_url == "https://github.com/run/222"


def test_dispatch_release_workflow_returns_none_when_no_new_run_appears(monkeypatch):
    tt_metal_sha = "a" * 40
    vllm_sha = "b" * 40
    previous_run = _workflow_run(111, "2000-01-01T00:00:00Z", status="completed")
    workflow_runs_responses = [
        [previous_run],
        *[[previous_run] for _ in range(dispatch_ci.WORKFLOW_RUN_LOOKUP_RETRIES)],
    ]

    def fake_github_api_request(url, token, method="GET", payload=None):
        if method == "GET":
            if "/repos/tenstorrent/tt-metal/commits/" in url:
                return 200, json.dumps({"sha": tt_metal_sha}).encode("utf-8")
            if "/repos/tenstorrent/vllm/commits/" in url:
                return 200, json.dumps({"sha": vllm_sha}).encode("utf-8")
            if "/actions/workflows/release.yml/runs" in url:
                return 200, json.dumps(
                    {"workflow_runs": workflow_runs_responses.pop(0)}
                ).encode("utf-8")
        return 204, b""

    monkeypatch.setattr(dispatch_ci, "get_github_token", lambda: "token-123")
    monkeypatch.setattr(dispatch_ci, "_github_api_request", fake_github_api_request)
    monkeypatch.setattr(dispatch_ci.time, "sleep", lambda _: None)

    run_url = dispatch_ci.dispatch_release_workflow(
        release_branch="stable",
        tt_metal_ref="metal-sha",
        vllm_ref="vllm-sha",
    )

    assert run_url is None


def test_resolve_commit_sha_falls_back_to_graphql_for_short_sha(monkeypatch):
    fallback_sha = "c" * 40

    def fake_github_api_request(url, token, method="GET", payload=None):
        raise RuntimeError(
            "GitHub API request failed (422) for "
            "https://api.github.com/repos/tenstorrent/tt-metal/commits/2f70ab2: "
            '{"message":"No commit found for SHA: 2f70ab2","status":"422"}'
        )

    monkeypatch.setattr(dispatch_ci, "_github_api_request", fake_github_api_request)
    monkeypatch.setattr(
        dispatch_ci,
        "_resolve_commit_sha_from_graphql",
        lambda owner, repo, ref, token: fallback_sha,
    )

    assert (
        dispatch_ci._resolve_commit_sha(
            "tenstorrent", "tt-metal", "2f70ab2", "token-123"
        )
        == fallback_sha
    )
