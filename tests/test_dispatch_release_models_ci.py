import json

import pytest

import scripts.release.dispatch_release_models_ci as dispatch_ci


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
    assert len(captured["requests"]) == 3
    assert captured["requests"][0]["token"] == "token-123"
    assert captured["requests"][0]["method"] == "GET"
    assert (
        "/repos/tenstorrent/tt-metal/commits/metal-sha"
        in captured["requests"][0]["url"]
    )
    assert captured["requests"][1]["token"] == "token-123"
    assert captured["requests"][1]["method"] == "GET"
    assert "/repos/tenstorrent/vllm/commits/vllm-sha" in captured["requests"][1]["url"]
    assert captured["requests"][2]["method"] == "POST"
    assert captured["requests"][2]["url"].endswith(
        "/repos/tenstorrent/tt-shield/actions/workflows/release.yml/dispatches"
    )
    assert captured["requests"][2]["payload"] == {
        "ref": "main",
        "inputs": {
            "tt-metal-commit": tt_metal_sha,
            "tt-inference-server-commit": "stable",
            "vllm-commit": vllm_sha,
            "workflow": "release",
        },
    }


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
