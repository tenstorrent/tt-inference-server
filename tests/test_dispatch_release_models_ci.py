import json

import pytest

import scripts.release.dispatch_release_models_ci as dispatch_ci


def test_normalize_dispatch_ref_rejects_commit_sha():
    with pytest.raises(ValueError, match="workflow_dispatch requires a branch or tag"):
        dispatch_ci.normalize_dispatch_ref("0123456789abcdef")


def test_dispatch_release_workflow_posts_expected_payload(monkeypatch):
    captured = {}

    def fake_github_api_request(url, token, method="GET", payload=None):
        captured["url"] = url
        captured["token"] = token
        captured["method"] = method
        captured["payload"] = payload
        return 200, json.dumps({"html_url": "https://github.com/run/123"}).encode(
            "utf-8"
        )

    monkeypatch.setattr(dispatch_ci, "get_github_token", lambda: "token-123")
    monkeypatch.setattr(dispatch_ci, "_github_api_request", fake_github_api_request)

    run_url = dispatch_ci.dispatch_release_workflow(
        base_ref="main",
        release_branch="stable",
        tt_metal_ref="metal-sha",
        vllm_ref="vllm-sha",
    )

    assert run_url == "https://github.com/run/123"
    assert captured["token"] == "token-123"
    assert captured["method"] == "POST"
    assert captured["url"].endswith(
        "/repos/tenstorrent/tt-shield/actions/workflows/release.yml/dispatches"
    )
    assert captured["payload"] == {
        "ref": "main",
        "inputs": {
            "tt-metal-commit": "metal-sha",
            "tt-inference-server-commit": "stable",
            "vllm-commit": "vllm-sha",
            "workflow": "release",
        },
    }
