import json

import pytest

import scripts.release.dispatch_release_models_ci as dispatch_ci


def test_normalize_dispatch_ref_rejects_commit_sha():
    with pytest.raises(ValueError, match="workflow_dispatch requires a branch or tag"):
        dispatch_ci.normalize_dispatch_ref("0123456789abcdef")


def test_resolve_release_workflow_refs_requires_single_unique_values(tmp_path):
    release_model_spec_path = tmp_path / "release_model_spec.json"
    release_model_spec_path.write_text(
        json.dumps(
            {
                "model_specs": {
                    "org/model": {
                        "n150": {
                            "vllm": {
                                "impl-a": {
                                    "model_id": "spec-a",
                                    "tt_metal_commit": "metal-sha",
                                    "vllm_commit": "vllm-sha",
                                },
                                "impl-b": {
                                    "model_id": "spec-b",
                                    "tt_metal_commit": "metal-sha",
                                    "vllm_commit": "vllm-sha",
                                },
                            }
                        }
                    }
                }
            }
        )
    )

    assert dispatch_ci.resolve_release_workflow_refs(release_model_spec_path) == (
        "metal-sha",
        "vllm-sha",
    )


def test_resolve_release_workflow_refs_raises_on_ambiguous_values(tmp_path):
    release_model_spec_path = tmp_path / "release_model_spec.json"
    release_model_spec_path.write_text(
        json.dumps(
            {
                "model_specs": {
                    "org/model": {
                        "n150": {
                            "vllm": {
                                "impl-a": {
                                    "model_id": "spec-a",
                                    "tt_metal_commit": "metal-a",
                                    "vllm_commit": "vllm-sha",
                                },
                                "impl-b": {
                                    "model_id": "spec-b",
                                    "tt_metal_commit": "metal-b",
                                    "vllm_commit": "vllm-sha",
                                },
                            }
                        }
                    }
                }
            }
        )
    )

    with pytest.raises(ValueError, match="multiple tt_metal_commit values"):
        dispatch_ci.resolve_release_workflow_refs(release_model_spec_path)


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
