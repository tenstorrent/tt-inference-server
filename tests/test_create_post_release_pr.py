# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import json
import subprocess
from types import SimpleNamespace

import pytest

from scripts.release.create_post_release_pr import (
    build_rows,
    load_prod_blocks_from_ref,
    render_body,
    render_table,
    resolve_release_scope,
)
from scripts.release.release_scope import (
    ProdLeaf,
    ProdPin,
    expand_raw_prod_blocks,
)
from workflows.model_spec import MODEL_SPEC_CATALOG_FILES


IDENTITY = ("Qwen/Qwen3-32B", "GALAXY", "vLLM", "qwen3_32b_galaxy")


def _write_dev(tmp_path):
    dev = tmp_path / "dev"
    dev.mkdir()
    for filename in MODEL_SPEC_CATALOG_FILES:
        (dev / filename).write_text("templates: []\n")
    (dev / "llm.yaml").write_text(
        """
templates:
- weights: [Qwen/Qwen3-32B]
  impl: qwen3_32b_galaxy
  inference_engine: VLLM
  device_model_specs:
    - {device: GALAXY, max_concurrency: 32, max_context: 131072, default_impl: true}
    - {device: BLACKHOLE_GALAXY, max_concurrency: 8, max_context: 32768, default_impl: true}
- weights: [Qwen/Qwen3-32B]
  impl: tt_transformers
  inference_engine: VLLM
  device_model_specs:
    - {device: GALAXY, max_concurrency: 1, max_context: 32768}
""".lstrip()
    )
    return dev


def _scope(tmp_path):
    config = {
        "models": {
            "Qwen3-32B": {
                "inference_engine": "vLLM",
                "ci": {"release": {"devices": ["GALAXY"]}},
            }
        }
    }
    return resolve_release_scope(config, _write_dev(tmp_path))


def _prod_leaf(version="1.2.3"):
    return ProdLeaf(
        identity=IDENTITY,
        pin=ProdPin(version, "metal", "vllm", "ghcr.io/image:tag"),
        status="FUNCTIONAL",
    )


def test_exact_release_row_uses_default_leaf_full_repo_and_pins(tmp_path):
    scope = _scope(tmp_path)
    base = expand_raw_prod_blocks(
        [
            {
                "weights": ["Qwen/Qwen3-32B", "Qwen/Qwen3-14B"],
                "impl": "qwen3_32b_galaxy",
                "inference_engine": "VLLM",
                "version": "1.0.0",
                "tt_metal_commit": "old-metal",
                "vllm_commit": "old-vllm",
                "status": "FUNCTIONAL",
                "device_model_specs": [{"device": "GALAXY"}],
            }
        ]
    )

    rows = build_rows(
        scope,
        {IDENTITY: _prod_leaf()},
        base,
        jobs=None,
        tt_shield_repo="tenstorrent/tt-shield",
        run_id=None,
        version="1.2.3",
    )
    table = render_table(rows)

    assert [item.identity for item in scope] == [IDENTITY]
    assert table.count("Qwen/Qwen3-32B") == 1
    assert "Qwen/Qwen3-14B" not in table
    # The base block bundled two weights under one pin; only the released leaf
    # is reported, and its commit is shown as the old -> new change.
    assert "`old-metal` → `metal`" in table
    assert "No change [FUNCTIONAL]" in table


def test_missing_or_wrong_version_current_prod_fails(tmp_path):
    scope = _scope(tmp_path)
    with pytest.raises(ValueError, match="missing"):
        build_rows(scope, {}, {}, None, "repo/name", None, "1.2.3")
    with pytest.raises(ValueError, match="has version"):
        build_rows(
            scope,
            {IDENTITY: _prod_leaf("other")},
            {},
            None,
            "repo/name",
            None,
            "1.2.3",
        )


def test_retained_same_version_leaf_outside_cleaned_scope_is_ignored(tmp_path):
    scope = _scope(tmp_path)
    extra_identity = ("org/deferred", "N150", "media", "whisper")
    prod = {
        IDENTITY: _prod_leaf(),
        extra_identity: ProdLeaf(
            extra_identity,
            ProdPin("1.2.3", "metal", None, None),
            "FUNCTIONAL",
        ),
    }

    rows = build_rows(scope, prod, {}, None, "repo/name", None, "1.2.3")

    assert [row["identity"] for row in rows] == [IDENTITY]


def test_ci_job_matching_supports_full_repo_and_rejects_ambiguity(tmp_path):
    scope = _scope(tmp_path)
    jobs = [
        {
            "id": 42,
            "name": "run-tests / run-release-Qwen__Qwen3-32B-runner-GALAXY",
        }
    ]
    rows = build_rows(
        scope,
        {IDENTITY: _prod_leaf()},
        {},
        jobs,
        "tenstorrent/tt-shield",
        "123",
        "1.2.3",
    )
    assert rows[0]["ci_url"].endswith("/job/42")

    with pytest.raises(ValueError, match="Multiple CI jobs"):
        build_rows(
            scope,
            {IDENTITY: _prod_leaf()},
            {},
            jobs * 2,
            "tenstorrent/tt-shield",
            "123",
            "1.2.3",
        )


def test_short_and_full_selectors_resolving_same_identity_fail(tmp_path):
    config = {
        "models": {
            "Qwen3-32B": {
                "inference_engine": "vLLM",
                "ci": {"release": {"devices": ["GALAXY"]}},
            },
            "Qwen/Qwen3-32B": {
                "inference_engine": "vLLM",
                "ci": {"release": {"devices": ["GALAXY"]}},
            },
        }
    }

    with pytest.raises(ValueError, match="duplicate identity"):
        resolve_release_scope(config, _write_dev(tmp_path))


def test_rendered_table_is_json_independent_and_exact(tmp_path):
    scope = _scope(tmp_path)
    rows = build_rows(
        scope,
        {IDENTITY: _prod_leaf()},
        {},
        None,
        "repo/name",
        None,
        "1.2.3",
    )

    assert json.dumps(rows, default=str)
    assert (
        "| Impl | Model Arch | Weights | Devices | TT-Metal Commit Change | "
        "Status Change | CI Job Link |"
    ) in render_table(rows)


def test_body_carries_release_metadata_and_promoted_images():
    """The release pipeline reads both back out of the PR body.

    The metadata comment identifies the run and version for the Release Object,
    and the promoted image list is the publish plan; neither is recoverable from
    the table, so a body missing them silently loses release provenance.
    """
    body = render_body(
        "1.2.3",
        "999",
        rows=[],
        promoted_images=["ghcr.io/tenstorrent/a:v1.2.3", "https://ghcr.io/b:v1.2.3"],
    )

    assert body.startswith(
        "<!--\nmetadata:run_id=999\nmetadata:version=v1.2.3\n-->\n\n"
    )
    assert "- https://ghcr.io/tenstorrent/a:v1.2.3" in body
    assert "- https://ghcr.io/b:v1.2.3" in body
    assert "**Total:** 2" in body


def test_invalid_base_ref_fails_instead_of_appearing_new(monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(128, args[0])

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(ValueError, match="Could not read prod catalog"):
        load_prod_blocks_from_ref("not-a-ref")


def test_one_legacy_basename_job_cannot_link_two_repositories():
    identity_a = ("org-a/shared", "N150", "vLLM", "impl-a")
    identity_b = ("org-b/shared", "N150", "vLLM", "impl-b")
    scope = [
        SimpleNamespace(identity=identity_a),
        SimpleNamespace(identity=identity_b),
    ]
    prod = {
        identity_a: ProdLeaf(
            identity_a, ProdPin("1.2.3", "metal", "vllm", None), "FUNCTIONAL"
        ),
        identity_b: ProdLeaf(
            identity_b, ProdPin("1.2.3", "metal", "vllm", None), "FUNCTIONAL"
        ),
    }
    jobs = [{"id": 1, "name": "run-release-shared-runner-N150"}]

    with pytest.raises(ValueError, match="ambiguously matches"):
        build_rows(scope, prod, {}, jobs, "repo/name", "123", "1.2.3")
