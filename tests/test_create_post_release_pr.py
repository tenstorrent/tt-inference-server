# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import json
import subprocess
import urllib.error
from types import SimpleNamespace

import pytest

from scripts.release.create_post_release_pr import (
    build_rows,
    fetch_run_jobs,
    jobs_that_ran,
    render_body,
    render_table,
    resolve_galaxy_sw_versions,
    resolve_release_scope,
)
from scripts.release.release_scope import (
    ProdLeaf,
    ProdPin,
    load_prod_leaves_from_ref,
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
    base = {
        IDENTITY: ProdLeaf(
            identity=IDENTITY,
            pin=ProdPin("1.0.0", "old-metal", "old-vllm", None),
            status="FUNCTIONAL",
        )
    }

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

    # The dev catalogue offers two impls on GALAXY; only the default one is the
    # release leaf, and its commit is reported as the old -> new change.
    assert [item.identity for item in scope] == [IDENTITY]
    assert table.count("Qwen/Qwen3-32B") == 1
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
        load_prod_leaves_from_ref("not-a-ref")


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


def test_ci_job_link_follows_the_configured_impl():
    identity = ("meta-llama/Llama-3.1-8B-Instruct", "P300X2", "vLLM", "llama31_8b_qb2")
    scope = [
        SimpleNamespace(identity=identity, combo=SimpleNamespace(impl="llama31-8b-qb2"))
    ]
    prod = {
        identity: ProdLeaf(
            identity, ProdPin("1.2.3", "metal", "vllm", None), "FUNCTIONAL"
        )
    }
    jobs = [
        {
            "id": 1,
            "name": "_ / vLLM / run-release-meta-llama__Llama-3.1-8B-Instruct-bh-qb-ge-p300x2",
        },
        {
            "id": 2,
            "name": "_ / vLLM / run-release-meta-llama__Llama-3.1-8B-Instruct@llama31-8b-qb2@-bh-qb-ge-p300x2",
        },
    ]
    rows = build_rows(scope, prod, {}, jobs, "repo/name", "123", "1.2.3")
    assert rows[0]["ci_url"].endswith("/job/2")


QB2 = ("meta-llama/Llama-3.1-8B-Instruct", "P300X2", "vLLM", "llama31_8b_qb2")
LEGACY_JOB = {
    "id": 7,
    "name": "_ / vLLM / run-release-meta-llama__Llama-3.1-8B-Instruct-bh-qb-ge-p300x2",
}


def _qb2_rows(jobs, job_log=None):
    scope = [
        SimpleNamespace(identity=QB2, combo=SimpleNamespace(impl="llama31-8b-qb2"))
    ]
    prod = {QB2: ProdLeaf(QB2, ProdPin("1.2.3", "metal", "vllm", None), "FUNCTIONAL")}
    return build_rows(
        scope, prod, {}, jobs, "repo/name", "123", "1.2.3", job_log=job_log
    )


def test_legacy_run_links_the_bare_model_job_its_log_proves_ran_the_impl():
    """Runs from before ``<model>@<impl>`` job names: the bare-model job is only
    linked when its log shows ``--impl <impl>`` -- it may be the default impl."""
    log = '  arguments+=("--impl" "llama31-8b-qb2")\n'
    assert _qb2_rows([LEGACY_JOB], job_log=lambda _id: log)[0]["ci_url"].endswith(
        "/job/7"
    )


@pytest.mark.parametrize(
    "log",
    [None, 'if [ "" != "" ]; then', '  arguments+=("--impl" "llama31-8b-qb2-fast")'],
)
def test_legacy_run_without_proof_of_the_impl_is_not_linked(log):
    assert _qb2_rows([LEGACY_JOB], job_log=lambda _id: log)[0]["ci_url"] is None
    assert _qb2_rows([LEGACY_JOB])[0]["ci_url"] is None


def test_a_run_with_leaf_job_names_never_falls_back_to_the_bare_model_job():
    other = {"id": 8, "name": "run-release-org__Other@impl-x@-p150-P150"}
    rows = _qb2_rows(
        [LEGACY_JOB, other], job_log=lambda _id: '"--impl" "llama31-8b-qb2"'
    )
    assert rows[0]["ci_url"] is None


def test_rows_record_the_matched_job_id():
    job = {
        "id": 9,
        "name": LEGACY_JOB["name"].replace("Instruct-", "Instruct@llama31-8b-qb2@-"),
    }
    assert _qb2_rows([job])[0]["ci_job_id"] == 9


def test_longer_impl_outside_release_scope_is_not_linked():
    job = {
        "id": 10,
        "name": LEGACY_JOB["name"].replace(
            "Instruct-", "Instruct@llama31-8b-qb2-fast@-"
        ),
    }
    assert _qb2_rows([job])[0]["ci_url"] is None


def test_galaxy_sw_versions_read_the_first_galaxy_row_that_has_a_job(monkeypatch):
    fetched = []
    monkeypatch.setattr(
        "scripts.release.create_post_release_pr.fetch_job_log",
        lambda _repo, job_id, _token: fetched.append(job_id),
    )
    rows = [
        {"device": "GALAXY", "ci_job_id": None},
        {"device": "GALAXY", "ci_job_id": 42},
    ]
    resolve_galaxy_sw_versions(rows, [{"id": 42}], "repo/name", "123", "token")
    assert fetched == [42]


TT_IDENTITY = ("Qwen/Qwen3-32B", "GALAXY", "vLLM", "tt_transformers")


def test_two_impls_of_one_model_device_are_released_and_linked_separately(tmp_path):
    config = {
        "models": {
            "Qwen3-32B": {
                "implementations": [
                    {
                        "inference_engine": "vLLM",
                        "ci": {"release": {"devices": ["GALAXY"]}},
                    },
                    {
                        "inference_engine": "vLLM",
                        "impl": "tt-transformers",
                        "ci": {"release": {"devices": ["GALAXY"]}},
                    },
                ]
            }
        }
    }
    scope = resolve_release_scope(config, _write_dev(tmp_path))
    assert {item.identity for item in scope} == {IDENTITY, TT_IDENTITY}

    pin = ProdPin("1.2.3", "metal", "vllm", None)
    prod = {i: ProdLeaf(i, pin, "FUNCTIONAL") for i in (IDENTITY, TT_IDENTITY)}
    jobs = [
        {"id": 1, "name": "_ / vLLM / run-release-Qwen__Qwen3-32B-6u-galaxy"},
        {
            "id": 2,
            "name": "_ / vLLM / run-release-Qwen__Qwen3-32B@tt-transformers@-6u-galaxy",
        },
    ]
    rows = build_rows(scope, prod, {}, jobs, "repo/name", "123", "1.2.3")
    assert {r["identity"]: r["ci_job_id"] for r in rows} == {
        IDENTITY: 1,
        TT_IDENTITY: 2,
    }


# A re-run ("re-run failed jobs") is a new attempt of the same run. GitHub copies
# every job that already passed into it under a NEW job id; the copy keeps the
# original's start time but has no artifacts, so nothing was collected under its
# id. These are the timestamps of tt-shield run 38572819455 (v0.25.0).
ATTEMPT_1 = "2026-10-14T13:33:49Z"
ATTEMPT_2 = "2026-10-15T01:15:15Z"
RAN_IN_1 = "2026-10-14T16:01:57Z"
RAN_IN_2 = "2026-10-15T01:15:19Z"


def _job(job_id, name, conclusion, started_at):
    return {
        "id": job_id,
        "name": name,
        "conclusion": conclusion,
        "started_at": started_at,
    }


def _ids(jobs):
    return sorted(job["id"] for job in jobs)


def test_a_rerun_keeps_the_jobs_that_ran_not_their_copies():
    attempts = [
        {
            "run_started_at": ATTEMPT_1,
            "jobs": [
                _job(1, "run-release-qwen", "success", RAN_IN_1),
                _job(2, "run-release-wan", "failure", RAN_IN_1),
            ],
        },
        {
            "run_started_at": ATTEMPT_2,
            "jobs": [
                _job(11, "run-release-qwen", "success", RAN_IN_1),  # the copy of job 1
                _job(12, "run-release-wan", "success", RAN_IN_2),  # re-ran
            ],
        },
    ]
    assert _ids(jobs_that_ran(attempts)) == [1, 12]


def test_a_job_that_fails_twice_is_taken_from_the_attempt_it_passed():
    attempt_3 = "2026-10-16T00:00:00Z"
    attempts = [
        {
            "run_started_at": ATTEMPT_1,
            "jobs": [
                _job(1, "a", "success", RAN_IN_1),
                _job(2, "b", "failure", RAN_IN_1),
            ],
        },
        {
            "run_started_at": ATTEMPT_2,
            "jobs": [
                _job(11, "a", "success", RAN_IN_1),
                _job(12, "b", "failure", RAN_IN_2),
            ],
        },
        {
            "run_started_at": attempt_3,
            "jobs": [
                _job(21, "a", "success", RAN_IN_1),
                _job(22, "b", "success", "2026-10-16T00:00:05Z"),
            ],
        },
    ]
    assert _ids(jobs_that_ran(attempts)) == [1, 22]


def test_a_success_wins_over_a_newer_failure_of_the_same_job():
    # "Re-run all jobs": every job runs again, and this one failed the second time.
    attempts = [
        {"run_started_at": ATTEMPT_1, "jobs": [_job(1, "a", "success", RAN_IN_1)]},
        {"run_started_at": ATTEMPT_2, "jobs": [_job(11, "a", "failure", RAN_IN_2)]},
    ]
    assert _ids(jobs_that_ran(attempts)) == [1]


def test_one_attempt_is_returned_as_listed():
    jobs = [
        _job(1, "a", "success", RAN_IN_1),
        _job(2, "b", "failure", RAN_IN_1),
        _job(3, "c", "skipped", None),
    ]
    assert jobs_that_ran([{"run_started_at": ATTEMPT_1, "jobs": jobs}]) == jobs


def test_two_jobs_of_one_name_in_an_attempt_are_both_kept_for_build_rows_to_refuse():
    jobs = [_job(1, "a", "success", RAN_IN_1), _job(2, "a", "success", RAN_IN_1)]
    assert _ids(jobs_that_ran([{"run_started_at": ATTEMPT_1, "jobs": jobs}])) == [1, 2]


@pytest.mark.parametrize("rerun", ["success", "failure"])
def test_a_partial_rerun_of_a_duplicated_name_stays_ambiguous(rerun):
    # Which of the two the re-run replaced is unknowable: every execution stays.
    attempts = [
        {
            "run_started_at": ATTEMPT_1,
            "jobs": [
                _job(1, "a", "success", RAN_IN_1),
                _job(2, "a", "failure", RAN_IN_1),
            ],
        },
        {
            "run_started_at": ATTEMPT_2,
            "jobs": [
                _job(11, "a", "success", RAN_IN_1),
                _job(12, "a", rerun, RAN_IN_2),
            ],
        },
    ]
    assert _ids(jobs_that_ran(attempts)) == [1, 2, 12]


class _FakeApi:
    """The GitHub REST paths fetch_run_jobs reads, for one run with two attempts."""

    def __init__(self):
        base = "https://api.github.com/repos/tenstorrent/tt-shield/actions/runs/7"
        self.responses = {
            base: {"run_attempt": 2},
            f"{base}/attempts/1": {"run_started_at": ATTEMPT_1},
            f"{base}/attempts/2": {"run_started_at": ATTEMPT_2},
            f"{base}/attempts/1/jobs?per_page=100&page=1": {
                "jobs": [
                    _job(1, "a", "success", RAN_IN_1),
                    _job(2, "b", "failure", RAN_IN_1),
                ]
            },
            f"{base}/attempts/2/jobs?per_page=100&page=1": {
                "jobs": [
                    _job(11, "a", "success", RAN_IN_1),
                    _job(12, "b", "success", RAN_IN_2),
                ]
            },
        }
        self.requested = []

    def __call__(self, url, token):
        self.requested.append(url)
        if url not in self.responses:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        return self.responses[url]


def test_fetch_run_jobs_reads_every_attempt(monkeypatch):
    api = _FakeApi()
    monkeypatch.setattr("scripts.release.create_post_release_pr._api_get", api)
    assert _ids(fetch_run_jobs("tenstorrent/tt-shield", "7", "token")) == [1, 12]
    assert not any("filter=" in url or url.endswith("/7/jobs") for url in api.requested)


def test_fetch_run_jobs_is_none_when_the_run_cannot_be_read(monkeypatch):
    api = _FakeApi()
    del api.responses[
        next(
            u
            for u in api.responses
            if u.endswith("/attempts/2/jobs?per_page=100&page=1")
        )
    ]
    monkeypatch.setattr("scripts.release.create_post_release_pr._api_get", api)
    assert fetch_run_jobs("tenstorrent/tt-shield", "7", "token") is None
