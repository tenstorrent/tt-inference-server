# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import io
import json
import zipfile

import pytest

from scripts.release.build_release_artifacts import (
    artifact_model_token,
    device_from_jobs,
    resolve_model,
    resolve_configured_scope,
    runner_of,
    validate_bundle_identity,
    validate_staged_identity_set,
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
""".lstrip()
    )
    return dev


def _bundle(tmp_path, identity=IDENTITY, name="bundle.zip"):
    path = tmp_path / name
    document = {
        "runtime_model_spec": {
            "hf_model_repo": identity[0],
            "device_type": identity[1],
            "inference_engine": identity[2],
            "impl": {"impl_id": identity[3]},
        }
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("runtime_model_specs/spec.json", json.dumps(document))
    return path


def test_config_scope_resolves_exact_artifact_identity(tmp_path):
    config = {
        "models": {
            "Qwen3-32B": {
                "inference_engine": "vLLM",
                "ci": {"release": {"devices": ["GALAXY"]}},
            }
        }
    }

    models, expected, kinds = resolve_configured_scope(config, _write_dev(tmp_path))

    assert models == {"Qwen/Qwen3-32B": ["galaxy"]}
    assert expected == {("Qwen/Qwen3-32B", "galaxy"): IDENTITY}
    assert kinds == {"Qwen/Qwen3-32B": ["release"]}


def test_artifact_and_job_matching_support_full_name_variants():
    artifact = "workflow_logs_release_Qwen__Qwen3-32B_runner_default"
    assert artifact_model_token(artifact, "Qwen/Qwen3-32B") == "Qwen__Qwen3-32B"
    assert runner_of(artifact, "Qwen/Qwen3-32B") == "runner"

    jobs = [{"name": "run-tests / run-release-Qwen/Qwen3-32B-runner-GALAXY"}]
    assert device_from_jobs(jobs, "Qwen/Qwen3-32B", "runner") == "GALAXY"


def test_bundle_runtime_identity_must_match_config(tmp_path):
    bundle = _bundle(tmp_path)
    assert validate_bundle_identity(bundle, IDENTITY) == IDENTITY

    with pytest.raises(ValueError, match="does not match"):
        validate_bundle_identity(
            bundle,
            ("Qwen/Qwen3-32B", "N150", "vLLM", "qwen3_32b_galaxy"),
        )


def test_artifact_resolution_selects_unique_exact_identity(tmp_path, monkeypatch):
    wrong = _bundle(
        tmp_path,
        ("Qwen/Qwen3-32B", "N150", "vLLM", "qwen3_32b_galaxy"),
        "wrong.zip",
    )
    correct = _bundle(tmp_path, name="correct.zip")
    artifacts = [
        {
            "id": 1,
            "name": "workflow_logs_release_Qwen__Qwen3-32B_old_default",
        },
        {
            "id": 2,
            "name": "workflow_logs_release_Qwen__Qwen3-32B_new_default",
        },
    ]
    jobs = [
        {"name": "run-release-Qwen/Qwen3-32B-old-galaxy"},
        {"name": "run-release-Qwen/Qwen3-32B-new-galaxy"},
    ]
    paths = {1: wrong, 2: correct}
    monkeypatch.setattr(
        "scripts.release.build_release_artifacts.download_artifact",
        lambda repo, artifact, tmp, cache: paths[artifact["id"]],
    )

    chosen = resolve_model(
        "Qwen/Qwen3-32B",
        ["galaxy"],
        artifacts,
        jobs,
        "org/repo",
        tmp_path,
        {},
        {("Qwen/Qwen3-32B", "galaxy"): IDENTITY},
    )

    assert chosen == {"galaxy": artifacts[1]}

    paths[1] = correct
    with pytest.raises(SystemExit, match="found 2"):
        resolve_model(
            "Qwen/Qwen3-32B",
            ["galaxy"],
            artifacts,
            jobs,
            "org/repo",
            tmp_path,
            {},
            {("Qwen/Qwen3-32B", "galaxy"): IDENTITY},
        )


def test_bundle_validator_rejects_missing_runtime_spec(tmp_path):
    path = tmp_path / "empty.zip"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("run_logs/log.txt", "nothing")
    path.write_bytes(buffer.getvalue())

    with pytest.raises(ValueError, match="no runtime model spec"):
        validate_bundle_identity(path, IDENTITY)


def test_staged_runtime_identities_must_equal_configured_scope():
    other = ("org/other", "N150", "vLLM", "tt_transformers")

    validate_staged_identity_set({IDENTITY}, {IDENTITY})

    with pytest.raises(ValueError, match="missing=.*Qwen/Qwen3-32B"):
        validate_staged_identity_set({IDENTITY}, set())
    with pytest.raises(ValueError, match="extra=.*org/other"):
        validate_staged_identity_set({IDENTITY}, {IDENTITY, other})


def test_config_scope_rejects_model_device_engine_collision(tmp_path):
    dev = _write_dev(tmp_path)
    with (dev / "llm.yaml").open("a") as file:
        file.write(
            """
- weights: [Qwen/Qwen3-32B]
  impl: forge_vllm_plugin
  inference_engine: FORGE
  device_model_specs:
    - {device: GALAXY, max_concurrency: 1, max_context: 1024, default_impl: true}
"""
        )
    config = {
        "models": {
            "Qwen3-32B": {
                "implementations": [
                    {
                        "inference_engine": "vLLM",
                        "ci": {"release": {"devices": ["GALAXY"]}},
                    },
                    {
                        "inference_engine": "FORGE",
                        "ci": {"release": {"devices": ["GALAXY"]}},
                    },
                ]
            }
        }
    }

    with pytest.raises(ValueError, match="collide on artifact filename"):
        resolve_configured_scope(config, dev)


def test_config_scope_rejects_short_and_full_selector_aliases(tmp_path):
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
        resolve_configured_scope(config, _write_dev(tmp_path))


def test_config_scope_rejects_non_injective_artifact_slugs(tmp_path):
    dev = tmp_path / "dev"
    dev.mkdir()
    for filename in MODEL_SPEC_CATALOG_FILES:
        (dev / filename).write_text("templates: []\n")
    (dev / "llm.yaml").write_text(
        """
templates:
- weights: [org/model]
  impl: tt_transformers
  inference_engine: VLLM
  device_model_specs:
    - {device: N150, max_concurrency: 1, max_context: 1024, default_impl: true}
- weights: [org__model]
  impl: qwen3_32b_galaxy
  inference_engine: VLLM
  device_model_specs:
    - {device: N150, max_concurrency: 1, max_context: 1024, default_impl: true}
""".lstrip()
    )
    config = {
        "models": {
            "org/model": {
                "inference_engine": "vLLM",
                "ci": {"release": {"devices": ["N150"]}},
            },
            "org__model": {
                "inference_engine": "vLLM",
                "ci": {"release": {"devices": ["N150"]}},
            },
        }
    }

    with pytest.raises(ValueError, match="collide on artifact filename"):
        resolve_configured_scope(config, dev)


def test_job_matching_uses_the_impl_from_the_logs_artifact(tmp_path, monkeypatch):
    """tt-shield names a non-default impl's job ``run-release-<model>@<impl>-...``;
    the impl comes from the logs artifact's ``_<impl-or-default>`` suffix."""
    artifacts = [
        {
            "id": 1,
            "name": "workflow_logs_release_meta-llama__Llama-3.1-8B-Instruct_bh-qb-ge_llama31-8b-qb2",
        },
    ]
    jobs = [
        {
            "name": "_ / vLLM / run-release-meta-llama__Llama-3.1-8B-Instruct@llama31-8b-qb2-bh-qb-ge-p300x2"
        },
    ]

    def no_download(*_args):
        raise AssertionError("device should come from the job name, not the bundle")

    monkeypatch.setattr(
        "scripts.release.build_release_artifacts.download_artifact", no_download
    )
    chosen = resolve_model(
        "meta-llama/Llama-3.1-8B-Instruct",
        ["p300x2"],
        artifacts,
        jobs,
        "org/repo",
        tmp_path,
        {},
    )
    assert chosen == {"p300x2": artifacts[0]}


def test_device_from_jobs_keeps_default_and_impl_jobs_apart():
    model = "meta-llama/Llama-3.1-8B-Instruct"
    jobs = [
        {
            "name": "run-release-meta-llama__Llama-3.1-8B-Instruct@llama31-8b-qb2-bh-qb-ge-p300x2"
        },
        {"name": "run-release-meta-llama__Llama-3.1-8B-Instruct-bh-qb-ge-p300"},
    ]
    assert device_from_jobs(jobs, model, "bh-qb-ge", impl="llama31-8b-qb2") == "p300x2"
    assert device_from_jobs(jobs, model, "bh-qb-ge") == "p300"


def test_device_from_jobs_reads_runs_from_before_leaf_job_names():
    """Before tt-shield named jobs ``<model>@<impl>``, an impl's job carried the
    bare model. The logs artifact already proves which impl ran, and the runner
    label pins its job, so such a run still resolves."""
    model = "meta-llama/Llama-3.1-8B-Instruct"
    legacy = [
        {
            "name": "_ / vLLM / run-training_tests-meta-llama__Llama-3.1-8B-Instruct-bh-qb-ge-p300x2"
        }
    ]
    assert (
        device_from_jobs(
            legacy, model, "bh-qb-ge", "training_tests", impl="trainer-training-lora"
        )
        == "p300x2"
    )
    # A run that already uses leaf names never falls back to the bare-model job.
    current = legacy + [
        {"name": "run-release-meta-llama__Llama-3.1-8B-Instruct@other-bh-qb-ge-p300"}
    ]
    assert (
        device_from_jobs(
            current, model, "bh-qb-ge", "training_tests", impl="trainer-training-lora"
        )
        is None
    )
