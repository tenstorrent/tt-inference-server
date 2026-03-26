import json

import pytest

import scripts.release.release_dispatch_inputs as dispatch_inputs


def _make_release_diff_record(
    *,
    template_key="template-1",
    model_arch="Llama-3.2-1B",
    inference_engine="vLLM",
    weights=None,
    devices=None,
    tt_metal_commit_after="metal-sha",
    vllm_commit_after="vllm-sha",
):
    return {
        "template_key": template_key,
        "impl": "tt-transformers",
        "impl_id": "tt_transformers",
        "model_arch": model_arch,
        "inference_engine": inference_engine,
        "weights": weights
        or ["meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct"],
        "devices": devices or ["N150", "N300", "T3K"],
        "status_before": "FUNCTIONAL",
        "status_after": "FUNCTIONAL",
        "tt_metal_commit_before": "old-metal-sha",
        "tt_metal_commit_after": tt_metal_commit_after,
        "vllm_commit_before": "old-vllm-sha",
        "vllm_commit_after": vllm_commit_after,
        "ci_job_url": None,
        "ci_run_number": None,
    }


def _write_release_diff_json(path, records):
    path.write_text(json.dumps(records), encoding="utf-8")


def _write_models_ci_config(path, models):
    path.write_text(json.dumps({"models": models}), encoding="utf-8")


def test_resolve_release_workflow_refs_reads_unique_after_values(tmp_path):
    release_diff_path = tmp_path / "pre_release_models_diff.json"
    _write_release_diff_json(
        release_diff_path,
        [
            _make_release_diff_record(template_key="template-a", devices=["N150"]),
            _make_release_diff_record(template_key="template-b", devices=["N300"]),
        ],
    )

    assert dispatch_inputs.resolve_release_workflow_refs(release_diff_path) == (
        "metal-sha",
        "vllm-sha",
    )


def test_load_release_diff_records_raises_on_missing_file(tmp_path):
    release_diff_path = tmp_path / "pre_release_models_diff.json"

    with pytest.raises(FileNotFoundError, match="Pre-release diff JSON not found"):
        dispatch_inputs.load_release_diff_records(release_diff_path)


def test_resolve_release_workflow_refs_raises_on_ambiguous_values(tmp_path):
    release_diff_path = tmp_path / "pre_release_models_diff.json"
    _write_release_diff_json(
        release_diff_path,
        [
            _make_release_diff_record(
                template_key="template-a", tt_metal_commit_after="metal-a"
            ),
            _make_release_diff_record(
                template_key="template-b", tt_metal_commit_after="metal-b"
            ),
        ],
    )

    with pytest.raises(ValueError, match="multiple tt_metal_commit_after values"):
        dispatch_inputs.resolve_release_workflow_refs(release_diff_path)


def test_resolve_release_workflow_refs_raises_on_empty_diff(tmp_path):
    release_diff_path = tmp_path / "pre_release_models_diff.json"
    _write_release_diff_json(release_diff_path, [])

    with pytest.raises(
        ValueError, match="does not contain any non-empty tt_metal_commit_after"
    ):
        dispatch_inputs.resolve_release_workflow_refs(release_diff_path)


def test_collect_release_devices_by_config_entry_matches_unique_weight_name(tmp_path):
    release_diff_path = tmp_path / "pre_release_models_diff.json"
    models_ci_config_path = tmp_path / "models-ci-config.json"
    _write_release_diff_json(release_diff_path, [_make_release_diff_record()])
    _write_models_ci_config(
        models_ci_config_path,
        {
            "Llama-3.2-1B-Instruct": {
                "inference_engine": "vLLM",
                "ci": {
                    "nightly": {"devices": ["N150"]},
                    "release": {"devices": ["N150", "N300", "T3K"]},
                },
            }
        },
    )

    assert dispatch_inputs.collect_release_devices_by_config_entry(
        release_diff_path, models_ci_config_path
    ) == {"Llama-3.2-1B-Instruct": ["N150", "N300", "T3K"]}


def test_collect_release_devices_by_config_entry_raises_on_ambiguous_match(tmp_path):
    release_diff_path = tmp_path / "pre_release_models_diff.json"
    models_ci_config_path = tmp_path / "models-ci-config.json"
    _write_release_diff_json(release_diff_path, [_make_release_diff_record()])
    _write_models_ci_config(
        models_ci_config_path,
        {
            "Llama-3.2-1B": {
                "inference_engine": "vLLM",
                "ci": {"nightly": {"devices": ["N150"]}},
            },
            "Llama-3.2-1B-Instruct": {
                "inference_engine": "vLLM",
                "ci": {"nightly": {"devices": ["N150"]}},
            },
        },
    )

    with pytest.raises(
        ValueError, match="Failed to match exactly one Models CI config entry"
    ):
        dispatch_inputs.collect_release_devices_by_config_entry(
            release_diff_path, models_ci_config_path
        )


def test_prune_release_models_ci_config_removes_unmatched_release_entries(tmp_path):
    release_diff_path = tmp_path / "pre_release_models_diff.json"
    models_ci_config_path = tmp_path / "models-ci-config.json"
    _write_release_diff_json(
        release_diff_path,
        [_make_release_diff_record(devices=["N150", "T3K"])],
    )
    _write_models_ci_config(
        models_ci_config_path,
        {
            "Llama-3.2-1B-Instruct": {
                "inference_engine": "vLLM",
                "ci": {
                    "nightly": {"devices": ["N150", "N300", "T3K"]},
                    "release": {
                        "devices": ["N150", "N300", "T3K"],
                        "device-args": {
                            "N150": {"additional-args": "--foo"},
                            "N300": {"additional-args": "--bar"},
                        },
                    },
                },
            },
            "Qwen3-8B": {
                "inference_engine": "vLLM",
                "ci": {
                    "nightly": {"devices": ["GALAXY"]},
                    "release": {"devices": ["GALAXY"]},
                },
            },
        },
    )

    assert dispatch_inputs.prune_release_models_ci_config(
        release_diff_path, models_ci_config_path
    ) == {"Llama-3.2-1B-Instruct": ["N150", "T3K"]}

    config = json.loads(models_ci_config_path.read_text(encoding="utf-8"))
    assert config["models"]["Llama-3.2-1B-Instruct"]["ci"]["release"] == {
        "devices": ["N150", "T3K"],
        "device-args": {"N150": {"additional-args": "--foo"}},
    }
    assert "release" not in config["models"]["Qwen3-8B"]["ci"]


def test_validate_release_models_ci_config_accepts_exact_release_devices(tmp_path):
    release_diff_path = tmp_path / "pre_release_models_diff.json"
    models_ci_config_path = tmp_path / "models-ci-config.json"
    _write_release_diff_json(release_diff_path, [_make_release_diff_record()])
    _write_models_ci_config(
        models_ci_config_path,
        {
            "Llama-3.2-1B-Instruct": {
                "inference_engine": "vLLM",
                "ci": {
                    "nightly": {"devices": ["N150", "N300", "T3K"]},
                    "release": {"devices": ["N150", "N300", "T3K"]},
                },
            }
        },
    )

    assert dispatch_inputs.validate_release_models_ci_config(
        release_diff_path, models_ci_config_path
    ) == {"Llama-3.2-1B-Instruct": ["N150", "N300", "T3K"]}


def test_validate_release_models_ci_config_raises_on_unmatched_release_entry(tmp_path):
    release_diff_path = tmp_path / "pre_release_models_diff.json"
    models_ci_config_path = tmp_path / "models-ci-config.json"
    _write_release_diff_json(release_diff_path, [_make_release_diff_record()])
    _write_models_ci_config(
        models_ci_config_path,
        {
            "Llama-3.2-1B-Instruct": {
                "inference_engine": "vLLM",
                "ci": {
                    "nightly": {"devices": ["N150", "N300", "T3K"]},
                    "release": {"devices": ["N150", "N300", "T3K"]},
                },
            },
            "Qwen3-8B": {
                "inference_engine": "vLLM",
                "ci": {
                    "nightly": {"devices": ["GALAXY"]},
                    "release": {"devices": ["GALAXY"]},
                },
            },
        },
    )

    with pytest.raises(ValueError, match="unexpected `ci.release` entries"):
        dispatch_inputs.validate_release_models_ci_config(
            release_diff_path, models_ci_config_path
        )


def test_validate_release_models_ci_config_raises_on_device_mismatch(tmp_path):
    release_diff_path = tmp_path / "pre_release_models_diff.json"
    models_ci_config_path = tmp_path / "models-ci-config.json"
    _write_release_diff_json(release_diff_path, [_make_release_diff_record()])
    _write_models_ci_config(
        models_ci_config_path,
        {
            "Llama-3.2-1B-Instruct": {
                "inference_engine": "vLLM",
                "ci": {
                    "nightly": {"devices": ["N150", "N300", "T3K"]},
                    "release": {"devices": ["N150", "N300"]},
                },
            }
        },
    )

    with pytest.raises(ValueError, match="do not match"):
        dispatch_inputs.validate_release_models_ci_config(
            release_diff_path, models_ci_config_path
        )
