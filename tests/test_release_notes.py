import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from scripts.release.generate_release_notes import build_release_notes
from scripts.release.generate_release_notes import main
from scripts.release.release_diff import build_template_key
from scripts.release.release_performance import build_release_performance_diff_data
from scripts.release.release_paths import (
    get_versioned_release_logs_dir,
    resolve_release_output_dir,
)


def test_release_paths_use_versioned_release_logs():
    assert get_versioned_release_logs_dir("0.10.0").as_posix() == "release_logs/v0.10.0"
    assert (
        resolve_release_output_dir(version="0.10.0")
        .as_posix()
        .endswith("/release_logs/v0.10.0")
    )


def make_release_diff_record(
    *, model_arch="DemoModel", devices=None, weights=None, impl_id="demo_impl"
):
    devices = devices or ["N150"]
    weights = weights or ["demo/model"]
    return {
        "template_key": build_template_key(impl_id, weights, devices, "vllm"),
        "impl": "demo-impl",
        "impl_id": impl_id,
        "model_arch": model_arch,
        "inference_engine": "vllm",
        "weights": weights,
        "devices": devices,
        "status_before": "ModelStatusTypes.EXPERIMENTAL",
        "status_after": "ModelStatusTypes.COMPLETE",
        "tt_metal_commit_before": "aaaaaaa",
        "tt_metal_commit_after": "bbbbbbb",
        "vllm_commit_before": None,
        "vllm_commit_after": None,
        "ci_job_url": "https://example.com/jobs/456",
        "ci_run_number": 123,
    }


def make_release_performance_data(
    *, perf_status="functional", ttft=45.0, test_status="passed"
):
    return {
        "schema_version": "0.1.0",
        "models": {
            "DemoModel": {
                "n150": {
                    "demo_impl": {
                        "vLLM": {
                            "perf_status": perf_status,
                            "ci_run_number": 123,
                            "ci_run_url": "https://example.com/runs/123",
                            "ci_job_url": "https://example.com/jobs/456",
                            "perf_target_results": [
                                {
                                    "is_summary_data_point": True,
                                    "config": {
                                        "task_type": "text",
                                        "isl": 128,
                                        "osl": 128,
                                        "max_concurrency": 1,
                                    },
                                    "targets": {
                                        "ttft_ms": 50.0,
                                        "ttft_streaming_ms": None,
                                        "tput_user": 10.0,
                                        "tput_prefill": None,
                                        "e2el_ms": None,
                                        "tput": None,
                                        "rtr": None,
                                        "tolerance": 0.05,
                                    },
                                    "measured_metrics": {
                                        "ttft": ttft,
                                        "tput_user": None,
                                        "tput": None,
                                        "ttft_streaming_ms": None,
                                        "tput_prefill": None,
                                        "e2el_ms": None,
                                        "rtr": None,
                                    },
                                    "benchmark_summary": {
                                        "task_type": "text",
                                        "isl": 128,
                                        "osl": 128,
                                        "max_concurrency": 1,
                                        "ttft": ttft,
                                    },
                                }
                            ],
                        }
                    }
                }
            }
        },
    }


def make_artifacts_summary_data():
    return {
        "generated_artifacts": [
            "release_model_spec.json",
            "docs/model_support/",
            "README.md",
        ],
        "built_images": ["ghcr.io/tenstorrent/built:tag"],
        "acceptance_warnings": [
            {
                "heading": "DemoModel on N150",
                "summary_markdown": "### Acceptance Criteria\n\n- Acceptance status: `FAIL`",
            }
        ],
        "images_to_build": ["ghcr.io/tenstorrent/build:tag"],
        "copied_images": {
            "ghcr.io/tenstorrent/release:tag": "ghcr.io/tenstorrent/ci:tag"
        },
        "existing_with_ci_ref": {
            "ghcr.io/tenstorrent/existing:tag": "ghcr.io/tenstorrent/existing-ci:tag"
        },
        "existing_without_ci_ref": ["ghcr.io/tenstorrent/manual:tag"],
    }


def make_performance_entry(
    model,
    device,
    *,
    perf_status="functional",
    ttft=45.0,
    test_status="passed",
):
    return {
        "perf_status": perf_status,
        "ci_run_number": 123,
        "ci_run_url": "https://example.com/runs/123",
        "ci_job_url": "https://example.com/jobs/456",
        "perf_target_results": [
            {
                "is_summary_data_point": True,
                "config": {
                    "task_type": "text",
                    "isl": 128,
                    "osl": 128,
                    "max_concurrency": 1,
                },
                "targets": {
                    "ttft_ms": 50.0,
                    "ttft_streaming_ms": None,
                    "tput_user": 10.0,
                    "tput_prefill": None,
                    "e2el_ms": None,
                    "tput": None,
                    "rtr": None,
                    "tolerance": 0.05,
                },
                "measured_metrics": {
                    "ttft": ttft,
                    "tput_user": None,
                    "tput": None,
                    "ttft_streaming_ms": None,
                    "tput_prefill": None,
                    "e2el_ms": None,
                    "rtr": None,
                },
                "benchmark_summary": {
                    "task_type": "text",
                    "isl": 128,
                    "osl": 128,
                    "max_concurrency": 1,
                    "ttft": ttft,
                },
            }
        ],
    }


def test_build_release_notes_renders_raw_inputs_and_performance_diff():
    notes = build_release_notes(
        version="0.10.0",
        model_diff_records=[make_release_diff_record()],
        artifacts_summary_data=make_artifacts_summary_data(),
        release_performance_data=make_release_performance_data(
            perf_status="target", ttft=52.0, test_status="failed"
        ),
        base_release_performance_data=make_release_performance_data(),
    )

    assert notes.startswith("# tt-inference-server v0.10.0\n")
    assert "## Summary of Changes\n" in notes
    assert "## Recommended system software versions\n" in notes
    assert "## Known Issues\n" in notes
    assert (
        "## Model and Hardware Support Diff\n\nThis document shows model specification updates."
        in notes
    )
    assert "| Performance Diff |" in notes
    assert "N150: Perf status: functional -> target; Perf targets ~1" in notes
    assert "## Performance\n" in notes
    assert "### DemoModel on n150 (demo_impl, vLLM)" in notes
    assert "## Scale Out\n" in notes
    assert "## Deprecations and breaking changes\n" in notes
    assert "### Generated Release Artifacts" in notes
    assert "### Docker Images Built Locally" in notes
    assert "### Release Acceptance Warnings" in notes
    assert "## Release Artifacts Summary\n" in notes
    assert "### Images Promoted from Models CI" in notes
    assert "## Contributors\n" in notes
    assert "## Assets\n" in notes
    assert "https://ghcr.io/tenstorrent/release:tag" in notes
    assert "https://ghcr.io/tenstorrent/build:tag" in notes
    assert "https://ghcr.io/tenstorrent/built:tag" in notes


def test_build_release_performance_diff_data_classifies_all_diff_states():
    release_diff_records = [
        make_release_diff_record(model_arch="ChangedModel", devices=["N150"]),
        make_release_diff_record(model_arch="NewModel", devices=["N150"]),
        make_release_diff_record(model_arch="RemovedModel", devices=["N150"]),
        make_release_diff_record(model_arch="SameModel", devices=["N150"]),
    ]
    current_release_performance_data = {
        "schema_version": "0.1.0",
        "models": {
            "ChangedModel": {
                "n150": {
                    "demo_impl": {
                        "vLLM": make_performance_entry(
                            "ChangedModel",
                            "n150",
                            perf_status="target",
                            ttft=52.0,
                            test_status="failed",
                        )
                    }
                }
            },
            "NewModel": {
                "n150": {
                    "demo_impl": {
                        "vLLM": make_performance_entry(
                            "NewModel", "n150", perf_status="target", ttft=60.0
                        )
                    }
                }
            },
            "SameModel": {
                "n150": {
                    "demo_impl": {"vLLM": make_performance_entry("SameModel", "n150")}
                }
            },
        },
    }
    base_release_performance_data = {
        "schema_version": "0.1.0",
        "models": {
            "ChangedModel": {
                "n150": {
                    "demo_impl": {
                        "vLLM": make_performance_entry("ChangedModel", "n150")
                    }
                }
            },
            "RemovedModel": {
                "n150": {
                    "demo_impl": {
                        "vLLM": make_performance_entry("RemovedModel", "n150")
                    }
                }
            },
            "SameModel": {
                "n150": {
                    "demo_impl": {"vLLM": make_performance_entry("SameModel", "n150")}
                }
            },
        },
    }

    diff_data = build_release_performance_diff_data(
        release_diff_records=release_diff_records,
        release_performance_data=current_release_performance_data,
        base_release_performance_data=base_release_performance_data,
    )

    assert [record["diff_status"] for record in diff_data["records"]] == [
        "changed",
        "new",
        "removed",
        "unchanged",
    ]
    assert [record["summary"] for record in diff_data["records"]] == [
        "Perf status: functional -> target; Perf targets ~1",
        "New release data",
        "Removed release data",
        "No performance changes",
    ]


def test_build_release_performance_diff_data_filters_real_release_slice():
    release_diff_path = (
        Path(__file__).resolve().parent.parent
        / "release_logs"
        / "v0.12.0"
        / "pre_release_models_diff.json"
    )
    release_diff_records = json.loads(release_diff_path.read_text())
    current_release_performance_data = {
        "schema_version": "0.1.0",
        "models": {
            "Llama-3.2-1B": {
                "n150": {
                    "tt_transformers": {
                        "vLLM": make_performance_entry(
                            "Llama-3.2-1B", "n150", perf_status="target", ttft=52.0
                        )
                    }
                },
                "n300": {
                    "tt_transformers": {
                        "vLLM": make_performance_entry(
                            "Llama-3.2-1B", "n300", perf_status="target", ttft=53.0
                        )
                    }
                },
                "t3k": {
                    "tt_transformers": {
                        "vLLM": make_performance_entry(
                            "Llama-3.2-1B", "t3k", perf_status="target", ttft=54.0
                        )
                    }
                },
            },
            "UnrelatedModel": {
                "n150": {
                    "demo_impl": {
                        "vLLM": make_performance_entry("UnrelatedModel", "n150")
                    }
                }
            },
        },
    }
    base_release_performance_data = {
        "schema_version": "0.1.0",
        "models": {
            "Llama-3.2-1B": {
                "n150": {
                    "tt_transformers": {
                        "vLLM": make_performance_entry("Llama-3.2-1B", "n150")
                    }
                },
                "n300": {
                    "tt_transformers": {
                        "vLLM": make_performance_entry("Llama-3.2-1B", "n300")
                    }
                },
                "t3k": {
                    "tt_transformers": {
                        "vLLM": make_performance_entry("Llama-3.2-1B", "t3k")
                    }
                },
            }
        },
    }
    for device in ("n150", "n300", "t3k"):
        current_release_performance_data["models"]["Llama-3.2-1B"][device][
            "tt_transformers"
        ]["vLLM"]["impl_id"] = "tt_transformers"
        base_release_performance_data["models"]["Llama-3.2-1B"][device][
            "tt_transformers"
        ]["vLLM"]["impl_id"] = "tt_transformers"

    diff_data = build_release_performance_diff_data(
        release_diff_records=release_diff_records,
        release_performance_data=current_release_performance_data,
        base_release_performance_data=base_release_performance_data,
    )

    assert [record["device"] for record in diff_data["records"]] == [
        "N150",
        "N300",
        "T3K",
    ]
    assert {record["model_arch"] for record in diff_data["records"]} == {"Llama-3.2-1B"}


def test_main_reads_raw_release_inputs_from_release_dir(tmp_path):
    version_file = tmp_path / "VERSION"
    version_file.write_text("0.10.0\n")

    release_dir = tmp_path / "release_logs" / "v0.10.0"
    release_dir.mkdir(parents=True)
    (release_dir / "pre_release_models_diff.json").write_text(
        json.dumps([make_release_diff_record()])
    )
    (release_dir / "release_artifacts_summary.json").write_text(
        json.dumps(make_artifacts_summary_data())
    )
    current_performance_path = tmp_path / "release_performance.json"
    current_performance_path.write_text(
        json.dumps(
            make_release_performance_data(
                perf_status="target", ttft=52.0, test_status="failed"
            )
        )
    )
    base_performance_path = tmp_path / "release_performance_base.json"
    base_performance_path.write_text(json.dumps(make_release_performance_data()))
    output_path = release_dir / "release_notes_v0.10.0.md"

    args = Namespace(
        version=None,
        version_file=str(version_file),
        artifacts_summary_json=None,
        model_diff_json=None,
        release_performance_json=str(current_performance_path),
        base_release_performance_json=str(base_performance_path),
        output=str(output_path),
    )

    with patch(
        "scripts.release.generate_release_notes.parse_args",
        return_value=args,
    ), patch(
        "scripts.release.generate_release_notes.get_versioned_release_logs_dir",
        return_value=release_dir,
    ):
        assert main() == 0

    notes = output_path.read_text()
    assert (
        "## Model and Hardware Support Diff\n\nThis document shows model specification updates."
        in notes
    )
    assert "N150: Perf status: functional -> target; Perf targets ~1" in notes
    assert "## Performance\n" in notes
