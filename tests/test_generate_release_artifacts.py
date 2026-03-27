import json
import subprocess
import sys
from argparse import Namespace
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch

import scripts.release.generate_release_artifacts as gra
from scripts.release.release_diff import build_template_key
from workflows.perf_targets import PerfTarget, PerfTargetSet


TT_METAL_COMMIT = "a" * 40
OTHER_TT_METAL_COMMIT = "b" * 40
VLLM_COMMIT = "1" * 7
OTHER_VLLM_COMMIT = "2" * 7


def make_image(
    name,
    *,
    channel="release",
    tt_metal_commit=TT_METAL_COMMIT,
    vllm_commit=VLLM_COMMIT,
    timestamp="123456",
):
    return (
        f"ghcr.io/tenstorrent/{name}-{channel}-image:"
        f"0.10.0-{tt_metal_commit}-{vllm_commit}-{timestamp}"
    )


def make_model_spec(
    docker_image,
    *,
    tt_metal_commit=TT_METAL_COMMIT,
    vllm_commit=VLLM_COMMIT,
    model_name="DemoModel",
    device_name="N150",
    impl_id="demo_impl",
    impl_name="demo-impl",
    inference_engine="vLLM",
):
    return SimpleNamespace(
        docker_image=docker_image,
        tt_metal_commit=tt_metal_commit,
        vllm_commit=vllm_commit,
        model_name=model_name,
        device_type=SimpleNamespace(name=device_name),
        impl=SimpleNamespace(impl_id=impl_id, impl_name=impl_name),
        inference_engine=inference_engine,
    )


def make_record(model_id, target_image, *, ci_image=None, model_spec=None):
    return gra.MergedModelRecord(
        model_id=model_id,
        model_spec=model_spec or make_model_spec(target_image),
        ci_data={"docker_image": ci_image} if ci_image else {},
        target_docker_image=target_image,
    )


def make_release_diff_record(*, model_arch="DemoModel", devices=None):
    devices = devices or ["N150"]
    return {
        "template_key": build_template_key(
            "demo_impl", ["demo/model"], devices, "vllm"
        ),
        "impl": "demo-impl",
        "impl_id": "demo_impl",
        "model_arch": model_arch,
        "inference_engine": "vllm",
        "weights": ["demo/model"],
        "devices": devices,
        "status_before": "EXPERIMENTAL",
        "status_after": "COMPLETE",
        "tt_metal_commit_before": "aaaaaaa",
        "tt_metal_commit_after": "bbbbbbb",
        "vllm_commit_before": None,
        "vllm_commit_after": None,
        "ci_job_url": "https://example.com/jobs/456",
        "ci_run_number": 123,
    }


def make_raw_ci_entry(
    *,
    run_number=123,
    docker_image=None,
    release_report=None,
    ci_job_url="https://example.com/jobs/456",
):
    return {
        "job_run_datetimestamp": "2026-03-27_10-00-00",
        "ci_metadata": {
            "run_number": run_number,
            "ci_run_url": "https://example.com/runs/123",
            "ci_job_metadata": {
                "job_id": 456,
                "job_url": ci_job_url,
                "job_name": "demo-job",
                "job_status": "completed",
                "job_conclusion": "success",
                "completed_at": "2026-03-27T10:00:00Z",
            },
        },
        "ci_logs": {"firmware_bundle": "fw", "kmd_version": "kmd"},
        "workflow_logs": {
            "summary": {
                "docker_image": docker_image or make_image("demo-ci", channel="dev"),
                "tt_metal_commit": TT_METAL_COMMIT,
                "vllm_commit": VLLM_COMMIT,
                "perf_status": "target",
                "benchmarks_completed": True,
                "accuracy_status": True,
                "evals_completed": True,
                "regression_checked": True,
                "regression_passed": True,
                "regression_ok": True,
                "is_passing": True,
            },
            "reports_output": release_report or {},
        },
    }


def make_release_report(*, functional_status="target", parameter_status="passed"):
    return {
        "metadata": {"model_name": "DemoModel", "device": "n150"},
        "benchmarks_summary": [
            {
                "task_type": "text",
                "isl": 128,
                "osl": 128,
                "max_concurrency": 1,
                "ttft": 45.0,
                "tput_user": 12.0,
            }
        ],
        "evals": [{"task_name": "demo_eval", "accuracy_check": 2, "score": 1.0}],
        "parameter_support_tests": {
            "results": {
                "test_smoke": [
                    {
                        "status": parameter_status,
                        "test_node_name": "test_smoke[param]",
                        "message": "",
                    }
                ]
            }
        },
        "benchmark_target_evaluation": {
            "status": functional_status,
            "reference_available": True,
            "support_levels": {
                "functional": {"checked": True, "passed": True, "failures": []},
                "complete": {"checked": True, "passed": True, "failures": []},
                "target": {"checked": True, "passed": True, "failures": []},
            },
            "regression": {"checked": True, "passed": True, "failures": []},
            "next_status": None,
            "next_status_failures": [],
            "errors": [],
        },
    }


def test_merge_specs_with_ci_data_derives_dev_image_without_mutating_model_specs():
    release_image = make_image("demo")
    model_spec = make_model_spec(release_image)
    ci_data = {"demo_model": {"docker_image": "ci-image"}}

    with patch.object(gra, "MODEL_SPECS", {"demo_model": model_spec}):
        merged = gra.merge_specs_with_ci_data(ci_data, is_dev=True)

    assert model_spec.docker_image == release_image
    assert merged["demo_model"].target_docker_image == release_image.replace(
        "-release-", "-dev-"
    )
    assert merged["demo_model"].ci_data == {"docker_image": "ci-image"}


def test_build_ci_data_from_raw_results_uses_latest_entry():
    older_entry = make_raw_ci_entry(
        run_number=100, ci_job_url="https://example.com/jobs/100"
    )
    newer_entry = make_raw_ci_entry(
        run_number=200, ci_job_url="https://example.com/jobs/200"
    )

    ci_data = gra.build_ci_data_from_raw_results(
        {"demo_model": [older_entry, newer_entry]}
    )

    assert ci_data["demo_model"]["ci_run_number"] == 200
    assert ci_data["demo_model"]["ci_job_url"] == "https://example.com/jobs/200"


def test_build_acceptance_warnings_logs_warning_only():
    target_image = make_image("demo")
    failing_report = make_release_report(parameter_status="failed")
    merged_spec = {
        "demo_model": gra.MergedModelRecord(
            model_id="demo_model",
            model_spec=make_model_spec(target_image),
            ci_data={
                "release_report": failing_report,
                "ci_job_url": "https://example.com/jobs/456",
            },
            target_docker_image=target_image,
        )
    }

    warnings = gra.build_acceptance_warnings(merged_spec)

    assert len(warnings) == 1
    assert warnings[0]["model_id"] == "demo_model"
    assert "Acceptance status: `FAIL`" in warnings[0]["summary_markdown"]


def test_extract_commits_from_tag_requires_expected_format():
    valid_image = make_image("demo")
    invalid_image = "ghcr.io/tenstorrent/demo-release-image:latest"

    assert gra.extract_commits_from_tag(valid_image) == (
        TT_METAL_COMMIT,
        VLLM_COMMIT,
    )
    assert gra.extract_commits_from_tag(invalid_image) is None


def test_run_registry_command_returns_none_on_timeout():
    with patch("scripts.release.generate_release_artifacts.subprocess.run") as run_mock:
        run_mock.side_effect = subprocess.TimeoutExpired(["docker"], timeout=30)

        result = gra.run_registry_command(["docker"], 30, "checking image")

    assert result is None


def test_copy_docker_image_returns_false_on_nonzero_exit():
    failed_process = subprocess.CompletedProcess(
        args=["crane", "copy"],
        returncode=1,
        stdout="",
        stderr="copy failed",
    )

    with patch.object(gra, "run_registry_command", return_value=failed_process):
        assert gra.copy_docker_image("src:image", "dst:image") is False


def test_write_output_writes_json_and_markdown(tmp_path):
    images_to_build = defaultdict(list, {"ghcr.io/tenstorrent/build:tag": ["model-a"]})
    copied_images = {"ghcr.io/tenstorrent/release:tag": "ghcr.io/tenstorrent/ci:tag"}
    existing_with_ci_ref = {
        "ghcr.io/tenstorrent/existing:tag": "ghcr.io/tenstorrent/existing-ci:tag"
    }
    existing_without_ci_ref = defaultdict(
        list, {"ghcr.io/tenstorrent/manual:tag": ["model-b"]}
    )

    summary = gra.write_output(
        images_to_build,
        copied_images,
        existing_with_ci_ref,
        existing_without_ci_ref,
        tmp_path,
        "release",
    )

    assert summary["summary"] == {
        "total_to_build": 1,
        "total_copied": 1,
        "total_existing_with_ci": 1,
        "total_existing_without_ci": 1,
        "total_generated_artifacts": 0,
        "total_acceptance_warnings": 0,
    }
    assert (tmp_path / "release_artifacts_summary.json").exists()
    markdown = (tmp_path / "release_artifacts_summary.md").read_text()
    assert "Generated Release Artifacts" in markdown
    assert "Release Acceptance Warnings" in markdown
    assert "Images Promoted from Models CI" in markdown
    assert "https://ghcr.io/tenstorrent/release:tag" in markdown
    assert "https://ghcr.io/tenstorrent/manual:tag" in markdown


def test_generate_release_artifacts_groups_shared_existing_image_with_ci():
    target_image = make_image("demo")
    ci_image = make_image("demo-ci", channel="dev")
    merged_spec = {
        "model-a": make_record("model-a", target_image, ci_image=ci_image),
        "model-b": make_record("model-b", target_image),
    }

    with patch.object(
        gra,
        "check_image_exists",
        side_effect=lambda image, cache=None: True,
    ) as exists_mock, patch.object(gra, "copy_docker_image") as copy_mock:
        (
            images_to_build,
            unique_images_count,
            copied_images,
            existing_with_ci_ref,
            existing_without_ci_ref,
        ) = gra.generate_release_artifacts(merged_spec, dry_run=False)

    assert dict(images_to_build) == {}
    assert unique_images_count == 0
    assert copied_images == {}
    assert existing_with_ci_ref == {target_image: ci_image}
    assert dict(existing_without_ci_ref) == {}
    assert exists_mock.call_count == 2
    copy_mock.assert_not_called()


def test_generate_release_artifacts_copies_shared_image_once():
    target_image = make_image("demo")
    ci_image = make_image("demo-ci", channel="dev")
    merged_spec = {
        "model-a": make_record("model-a", target_image, ci_image=ci_image),
        "model-b": make_record("model-b", target_image),
    }
    image_exists = {target_image: False, ci_image: True}

    with patch.object(
        gra,
        "check_image_exists",
        side_effect=lambda image, cache=None: image_exists[image],
    ), patch.object(gra, "copy_docker_image", return_value=True) as copy_mock:
        (
            images_to_build,
            unique_images_count,
            copied_images,
            existing_with_ci_ref,
            existing_without_ci_ref,
        ) = gra.generate_release_artifacts(merged_spec, dry_run=False)

    assert dict(images_to_build) == {}
    assert unique_images_count == 0
    assert copied_images == {target_image: ci_image}
    assert existing_with_ci_ref == {}
    assert dict(existing_without_ci_ref) == {}
    copy_mock.assert_called_once_with(ci_image, target_image, False)


def test_generate_release_artifacts_marks_shared_image_for_build_on_commit_mismatch():
    target_image = make_image("demo")
    ci_image = make_image(
        "demo-ci",
        channel="dev",
        tt_metal_commit=OTHER_TT_METAL_COMMIT,
        vllm_commit=OTHER_VLLM_COMMIT,
    )
    shared_spec = make_model_spec(target_image)
    merged_spec = {
        "model-a": make_record(
            "model-a",
            target_image,
            ci_image=ci_image,
            model_spec=shared_spec,
        ),
        "model-b": make_record(
            "model-b",
            target_image,
            model_spec=shared_spec,
        ),
    }
    image_exists = {target_image: False, ci_image: True}

    with patch.object(
        gra,
        "check_image_exists",
        side_effect=lambda image, cache=None: image_exists[image],
    ), patch.object(gra, "copy_docker_image") as copy_mock:
        (
            images_to_build,
            unique_images_count,
            copied_images,
            existing_with_ci_ref,
            existing_without_ci_ref,
        ) = gra.generate_release_artifacts(merged_spec, dry_run=False)

    assert dict(images_to_build) == {target_image: ["model-a", "model-b"]}
    assert unique_images_count == 1
    assert copied_images == {}
    assert existing_with_ci_ref == {}
    assert dict(existing_without_ci_ref) == {}
    copy_mock.assert_not_called()


def test_write_release_performance_outputs_returns_raw_data_and_writes_baseline(
    tmp_path,
):
    output_dir = tmp_path / "release_logs" / "v0.10.0"
    output_dir.mkdir(parents=True)
    target_image = make_image("demo")
    model_spec = make_model_spec(target_image)
    merged_spec = {
        "model-a": gra.MergedModelRecord(
            model_id="model-a",
            model_spec=model_spec,
            ci_data={
                "perf_status": "target",
                "accuracy_status": True,
                "ci_run_number": 123,
                "ci_run_url": "https://example.com/runs/123",
                "ci_job_url": "https://example.com/jobs/456",
                "release_report": {
                    "benchmarks_summary": [
                        {
                            "task_type": "text",
                            "isl": 128,
                            "osl": 128,
                            "max_concurrency": 1,
                            "ttft": 45.0,
                            "tput_user": 12.0,
                            "target_checks": {
                                "target": {
                                    "ttft_check": 2,
                                    "tput_user_check": 2,
                                    "ttft": 50.0,
                                    "tput_user": 10.0,
                                }
                            },
                        }
                    ],
                    "benchmark_target_evaluation": {
                        "status": "target",
                        "reference_available": True,
                        "support_levels": {
                            "functional": {
                                "checked": True,
                                "passed": True,
                                "failures": [],
                            },
                            "complete": {
                                "checked": True,
                                "passed": True,
                                "failures": [],
                            },
                            "target": {"checked": True, "passed": True, "failures": []},
                        },
                        "regression": {
                            "checked": False,
                            "passed": True,
                            "failures": [],
                        },
                        "next_status": None,
                        "next_status_failures": [],
                        "errors": [],
                    },
                    "parameter_support_tests": {
                        "endpoint_url": "http://localhost:8000",
                        "test_run_timestamp_utc": "2026-03-11T12:00:00",
                        "results": {
                            "test_smoke": [
                                {
                                    "status": "passed",
                                    "test_node_name": "test_smoke[param]",
                                    "message": "",
                                }
                            ]
                        },
                    },
                },
            },
            target_docker_image=target_image,
        )
    }
    baseline_path = (
        tmp_path / "benchmarking" / "benchmark_targets" / "release_performance.json"
    )

    perf_target = PerfTargetSet(
        model_name="DemoModel",
        device=SimpleNamespace(name="N150"),
        perf_targets=[
            PerfTarget(
                isl=128,
                osl=128,
                max_concurrency=1,
                task_type="text",
                ttft_ms=50.0,
                tput_user=10.0,
                tput=12.0,
                is_summary=True,
            )
        ],
    )

    release_performance_module = sys.modules[
        gra.build_release_performance_data.__module__
    ]

    with patch.object(
        gra, "get_release_performance_path", return_value=baseline_path
    ), patch.object(
        release_performance_module, "get_perf_target", return_value=perf_target
    ):
        release_performance_data = gra.write_release_performance_outputs(
            merged_spec, output_dir, dry_run=False
        )

    entry = release_performance_data["models"]["DemoModel"]["n150"]["demo_impl"]["vLLM"]
    assert entry["benchmarks_summary"][0]["ttft"] == 45.0
    assert not (output_dir / "release_performance.md").exists()
    baseline = json.loads(baseline_path.read_text())
    baseline_entry = baseline["models"]["DemoModel"]["n150"]["demo_impl"]["vLLM"]
    assert baseline_entry["benchmarks_summary"][0]["ttft"] == 45.0
    assert baseline_entry["benchmarks_summary"][0]["isl"] == 128
    assert baseline_entry["perf_target_results"][0]["config"]["isl"] == 128
    assert baseline_entry["perf_target_summary"]["measured_metrics"]["ttft"] == 45.0
    assert (
        baseline_entry["perf_target_summary"]["measured_metrics"]["tput_user"] == 12.0
    )
    assert baseline_entry["report_data"]["benchmarks_summary"][0]["ttft"] == 45.0
    assert (
        "test_smoke"
        in baseline_entry["report_data"]["parameter_support_tests"]["results"]
    )
    assert (
        baseline_entry["report_data"]["benchmark_target_evaluation"]["status"]
        == "target"
    )


def test_write_release_performance_diff_output_filters_to_release_models(tmp_path):
    output_dir = tmp_path / "release_logs" / "v0.10.0"
    output_dir.mkdir(parents=True)
    (output_dir / "pre_release_models_diff.json").write_text(
        json.dumps([make_release_diff_record()])
    )
    release_performance_data = {
        "schema_version": "0.1.0",
        "models": {
            "DemoModel": {
                "n150": {
                    "demo_impl": {
                        "vLLM": {
                            "model": "DemoModel",
                            "device": "n150",
                            "impl_id": "demo_impl",
                            "inference_engine": "vLLM",
                            "perf_status": "target",
                            "accuracy_status": True,
                            "benchmarks_summary": [
                                {
                                    "task_type": "text",
                                    "isl": 128,
                                    "osl": 128,
                                    "max_concurrency": 1,
                                    "ttft": 52.0,
                                }
                            ],
                            "report_data": {
                                "parameter_support_tests": {
                                    "results": {"test_smoke": [{"status": "failed"}]}
                                }
                            },
                        }
                    }
                }
            },
            "UnrelatedModel": {
                "n300": {
                    "demo_impl": {
                        "vLLM": {
                            "model": "UnrelatedModel",
                            "device": "n300",
                            "impl_id": "demo_impl",
                            "inference_engine": "vLLM",
                            "perf_status": "target",
                            "accuracy_status": True,
                            "benchmarks_summary": [],
                            "report_data": {},
                        }
                    }
                }
            },
        },
    }
    base_release_performance_data = {
        "schema_version": "0.1.0",
        "models": {
            "DemoModel": {
                "n150": {
                    "demo_impl": {
                        "vLLM": {
                            "model": "DemoModel",
                            "device": "n150",
                            "impl_id": "demo_impl",
                            "inference_engine": "vLLM",
                            "perf_status": "functional",
                            "accuracy_status": True,
                            "benchmarks_summary": [
                                {
                                    "task_type": "text",
                                    "isl": 128,
                                    "osl": 128,
                                    "max_concurrency": 1,
                                    "ttft": 45.0,
                                }
                            ],
                            "report_data": {
                                "parameter_support_tests": {
                                    "results": {"test_smoke": [{"status": "passed"}]}
                                }
                            },
                        }
                    }
                }
            }
        },
    }

    with patch.object(
        gra,
        "load_git_release_performance_data",
        return_value=base_release_performance_data,
    ), patch.object(
        gra,
        "get_release_performance_path",
        return_value=tmp_path
        / "benchmarking"
        / "benchmark_targets"
        / "release_performance.json",
    ):
        output_path = gra.write_release_performance_diff_output(
            output_dir, release_performance_data
        )

    diff_data = json.loads(output_path.read_text())
    assert diff_data["records"] == [
        {
            "after_entry": release_performance_data["models"]["DemoModel"]["n150"][
                "demo_impl"
            ]["vLLM"],
            "before_entry": base_release_performance_data["models"]["DemoModel"][
                "n150"
            ]["demo_impl"]["vLLM"],
            "device": "N150",
            "diff_status": "changed",
            "impl_id": "demo_impl",
            "inference_engine": "vllm",
            "model_arch": "DemoModel",
            "summary": "Perf status: functional -> target; Benchmarks ~1; LLM API tests ~1",
            "template_key": build_template_key(
                "demo_impl", ["demo/model"], ["N150"], "vllm"
            ),
            "weights": ["demo/model"],
        }
    ]


def test_write_release_performance_diff_output_requires_pre_release_diff_json(tmp_path):
    output_dir = tmp_path / "release_logs" / "v0.10.0"
    output_dir.mkdir(parents=True)

    try:
        gra.write_release_performance_diff_output(
            output_dir, {"schema_version": "0.1.0", "models": {}}
        )
    except FileNotFoundError as exc:
        assert "pre_release_models_diff.json" in str(exc)
    else:
        raise AssertionError(
            "Expected FileNotFoundError when pre-release diff is missing"
        )


def test_write_release_notes_uses_raw_json_inputs(tmp_path):
    output_dir = tmp_path / "release_logs" / "v0.10.0"
    output_dir.mkdir(parents=True)
    (output_dir / "pre_release_models_diff.json").write_text(
        json.dumps(
            [
                {
                    "template_key": build_template_key(
                        "demo_impl", ["demo/model"], ["N150"], "vllm"
                    ),
                    "impl": "demo-impl",
                    "impl_id": "demo_impl",
                    "model_arch": "DemoModel",
                    "inference_engine": "vllm",
                    "weights": ["demo/model"],
                    "devices": ["N150"],
                    "status_before": "EXPERIMENTAL",
                    "status_after": "COMPLETE",
                    "tt_metal_commit_before": "aaaaaaa",
                    "tt_metal_commit_after": "bbbbbbb",
                    "vllm_commit_before": None,
                    "vllm_commit_after": None,
                    "ci_job_url": "https://example.com/jobs/456",
                    "ci_run_number": 123,
                }
            ]
        )
    )
    (output_dir / "release_artifacts_summary.json").write_text(
        json.dumps(
            {
                "images_to_build": ["ghcr.io/tenstorrent/build:tag"],
                "copied_images": {
                    "ghcr.io/tenstorrent/release:tag": "ghcr.io/tenstorrent/ci:tag"
                },
                "existing_with_ci_ref": {},
                "existing_without_ci_ref": [],
            }
        )
    )
    release_performance_data = {
        "schema_version": "0.1.0",
        "models": {
            "DemoModel": {
                "n150": {
                    "demo_impl": {
                        "vLLM": {
                            "model": "DemoModel",
                            "device": "n150",
                            "impl_id": "demo_impl",
                            "inference_engine": "vLLM",
                            "perf_status": "target",
                            "accuracy_status": True,
                            "benchmarks_summary": [
                                {
                                    "task_type": "text",
                                    "isl": 128,
                                    "osl": 128,
                                    "max_concurrency": 1,
                                    "ttft": 52.0,
                                }
                            ],
                            "report_data": {
                                "parameter_support_tests": {
                                    "results": {"test_smoke": [{"status": "failed"}]}
                                }
                            },
                        }
                    }
                }
            }
        },
    }
    base_release_performance_data = {
        "schema_version": "0.1.0",
        "models": {
            "DemoModel": {
                "n150": {
                    "demo_impl": {
                        "vLLM": {
                            "model": "DemoModel",
                            "device": "n150",
                            "impl_id": "demo_impl",
                            "inference_engine": "vLLM",
                            "perf_status": "functional",
                            "accuracy_status": True,
                            "benchmarks_summary": [
                                {
                                    "task_type": "text",
                                    "isl": 128,
                                    "osl": 128,
                                    "max_concurrency": 1,
                                    "ttft": 45.0,
                                }
                            ],
                            "report_data": {
                                "parameter_support_tests": {
                                    "results": {"test_smoke": [{"status": "passed"}]}
                                }
                            },
                        }
                    }
                }
            }
        },
    }

    with patch.object(
        gra,
        "load_git_release_performance_data",
        return_value=base_release_performance_data,
    ):
        notes_path = gra.write_release_notes(
            output_dir,
            "0.10.0",
            release_performance_data=release_performance_data,
        )

    notes = notes_path.read_text()
    assert "## Model and Hardware Support Diff" in notes
    assert (
        "N150: Perf status: functional -> target; Benchmarks ~1; LLM API tests ~1"
        in notes
    )
    assert "## Release Artifacts Summary" in notes
    assert "## Performance\n" in notes
    assert "### DemoModel on n150 (demo_impl, vLLM)" in notes


def test_main_wires_release_flow_and_emits_summary(tmp_path):
    ci_artifacts_path = tmp_path / "release_logs" / "v0.10.0"
    ci_artifacts_path.mkdir(parents=True)
    output_dir = tmp_path / "release_logs" / "v0.10.0"
    model_spec_path = tmp_path / "workflows" / "model_spec.py"
    model_spec_path.parent.mkdir(parents=True)
    model_spec_path.write_text("MODEL_SPECS = {}\n")
    readme_path = tmp_path / "README.md"
    readme_path.write_text("# README\n")
    release_model_spec_path = tmp_path / "release_model_spec.json"
    merged_spec = {"model-a": make_record("model-a", make_image("demo"))}
    result_tuple = (
        defaultdict(list),
        0,
        {"ghcr.io/tenstorrent/release:tag": "ghcr.io/tenstorrent/ci:tag"},
        {},
        defaultdict(list),
    )
    args = Namespace(
        ci_artifacts_path=str(ci_artifacts_path),
        models_ci_run_id=None,
        out_root=None,
        dev=False,
        release=True,
        output_dir=str(output_dir),
        model_spec_path=str(model_spec_path),
        readme_path=str(readme_path),
        release_model_spec_path=str(release_model_spec_path),
        dry_run=False,
    )
    event_order = []

    def mark_event(name):
        def _inner(*args, **kwargs):
            event_order.append(name)
            if name == "write_release_notes":
                return output_dir / "release_notes.md"
            if name == "write_release_performance_outputs":
                return {"schema_version": "0.1.0", "models": {}}
            if name == "write_release_performance_diff_output":
                return output_dir / "release_performance_diff.json"
            return None

        return _inner

    with patch.object(gra, "configure_logging"), patch.object(
        gra, "get_versioned_release_logs_dir", return_value=output_dir
    ), patch(
        "argparse.ArgumentParser.parse_args",
        return_value=args,
    ), patch.object(
        gra, "resolve_release_output_dir", return_value=output_dir
    ), patch.object(gra, "check_docker_installed", return_value=True), patch.object(
        gra, "check_crane_installed", return_value=True
    ), patch.object(
        gra, "load_ci_data_from_artifacts_path", return_value={"model-a": {}}
    ) as load_ci_data_mock, patch.object(
        gra, "merge_specs_with_ci_data", return_value=merged_spec
    ) as merge_mock, patch.object(
        gra, "generate_release_artifacts", return_value=result_tuple
    ) as generate_mock, patch.object(
        gra, "write_output", side_effect=mark_event("write_output")
    ) as write_output_mock, patch.object(
        gra,
        "write_release_performance_outputs",
        side_effect=mark_event("write_release_performance_outputs"),
    ) as performance_mock, patch.object(
        gra,
        "write_release_performance_diff_output",
        side_effect=mark_event("write_release_performance_diff_output"),
    ) as performance_diff_mock, patch.object(
        gra,
        "write_release_model_spec_output",
        side_effect=mark_event("write_release_model_spec_output"),
    ) as export_mock, patch.object(
        gra,
        "regenerate_model_support_docs_and_update_readme",
        side_effect=mark_event("regenerate_model_support_docs_and_update_readme"),
    ) as docs_mock, patch.object(
        gra, "write_release_notes", side_effect=mark_event("write_release_notes")
    ) as release_notes_mock, patch.object(
        gra, "build_acceptance_warnings", return_value=[]
    ) as acceptance_mock, patch.object(
        gra, "emit_markdown_summary"
    ) as emit_mock, patch.object(gra, "get_version", return_value="0.10.0"):
        assert gra.main() == 0

    load_ci_data_mock.assert_called_once_with(ci_artifacts_path)
    merge_mock.assert_called_once_with({"model-a": {}}, False)
    generate_mock.assert_called_once_with(merged_spec, False)
    acceptance_mock.assert_called_once_with(merged_spec)
    write_output_mock.assert_called_once()
    performance_mock.assert_called_once_with(merged_spec, output_dir, False)
    performance_diff_mock.assert_called_once_with(
        output_dir, {"schema_version": "0.1.0", "models": {}}
    )
    export_mock.assert_called_once_with(
        model_spec_path=model_spec_path,
        output_path=release_model_spec_path,
        dry_run=False,
    )
    docs_mock.assert_called_once_with(
        model_spec_path=model_spec_path,
        readme_path=readme_path,
        release_performance_data={"schema_version": "0.1.0", "models": {}},
        dry_run=False,
    )
    release_notes_mock.assert_called_once_with(
        output_dir,
        "0.10.0",
        release_performance_data={"schema_version": "0.1.0", "models": {}},
    )
    emit_mock.assert_called_once_with(output_dir / "release_artifacts_summary.md")
    assert event_order == [
        "write_release_performance_outputs",
        "write_release_performance_diff_output",
        "write_release_model_spec_output",
        "regenerate_model_support_docs_and_update_readme",
        "write_output",
        "write_release_notes",
    ]


def test_main_release_run_id_reads_release_workflow(tmp_path):
    output_dir = tmp_path / "release_logs" / "v0.10.0"
    args = Namespace(
        ci_artifacts_path=None,
        models_ci_run_id=23578993514,
        out_root=None,
        dev=False,
        release=True,
        output_dir=str(output_dir),
        model_spec_path=str(tmp_path / "workflows" / "model_spec.py"),
        readme_path=str(tmp_path / "README.md"),
        release_model_spec_path=str(tmp_path / "release_model_spec.json"),
        dry_run=False,
    )
    merged_spec = {"model-a": make_record("model-a", make_image("demo"))}
    result_tuple = (defaultdict(list), 0, {}, {}, defaultdict(list))

    with patch.object(gra, "configure_logging"), patch.object(
        gra, "get_versioned_release_logs_dir", return_value=output_dir
    ), patch(
        "argparse.ArgumentParser.parse_args",
        return_value=args,
    ), patch.object(
        gra, "resolve_release_output_dir", return_value=output_dir
    ), patch.object(gra, "check_docker_installed", return_value=True), patch.object(
        gra, "check_crane_installed", return_value=True
    ), patch.object(
        gra, "load_ci_data_from_run_id", return_value={"model-a": {}}
    ) as load_ci_data_mock, patch.object(
        gra, "merge_specs_with_ci_data", return_value=merged_spec
    ), patch.object(
        gra, "generate_release_artifacts", return_value=result_tuple
    ), patch.object(gra, "write_output"), patch.object(
        gra,
        "write_release_performance_outputs",
        return_value={"schema_version": "0.1.0", "models": {}},
    ), patch.object(gra, "write_release_performance_diff_output"), patch.object(
        gra, "write_release_model_spec_output"
    ), patch.object(
        gra, "regenerate_model_support_docs_and_update_readme"
    ), patch.object(
        gra, "write_release_notes", return_value=output_dir / "release_notes.md"
    ), patch.object(gra, "emit_markdown_summary"), patch.object(
        gra, "get_version", return_value="0.10.0"
    ):
        assert gra.main() == 0

    load_ci_data_mock.assert_called_once_with(
        23578993514, output_dir, workflow_file=gra.RELEASE_WORKFLOW_FILE
    )


def test_main_dev_run_id_keeps_default_workflow(tmp_path):
    output_dir = tmp_path / "release_logs" / "v0.10.0"
    args = Namespace(
        ci_artifacts_path=None,
        models_ci_run_id=23578993514,
        out_root=None,
        dev=True,
        release=False,
        output_dir=str(output_dir),
        model_spec_path=str(tmp_path / "workflows" / "model_spec.py"),
        readme_path=str(tmp_path / "README.md"),
        release_model_spec_path=str(tmp_path / "release_model_spec.json"),
        dry_run=False,
    )
    merged_spec = {"model-a": make_record("model-a", make_image("demo", channel="dev"))}
    result_tuple = (defaultdict(list), 0, {}, {}, defaultdict(list))

    with patch.object(gra, "configure_logging"), patch.object(
        gra, "get_versioned_release_logs_dir", return_value=output_dir
    ), patch(
        "argparse.ArgumentParser.parse_args",
        return_value=args,
    ), patch.object(
        gra, "resolve_release_output_dir", return_value=output_dir
    ), patch.object(gra, "check_docker_installed", return_value=True), patch.object(
        gra, "check_crane_installed", return_value=True
    ), patch.object(
        gra, "load_ci_data_from_run_id", return_value={"model-a": {}}
    ) as load_ci_data_mock, patch.object(
        gra, "merge_specs_with_ci_data", return_value=merged_spec
    ), patch.object(
        gra, "generate_release_artifacts", return_value=result_tuple
    ), patch.object(gra, "write_output"), patch.object(gra, "emit_markdown_summary"):
        assert gra.main() == 0

    load_ci_data_mock.assert_called_once_with(
        23578993514, output_dir, workflow_file="on-nightly.yml"
    )


def test_main_dev_flow_skips_release_only_compatibility_outputs(tmp_path):
    ci_artifacts_path = tmp_path / "release_logs" / "v0.10.0"
    ci_artifacts_path.mkdir(parents=True)
    output_dir = tmp_path / "release_logs" / "v0.10.0"
    args = Namespace(
        ci_artifacts_path=str(ci_artifacts_path),
        models_ci_run_id=None,
        out_root=None,
        dev=True,
        release=False,
        output_dir=str(output_dir),
        model_spec_path=str(tmp_path / "workflows" / "model_spec.py"),
        readme_path=str(tmp_path / "README.md"),
        release_model_spec_path=str(tmp_path / "release_model_spec.json"),
        dry_run=False,
    )
    merged_spec = {"model-a": make_record("model-a", make_image("demo", channel="dev"))}
    result_tuple = (defaultdict(list), 0, {}, {}, defaultdict(list))

    with patch.object(gra, "configure_logging"), patch.object(
        gra, "get_versioned_release_logs_dir", return_value=output_dir
    ), patch(
        "argparse.ArgumentParser.parse_args",
        return_value=args,
    ), patch.object(
        gra, "resolve_release_output_dir", return_value=output_dir
    ), patch.object(gra, "check_docker_installed", return_value=True), patch.object(
        gra, "check_crane_installed", return_value=True
    ), patch.object(
        gra, "load_ci_data_from_artifacts_path", return_value={"model-a": {}}
    ), patch.object(
        gra, "merge_specs_with_ci_data", return_value=merged_spec
    ), patch.object(
        gra, "generate_release_artifacts", return_value=result_tuple
    ), patch.object(gra, "write_output"), patch.object(
        gra, "write_release_model_spec_output"
    ) as export_mock, patch.object(
        gra, "regenerate_model_support_docs_and_update_readme"
    ) as docs_mock, patch.object(
        gra, "write_release_notes"
    ) as release_notes_mock, patch.object(gra, "emit_markdown_summary"):
        assert gra.main() == 0

    export_mock.assert_not_called()
    docs_mock.assert_not_called()
    release_notes_mock.assert_not_called()
