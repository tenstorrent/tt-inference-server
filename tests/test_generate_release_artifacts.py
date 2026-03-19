import json
import subprocess
from argparse import Namespace
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch

import scripts.release.generate_release_artifacts as gra
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


def test_merge_specs_with_ci_data_derives_dev_image_without_mutating_model_specs(
    tmp_path,
):
    release_image = make_image("demo")
    ci_json_path = tmp_path / "last_good.json"
    ci_json_path.write_text(json.dumps({"demo_model": {"docker_image": "ci-image"}}))
    model_spec = make_model_spec(release_image)

    with patch.object(gra, "MODEL_SPECS", {"demo_model": model_spec}):
        merged = gra.merge_specs_with_ci_data(ci_json_path, is_dev=True)

    assert model_spec.docker_image == release_image
    assert merged["demo_model"].target_docker_image == release_image.replace(
        "-release-", "-dev-"
    )
    assert merged["demo_model"].ci_data == {"docker_image": "ci-image"}


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
    }
    assert (tmp_path / "release_artifacts_summary.json").exists()
    markdown = (tmp_path / "release_artifacts_summary.md").read_text()
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

    with patch.object(
        gra, "get_release_performance_path", return_value=baseline_path
    ), patch(
        "scripts.release.release_performance.get_perf_target", return_value=perf_target
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


def test_write_release_notes_uses_raw_json_inputs(tmp_path):
    output_dir = tmp_path / "release_logs" / "v0.10.0"
    output_dir.mkdir(parents=True)
    (output_dir / "pre_release_models_diff.json").write_text(
        json.dumps(
            [
                {
                    "template_key": "template:demo",
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
        "schema_version": 1,
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
        "schema_version": 1,
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
    assert "## Performance\n" not in notes


def test_main_wires_release_flow_and_emits_summary(tmp_path):
    ci_json_path = tmp_path / "last_good.json"
    ci_json_path.write_text("{}")
    output_dir = tmp_path / "release_logs" / "v0.10.0"
    merged_spec = {"model-a": make_record("model-a", make_image("demo"))}
    result_tuple = (
        defaultdict(list),
        0,
        {"ghcr.io/tenstorrent/release:tag": "ghcr.io/tenstorrent/ci:tag"},
        {},
        defaultdict(list),
    )
    args = Namespace(
        models_ci_last_good_json=str(ci_json_path),
        models_ci_run_id=None,
        out_root=None,
        dev=False,
        release=True,
        output_dir=str(output_dir),
        dry_run=False,
    )

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
        gra, "merge_specs_with_ci_data", return_value=merged_spec
    ) as merge_mock, patch.object(
        gra, "generate_release_artifacts", return_value=result_tuple
    ) as generate_mock, patch.object(
        gra, "write_output"
    ) as write_output_mock, patch.object(
        gra,
        "write_release_performance_outputs",
        return_value={"schema_version": 1, "models": {}},
    ) as performance_mock, patch.object(
        gra, "write_release_notes", return_value=output_dir / "release_notes.md"
    ) as release_notes_mock, patch.object(
        gra, "emit_markdown_summary"
    ) as emit_mock, patch.object(gra, "get_version", return_value="0.10.0"):
        assert gra.main() == 0

    merge_mock.assert_called_once_with(ci_json_path.resolve(), False)
    generate_mock.assert_called_once_with(merged_spec, False)
    write_output_mock.assert_called_once()
    performance_mock.assert_called_once_with(merged_spec, output_dir, False)
    release_notes_mock.assert_called_once_with(
        output_dir,
        "0.10.0",
        release_performance_data={"schema_version": 1, "models": {}},
    )
    emit_mock.assert_called_once_with(output_dir / "release_artifacts_summary.md")
