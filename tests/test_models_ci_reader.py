import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "release"))
import scripts.release.models_ci_reader as mcr


def test_write_summary_output_preserves_release_report_payload(tmp_path):
    all_models_dict = {
        "demo_model": [
            {
                "workflow_logs": {
                    "summary": {
                        "tt_metal_commit": "a" * 40,
                        "vllm_commit": "1" * 7,
                        "docker_image": "ghcr.io/tenstorrent/demo:tag",
                        "perf_status": "target",
                        "benchmarks_completed": True,
                        "accuracy_status": True,
                        "evals_completed": True,
                        "regression_checked": True,
                        "regression_passed": True,
                        "regression_ok": True,
                        "is_passing": True,
                    },
                    "reports_output": {
                        "metadata": {"model_name": "DemoModel", "device": "n150"},
                        "benchmarks_summary": [
                            {
                                "task_type": "text",
                                "isl": 128,
                                "osl": 128,
                                "max_concurrency": 1,
                                "ttft": 42.0,
                                "tput_user": 10.5,
                                "target_checks": {
                                    "target": {
                                        "ttft_check": 2,
                                        "tput_user_check": 2,
                                    }
                                },
                            }
                        ],
                        "evals": [{"task_name": "demo_eval", "accuracy_check": 2}],
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
                                "target": {
                                    "checked": True,
                                    "passed": True,
                                    "failures": [],
                                },
                            },
                            "regression": {
                                "checked": True,
                                "passed": True,
                                "failures": [],
                            },
                            "next_status": None,
                            "next_status_failures": [],
                            "errors": [],
                        },
                        "parameter_support_tests": {
                            "results": {
                                "test_smoke": [
                                    {
                                        "status": "passed",
                                        "test_node_name": "test_smoke[param]",
                                        "message": "",
                                    }
                                ]
                            }
                        },
                    },
                },
                "ci_metadata": {
                    "run_id": 55,
                    "run_number": 123,
                    "ci_run_url": "https://example.com/runs/55",
                    "ci_job_metadata": {
                        "job_id": 88,
                        "job_url": "https://example.com/jobs/88",
                        "job_name": "demo-job",
                        "job_status": "completed",
                        "job_conclusion": "success",
                    },
                },
                "ci_logs": {
                    "firmware_bundle": "fw",
                    "kmd_version": "kmd",
                },
            }
        ]
    }
    model_spec = SimpleNamespace(
        device_type=SimpleNamespace(name="N150"),
        impl=SimpleNamespace(impl_name="demo-impl"),
    )

    with patch.object(mcr, "MODEL_SPECS", {"demo_model": model_spec}):
        output_path = mcr.write_summary_output(all_models_dict, ["123"], tmp_path)

    output_data = json.loads(output_path.read_text())
    assert (
        output_data["demo_model"]["release_report"]["benchmarks_summary"][0]["isl"]
        == 128
    )
    assert (
        output_data["demo_model"]["release_report"]["parameter_support_tests"][
            "results"
        ]["test_smoke"][0]["status"]
        == "passed"
    )
    assert output_data["demo_model"]["regression_checked"] is True
    assert output_data["demo_model"]["regression_passed"] is True
    assert output_data["demo_model"]["regression_ok"] is True
    assert output_data["demo_model"]["is_passing"] is True


def test_write_summary_output_prefers_latest_accepted_passing_entry(tmp_path):
    older_passing_entry = {
        "workflow_logs": {
            "summary": {
                "tt_metal_commit": "a" * 40,
                "vllm_commit": "1" * 7,
                "docker_image": "ghcr.io/tenstorrent/demo:old",
                "perf_status": "target",
                "benchmarks_completed": True,
                "accuracy_status": True,
                "evals_completed": True,
                "regression_checked": True,
                "regression_passed": True,
                "regression_ok": True,
                "is_passing": True,
            },
            "reports_output": {
                "metadata": {"model_name": "DemoModel", "device": "n150"},
                "benchmarks_summary": [
                    {
                        "task_type": "text",
                        "isl": 128,
                        "osl": 128,
                        "max_concurrency": 1,
                        "ttft": 42.0,
                    }
                ],
                "evals": [{"task_name": "demo_eval", "accuracy_check": 2}],
                "parameter_support_tests": {
                    "results": {
                        "test_smoke": [
                            {
                                "status": "passed",
                                "test_node_name": "test_smoke[param]",
                                "message": "",
                            }
                        ]
                    }
                },
                "benchmark_target_evaluation": {
                    "status": "target",
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
            },
        },
        "ci_metadata": {
            "run_id": 1,
            "run_number": 100,
            "ci_run_url": "https://example.com/runs/100",
            "ci_job_metadata": {"job_id": 1, "job_url": "https://example.com/jobs/1"},
        },
        "ci_logs": {},
    }
    newer_failed_entry = {
        "workflow_logs": {
            "summary": {
                "tt_metal_commit": "b" * 40,
                "vllm_commit": "2" * 7,
                "docker_image": "ghcr.io/tenstorrent/demo:new",
                "perf_status": "target",
                "benchmarks_completed": True,
                "accuracy_status": True,
                "evals_completed": True,
                "regression_checked": True,
                "regression_passed": True,
                "regression_ok": True,
                "is_passing": True,
            },
            "reports_output": {
                "metadata": {"model_name": "DemoModel", "device": "n150"},
                "benchmarks_summary": [
                    {
                        "task_type": "text",
                        "isl": 128,
                        "osl": 128,
                        "max_concurrency": 1,
                        "ttft": 42.0,
                    }
                ],
                "evals": [{"task_name": "demo_eval", "accuracy_check": 2}],
                "parameter_support_tests": {
                    "results": {
                        "test_smoke": [
                            {
                                "status": "failed",
                                "test_node_name": "test_smoke[param]",
                                "message": "",
                            }
                        ]
                    }
                },
                "benchmark_target_evaluation": {
                    "status": "target",
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
            },
        },
        "ci_metadata": {
            "run_id": 2,
            "run_number": 200,
            "ci_run_url": "https://example.com/runs/200",
            "ci_job_metadata": {"job_id": 2, "job_url": "https://example.com/jobs/2"},
        },
        "ci_logs": {},
    }
    model_spec = SimpleNamespace(
        device_type=SimpleNamespace(name="N150"),
        impl=SimpleNamespace(impl_name="demo-impl"),
    )

    with patch.object(mcr, "MODEL_SPECS", {"demo_model": model_spec}):
        output_path = mcr.write_summary_output(
            {"demo_model": [older_passing_entry, newer_failed_entry]},
            ["100", "200"],
            tmp_path,
        )

    output_data = json.loads(output_path.read_text())
    assert output_data["demo_model"]["ci_run_number"] == 100
    assert output_data["demo_model"]["docker_image"] == "ghcr.io/tenstorrent/demo:old"


def test_write_summary_output_only_writes_requested_files(tmp_path):
    model_spec = SimpleNamespace(
        device_type=SimpleNamespace(name="N150"),
        impl=SimpleNamespace(impl_name="demo-impl"),
    )

    with patch.object(mcr, "MODEL_SPECS", {"demo_model": model_spec}):
        output_path = mcr.write_summary_output(
            {"demo_model": []},
            ["123"],
            tmp_path,
            write_all_results=True,
            write_last_good=False,
        )

    assert output_path is None
    assert (tmp_path / "models_ci_all_results_123_to_123.json").exists()
    assert not (tmp_path / "models_ci_last_good_123_to_123.json").exists()


def test_parse_job_name_supports_dynamic_release_jobs():
    parsed_job = mcr.parse_job_name(
        "_ / vLLM / run-release-Llama-3.2-1B-Instruct-llmbox-t3k"
    )

    assert parsed_job == {
        "workflow_type": "release",
        "model_name": "Llama-3.2-1B-Instruct",
        "hardware_name": "llmbox",
        "hardware": "t3k",
    }


def test_match_jobs_to_workflow_logs_supports_dynamic_release_jobs():
    jobs_ci_metadata = [
        {
            "job_id": 68659809626,
            "job_name": "_ / vLLM / run-release-Llama-3.2-1B-Instruct-llmbox-t3k",
            "job_status": "completed",
            "job_conclusion": "cancelled",
            "job_url": "https://example.com/jobs/68659809626",
            "started_at": "2026-03-26T08:41:43Z",
            "completed_at": "2026-03-26T14:43:11Z",
        }
    ]

    matched_job = mcr.match_jobs_to_workflow_logs(
        jobs_ci_metadata, "workflow_logs_release_Llama-3.2-1B-Instruct_llmbox"
    )

    assert matched_job == {
        "job_id": 68659809626,
        "job_name": "_ / vLLM / run-release-Llama-3.2-1B-Instruct-llmbox-t3k",
        "job_status": "completed",
        "job_conclusion": "cancelled",
        "job_url": "https://example.com/jobs/68659809626",
        "started_at": "2026-03-26T08:41:43Z",
        "completed_at": "2026-03-26T14:43:11Z",
        "model_name": "Llama-3.2-1B-Instruct",
        "hardware": "t3k",
        "hardware_name": "llmbox",
    }


def test_collect_artifact_run_ids_keeps_parent_and_child_runs_once():
    jobs = [
        {"id": 1, "run_id": 23578993514},
        {"id": 2, "run_id": 24000000001},
        {"id": 3, "run_id": 24000000001},
        {"id": 4, "run_id": 24000000002},
        {"id": 5},
    ]

    assert mcr._collect_artifact_run_ids(23578993514, jobs) == [
        23578993514,
        24000000001,
        24000000002,
    ]


def test_download_runs_fetches_workflow_logs_from_child_run_ids(tmp_path):
    out_root = tmp_path / "release_logs" / "v0.12.0"
    parent_run_id = 23578993514
    child_run_id = 24000000001
    jobs = [
        {
            "id": 68659809607,
            "run_id": parent_run_id,
            "name": "_ / vLLM / run-release-Llama-3.2-1B-Instruct-n150-n150",
            "status": "completed",
            "conclusion": "failure",
            "html_url": "https://example.com/jobs/68659809607",
            "started_at": "2026-03-26T09:04:27Z",
            "completed_at": "2026-03-26T10:00:36Z",
        },
        {
            "id": 68659809595,
            "run_id": child_run_id,
            "name": "_ / vLLM / run-release-Llama-3.2-1B-Instruct-n300-n300",
            "status": "completed",
            "conclusion": "cancelled",
            "html_url": "https://example.com/jobs/68659809595",
            "started_at": "2026-03-26T11:15:17Z",
            "completed_at": "2026-03-26T14:43:22Z",
        },
    ]

    def fake_extract_zip_to_dir(_zip_bytes, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)

    with patch.object(mcr, "get_workflow", return_value={"id": 123}), patch.object(
        mcr,
        "get_workflow_run",
        return_value={
            "id": parent_run_id,
            "run_number": 259,
            "created_at": "2026-03-26T05:26:02Z",
            "updated_at": "2026-03-26T14:43:31Z",
            "name": "Release",
        },
    ), patch.object(mcr, "list_run_jobs", return_value=jobs), patch.object(
        mcr, "download_ci_job_logs"
    ), patch.object(
        mcr,
        "list_run_artifacts",
        side_effect=lambda run_id, owner, repo, token: (
            [
                {
                    "id": 1,
                    "name": "workflow_logs_release_Llama-3.2-1B-Instruct_n150",
                    "archive_download_url": "https://example.com/a1.zip",
                }
            ]
            if run_id == parent_run_id
            else [
                {
                    "id": 2,
                    "name": "workflow_logs_release_Llama-3.2-1B-Instruct_n300",
                    "archive_download_url": "https://example.com/a2.zip",
                }
            ]
        ),
    ) as list_artifacts_mock, patch.object(
        mcr, "download_artifact_zip", return_value=b"zip-bytes"
    ), patch.object(mcr, "extract_zip_to_dir", side_effect=fake_extract_zip_to_dir):
        mcr.download_runs(
            owner="tenstorrent",
            repo="tt-shield",
            workflow_file="release.yml",
            token="token",
            out_root=out_root,
            max_runs=1,
            run_id=parent_run_id,
        )

    assert list_artifacts_mock.call_args_list[0].args[0] == parent_run_id
    assert list_artifacts_mock.call_args_list[1].args[0] == child_run_id
    run_out_dir = out_root / "ci_run_logs" / f"On_nightly_259_{parent_run_id}"
    assert (run_out_dir / "workflow_logs_release_Llama-3.2-1B-Instruct_n150").is_dir()
    assert (run_out_dir / "workflow_logs_release_Llama-3.2-1B-Instruct_n300").is_dir()
    jobs_metadata = json.loads((run_out_dir / "jobs_ci_metadata.json").read_text())
    assert jobs_metadata[0]["run_id"] == parent_run_id
    assert jobs_metadata[1]["run_id"] == child_run_id


def test_download_runs_does_not_skip_explicit_partial_run_dir(tmp_path):
    out_root = tmp_path / "release_logs" / "v0.12.0"
    parent_run_id = 23578993514
    run_out_dir = out_root / "ci_run_logs" / f"On_nightly_259_{parent_run_id}"
    (run_out_dir / "logs").mkdir(parents=True)
    (run_out_dir / "workflow_logs_release_Llama-3.2-1B-Instruct_n150").mkdir()

    with patch.object(mcr, "get_workflow", return_value={"id": 123}), patch.object(
        mcr,
        "get_workflow_run",
        return_value={
            "id": parent_run_id,
            "run_number": 259,
            "created_at": "2026-03-26T05:26:02Z",
            "updated_at": "2026-03-26T14:43:31Z",
            "name": "Release",
        },
    ), patch.object(
        mcr, "list_run_jobs", return_value=[]
    ) as list_jobs_mock, patch.object(mcr, "download_ci_job_logs"), patch.object(
        mcr, "list_run_artifacts", return_value=[]
    ), patch.object(mcr, "extract_zip_to_dir"):
        mcr.download_runs(
            owner="tenstorrent",
            repo="tt-shield",
            workflow_file="release.yml",
            token="token",
            out_root=out_root,
            max_runs=1,
            run_id=parent_run_id,
        )

    list_jobs_mock.assert_called_once_with(
        parent_run_id, "tenstorrent", "tt-shield", "token"
    )


def test_download_runs_extracts_report_artifacts(tmp_path):
    out_root = tmp_path / "release_logs" / "v0.12.0"
    parent_run_id = 23578993514

    def fake_extract_zip_to_dir(_zip_bytes, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)

    with patch.object(mcr, "get_workflow", return_value={"id": 123}), patch.object(
        mcr,
        "get_workflow_run",
        return_value={
            "id": parent_run_id,
            "run_number": 259,
            "created_at": "2026-03-26T05:26:02Z",
            "updated_at": "2026-03-26T14:43:31Z",
            "name": "Release",
        },
    ), patch.object(
        mcr,
        "list_run_jobs",
        return_value=[],
    ), patch.object(mcr, "download_ci_job_logs"), patch.object(
        mcr,
        "list_run_artifacts",
        return_value=[
            {
                "id": 1,
                "name": "workflow_logs_release_Llama-3.2-1B-Instruct_n150",
                "archive_download_url": "https://example.com/a1.zip",
            },
            {
                "id": 2,
                "name": "report_release_Llama-3.2-1B-Instruct_n300_68659809595",
                "archive_download_url": "https://example.com/a2.zip",
            },
        ],
    ), patch.object(
        mcr, "download_artifact_zip", return_value=b"zip-bytes"
    ), patch.object(mcr, "extract_zip_to_dir", side_effect=fake_extract_zip_to_dir):
        mcr.download_runs(
            owner="tenstorrent",
            repo="tt-shield",
            workflow_file="release.yml",
            token="token",
            out_root=out_root,
            max_runs=1,
            run_id=parent_run_id,
        )

    run_out_dir = out_root / "ci_run_logs" / f"On_nightly_259_{parent_run_id}"
    assert (run_out_dir / "workflow_logs_release_Llama-3.2-1B-Instruct_n150").is_dir()
    assert (
        run_out_dir / "report_release_Llama-3.2-1B-Instruct_n300_68659809595"
    ).is_dir()


def test_parse_report_artifact_dir_builds_workflow_style_record(tmp_path):
    report_dir = tmp_path / "report_release_Llama-3.2-1B-Instruct_n300_68659809595"
    report_dir.mkdir()
    (report_dir / "report_68659809595.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "model_id": "id_tt-transformers_Llama-3.2-1B-Instruct_n300"
                },
                "evals": [{"accuracy_check": 2}],
                "benchmark_target_evaluation": {
                    "status": "functional",
                    "reference_available": True,
                    "errors": [],
                    "regression": {"checked": False, "passed": True, "failures": []},
                },
            }
        )
    )
    (report_dir / "model_spec_68659809595.json").write_text(
        json.dumps(
            {
                "model_id": "id_tt-transformers_Llama-3.2-1B-Instruct_n300",
                "docker_image": "ghcr.io/tenstorrent/demo:tag",
                "cli_args": {
                    "override_docker_image": "ghcr.io/tenstorrent/demo:override"
                },
            }
        )
    )

    workflow_logs_parser_module = sys.modules[
        mcr.build_parsed_workflow_logs_data.__module__
    ]

    with patch.object(
        workflow_logs_parser_module,
        "parse_commits_from_docker_image",
        return_value=("a" * 40, "1" * 7),
    ):
        parsed = mcr._parse_report_artifact_dir(report_dir)

    assert parsed is not None
    assert (
        parsed["summary"]["model_id"] == "id_tt-transformers_Llama-3.2-1B-Instruct_n300"
    )
    assert parsed["summary"]["perf_status"] == "functional"
    assert parsed["summary"]["docker_image"] == "ghcr.io/tenstorrent/demo:override"


def test_process_run_directory_uses_report_artifact_fallback(tmp_path):
    run_out_dir = tmp_path / "On_nightly_259_23578993514"
    run_out_dir.mkdir()
    (run_out_dir / "logs").mkdir()
    jobs_ci_metadata = [
        {
            "job_id": 68659809595,
            "run_id": 23578993514,
            "job_name": "_ / vLLM / run-release-Llama-3.2-1B-Instruct-n300-n300",
            "job_status": "completed",
            "job_conclusion": "cancelled",
            "job_url": "https://example.com/jobs/68659809595",
            "started_at": "2026-03-26T11:15:17Z",
            "completed_at": "2026-03-26T14:43:22Z",
        }
    ]
    (run_out_dir / "jobs_ci_metadata.json").write_text(json.dumps(jobs_ci_metadata))
    report_dir = run_out_dir / "report_release_Llama-3.2-1B-Instruct_n300_68659809595"
    report_dir.mkdir()
    (report_dir / "report_68659809595.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "model_id": "id_tt-transformers_Llama-3.2-1B-Instruct_n300"
                },
                "evals": [{"accuracy_check": 2}],
                "benchmark_target_evaluation": {
                    "status": "functional",
                    "reference_available": True,
                    "errors": [],
                    "regression": {"checked": False, "passed": True, "failures": []},
                },
            }
        )
    )
    (report_dir / "model_spec_68659809595.json").write_text(
        json.dumps(
            {
                "model_id": "id_tt-transformers_Llama-3.2-1B-Instruct_n300",
                "docker_image": "ghcr.io/tenstorrent/demo:tag",
            }
        )
    )
    run_ci_metadata = {
        "run_id": 23578993514,
        "run_number": 259,
        "owner": "tenstorrent",
        "repo": "tt-shield",
    }

    workflow_logs_parser_module = sys.modules[
        mcr.build_parsed_workflow_logs_data.__module__
    ]

    with patch.object(
        workflow_logs_parser_module,
        "parse_commits_from_docker_image",
        return_value=("a" * 40, "1" * 7),
    ), patch.object(mcr, "parse_ci_job_log", return_value={"docker_image": None}):
        all_models_dict = mcr.process_run_directory(
            run_out_dir, "2026-03-26_21-44-13", run_ci_metadata
        )

    assert "id_tt-transformers_Llama-3.2-1B-Instruct_n300" in all_models_dict
    entry = all_models_dict["id_tt-transformers_Llama-3.2-1B-Instruct_n300"][0]
    assert entry["workflow_logs"]["summary"]["perf_status"] == "functional"
    assert entry["ci_metadata"]["ci_job_metadata"]["job_id"] == 68659809595
