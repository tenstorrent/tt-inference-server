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
                        "regression_passed": False,
                        "regression_ok": False,
                        "is_passing": False,
                    },
                    "reports_output": {
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
    assert output_data["demo_model"]["regression_passed"] is False
    assert output_data["demo_model"]["regression_ok"] is False
    assert output_data["demo_model"]["is_passing"] is False
