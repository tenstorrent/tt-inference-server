import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "release"))
import scripts.release.workflow_logs_parser as workflow_logs_parser


def test_parse_perf_status_prefers_persisted_benchmark_target_evaluation():
    report_data = {
        "benchmark_target_evaluation": {
            "status": "complete",
            "reference_available": True,
            "errors": [],
        }
    }

    assert workflow_logs_parser.parse_perf_status(report_data) == "complete"
    assert workflow_logs_parser.parse_benchmarks_completed(report_data) is True


def test_parse_perf_status_falls_back_to_shared_benchmark_evaluation():
    with patch.object(
        workflow_logs_parser,
        "evaluate_benchmark_targets",
        return_value={
            "status": "functional",
            "reference_available": True,
            "errors": [],
        },
    ):
        assert workflow_logs_parser.parse_perf_status({}) == "functional"
        assert workflow_logs_parser.parse_benchmarks_completed({}) is True


def test_parse_workflow_logs_dir_blocks_regression_failures_from_passing(tmp_path):
    workflow_logs_dir = tmp_path / "workflow_logs_demo"
    runtime_model_specs_dir = workflow_logs_dir / "runtime_model_specs"
    reports_data_dir = workflow_logs_dir / "reports_output" / "release" / "data"
    runtime_model_specs_dir.mkdir(parents=True)
    reports_data_dir.mkdir(parents=True)

    (runtime_model_specs_dir / "model_spec.json").write_text(
        json.dumps(
            {
                "model_id": "demo-model",
                "docker_image": "ghcr.io/tenstorrent/demo:tag",
            }
        )
    )
    (reports_data_dir / "report_data_demo-model_123.json").write_text(
        json.dumps(
            {
                "evals": [{"accuracy_check": 2}],
                "benchmark_target_evaluation": {
                    "status": "target",
                    "reference_available": True,
                    "errors": [],
                    "regression": {
                        "checked": True,
                        "passed": False,
                        "failures": [
                            {
                                "key": "benchmarks.regression.demo",
                                "message": "Regression target failed.",
                            }
                        ],
                    },
                },
            }
        )
    )

    with patch.object(
        workflow_logs_parser,
        "parse_commits_from_docker_image",
        return_value=("a" * 40, "1" * 7),
    ):
        parsed = workflow_logs_parser.parse_workflow_logs_dir(workflow_logs_dir)

    assert parsed is not None
    assert parsed["summary"]["perf_status"] == "target"
    assert parsed["summary"]["benchmarks_completed"] is True
    assert parsed["summary"]["accuracy_status"] is True
    assert parsed["summary"]["evals_completed"] is True
    assert parsed["summary"]["regression_checked"] is True
    assert parsed["summary"]["regression_passed"] is False
    assert parsed["summary"]["regression_ok"] is False
    assert parsed["summary"]["is_passing"] is False
