#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.agentic.report import is_harbor_result, process_agentic_eval_files  # noqa: E402
from workflows.run_reports import separate_files_by_format  # noqa: E402


def test_process_agentic_eval_files_extracts_harbor_accuracy(tmp_path):
    job_dir = tmp_path / "terminal_bench_2"
    job_dir.mkdir()
    result_path = job_dir / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "config": {
                    "datasets": [
                        {
                            "name": "terminal-bench/terminal-bench-2",
                            "ref": "latest",
                        }
                    ]
                },
                "stats": {
                    "evals": {
                        "terminus-2__openai/Qwen/Qwen3.6-27B__terminal-bench/terminal-bench-2": {
                            "pass_at_k": {"2": 0.75}
                        }
                    }
                },
                "trial_results": [
                    {"verifier_result": {"rewards": {"reward": 1}}},
                    {"verifier_result": {"rewards": {"reward": 0}}},
                ],
            }
        ),
        encoding="utf-8",
    )

    results, meta_data = process_agentic_eval_files([str(result_path)])

    assert results["terminal_bench_2"]["accuracy"] == 0.5
    assert results["terminal_bench_2"]["pass_at_1"] == 0.5
    assert results["terminal_bench_2"]["pass_at_2"] == 0.75
    assert results["terminal_bench_2"]["n_trials"] == 2
    assert meta_data["terminal_bench_2"]["dataset_path"] == (
        "terminal-bench/terminal-bench-2@latest"
    )


def test_process_agentic_eval_files_extracts_harbor_summary_metrics(tmp_path):
    job_dir = tmp_path / "terminal_bench_2_smoke_test"
    job_dir.mkdir()
    result_path = job_dir / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "stats": {
                    "evals": {
                        "terminus-2__Qwen/Qwen3.6-27B__terminal-bench/terminal-bench-2": {
                            "n_trials": 10,
                            "metrics": [{"mean": 0.45454545454545453}],
                            "pass_at_k": {"2": 0.8},
                            "reward_stats": {
                                "reward": {
                                    "0.0": ["failed-task"],
                                    "1.0": ["resolved-a", "resolved-b"],
                                }
                            },
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    dict_files, list_files, agentic_files = separate_files_by_format([str(result_path)])
    results, meta_data = process_agentic_eval_files(agentic_files)

    assert dict_files == []
    assert list_files == []
    assert agentic_files == [str(result_path)]
    assert results["terminal_bench_2_smoke_test"]["accuracy"] == 0.45454545454545453
    assert results["terminal_bench_2_smoke_test"]["pass_at_1"] == 0.45454545454545453
    assert results["terminal_bench_2_smoke_test"]["pass_at_2"] == 0.8
    assert results["terminal_bench_2_smoke_test"]["n_trials"] == 10
    assert results["terminal_bench_2_smoke_test"]["n_resolved"] == 2
    assert meta_data["terminal_bench_2_smoke_test"]["dataset_path"] == "N/A"


def test_process_agentic_eval_files_extracts_swebench_metrics(tmp_path):
    job_dir = tmp_path / "swe_bench_verified"
    job_dir.mkdir()
    result_path = job_dir / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "config": {
                    "datasets": [
                        {
                            "name": "SWE-bench/SWE-bench_Verified",
                            "split": "test",
                        }
                    ]
                },
                "stats": {
                    "evals": {
                        "swe-agent__openai/Qwen/Qwen3.6-27B__SWE-bench/SWE-bench_Verified": {
                            "n_trials": 3,
                            "metrics": [{"name": "accuracy", "mean": 2 / 3}],
                            "pass_at_k": {"1": 2 / 3},
                        }
                    }
                },
                "trial_results": [
                    {"verifier_result": {"rewards": {"reward": 1.0}}},
                    {"verifier_result": {"rewards": {"reward": 0.0}}},
                    {"verifier_result": {"rewards": {"reward": 1.0}}},
                ],
            }
        ),
        encoding="utf-8",
    )

    dict_files, list_files, agentic_files = separate_files_by_format([str(result_path)])
    results, meta_data = process_agentic_eval_files(agentic_files)

    assert dict_files == []
    assert list_files == []
    assert agentic_files == [str(result_path)]
    assert results["swe_bench_verified"]["accuracy"] == 2 / 3
    assert results["swe_bench_verified"]["pass_at_1"] == 2 / 3
    assert results["swe_bench_verified"]["n_trials"] == 3
    assert results["swe_bench_verified"]["n_resolved"] == 2
    assert (
        meta_data["swe_bench_verified"]["dataset_path"]
        == "SWE-bench/SWE-bench_Verified"
    )
