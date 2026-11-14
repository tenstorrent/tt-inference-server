# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import json
from pathlib import Path

import pytest

from workflows.accuracy_campaign import assess_task, build_command, digest
from workflows.isolated_eval_docker import IsolatedDocker


def write_report(root, task, score):
    directory = root / f"workflow_logs/reports_output/{task['workflow']}/data"
    directory.mkdir(parents=True)
    (directory / "report_data_test.json").write_text(
        json.dumps(
            {
                "blocks": [
                    {
                        "data": {
                            "task_name": task["name"],
                            "score": score,
                            "gpu_reference_score": 80.0,
                            "tolerance": 0.05,
                        }
                    }
                ]
            }
        )
    )


@pytest.mark.parametrize("score,passed", [(90.0, True), (50.0, False)])
def test_full_coverage_and_accuracy_are_independent(tmp_path, score, passed):
    task = {"name": "test", "workflow": "evals", "samples": 3}
    write_report(tmp_path, task, score)
    samples = tmp_path / "workflow_logs/reports_output/evals/samples_test_2026-11.jsonl"
    samples.write_text("\n".join(json.dumps({"doc_id": i}) for i in range(3)))
    result = assess_task(task, tmp_path, 0 if passed else 1)
    assert result["operationally_complete"]
    assert result["accuracy_pass"] is passed


@pytest.mark.parametrize("ids", [[0, 1], [0, 0, 1]])
def test_partial_or_duplicate_samples_are_not_reportable(tmp_path, ids):
    task = {"name": "test", "workflow": "evals", "samples": 3}
    write_report(tmp_path, task, 100.0)
    samples = tmp_path / "workflow_logs/reports_output/evals/samples_test_2026-11.jsonl"
    samples.write_text("\n".join(json.dumps({"doc_id": i}) for i in ids))
    result = assess_task(task, tmp_path, 0)
    assert not result["operationally_complete"]
    assert result["accuracy_pass"] is None


def test_agentic_exact_repeats(tmp_path):
    task = {"name": "terminal_bench_2_1", "workflow": "agentic", "samples": 6}
    write_report(tmp_path, task, 90.0)
    parent = (
        tmp_path
        / f"workflow_logs/reports_output/agentic/eval_model/agentic/{task['name']}"
    )
    parent.mkdir(parents=True)
    (parent / "result.json").write_text(json.dumps({"n_total_trials": 6}))
    for i in range(6):
        trial = parent / str(i)
        trial.mkdir()
        (trial / "result.json").write_text(
            json.dumps({"task_name": str(i // 3), "finished_at": "done"})
        )
    assert assess_task(task, tmp_path, 0)["operationally_complete"]
    (parent / "0/result.json").write_text(
        json.dumps({"task_name": "extra", "finished_at": "done"})
    )
    assert not assess_task(task, tmp_path, 0)["operationally_complete"]


def test_commands_never_limit_or_change_task_budgets():
    profile = json.loads(
        (
            Path(__file__).parents[2]
            / "reference_config/evals/deepseek_v41_accuracy.json"
        ).read_text()
    )
    assert [task["samples"] for task in profile["tasks"]] == sorted(
        task["samples"] for task in profile["tasks"]
    )
    for task in profile["tasks"]:
        argv = build_command(Path("/source"), task, profile, 8014)
        assert argv[argv.index("--eval-tasks") + 1] == task["name"]
        assert not any(
            flag in argv
            for flag in ("--eval-samples", "--limit-samples-mode", "--ci-mode")
        )
        overrides = json.loads(argv[argv.index("--vllm-override-args") + 1])
        assert "max_model_len" not in overrides
        assert "max_tokens" not in overrides
        assert overrides["tokenizer-mode"] == "deepseek_v41"
        assert (
            json.loads(overrides["default-chat-template-kwargs"])["reasoning_effort"]
            == "max"
        )


def test_no_cleanup_of_uncreated_namespace_or_daemons(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(
        "workflows.isolated_eval_docker.subprocess.run",
        lambda args, **kwargs: commands.append(args),
    )
    docker = IsolatedDocker(tmp_path / "storage", tmp_path / "logs")
    docker.close()
    assert commands == []


def test_digest_on_supported_python_versions(tmp_path):
    path = tmp_path / "file"
    path.write_bytes(b"abc")
    assert (
        digest(path)
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
