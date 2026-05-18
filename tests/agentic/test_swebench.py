#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.agentic.swebench import (  # noqa: E402
    SWEbenchRunConfig,
    _add_swebench_container_name_patch_to_env,
    _write_mini_sweagent_model_config,
    build_mini_sweagent_command,
    build_sweagent_command,
    build_swebench_harness_command,
    convert_sweagent_preds_to_jsonl,
    normalize_swebench_report,
    run,
)


def _swebench_config(tmp_path):
    return SWEbenchRunConfig(
        task_name="swe_bench_verified",
        dataset_name="SWE-bench/SWE-bench_Verified",
        dataset_split="test",
        sweagent_subset="verified",
        agent_backend="mini-swe-agent",
        model_name="openai/Qwen/Qwen3.6-27B",
        api_base="http://127.0.0.1:8000/v1",
        output_dir=tmp_path,
        sweagent_config="config/default.yaml",
        mini_config="swebench.yaml",
        mini_model_class="litellm",
        mini_environment_class="docker",
        n_concurrent_trials=2,
        max_workers=3,
        n_tasks=5,
        temperature=1.0,
        top_p=0.95,
        max_input_tokens=200 * 1024,
        max_output_tokens=None,
        completion_kwargs={"extra_body": {"top_k": 20}},
        swebench_timeout_sec=1800,
        shuffle=True,
        random_delay_multiplier=0.3,
        score_existing_predictions=False,
    )


def test_build_sweagent_command_uses_verified_subset_and_slice(tmp_path):
    config = _swebench_config(tmp_path)

    cmd = build_sweagent_command(
        config,
        sweagent_config_path=tmp_path / "sweagent_model_config.yaml",
        sweagent_output_dir=tmp_path / "sweagent",
    )

    assert cmd[1] == "run-batch"
    assert cmd[cmd.index("--instances.type") + 1] == "swe_bench"
    assert cmd[cmd.index("--instances.subset") + 1] == "verified"
    assert cmd[cmd.index("--instances.split") + 1] == "test"
    assert cmd[cmd.index("--instances.slice") + 1] == ":5"
    assert cmd[cmd.index("--num_workers") + 1] == "2"
    assert "--instances.shuffle=true" in cmd


def test_build_sweagent_command_resolves_source_config(monkeypatch, tmp_path):
    import evals.agentic.swebench as swebench_module

    config = _swebench_config(tmp_path)
    sweagent_source_dir = tmp_path / "SWE-agent"
    (sweagent_source_dir / "config").mkdir(parents=True)
    monkeypatch.setattr(
        swebench_module,
        "_get_sweagent_source_dir",
        lambda: sweagent_source_dir,
    )

    cmd = build_sweagent_command(
        config,
        sweagent_config_path=tmp_path / "sweagent_model_config.yaml",
        sweagent_output_dir=tmp_path / "sweagent",
    )

    assert cmd[cmd.index("--config") + 1] == str(
        sweagent_source_dir / "config" / "default.yaml"
    )


def test_build_mini_sweagent_command_uses_tool_config_and_slice(tmp_path):
    config = _swebench_config(tmp_path)

    cmd = build_mini_sweagent_command(
        config,
        mini_config_path=tmp_path / "mini_sweagent_model_config.yaml",
        mini_output_dir=tmp_path / "mini_sweagent",
    )

    assert cmd[1] == "swebench"
    assert cmd[cmd.index("--model") + 1] == "openai/Qwen/Qwen3.6-27B"
    assert cmd[cmd.index("--subset") + 1] == "verified"
    assert cmd[cmd.index("--split") + 1] == "test"
    assert cmd[cmd.index("--workers") + 1] == "2"
    assert cmd[cmd.index("--output") + 1] == str(tmp_path / "mini_sweagent")
    assert cmd[cmd.index("--environment-class") + 1] == "docker"
    assert cmd[cmd.index("--slice") + 1] == ":5"
    assert "--shuffle" in cmd
    assert "swebench.yaml" in cmd


def test_write_mini_sweagent_model_config_uses_tool_call_model(tmp_path):
    config = _swebench_config(tmp_path)

    config_path = _write_mini_sweagent_model_config(config)
    model_config = json.loads(config_path.read_text(encoding="utf-8"))

    assert model_config["model"]["model_class"] == "litellm"
    assert model_config["model"]["cost_tracking"] == "ignore_errors"
    assert model_config["model"]["model_kwargs"]["api_base"] == (
        "http://127.0.0.1:8000/v1"
    )
    assert model_config["model"]["model_kwargs"]["extra_body"] == {"top_k": 20}


def test_convert_sweagent_preds_to_official_jsonl(tmp_path):
    preds_path = tmp_path / "preds.json"
    predictions_path = tmp_path / "predictions.jsonl"
    preds_path.write_text(
        json.dumps(
            {
                "sympy__sympy-20590": {
                    "model_patch": "diff --git a/file.py b/file.py\n",
                }
            }
        ),
        encoding="utf-8",
    )

    records = convert_sweagent_preds_to_jsonl(
        preds_path, predictions_path, "openai/Qwen/Qwen3.6-27B"
    )

    assert records == [
        {
            "instance_id": "sympy__sympy-20590",
            "model_name_or_path": "openai/Qwen/Qwen3.6-27B",
            "model_patch": "diff --git a/file.py b/file.py\n",
        }
    ]
    assert json.loads(predictions_path.read_text(encoding="utf-8")) == records[0]


def test_build_swebench_harness_command_uses_official_runner(tmp_path):
    config = _swebench_config(tmp_path)

    cmd = build_swebench_harness_command(
        config,
        predictions_path=tmp_path / "predictions.jsonl",
        run_id="swe_bench_verified",
    )

    assert cmd[1:4] == ["-m", "swebench.harness.run_evaluation", "--dataset_name"]
    assert cmd[cmd.index("--dataset_name") + 1] == "SWE-bench/SWE-bench_Verified"
    assert cmd[cmd.index("--predictions_path") + 1] == str(
        tmp_path / "predictions.jsonl"
    )
    assert cmd[cmd.index("--max_workers") + 1] == "3"
    assert cmd[cmd.index("--timeout") + 1] == "1800"


def test_swebench_container_name_patch_prepends_pythonpath(tmp_path):
    env = _add_swebench_container_name_patch_to_env(
        tmp_path,
        {"PYTHONPATH": "/existing/path"},
    )

    patch_path = tmp_path / "swebench_harness_patch" / "sitecustomize.py"
    assert patch_path.exists()
    assert env["PYTHONPATH"].startswith(str(patch_path.parent))
    assert env["PYTHONPATH"].endswith("/existing/path")
    assert "get_safe_instance_container_name" in patch_path.read_text(encoding="utf-8")


def test_run_scores_existing_predictions_without_converting(monkeypatch, tmp_path):
    import evals.agentic.swebench as swebench_module

    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps(
            {
                "instance_id": "task-a",
                "model_name_or_path": "openai/Qwen/Qwen3.6-27B",
                "model_patch": "diff --git a/file.py b/file.py\n",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    harness_report_path = tmp_path / "openai__Qwen__Qwen3.6-27B.swe_bench_verified.json"
    harness_report_path.write_text(
        json.dumps(
            {
                "submitted_instances": 1,
                "resolved_instances": 0,
                "submitted_ids": ["task-a"],
                "resolved_ids": [],
            }
        ),
        encoding="utf-8",
    )
    config = SWEbenchRunConfig(
        **{
            **_swebench_config(tmp_path).__dict__,
            "score_existing_predictions": True,
        }
    )

    monkeypatch.setattr(swebench_module, "_run_command", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        swebench_module,
        "convert_sweagent_preds_to_jsonl",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("conversion should be skipped")
        ),
    )

    assert run(config) == 0
    assert (tmp_path / "result.json").exists()


def test_normalize_swebench_report_writes_agentic_result(tmp_path):
    config = _swebench_config(tmp_path)
    harness_report_path = tmp_path / "openai__Qwen__Qwen3.6-27B.swe_bench_verified.json"
    predictions_path = tmp_path / "predictions.jsonl"
    result_path = tmp_path / "result.json"
    harness_report_path.write_text(
        json.dumps(
            {
                "submitted_instances": 3,
                "resolved_instances": 2,
                "submitted_ids": ["task-a", "task-b", "task-c"],
                "resolved_ids": ["task-a", "task-c"],
                "unresolved_ids": ["task-b"],
            }
        ),
        encoding="utf-8",
    )

    result = normalize_swebench_report(
        harness_report_path,
        result_path,
        config,
        predictions_path,
    )

    eval_stats = next(iter(result["stats"]["evals"].values()))
    assert eval_stats["n_trials"] == 3
    assert eval_stats["metrics"][0]["mean"] == 2 / 3
    assert eval_stats["pass_at_k"]["1"] == 2 / 3
    assert len(result["trial_results"]) == 3
    assert result_path.exists()
