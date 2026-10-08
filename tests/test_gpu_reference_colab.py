# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for scripts/gpu_reference_colab (the Colab GPU-reference workflow).

Covers the pure-Python parts of gpuref.py (spec -> vllm serve args, score
summary parsing from a real Shield eval report, provenance helpers) and a
syntax check of the two shell scripts. Nothing here talks to Colab.
"""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = REPO_ROOT / "scripts" / "gpu_reference_colab"
GPUREF_PATH = SCRIPT_DIR / "gpuref.py"
FIXTURE = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "gpu_reference_colab"
    / "report_evals_solar_p300x2.json"
)


def _load_gpuref():
    spec = importlib.util.spec_from_file_location("gpuref_under_test", GPUREF_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gpuref = _load_gpuref()


# --- spec reading -> vllm serve args ---------------------------------------


def test_vllm_flags_skips_explicit_and_tt_only_keys():
    args = {
        "model": "org/m",
        "served_model_name": "org/m",
        "max_model_len": "4096",
        "max_num_seqs": "32",
        "dtype": "float16",
        "block_size": "64",
        "max-log-len": "32",
        "additional_config": '{"tt": {}}',
        "max_num_batched_tokens": "4096",
        "seed": "9472",
        "revision": "abc",
    }
    assert gpuref.vllm_flags(args) == [
        "--max-num-batched-tokens",
        "4096",
        "--seed",
        "9472",
        "--revision",
        "abc",
    ]


def test_vllm_flags_bools_dicts_and_none():
    args = {
        "enable_auto_tool_choice": True,
        "enable_prefix_caching": False,
        "hf_overrides": {"a": 1},
        "tool_call_parser": None,
    }
    assert gpuref.vllm_flags(args) == [
        "--enable-auto-tool-choice",
        "--no-enable-prefix-caching",
        "--hf-overrides",
        '{"a": 1}',
    ]


# The GPU entries mirror the Quetzal P300X2 rows: max_context = min(row,
# native max_position_embeddings), max_concurrency = row, same HF revision.
EXPECTED_GPU_SPECS = {
    "upstage/SOLAR-10.7B-Instruct-v1.0": (
        4096,
        32,
        "090d405e6fe7b4c7b0b999a67f5c6dee0514a7ad",
    ),
    "meta-llama/Llama-3.2-1B": (
        131072,
        32,
        "4e20de362430cd3b72f300e6b0f18e50e7166e08",
    ),
    "Qwen/Qwen1.5-0.5B-Chat": (
        32768,
        32,
        "4d14e384a4b037942bb3f3016665157c8bcb70ea",
    ),
}


@pytest.mark.parametrize("model", sorted(EXPECTED_GPU_SPECS))
def test_serve_plan_from_dev_gpu_spec(model):
    # A fresh interpreter, as on the VM: MODEL_SPECS is fixed at import time
    # and other tests import the prod catalog.
    out = subprocess.run(
        [sys.executable, str(GPUREF_PATH), "serve-plan", "--model", model],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    plan = json.loads(out.stdout)
    max_context, max_concurrency, revision = EXPECTED_GPU_SPECS[model]
    name = model.split("/")[1]

    assert plan["model_id"] == f"id_quetzal_{name}_gpu"
    assert plan["max_context"] == max_context
    assert plan["max_concurrency"] == max_concurrency
    assert plan["revision"] == revision
    argv = plan["vllm_serve_args"]
    assert argv[:12] == [
        "serve",
        model,
        "--served-model-name",
        model,
        "--port",
        "8000",
        "--max-model-len",
        str(max_context),
        "--max-num-seqs",
        str(max_concurrency),
        "--dtype",
        "bfloat16",
    ]
    assert argv[argv.index("--revision") + 1] == revision
    assert argv[argv.index("--tokenizer-revision") + 1] == revision
    for dropped in ("--block-size", "--additional-config", "--max-log-len"):
        assert dropped not in argv
    assert set(plan["dropped_tt_only_vllm_args"]) == {
        "block_size",
        "max-log-len",
        "additional_config",
    }


def test_serve_plan_rejects_model_without_gpu_spec():
    out = subprocess.run(
        [sys.executable, str(GPUREF_PATH), "serve-plan", "--model", "org/not-a-model"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert out.returncode != 0
    assert "No model spec matches" in out.stderr


# --- score summary ----------------------------------------------------------


def test_summary_rows_from_shield_report_layout(tmp_path):
    report_dir = tmp_path / "report_evals_upstage__SOLAR-10.7B-Instruct-v1.0_X_1"
    report_dir.mkdir()
    shutil.copy(FIXTURE, report_dir / "report_113240547276.json")
    (report_dir / "model_spec_113240547276.json").write_text("{}")

    rows = gpuref.summary_rows(tmp_path)
    by_task = {r["task"]: r for r in rows}
    assert set(by_task) == {"leaderboard_ifeval", "leaderboard_math_hard", "mmlu_pro"}
    assert {r["model"] for r in rows} == {"upstage/SOLAR-10.7B-Instruct-v1.0"}
    assert {r["device"] for r in rows} == {"P300X2"}
    assert by_task["mmlu_pro"]["score"] == pytest.approx(13.887965425531915)
    assert by_task["leaderboard_ifeval"]["score"] == pytest.approx(42.329020332717185)
    assert by_task["leaderboard_math_hard"]["published_score"] == 5.22
    assert by_task["mmlu_pro"]["gpu_reference_score"] is None

    text = gpuref.format_summary(rows)
    assert "mmlu_pro" in text and "13.89" in text and "42.33" in text


def test_summary_local_layout_newest_report_wins(tmp_path):
    payload = json.loads(FIXTURE.read_text())
    data_dir = tmp_path / "workflow_logs" / "reports_output" / "evals" / "data"
    data_dir.mkdir(parents=True)
    old = json.loads(json.dumps(payload))
    new = json.loads(json.dumps(payload))
    old["metadata"]["generated_at"] = "2026-10-01T00:00:00+00:00"
    new["metadata"]["generated_at"] = "2026-10-09T00:00:00+00:00"
    new["metadata"]["device"] = "GPU"
    for section in new["sections"]:
        section["data"]["score"] = 50.0
    (data_dir / "report_data_old.json").write_text(json.dumps(old))
    (data_dir / "report_data_new.json").write_text(json.dumps(new))
    # Non-report JSON and reports without eval blocks are ignored.
    (data_dir / "report_data_bench.json").write_text(
        json.dumps(
            {"metadata": {"model_repo": "x/y"}, "sections": [{"kind": "benchmarks"}]}
        )
    )
    (tmp_path / "report_status.json").write_text("[1, 2]")

    rows = gpuref.summary_rows(tmp_path)
    assert len(rows) == 3
    assert {r["device"] for r in rows} == {"GPU"}
    assert {r["score"] for r in rows} == {50.0}


def test_summary_empty(tmp_path):
    assert gpuref.summary_rows(tmp_path) == []
    assert gpuref.format_summary([]) == "No TTIS eval reports found."


# --- provenance helpers ------------------------------------------------------


def test_lm_eval_commit_pinned_matches_requirements():
    commit = gpuref.lm_eval_commit_pinned(REPO_ROOT)
    assert commit is not None and len(commit) == 40
    assert commit in (REPO_ROOT / "requirements" / "evals-common.txt").read_text()


def test_lm_eval_commit_installed_reads_hidden_venv(tmp_path):
    dist = (
        tmp_path
        / ".workflow_venvs"
        / ".venv_evals_common"
        / "lib"
        / "python3.10"
        / "site-packages"
        / "lm_eval-0.4.8.dist-info"
    )
    dist.mkdir(parents=True)
    (dist / "direct_url.json").write_text(
        json.dumps(
            {"url": "https://github.com/x/y.git", "vcs_info": {"commit_id": "f" * 40}}
        )
    )
    assert gpuref.lm_eval_commit_installed(tmp_path) == "f" * 40
    assert gpuref.lm_eval_commit_installed(tmp_path / "missing") is None


def test_hf_token_never_in_argv():
    """The CLI takes no token argument; the token is read from env/file only."""
    source = GPUREF_PATH.read_text()
    assert "--token" not in source
    assert "--hf-token" not in source


# --- shell scripts -----------------------------------------------------------


@pytest.mark.parametrize("script", ["colab_gpu_reference.sh", "remote_runner.sh"])
def test_shell_scripts_parse(script):
    subprocess.run(["bash", "-n", str(SCRIPT_DIR / script)], check=True)


@pytest.mark.skipif(
    shutil.which("shellcheck") is None, reason="shellcheck not installed"
)
@pytest.mark.parametrize("script", ["colab_gpu_reference.sh", "remote_runner.sh"])
def test_shell_scripts_shellcheck(script):
    subprocess.run(["shellcheck", str(SCRIPT_DIR / script)], check=True)
