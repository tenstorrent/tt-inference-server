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
import os
import shlex
import shutil
import stat
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
        str(plan["max_num_seqs"]),
        "--dtype",
        "bfloat16",
    ]
    # Both are small, so the GPU throughput policy raises max_num_seqs -- when
    # the Hub (or the token, for gated Llama) is reachable to size the model.
    assert plan["max_num_seqs"] in (max_concurrency, gpuref.GPU_HIGH_CONCURRENCY)
    assert plan["client_concurrency"] == plan["max_num_seqs"]
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


def test_summary_reports_gpu_from_provenance(tmp_path):
    report_dir = tmp_path / "workflow_logs" / "reports_output" / "evals" / "data"
    report_dir.mkdir(parents=True)
    shutil.copy(FIXTURE, report_dir / "report_data_x.json")
    model_dir = tmp_path / "upstage__SOLAR-10.7B-Instruct-v1.0"
    model_dir.mkdir()
    (model_dir / "provenance.json").write_text(
        json.dumps(
            {
                "model": "upstage/SOLAR-10.7B-Instruct-v1.0",
                "gpu": {"name": "NVIDIA A100-SXM4-40GB"},
            }
        )
    )
    rows = gpuref.summary_rows(tmp_path)
    assert {r["gpu"] for r in rows} == {"NVIDIA A100-SXM4-40GB"}
    assert "NVIDIA A100-SXM4-40GB" in gpuref.format_summary(rows)


def test_memory_estimate_and_fit():
    # SOLAR-10.7B: 48 layers, 32 heads, 8 KV heads, hidden 4096 (head_dim 128).
    config = {
        "num_hidden_layers": 48,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 4096,
    }
    assert gpuref.kv_bytes_per_token(config) == 2 * 48 * 8 * 128 * 2
    est = gpuref.memory_estimate(10_731_524_096, config, 4096, 32)
    assert est["weights_gib"] == pytest.approx(19.99, abs=0.01)
    assert est["kv_gib_one_seq"] == pytest.approx(0.75, abs=0.01)
    assert est["kv_gib_all_seqs"] == pytest.approx(24.0, abs=0.01)
    verdicts = gpuref.gpu_verdicts(est, 4096)
    assert verdicts["H100"] == verdicts["A100"] == "fits" and verdicts["T4"] == "no"
    # Explicit head_dim wins over hidden_size // heads; MHA defaults kv_heads.
    assert gpuref.kv_bytes_per_token(
        {
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "hidden_size": 64,
            "head_dim": 32,
        }
    ) == (2 * 2 * 4 * 32 * 2)


def test_gpu_verdicts_fit_cap_and_no():
    config = {  # Qwen2.5-32B-like: 64 layers, 40 heads, 8 KV heads, head_dim 128
        "num_hidden_layers": 64,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "hidden_size": 5120,
    }
    est = gpuref.memory_estimate(32_763_876_352, config, 32768, 32)
    verdicts = gpuref.gpu_verdicts(est, 32768)
    assert verdicts["A100"] == "no"
    assert (
        isinstance(verdicts["H100"], dict) and 2048 <= verdicts["H100"]["cap"] < 32768
    )
    small = gpuref.memory_estimate(619_570_176, config, 4096, 32)
    assert gpuref.gpu_verdicts(small, 4096)["A100"] == "fits"
    # Weights alone (KV unknown) too big for every GPU: a 70B.
    weights_only = {
        "weights_gib": 131.4,
        "need_gib": 135.4,
        "kv_bytes_per_token": None,
    }
    assert set(gpuref.gpu_verdicts(weights_only, 131072).values()) == {"no"}


def test_summary_empty(tmp_path):
    assert gpuref.summary_rows(tmp_path) == []
    assert gpuref.format_summary([]) == "No TTIS eval reports found."


# --- provenance helpers ------------------------------------------------------


def test_provenance_records_pinned_and_installed_lm_eval(tmp_path):
    """The installed commit lives in a hidden .venv_* dir, which glob.glob
    would skip; build_provenance must still find it."""
    ttis = tmp_path / "ttis"
    (ttis / "requirements").mkdir(parents=True)
    (ttis / "requirements" / "evals-common.txt").write_text(
        "git+https://github.com/x/lm-evaluation-harness.git@"
        + "e" * 40
        + "#egg=lm-eval\n"
    )
    dist = ttis / ".workflow_venvs" / ".venv_evals_common" / "lib" / "python3.10"
    dist = dist / "site-packages" / "lm_eval-0.4.8.dist-info"
    dist.mkdir(parents=True)
    (dist / "direct_url.json").write_text(
        json.dumps({"vcs_info": {"commit_id": "f" * 40}})
    )
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps({"model_id": "id_x", "vllm_serve_args": ["serve", "a/b"]})
    )
    out = tmp_path / "provenance.json"
    gpuref.main(
        ["provenance", "--out", str(out), "--model", "a/b", "--ttis-dir", str(ttis),
         "--plan", str(plan), "--vllm-python", sys.executable, "--vllm-bin", "vllm",
         "--status", "ok", "--started-at", "t0", "--finished-at", "t1",
         "--max-model-len-cap", "22960"]
    )  # fmt: skip
    prov = json.loads(out.read_text())
    assert prov["lm_eval_commit_pinned"] == "e" * 40
    assert prov["lm_eval_commit_installed"] == "f" * 40
    assert prov["max_model_len_cap"] == 22960
    assert prov["vllm_serve_command"] == ["vllm", "serve", "a/b"]


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


# --- driver-side helpers ------------------------------------------------------


def _row(model, verdicts, **extra):
    row = {
        "model": model,
        "ok": True,
        "model_id": "id_x",
        "max_context": 4096,
        "max_concurrency": 32,
        "revision": "r",
        "gated": False,
        "needs_token": False,
        "have_token": True,
        "memory": {
            "weights_gib": 20.0,
            "kv_gib_one_seq": 1.0,
            "need_gib": 25.0,
            "verdicts": verdicts,
        },
    }
    row.update(extra)
    return row


def test_preflight_keeps_gpus_that_hold_every_model():
    small = _row("a/small", {"H100": "fits", "A100": "fits"})
    big = _row("a/big", {"H100": "fits", "A100": {"cap": 20000}})
    huge = _row("a/huge", {"H100": "fits", "A100": "no"})
    usable, ok, _ = gpuref.preflight([small, big], ["H100", "A100"])
    assert ok and usable == [
        "H100",
        "A100",
    ]  # capped A100 is a fallback, checked by evidence
    usable, ok, _ = gpuref.preflight([small, huge], ["H100", "A100"])
    assert ok and usable == ["H100"]
    # With A100 alone, a capped fit is accepted (and reported).
    usable, ok, msgs = gpuref.preflight([big], ["A100"])
    assert ok and usable == ["A100"] and any("cap" in m for m in msgs)


def test_preflight_refuses_models_that_fit_no_listed_gpu():
    seventy_b = _row("a/70b", {"H100": "no", "A100": "no"})
    _, ok, msgs = gpuref.preflight([seventy_b], ["H100", "A100"])
    assert not ok and any("multi-GPU" in m for m in msgs)
    thirty_b = _row("a/32b", {"H100": "fits", "A100": "no"})
    _, ok, msgs = gpuref.preflight([thirty_b], ["A100"])
    assert not ok and any("add H100" in m for m in msgs)


def test_preflight_access_and_spec_failures():
    no_spec = {"model": "a/b", "ok": False, "error": "no GPU spec"}
    assert not gpuref.preflight([no_spec], ["H100"])[1]
    gated = _row(
        "a/g", {"H100": "fits"}, gated=True, needs_token=True, have_token=False
    )
    assert not gpuref.preflight([gated], ["H100"])[1]
    unreadable = _row(
        "a/g", {"H100": "fits"}, gated=True, needs_token=True, token_can_read=False
    )
    assert not gpuref.preflight([unreadable], ["H100"])[1]
    assert not gpuref.preflight([_row("a/m", {}, memory=None)], ["H100"])[1]


SNIPPETS = SCRIPT_DIR / "snippets"


def _snippet(name, args=None):
    """What the driver sends: snippets/NAME.py, with launch's ARGS prepended
    the way colab_gpu_reference.sh writes it."""
    head = "ARGS = [" + "".join(f"'{a}', " for a in args) + "]\n" if args else ""
    return head + (SNIPPETS / f"{name}.py").read_text()


@pytest.mark.parametrize(
    "name", ["prepare", "install_token", "launch", "status", "pack"]
)
def test_snippets_compile_and_never_carry_a_token(name):
    args = ["--ttis-sha", "a" * 40, "org/model"] if name == "launch" else None
    src = _snippet(name, args)
    compile(src, name, "exec")
    assert 'W = "/content/gpuref"' in src
    assert "HF_TOKEN" not in src


def test_launch_snippet_detaches_with_runner_args():
    src = _snippet("launch", ["--ttis-sha", "b" * 40, "org/model"])
    assert src.startswith("ARGS = ['--ttis-sha', '" + "b" * 40 + "', 'org/model', ]")
    assert "start_new_session=True" in src
    driver = (SCRIPT_DIR / "colab_gpu_reference.sh").read_text()
    assert 'printf "\'%s\', " "${arg}"' in driver


def test_status_snippet_reports_state(tmp_path):
    src = _snippet("status").replace("/content/gpuref", str(tmp_path))

    def state():
        out = subprocess.run(
            [sys.executable, "-c", src], capture_output=True, text=True, check=True
        )
        return gpuref.parse_status(out.stdout)

    assert state()[0] == "ABSENT"
    (tmp_path / "runner.pid").write_text("999999")  # launched; no such process
    assert state()[0] == "DIED"
    (tmp_path / "FAILED").write_text("x")
    (tmp_path / "runner.log").write_text("line1\nline2\n")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "status.json").write_text(
        json.dumps({"a/b": {"status": "ok"}, "c/d": {"status": "failed"}})
    )
    status, lines, done, _ = state()
    assert status == "FAILED" and "  | line2" in lines and done == 2


def test_parse_status_and_balance():
    state, lines, done, phase = gpuref.parse_status(
        "noise\nGPUREF_STATE=RUNNING\nGPUREF_DONE=1\nGPUREF_PHASE=model 1/3 x: evals\nGPUREF_LOG| hello\n"
    )
    assert state == "RUNNING" and done == 1 and phase == "model 1/3 x: evals"
    assert lines == ["  phase: model 1/3 x: evals", "  | hello"]
    assert gpuref.parse_status("garbage")[0] is None
    usage = "Current balance: 4988.40 compute units\nUsage rate: 5.30/hr\n"
    assert gpuref.parse_balance(usage) == 4988.40
    assert gpuref.parse_balance("nothing") is None


def test_runner_helpers(tmp_path):
    path = tmp_path / "status.json"
    gpuref.record_status(path, "a/b", "ok", "")
    gpuref.record_status(path, "c/d", "failed", "run.py exited 1")
    assert json.loads(path.read_text()) == {
        "a/b": {"status": "ok", "note": None},
        "c/d": {"status": "failed", "note": "run.py exited 1"},
    }
    for body, rc in (('{"data": [{"id": "a/b"}]}', 0), ("{}", 1), ("<html>", 1)):
        served = subprocess.run(
            [sys.executable, str(GPUREF_PATH), "served", "a/b"], input=body, text=True
        )
        assert served.returncode == rc


# --- request-length evidence for capped runs ---------------------------------


def _write_samples(path, rows):
    path.write_text(
        "".join(
            json.dumps(
                {"arguments": {"gen_args_0": {"arg_0": p, "arg_1": {}}}, "resps": [[r]]}
            )
            + "\n"
            for p, r in rows
        )
    )


def test_request_token_stats_per_task(tmp_path):
    _write_samples(
        tmp_path / "samples_mmlu_pro_law_2026-10-09T00-14-03.525271.jsonl",
        [("a b c", "d e"), ("a", "")],
    )
    _write_samples(
        tmp_path / "samples_leaderboard_ifeval_2026-10-08T23-28-26.301824.jsonl",
        [("x " * 10, "y " * 5)],
    )
    stats = gpuref.request_token_stats(tmp_path, lambda t: len(t.split()))
    assert stats["mmlu_pro_law"] == {"samples": 2, "max_tokens": 5, "empty": 1}
    assert stats["leaderboard_ifeval"]["max_tokens"] == 15


def test_cap_bound_needs_clean_logs_and_headroom(tmp_path):
    ok = {
        "max_request_tokens": {"mmlu_pro_law": {"max_tokens": 4033}},
        "rejection_log_lines": [],
    }
    assert not gpuref.cap_bound(ok, 22960)
    assert gpuref.cap_bound(ok, 4090)  # within the 64-token reserve of the cap
    assert gpuref.cap_bound(dict(ok, rejection_log_lines=["x: 400 Bad Request"]), 22960)
    assert gpuref.cap_bound(None, 22960)
    log = tmp_path / "run.log"
    log.write_text("fine\nThis model's maximum context length is 22960 tokens\n")
    assert gpuref.rejection_lines([log, tmp_path / "missing.log"]) == [
        "run.log: This model's maximum context length is 22960 tokens"
    ]


# --- sweep: target selection, placement, derived specs, proposed patch --------


def _task(task, published=None, gpu_ref=None, requested=None, venv="EVALS_COMMON"):
    return {"task": task, "published": published, "gpu_ref": gpu_ref,
            "requested": requested, "venv": venv}  # fmt: skip


def _spec_row(device, impl="quetzal", default=False, ctx=32768, rev="r1"):
    return {"device": device, "impl": impl, "default": default, "max_context": ctx,
            "max_concurrency": 32, "revision": rev}  # fmt: skip


SWEEP_TASKS = {
    "org/gpu-model": [
        _task("ungated"),  # no published, no GPU reference
        _task("published", published=40.0),  # gated by a published score
        _task("referenced", gpu_ref=50.0),  # already has a GPU reference
        _task(
            "flagged", published=56.7, gpu_ref=66.1, requested="different checkpoint"
        ),
        _task("agentic", venv="EVALS_AGENTIC"),
    ],
    "org/quetzal-only": [_task("ungated")],
    "org/no-rows": [_task("ungated")],
}
SWEEP_ROWS = {
    "org/gpu-model": [_spec_row("GPU"), _spec_row("P300X2")],
    "org/quetzal-only": [_spec_row("P300X2", ctx=131072)],
}


def test_select_targets_ungated_flagged_and_exclusions():
    targets, skipped = gpuref.select_targets(SWEEP_TASKS, SWEEP_ROWS)
    picked = {(t["model"], t["task"]): t for t in targets}
    assert set(picked) == {
        ("org/gpu-model", "ungated"),
        ("org/gpu-model", "flagged"),
        ("org/quetzal-only", "ungated"),
    }
    assert picked[("org/gpu-model", "ungated")]["reason"] == "ungated"
    assert (
        picked[("org/gpu-model", "flagged")]["reason"]
        == "requested: different checkpoint"
    )
    assert picked[("org/gpu-model", "ungated")]["spec"] == "existing"
    assert picked[("org/quetzal-only", "ungated")]["spec"] == "derived"
    why = {(s["model"], s["task"]): s["skip"] for s in skipped}
    assert "EVALS_AGENTIC" in why[("org/gpu-model", "agentic")]
    assert "no GPU spec" in why[("org/no-rows", "ungated")]
    # Already referenced: only with --refresh.
    targets, _ = gpuref.select_targets(SWEEP_TASKS, SWEEP_ROWS, refresh=True)
    refreshed = {
        t["task"]: t["reason"] for t in targets if t["model"] == "org/gpu-model"
    }
    assert refreshed["referenced"] == "refresh"
    assert "published" not in refreshed


def test_gpu_group_placement():
    assert gpuref.gpu_group({"A100": "fits", "H100": "fits"}) == "A100"
    assert gpuref.gpu_group({"A100": {"cap": 20000}, "H100": "fits"}) == "A100"
    assert gpuref.gpu_group({"A100": "no", "H100": {"cap": 21000}}) == "H100"
    assert gpuref.gpu_group({"A100": "no", "H100": "no"}) is None  # 70B: skipped


def test_mid_size_models_short_of_a100_kv_room_go_to_h100():
    config = {  # Qwen3-14B
        "num_attention_heads": 40, "num_key_value_heads": 8, "head_dim": 128,
        "num_hidden_layers": 40, "hidden_size": 5120, "intermediate_size": 17408,
    }  # fmt: skip
    est = gpuref.memory_estimate(14_770_000_000, config, 40960, 32)
    verdicts = gpuref.gpu_verdicts(est, 40960)
    assert gpuref.gpu_group(verdicts) == "A100"
    group, why = gpuref.gpu_placement(verdicts, 14_770_000_000, est, 32)
    assert group == "H100" and "< row 32" in why
    # Row value already within the A100's room, or outside 9e9-16e9: unchanged.
    assert gpuref.gpu_placement(verdicts, 14_770_000_000, est, 4)[0] == "A100"
    assert gpuref.gpu_placement(verdicts, 8_000_000_000, est, 32)[0] == "A100"
    assert gpuref.gpu_placement(verdicts)[0] == "A100"
    assert gpuref.gpu_placement({"A100": "no", "H100": "fits"}) == (
        "H100", "needs an 80 GB GPU",
    )  # fmt: skip


def test_probe_first_runs_the_costliest_moved_model_first():
    groups = {"H100": ["a-1m", "b-32b", "c-14b", "d-14b"], "A100": ["e", "f"]}
    placed = {
        "a-1m": {"moved_to_h100": True, "est_hours": 1.45},
        "b-32b": {"moved_to_h100": False, "est_hours": 3.0},
        "c-14b": {"moved_to_h100": True, "est_hours": 1.85},
        "d-14b": {"moved_to_h100": True, "est_hours": 0.93},
        "e": {"moved_to_h100": False, "est_hours": 1.0},
        "f": {"moved_to_h100": False, "est_hours": 2.0},
    }
    gpuref.probe_first(groups, placed)
    assert groups == {"H100": ["c-14b", "a-1m", "b-32b", "d-14b"], "A100": ["e", "f"]}


def test_too_big_for_any_gpu_is_skipped_by_verdicts():
    config = {"num_hidden_layers": 80, "num_attention_heads": 64,
              "num_key_value_heads": 8, "hidden_size": 8192, "intermediate_size": 29568}  # fmt: skip
    est = gpuref.memory_estimate(72_700_000_000, config, 32768, 32)  # a 72B model
    assert gpuref.gpu_group(gpuref.gpu_verdicts(est, 32768)) is None


def test_derive_gpu_spec_prefers_quetzal_row_and_caps_native_context():
    rows = [_spec_row("P300X2", ctx=131072, rev="abc"),
            _spec_row("T3K", impl="tt_transformers", default=True, ctx=8192, rev=None)]  # fmt: skip
    spec = gpuref.derive_gpu_spec(rows, native_ctx=32768)
    assert spec["impl"] == "quetzal" and spec["max_context"] == 32768
    assert spec["max_concurrency"] == 32 and spec["revision"] == "abc"
    tt_only = [
        _spec_row("T3K", impl="tt_transformers", default=True, ctx=131072, rev=None)
    ]
    spec = gpuref.derive_gpu_spec(tt_only, native_ctx=32768)
    assert spec["impl"] == "tt_transformers" and spec["max_context"] == 32768
    assert gpuref.derive_gpu_spec([_spec_row("GPU")], 4096) is None


def test_write_derived_specs_regenerates_a_loadable_block():
    import yaml

    base = {"impl": "quetzal", "max_context": 4096, "max_concurrency": 32,
            "revision": "abc", "source": "quetzal P300X2 row"}  # fmt: skip
    one_m = dict(base, hf_overrides={"dual_chunk_attention_config": None})
    text = gpuref.write_derived_specs(
        "templates:\n- weights:\n  - org/x\n", {"org/m": base}
    )
    assert gpuref.block_models(text) == ["org/m"]
    # Merged by model: org/m is rewritten in place, org/one-m is added.
    text = gpuref.write_derived_specs(text, {"org/m": base, "org/one-m": one_m})
    assert text.count(gpuref.SPEC_BEGIN) == 1
    assert gpuref.block_models(text) == ["org/m", "org/one-m"]
    entries = {
        e["weights"][0]: e["device_model_specs"][0]
        for e in yaml.safe_load(text)["templates"]
        if "device_model_specs" in e
    }
    assert entries["org/m"] == {"device": "GPU", "max_concurrency": 32, "max_context": 4096,
                                "default_impl": True,
                                "vllm_args": {"revision": "abc", "tokenizer_revision": "abc"}}  # fmt: skip
    assert entries["org/one-m"]["vllm_args"]["hf_overrides"] == {
        "dual_chunk_attention_config": None
    }
    # The overrides reach vllm serve as JSON.
    flags = gpuref.vllm_flags(entries["org/one-m"]["vllm_args"])
    assert (
        flags[flags.index("--hf-overrides") + 1]
        == '{"dual_chunk_attention_config": null}'
    )


def test_write_specs_for_subset_b_keeps_a():
    """A sweep derives specs only for the models it selects; the others'
    generated entries survive unless --prune."""
    import yaml

    base = {"impl": "quetzal", "max_context": 4096, "max_concurrency": 32,
            "revision": "abc", "source": "quetzal P300X2 row"}  # fmt: skip
    a, b = dict(base), dict(base, max_context=8192)
    head = "templates:\n- weights:\n  - org/x\n"
    text = gpuref.write_derived_specs(head, {"org/a": a})
    text = gpuref.write_derived_specs(text, {"org/b": b})
    assert gpuref.block_models(text) == ["org/a", "org/b"]
    assert (
        gpuref.write_derived_specs(text, {"org/b": b}) == text
    )  # idempotent: not stale
    text = gpuref.write_derived_specs(text, {"org/a": dict(a, max_context=2048)})
    assert "max_context: 2048" in text
    assert gpuref.block_models(text) == ["org/a", "org/b"]
    assert "max_context: 8192" in text  # b untouched
    pruned = gpuref.write_derived_specs(text, {"org/b": b}, prune=True)
    assert gpuref.block_models(pruned) == ["org/b"]
    assert gpuref.write_derived_specs(text, {}, prune=True) == head + "\n"
    yaml.safe_load(text)  # the block still loads


def test_generator_never_duplicates_a_handwritten_gpu_spec():
    """A model with a reviewed GPU template outside the generated block gets no
    generated entry: not added by a merge, and an existing one is dropped."""
    import yaml

    base = {"impl": "quetzal", "max_context": 4096, "max_concurrency": 32,
            "revision": "abc", "source": "quetzal P300X2 row"}  # fmt: skip
    head = (
        "templates:\n- weights:\n  - org/reviewed\n  impl: quetzal\n"
        "  device_model_specs:\n  - device: GPU\n    max_context: 2048\n"
    )
    assert gpuref.handwritten_gpu_models(head) == {"org/reviewed"}
    text = gpuref.write_derived_specs(head, {"org/reviewed": base, "org/new": base})
    assert gpuref.block_models(text) == ["org/new"]
    # An older block that already holds a copy loses it on the next write.
    stale = (
        head
        + gpuref.SPEC_BEGIN
        + "\n"
        + gpuref.derived_specs_yaml({"org/reviewed": base, "org/new": base})
        + "\n"
        + gpuref.SPEC_END
        + "\n"
    )
    assert gpuref.handwritten_gpu_models(stale) == {"org/reviewed"}  # block excluded
    assert gpuref.block_models(gpuref.write_derived_specs(stale, {})) == ["org/new"]
    yaml.safe_load(text)


def test_multimodal_checkpoints_are_served_text_only():
    import yaml

    assert gpuref.derive_text_only({"vision_config": {}, "text_config": {}}) == {
        "image": 0,
        "video": 0,
    }
    assert gpuref.derive_text_only({"hidden_size": 8}) is None
    d = {"impl": "quetzal", "max_context": 4096, "max_concurrency": 32,
         "revision": "abc", "source": "row",
         "limit_mm_per_prompt": {"image": 0, "video": 0}}  # fmt: skip
    entry = yaml.safe_load("templates:\n" + gpuref.derived_specs_yaml({"org/mm": d}))
    args = entry["templates"][0]["device_model_specs"][0]["vllm_args"]
    assert args["limit_mm_per_prompt"] == {"image": 0, "video": 0}
    flags = gpuref.vllm_flags(args)
    assert flags[flags.index("--limit-mm-per-prompt") + 1] == '{"image": 0, "video": 0}'


def test_write_specs_keeps_every_committed_entry():
    """The committed block is unchanged by a write for one of its models."""
    text = (REPO_ROOT / "workflows" / "model_specs" / "dev" / "llm.yaml").read_text()
    models = gpuref.block_models(text)
    entries = gpuref.block_entries(text)
    assert len(models) > 1 and set(entries) == set(models)
    assert gpuref.write_derived_specs(text, {}) == text


def test_derive_overrides_disables_dual_chunk_attention_only_when_present():
    assert gpuref.derive_overrides(
        {"dual_chunk_attention_config": {"chunk_size": 262144}}
    ) == {"dual_chunk_attention_config": None}
    assert gpuref.derive_overrides({"max_position_embeddings": 32768}) == {}


def test_resume_needs_the_same_serve_args_and_every_task(tmp_path):
    session = tmp_path / "gpuref-sweep-a100-x"
    data = session / "workflow_logs" / "reports_output" / "evals" / "data"
    data.mkdir(parents=True)
    report = json.loads(FIXTURE.read_text())  # SOLAR: ifeval, math_hard, mmlu_pro
    (data / "report_data_x.json").write_text(json.dumps(report))
    model = report["metadata"]["model_repo"]
    prov = {
        "model": model,
        "status": "ok",
        "vllm_serve_command": ["vllm", "serve", model, "--x", "1"],
        "eval_fingerprint": "fp-a",
    }
    results = session / "results"
    argv = ["serve", model, "--x", "1"]
    assert gpuref.run_covers(prov, results, argv, ["mmlu_pro"], "fp-a")
    assert not gpuref.run_covers(
        prov, results, ["serve", model, "--x", "2"], ["mmlu_pro"], "fp-a"
    )
    assert not gpuref.run_covers(prov, results, argv, ["humaneval"], "fp-a")
    # --max-num-seqs only schedules requests: a run at another value still counts.
    prov["vllm_serve_command"] += ["--max-num-seqs", "32"]
    assert gpuref.run_covers(
        prov, results, argv + ["--max-num-seqs", "256"], ["mmlu_pro"], "fp-a"
    )


def test_resume_needs_the_same_eval_fingerprint(tmp_path):
    """Other eval settings, lm-eval or vLLM (another fingerprint) never count,
    nor does a run recorded before fingerprints existed, nor no fingerprint."""
    session = tmp_path / "gpuref-sweep-a100-x"
    data = session / "workflow_logs" / "reports_output" / "evals" / "data"
    data.mkdir(parents=True)
    report = json.loads(FIXTURE.read_text())
    (data / "report_data_x.json").write_text(json.dumps(report))
    model = report["metadata"]["model_repo"]
    argv = ["serve", model, "--x", "1"]
    prov = {"model": model, "status": "ok", "vllm_serve_command": ["vllm"] + argv,
            "eval_fingerprint": "fp-a"}  # fmt: skip
    results = session / "results"
    assert gpuref.run_covers(prov, results, argv, ["mmlu_pro"], "fp-a")
    assert not gpuref.run_covers(prov, results, argv, ["mmlu_pro"], "fp-b")
    assert not gpuref.run_covers(prov, results, argv, ["mmlu_pro"], None)
    del prov["eval_fingerprint"]  # legacy run
    assert not gpuref.run_covers(prov, results, argv, ["mmlu_pro"], "fp-a")


def _patch_eval_config(monkeypatch, model, task):
    from reference_config.evals import eval_config

    monkeypatch.setitem(
        eval_config._eval_config_map,
        model,
        eval_config.EvalConfig(hf_model_repo=model, tasks=[task]),
    )


def test_eval_fingerprint_tracks_score_inputs_only(monkeypatch):
    from dataclasses import replace

    from reference_config.evals.eval_config import EvalTask, EvalTaskScore
    from reference_config.evals.eval_utils import score_task_single_key
    from workflow_module.engine_types import EvalLimitMode

    model = "org/fingerprint-test"
    base = EvalTask(
        task_name="mmlu_pro",
        num_fewshot=5,
        score=EvalTaskScore(
            published_score=None, published_score_ref="x",
            score_func=score_task_single_key,
            score_func_kwargs={"result_keys": ["exact_match,custom-extract"], "unit": "percent"},
        ),
    )  # fmt: skip

    def fp(task, lm="be23028b", vllm="0.13.0", **stack):
        _patch_eval_config(monkeypatch, model, task)
        return gpuref.eval_fingerprint(
            model, lm, dict(gpuref.serving_stack(None), vllm=vllm, **stack)
        )

    f0 = fp(base)
    assert f0 and len(f0) == 64 and fp(base) == f0  # deterministic
    assert fp(replace(base, num_fewshot=0)) != f0
    assert fp(replace(base, fewshot_as_multiturn=False)) != f0
    assert fp(replace(base, gen_kwargs={"stream": "False", "max_gen_toks": 1024})) != f0
    assert fp(replace(base, thinking=False)) != f0
    assert fp(base, lm="deadbeef") != f0
    assert fp(base, vllm="0.14.0") != f0
    assert fp(base, transformers="5.15.0") != f0  # the whole stack counts
    assert fp(base, env={}) != f0
    # GPU device overrides are applied before hashing.
    assert fp(replace(base, device_overrides={"GPU": {"use_chat_api": True}})) != f0
    assert fp(replace(base, device_overrides={"GALAXY": {"use_chat_api": True}})) == f0
    # Transport and limits do not change a full run's score.
    assert fp(replace(base, max_concurrent=256)) == f0
    assert fp(replace(base, limit_samples_map={EvalLimitMode.CI_NIGHTLY: 0.2})) == f0
    # Unknown versions or model: no fingerprint, so nothing can match it.
    assert fp(base, lm=None) is None
    assert (
        gpuref.eval_fingerprint(
            "org/not-in-catalog", "be23028b", gpuref.serving_stack(None)
        )
        is None
    )


def test_report_ignores_runs_with_another_fingerprint(tmp_path):
    session = tmp_path / "gpuref-sweep-a100-x"
    data = session / "workflow_logs" / "reports_output" / "evals" / "data"
    data.mkdir(parents=True)
    report = json.loads(FIXTURE.read_text())
    (data / "report_data_x.json").write_text(json.dumps(report))
    model = report["metadata"]["model_repo"]
    mdir = session / "results" / model.replace("/", "__")
    mdir.mkdir(parents=True)
    argv = ["serve", model, "--max-model-len", "4096", "--max-num-seqs", "256"]
    prov = {"model": model, "status": "ok", "finished_at": "2026-10-10T00:00:00Z",
            "eval_fingerprint": "fp-old", "vllm_serve_command": ["vllm"] + argv}  # fmt: skip
    (mdir / "provenance.json").write_text(json.dumps(prov))
    target = {"model": model, "task": "mmlu_pro", "old_gpu_ref": None,
              "published": None, "tolerance": 0.05, "vllm_serve_args": argv}  # fmt: skip

    def report(**changes):
        plan = {"targets": [{**target, "eval_fingerprint": "fp-old", **changes}]}
        return gpuref.sweep_results(plan, tmp_path, None)[0]

    stale = report(eval_fingerprint="fp-new")
    assert stale["score"] is None and "eval fingerprint" in stale["note"]
    assert report()["score"] is not None
    # Same eval settings, other serving: the old run is not this plan's reference.
    longer = [a if a != "4096" else "8192" for a in argv]
    assert "serve args" in report(vllm_serve_args=longer)["note"]
    assert report(vllm_serve_args=longer)["score"] is None
    assert report(vllm_serve_args=None)["score"] is None  # plan without them
    # --max-num-seqs only schedules requests.
    assert report(vllm_serve_args=argv[:-1] + ["32"])["score"] is not None


def test_tt_grade_uses_the_tt_reports_noise_n(tmp_path):
    """TT 15 vs GPU 20: within noise at the TT run's n=100 (margin ~7.8 pts),
    a FAIL at the GPU run's n=541 (margin ~3.4 pts). TTIS grades with its own n."""
    tt = tmp_path / "tt" / "data"
    tt.mkdir(parents=True)
    report = json.loads(FIXTURE.read_text())
    ifeval = next(s for s in report["sections"] if s.get("kind") == "evals")
    ifeval["data"].update(score=15.0, noise_n=100.0)
    report["sections"] = [ifeval]
    (tt / "report_x.json").write_text(json.dumps(report))
    model = report["metadata"]["model_repo"]
    target = {"model": model, "task": "leaderboard_ifeval", "old_gpu_ref": 20.0,
              "published": None, "tolerance": 0.05}  # fmt: skip
    row = gpuref.sweep_results(
        {"targets": [target]}, tmp_path / "none", tmp_path / "tt"
    )[0]
    assert row["tt_score"] == 15.0 and row["tt_noise_n"] == 100.0
    assert row["tt_now"] == "PASS"
    assert gpuref.tt_grade(15.0, 20.0, 0.05, 541) == "FAIL"
    # A report without noise_n (ratio rule only) grades by the ratio alone.
    ifeval["data"].pop("noise_n")
    (tt / "report_x.json").write_text(json.dumps(report))
    row = gpuref.sweep_results(
        {"targets": [target]}, tmp_path / "none", tmp_path / "tt"
    )[0]
    assert row["tt_noise_n"] is None and row["tt_now"] == "FAIL"


def test_sweep_planner_gets_the_transformers_override():
    driver = (SCRIPT_DIR / "colab_gpu_reference.sh").read_text()
    assert 'plan_args+=(--transformers-version "${TRANSFORMERS_VERSION}")' in driver
    import inspect

    assert "transformers_version" in inspect.signature(gpuref.sweep_plan).parameters
    stack = gpuref.serving_stack(None, None, "4.57.6")
    assert stack["transformers"] == "4.57.6"  # what the planner fingerprints


def test_gpu_concurrency_raises_small_models_only():
    plan = {
        "max_concurrency": 32,
        "vllm_serve_args": [
            "serve",
            "m",
            "--max-num-seqs",
            "32",
            "--dtype",
            "bfloat16",
        ],
    }
    small = gpuref.gpu_concurrency(plan, 8_190_000_000)  # Qwen3-8B
    assert small["vllm_serve_args"][3] == str(gpuref.GPU_HIGH_CONCURRENCY)
    assert small["max_num_seqs"] == small["client_concurrency"] == 256
    assert plan["vllm_serve_args"][3] == "32"  # the input plan is not mutated
    for params in (14_800_000_000, None):  # Qwen3-14B; unknown size
        big = gpuref.gpu_concurrency(plan, params)
        assert big["vllm_serve_args"] == plan["vllm_serve_args"]
        assert big["max_num_seqs"] == big["client_concurrency"] == 32
    assert gpuref.model_params({"safetensors": {"total": 7}}, None) == 7


def test_gpu_concurrency_mid_tier_needs_kv_room_for_128_sequences():
    """10-16B models get as many sequences as fit at KV_TOKENS_PER_SEQ tokens,
    at most 128: ~126 for Qwen3-14B on an 80 GB H100; on an A100-40GB that is
    below the row's 32, so the row value stays."""
    config = {  # Qwen3-14B
        "num_attention_heads": 40, "num_key_value_heads": 8, "head_dim": 128,
        "num_hidden_layers": 40, "hidden_size": 5120, "intermediate_size": 17408,
    }  # fmt: skip
    params = 14_770_000_000
    estimate = gpuref.memory_estimate(params, config, 40960, 32)
    plan = {
        "max_concurrency": 32,
        "vllm_serve_args": ["serve", "m", "--max-num-seqs", "32"],
    }
    h100 = gpuref.gpu_concurrency(plan, params, estimate, 80.0)
    assert 64 < h100["max_num_seqs"] == h100["client_concurrency"] <= 128
    assert "KV fits" in h100["concurrency_rule"]
    a100 = gpuref.gpu_concurrency(plan, params, estimate, 40.0)
    assert a100["max_num_seqs"] == 32
    assert a100["concurrency_rule"] == "row max_concurrency"
    assert gpuref.gpu_concurrency(plan, params)["max_num_seqs"] == 32  # GPU unknown
    big = gpuref.gpu_concurrency(plan, 24_000_000_000, estimate, 80.0)
    assert big["max_num_seqs"] == 32


def test_patch_eval_config_sets_reference_and_clears_request():
    text = (
        '    EvalConfig(\n        hf_model_repo="org/m",\n        tasks=[\n'
        '            EvalTask(\n                task_name="a",\n'
        '                gpu_reference_requested="why",\n'
        "                score=EvalTaskScore(\n                    gpu_reference_score=1.0,\n"
        '                    gpu_reference_score_ref="old",\n'
        '            EvalTask(\n                task_name="b",\n'
        "                score=EvalTaskScore(\n                    gpu_reference_score=None,\n"
        '                    gpu_reference_score_ref="TBD",\n'
        "    EvalConfig(\n"
    )
    new = gpuref.patch_eval_config(
        text, [{"model": "org/m", "task": "a", "score": 12.345}], "URL"
    )
    assert "gpu_reference_requested" not in new
    assert (
        "gpu_reference_score=12.35," in new and 'gpu_reference_score_ref="URL",' in new
    )
    assert 'gpu_reference_score_ref="TBD"' in new  # task b untouched


def test_patch_adds_reference_fields_a_task_omits():
    text = (
        '    EvalConfig(\n        hf_model_repo="org/m",\n        tasks=[\n'
        '            EvalTask(\n                task_name="a",\n'
        "                score=EvalTaskScore(\n"
        "                    published_score=None,\n"
        "                    published_score_ref=None,\n"
        "                    score_func=f,\n"
        '            EvalTask(\n                task_name="b",\n'
    )
    row = {"model": "org/m", "task": "a", "score": 46.578, "verdict": "new"}
    new = gpuref.patch_eval_config(text, [row], "URL")
    assert (
        "                    published_score_ref=None,\n"
        "                    gpu_reference_score=46.58,\n"
        '                    gpu_reference_score_ref="URL",\n'
        "                    score_func=f,\n"
    ) in new
    assert new.count("gpu_reference_score=") == 1  # task b untouched


def test_reference_verdict_uses_the_noise_rule():
    assert gpuref.reference_verdict(None, 40.0, 541) == "new"
    # mmlu_pro-sized n: SE at 66.07 is ~0.43 pts, so 66.5 is within noise.
    assert gpuref.reference_verdict(66.07, 66.5, 12032) == "keep"
    assert gpuref.reference_verdict(66.07, 70.0, 12032) == "replace"
    assert gpuref.reference_verdict(66.07, None, 12032) is None


def test_patch_keeps_old_reference_within_noise_but_clears_flag():
    text = (
        '    EvalConfig(\n        hf_model_repo="org/m",\n        tasks=[\n'
        '            EvalTask(\n                task_name="a",\n'
        '                gpu_reference_requested="why",\n'
        "                score=EvalTaskScore(\n                    gpu_reference_score=66.07,\n"
        '                    gpu_reference_score_ref="old",\n'
    )
    row = {"model": "org/m", "task": "a", "score": 66.3, "verdict": "keep"}
    new = gpuref.patch_eval_config(text, [row], "URL")
    assert "gpu_reference_requested" not in new
    assert (
        "gpu_reference_score=66.07," in new and 'gpu_reference_score_ref="old",' in new
    )


def test_quetzal_audit_selects_every_task_of_audited_models():
    targets, _ = gpuref.select_targets(
        SWEEP_TASKS, SWEEP_ROWS, audit=frozenset({"org/gpu-model"})
    )
    reasons = {t["task"]: t["reason"] for t in targets if t["model"] == "org/gpu-model"}
    assert (
        reasons["published"] == "audit: Quetzal P300X2 row"
    )  # gated by a published score
    assert (
        reasons["referenced"] == "audit: Quetzal P300X2 row"
    )  # already has a GPU reference
    assert reasons["ungated"] == "ungated" and reasons["flagged"].startswith(
        "requested"
    )
    assert "agentic" not in reasons  # still only lm-eval tasks


def test_estimate_hours_and_param_fallback():
    assert gpuref.estimate_hours(["mmlu_pro"], 15.0) == round((25 + 10) / 60, 2)
    assert gpuref.estimate_hours(["mmlu_pro"], 60.0) > gpuref.estimate_hours(
        ["mmlu_pro"], 15.0
    )
    tinyllama = {"hidden_size": 2048, "num_hidden_layers": 22, "num_attention_heads": 32,
                 "num_key_value_heads": 4, "intermediate_size": 5632, "vocab_size": 32000}  # fmt: skip
    assert gpuref.params_from_config(tinyllama) == pytest.approx(1.1e9, rel=0.02)
    assert gpuref.params_from_config({}) is None


MATH_HARD_SIZES = {
    "leaderboard_math_algebra_hard": 307,
    "leaderboard_math_counting_and_prob_hard": 123,
    "leaderboard_math_geometry_hard": 132,
    "leaderboard_math_intermediate_algebra_hard": 280,
    "leaderboard_math_num_theory_hard": 154,
    "leaderboard_math_prealgebra_hard": 193,
    "leaderboard_math_precalculus_hard": 135,
}


def test_noise_n_is_ttis_effective_n():
    """math_hard is an unweighted mean of 7 subtasks: TTIS's n is
    k**2 / sum(1/n_i) = 1177.4, not the 1324 samples."""
    target = {"task": "leaderboard_math_hard", "unit": "percent",
              # as stored in the plan JSON: tuples become lists
              "result_keys": [[name, "exact_match,none"] for name in MATH_HARD_SIZES]}  # fmt: skip
    n = gpuref.noise_n(target, dict(MATH_HARD_SIZES))
    assert n == pytest.approx(49 / sum(1 / v for v in MATH_HARD_SIZES.values()))
    assert n == pytest.approx(1177.4, abs=0.1)
    # A group's single key: n is the group's count (the sum of its leaves).
    mmlu = {
        "task": "mmlu_pro",
        "unit": "percent",
        "result_keys": ["exact_match,custom-extract"],
    }
    assert gpuref.noise_n(mmlu, {"mmlu_pro": 12032}) == 12032
    # Not binomial (LongBench F1) or not percent: ratio rule only.
    lb = {
        "task": "longbench_single_e",
        "unit": "percent",
        "result_keys": ["qa_f1_score,none"],
    }
    assert gpuref.noise_n(lb, {"longbench_single_e": 450}) is None
    assert gpuref.noise_n(dict(mmlu, unit="WER"), {"mmlu_pro": 12032}) is None


def test_tt_grade_matches_ttis_grade_one_on_math_hard():
    """Yi-1.5-6B-Chat math_hard, TT 12.26 against the GPU reference 13.58:
    gpuref's grade and TTIS's _grade_one agree (within noise)."""
    from types import SimpleNamespace

    from reference_config.evals.eval_utils import score_multilevel_keys_mean
    from test_module.llm_tests import llm_eval_tests as mod

    keys = [(name, "exact_match,none") for name in MATH_HARD_SIZES]
    task = SimpleNamespace(
        task_name="leaderboard_math_hard",
        score=SimpleNamespace(
            published_score=None, score_func=score_multilevel_keys_mean,
            score_func_kwargs={"result_keys": keys, "unit": "percent"},
        ),
    )  # fmt: skip
    results = {name: {"exact_match,none": 0.1226} for name in MATH_HARD_SIZES}
    ref = {"reference_score": 13.58, "tolerance": 0.05}
    _, _, _, check, evidence = mod._grade_one(
        task, results, "leaderboard_math_hard", ref, sample_counts=dict(MATH_HARD_SIZES)
    )
    target = {"task": "leaderboard_math_hard", "unit": "percent", "result_keys": keys}
    n = gpuref.noise_n(target, dict(MATH_HARD_SIZES))
    assert evidence["noise_n"] == pytest.approx(n)
    assert evidence["accuracy_rule"] == "within_noise"
    assert gpuref.tt_grade(12.26, 13.58, 0.05, n) == "PASS"
    # Without an n (non-binomial) only the ratio rule applies: 0.90 fails.
    assert gpuref.tt_grade(12.26, 13.58, 0.05, None) == "FAIL"


def test_tt_grade_ratio_and_noise_rules():
    assert gpuref.tt_grade(13.89, 13.54, 0.05, 12032) == "PASS"  # SOLAR mmlu_pro, ratio
    assert (
        gpuref.tt_grade(11.83, 13.12, 0.05, 541) == "PASS"
    )  # Qwen1.5 ifeval, within noise
    assert gpuref.tt_grade(52.59, 57.39, 0.05, 12032) == "FAIL"  # Qwen2.5-7B mmlu_pro
    assert (
        gpuref.tt_grade(52.59, 28.09, 0.05, 12032) == "PASS"
    )  # ... against the old ref
    assert gpuref.tt_grade(None, 50.0, 0.05, 100) == "NA"
    assert gpuref.tt_grade(50.0, None, 0.05, 100) == "NA"


def test_none_override_serves_from_a_config_without_the_key():
    """vLLM dereferences dual_chunk_attention_config whenever the attribute
    exists, so removing it means serving from an edited config.json."""
    from types import SimpleNamespace

    dms = SimpleNamespace(max_context=131072, max_concurrency=32,
                          vllm_args={"revision": "r", "hf_overrides": {"dual_chunk_attention_config": None}})  # fmt: skip
    plan = gpuref.serve_plan(
        SimpleNamespace(device_model_spec=dms, hf_model_repo="org/m-1M", model_id="x")
    )
    argv = plan["vllm_serve_args"]
    assert "--hf-overrides" not in argv
    assert (
        argv[argv.index("--hf-config-path") + 1]
        == "/content/gpuref/hf_configs/org__m-1M"
    )
    assert plan["hf_config"]["drop"] == ["dual_chunk_attention_config"]
    assert argv[argv.index("--max-num-batched-tokens") + 1] == "16384"


# --- hardening (Cycode review) ----------------------------------------------


@pytest.mark.parametrize(
    "url, ok",
    [
        ("https://huggingface.co/api/models/org/name", True),
        ("https://huggingface.co/api/models/org/name/revision/abc123", True),
        ("https://huggingface.co/org/na.me-1/resolve/main/config.json", True),
        ("http://huggingface.co/api/models/org/name", False),
        ("https://evil.com/api/models/org/name", False),
        ("https://huggingface.co@evil.com/api/models/org/name", False),
        ("https://huggingface.co/api/models/org/../../x", False),
        ("https://huggingface.co/api/models/org/name?x=1", False),
        ("https://huggingface.co/org/name/resolve/main/other.bin", False),
        ("https://huggingface.co/api/datasets/org/name", False),
    ],
)
def test_hf_get_only_reaches_hub_model_documents(url, ok, monkeypatch):
    assert gpuref.hf_url_ok(url) is ok
    if not ok:  # refused before any request is made
        monkeypatch.setattr(gpuref.urllib.request, "urlopen", lambda *a, **k: 1 / 0)
        assert gpuref.hf_get(url, "token") is None


def test_cli_rejects_bad_repo_ids_and_revisions():
    for argv in (["serve-plan", "--model", "../x"], ["preflight", "--gpus", "A100", "a/b?c"],
                 ["served", "-x/y"]):  # fmt: skip
        with pytest.raises(SystemExit):
            gpuref.main(argv)
    with pytest.raises(SystemExit):
        gpuref.main(["request-evidence", "--model", "a/b", "--revision", "../x",
                     "--samples-root", "s", "--out", "o"])  # fmt: skip


def test_audit_refuses_refs_that_look_like_options():
    for ref in ("--output=/tmp/x", "-p", "main;rm"):
        with pytest.raises(ValueError):
            gpuref.quetzal_models({}, [ref])


def test_run_out_times_out_and_never_uses_a_shell():
    assert (
        gpuref.run_out([sys.executable, "-c", "import time; time.sleep(5)"], timeout=1)
        is None
    )
    assert gpuref.run_out([sys.executable, "-c", "print('a; echo b')"]) == "a; echo b"


def test_prepare_snippet_does_not_chmod():
    text = (SCRIPT_DIR / "snippets" / "prepare.py").read_text()
    assert "chmod" not in text.split("print(")[0] and "mode=0o700" in text


# --- unattended runs: status, capacity, publish -------------------------------


def test_status_view_prints_state_done_and_phase():
    out = subprocess.run(
        [sys.executable, str(GPUREF_PATH), "status-view"],
        input="GPUREF_STATE=RUNNING\nGPUREF_DONE=2\nGPUREF_PHASE=model 3/9 org/m: evals\n",
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    state, done, phase = out.stdout.strip().split(" ", 2)
    assert (state, done, phase) == ("RUNNING", "2", "model 3/9 org/m: evals")


def test_driver_status_writes_json_and_one_log_line(tmp_path):
    path, log = tmp_path / "s" / "driver_status.json", tmp_path / "gpuref_status.log"
    gpuref.main([
        "driver-status", str(path), "--log", str(log), "state=RUNNING",
        "session=gpuref-x", "phase=model 1/2 org/m: evals", "note=",
    ])  # fmt: skip
    record = json.loads(path.read_text())
    assert record["state"] == "RUNNING" and record["session"] == "gpuref-x"
    assert "note" not in record and record["updated_at"].endswith("Z")
    gpuref.main(["driver-status", str(path), "--log", str(log), "state=DONE"])
    lines = log.read_text().splitlines()
    assert len(lines) == 2 and "state=RUNNING" in lines[0]
    assert "phase=model_1/2_org/m:_evals" in lines[0]  # one token per field
    assert json.loads(path.read_text())["state"] == "DONE"
    # Options first, as the driver calls it: the same record.
    gpuref.main(["driver-status", "--log", str(log), str(path), "state=FAILED", "rc=1"])
    assert json.loads(path.read_text())["state"] == "FAILED"
    assert len(log.read_text().splitlines()) == 3


def test_driver_capacity_retry_and_group_fallback_are_wired():
    driver = (SCRIPT_DIR / "colab_gpu_reference.sh").read_text()
    assert "EX_NO_CAPACITY=75" in driver
    assert (
        'grep -qiE "Precondition Failed|\\b412\\b"' in driver
    )  # concurrent-session limit
    assert (
        'status NO_CAPACITY "gpus=${GPU}"' in driver
        and 'exit "${EX_NO_CAPACITY}"' in driver
    )
    assert '"${grc}" -eq "${EX_NO_CAPACITY}"' in driver  # sweep defers that group
    assert (
        'run_group "${gpus}" "${models}" "${RETRY_CAPACITY}"' in driver
    )  # retried last
    assert (
        'gpuref "${publish[@]}"' in driver
        and "--post" not in driver.split("publish=(")[1].split(")")[0]
    )


def test_driver_rejects_a_malformed_retry_capacity():
    out = subprocess.run(
        ["bash", str(SCRIPT_DIR / "colab_gpu_reference.sh"), "--retry-capacity", "15", "org/m"],
        capture_output=True, text=True,
    )  # fmt: skip
    assert out.returncode == 1 and "MINUTES,HOURS" in out.stderr


def test_task_selection_runs_only_named_tasks_in_full(monkeypatch):
    from reference_config.evals import eval_config
    from reference_config.evals.eval_config import EvalConfig, EvalTask

    model = "org/task-select"
    tasks = [EvalTask(task_name="mmlu_pro"), EvalTask(task_name="r1_gpqa_diamond")]
    monkeypatch.setitem(
        eval_config._eval_config_map,
        model,
        EvalConfig(hf_model_repo=model, tasks=tasks),
    )
    # null = selected, no --samples: lm-eval runs every doc of the task.
    assert gpuref.task_selection(model, ["mmlu_pro"]) == '{"mmlu_pro": null}'
    with pytest.raises(ValueError, match="not among"):
        gpuref.task_selection(model, ["ifeval"])
    with pytest.raises(ValueError):
        gpuref.task_selection(model, [])


def test_driver_task_flag_is_validated_and_explicit_only():
    driver = SCRIPT_DIR / "colab_gpu_reference.sh"

    def run(*argv):
        return subprocess.run(
            ["bash", str(driver), *argv], capture_output=True, text=True
        )

    out = run("--task", "org/m=mmlu_pro", "--sweep")
    assert out.returncode == 1 and "explicit runs" in out.stderr
    out = run("--task", "org/m=mmlu'pro", "org/m")
    assert out.returncode == 1 and "MODEL=TASK" in out.stderr
    out = run("--task", "org/other=mmlu_pro", "org/m")
    assert out.returncode == 1 and "not being run" in out.stderr
    runner = (SCRIPT_DIR / "remote_runner.sh").read_text()
    assert '--task) TASKS["${2%%=*}"]="${2#*=}"' in runner
    assert 'select=(--eval-samples "${eval_samples}")' in runner


def _results_with_solar(tmp_path):
    session = tmp_path / "gpuref-x"
    data = session / "workflow_logs" / "reports_output" / "evals" / "data"
    data.mkdir(parents=True)
    report = json.loads(FIXTURE.read_text())
    (data / "report_data_x.json").write_text(json.dumps(report))
    model = report["metadata"]["model_repo"]
    mdir = session / "results" / model.replace("/", "__")
    mdir.mkdir(parents=True)
    prov = {"model": model, "status": "ok", "finished_at": "2026-10-11T00:00:00Z",
            "eval_fingerprint": "fp", "packages": {"vllm": "0.28.0", "transformers": "5.15.0"},
            "vllm_env": {"VLLM_USE_FLASHINFER_SAMPLER": "0"}}  # fmt: skip
    (mdir / "provenance.json").write_text(json.dumps(prov))
    return model


def test_publish_explicit_models_renders_comment_and_patch(tmp_path):
    model = _results_with_solar(tmp_path)
    config = tmp_path / "eval_config.py"
    config.write_text(
        f'    EvalConfig(\n        hf_model_repo="{model}",\n        tasks=[\n'
        '            EvalTask(\n                task_name="mmlu_pro",\n'
        "                score=EvalTaskScore(\n                    gpu_reference_score=None,\n"
        '                    gpu_reference_score_ref="TBD",\n'
    )
    out = tmp_path / "pub"
    result = gpuref.render_publication(
        out, tmp_path, models=[model], eval_config=config, ref_url="URL"
    )
    assert result["finished"] >= 1
    md = (out / "sweep_5353.md").read_text()
    assert "vLLM 0.28.0 / transformers 5.15.0, VLLM_USE_FLASHINFER_SAMPLER=0" in md
    patch = (out / "eval_config.patch").read_text()
    assert '+                    gpu_reference_score_ref="URL",' in patch
    assert config.read_text().count("TBD") == 1  # never applied
    gpuref.render_publication(
        out, tmp_path, models=[model], eval_config=config, note="Why: run 123."
    )
    assert (
        (out / "sweep_5353.md")
        .read_text()
        .startswith("## GPU reference sweep\n\nWhy: run 123.\n\n| model |")
    )


def test_publish_post_needs_gh_once_and_refuses_tokens(tmp_path, monkeypatch):
    model = _results_with_solar(tmp_path)
    config = tmp_path / "eval_config.py"
    config.write_text(
        f'    EvalConfig(\n        hf_model_repo="{model}",\n        tasks=[\n'
        '            EvalTask(\n                task_name="mmlu_pro",\n'
        "                score=EvalTaskScore(\n                    gpu_reference_score=None,\n"
        '                    gpu_reference_score_ref="TBD",\n'
    )
    out = tmp_path / "pub"

    def render(url):
        return gpuref.render_publication(
            out, tmp_path, models=[model], eval_config=config, ref_url=url
        )

    render("URL")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="https://github.com/o/r/issues/5353#issuecomment-1\n",
            stderr="",
        )

    monkeypatch.setattr(gpuref.subprocess, "run", fake_run)
    url = gpuref.post_publication(out, 5353, render)
    assert url.endswith("#issuecomment-1")
    assert (
        calls[0][:4] == ["gh", "issue", "comment", "5353"] and "--body-file" in calls[0]
    )
    assert "#issuecomment-1" in (out / "eval_config.patch").read_text()
    with pytest.raises(SystemExit, match="already posted"):
        gpuref.post_publication(out, 5353, render)
    (out / "publish.json").unlink()
    (out / "sweep_5353.md").write_text("leak hf_" + "a" * 30)
    with pytest.raises(SystemExit, match="token-like"):
        gpuref.post_publication(out, 5353, render)
    assert len(calls) == 1


def test_publish_cli_needs_a_plan_or_models(tmp_path):
    with pytest.raises(SystemExit):
        gpuref.main(["publish", "--results-root", str(tmp_path)])


def test_no_capacity_exits_75_with_status(tmp_path):
    """A fake `colab` that always answers 503: with --retry-capacity the driver
    makes one more pass, records WAITING_CAPACITY then NO_CAPACITY, and exits 75."""
    if shutil.which("git") is None:
        pytest.skip("needs git")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "colab").write_text(
        "#!/bin/bash\n"
        'case "$1" in\n'
        '  new) echo "ColabRequestError: ... accelerator=A100: Service Unavailable" >&2; exit 1 ;;\n'
        "  sessions) exit 0 ;;\n"
        "  *) exit 0 ;;\n"
        "esac\n"
    )
    (bin_dir / "colab").chmod(0o755)
    home = tmp_path / "home"
    (home / ".config" / "colab-cli").mkdir(parents=True)
    (home / ".config" / "colab-cli" / "token.json").write_text("{}")
    root = tmp_path / "results"
    env = dict(
        os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", HOME=str(home),
        GPUREF_RESULTS_ROOT=str(root), GPUREF_NEW_RETRY_SECONDS="0",
    )  # fmt: skip
    driver = SCRIPT_DIR / "colab_gpu_reference.sh"
    text = driver.read_text()
    # Isolate the capacity loop: run it with the preflight stubbed out.
    start = text.index("# 2 start VM (or re-attach)")
    end = text.index("upload_secret() {")  # start_session(), the capacity loop
    head = text[: text.index("# 1 preflight")]
    script = tmp_path / "capacity.sh"
    script.write_text(
        head
        + f'SCRIPT_DIR="{SCRIPT_DIR}"\n'
        + 'TTIS_SHA=deadbeefdeadbeef SESSION=gpuref-test OUT_DIR="${RESULTS_ROOT}/gpuref-test"\n'
        + 'STATUS_FILE="${OUT_DIR}/driver_status.json" GPUS=(A100) SESSIONS_OUT=""\n'
        + text[start:end]
        + "start_session\nexit 0\n"
    )
    out = subprocess.run(
        ["bash", str(script), "--retry-capacity", "1,0", "--gpu", "A100", "org/m"],
        capture_output=True, text=True, env=env, timeout=120,
    )  # fmt: skip
    assert out.returncode == 75, out.stderr
    record = json.loads((root / "gpuref-test" / "driver_status.json").read_text())
    assert record["state"] == "NO_CAPACITY"
    assert "state=NO_CAPACITY" in (root / "gpuref_status.log").read_text()
    assert out.stderr.count("creating 'gpuref-test' on A100") == 3


# --- serving stack per model ---------------------------------------------------


def test_serving_stack_follows_the_architecture():
    default = gpuref.serving_stack({"architectures": ["LlamaForCausalLM"]})
    assert default == {"vllm": "0.13.0", "transformers": None,
                       "env": {"VLLM_USE_FLASHINFER_SAMPLER": "0"}}  # fmt: skip
    muse = gpuref.serving_stack(
        {"architectures": ["MuseGlimmerForConditionalGeneration"]}
    )
    assert (muse["vllm"], muse["transformers"]) == ("0.28.0", "5.15.0")
    assert muse["env"] == default["env"]
    gemma = gpuref.serving_stack({"architectures": ["Gemma4ForConditionalGeneration"]})
    assert (gemma["vllm"], gemma["transformers"]) == ("0.20.0", "5.5.3")
    assert gpuref.serving_stack(None) == default  # no config: the default
    forced = gpuref.serving_stack(None, vllm="0.14.1", transformers="4.57.6")
    assert (forced["vllm"], forced["transformers"]) == ("0.14.1", "4.57.6")
    gpuref.serving_stack(None)["env"]["X"] = "1"  # callers get a copy
    assert "X" not in gpuref.DEFAULT_STACK["env"]


def test_fingerprint_needs_the_requested_stack_installed():
    stack = gpuref.serving_stack({"architectures": ["Gemma4ForConditionalGeneration"]})
    assert gpuref.stack_installed(stack, {"vllm": "0.20.0", "transformers": "5.5.3"})
    assert not gpuref.stack_installed(
        stack, {"vllm": "0.20.0", "transformers": "5.15.0"}
    )
    assert not gpuref.stack_installed(
        stack, {"vllm": "0.13.0", "transformers": "5.5.3"}
    )
    default = gpuref.serving_stack(None)  # no transformers pin: any resolves
    assert gpuref.stack_installed(default, {"vllm": "0.13.0", "transformers": "4.57.6"})


def test_plan_stack_cli_for_the_runner(tmp_path, capsys):
    plan = tmp_path / "serve_plan.json"
    stack = gpuref.serving_stack({"architectures": ["MuseGlimmerForCausalLM"]})
    plan.write_text(json.dumps({"stack": stack}))
    gpuref.main(["plan-stack", str(plan)])
    gpuref.main(["plan-stack", "--env", str(plan)])
    assert capsys.readouterr().out.splitlines() == [
        "0.28.0 5.15.0", "VLLM_USE_FLASHINFER_SAMPLER=0",
    ]  # fmt: skip
    plan.write_text("{}")  # a plan from older tooling: the default stack
    gpuref.main(["plan-stack", str(plan)])
    assert capsys.readouterr().out.strip() == "0.13.0"


def test_runner_builds_one_venv_per_stack_and_serves_with_its_env():
    runner = (SCRIPT_DIR / "remote_runner.sh").read_text()
    assert 'VENV="${WORK_DIR}/venvs/vllm-${vllm}${tf:+-transformers-${tf}}"' in runner
    assert 'ensure_stack "${stack_vllm}" "${stack_tf}"' in runner
    assert 'plan-stack --env "${mdir}/serve_plan.json"' in runner
    assert 'setsid env ${SERVER_ENV[@]+"${SERVER_ENV[@]}"}' in runner
    assert "export VLLM_USE_FLASHINFER_SAMPLER" not in runner  # it is per stack now
    driver = (SCRIPT_DIR / "colab_gpu_reference.sh").read_text()
    assert 'runner_args+=(--transformers-version "${TRANSFORMERS_VERSION}")' in driver


# --- GPU spec derived at run time ---------------------------------------------


def test_ensure_gpu_spec_derives_from_the_tt_row_once(tmp_path, monkeypatch):
    """A pinned commit without a GPU spec gets one derived from its TT row,
    merged into the checkout's llm.yaml; once present nothing is derived."""
    import yaml

    spec_dir = tmp_path / "workflows" / "model_specs" / "dev"
    spec_dir.mkdir(parents=True)
    llm = spec_dir / "llm.yaml"
    llm.write_text("templates:\n- weights:\n  - org/other\n")
    rows = {"org/m": [{"device": "P300X2", "impl": "quetzal", "default": False,
                       "max_context": 131072, "max_concurrency": 32, "revision": "abc"}]}  # fmt: skip

    def load(model, ttis_dir):
        if model in gpuref.block_models(llm.read_text()):
            return object()
        raise ValueError("no GPU spec")

    monkeypatch.setattr(gpuref, "load_gpu_spec", load)
    monkeypatch.setattr(gpuref, "catalog_facts", lambda ttis_dir=None: ({}, rows))
    monkeypatch.setattr(
        gpuref, "hf_get", lambda url, token=None: {"max_position_embeddings": 32768}
    )
    d = gpuref.ensure_gpu_spec("org/m", tmp_path)
    assert (d["max_context"], d["max_concurrency"], d["revision"]) == (32768, 32, "abc")
    assert gpuref.block_models(llm.read_text()) == ["org/m"]
    yaml.safe_load(llm.read_text())
    assert gpuref.ensure_gpu_spec("org/m", tmp_path) is None  # already there
    with pytest.raises(ValueError, match="no TT row"):
        gpuref.ensure_gpu_spec("org/unknown", tmp_path)


def test_runner_derives_a_missing_gpu_spec_before_planning():
    runner = (SCRIPT_DIR / "remote_runner.sh").read_text()
    ensure = runner.index('ensure-gpu-spec --ttis-dir "${TTIS_DIR}" --model "${model}"')
    assert ensure < runner.index('serve-plan --ttis-dir "${TTIS_DIR}"')
    assert '--out "${mdir}/derived_gpu_spec.json"' in runner


def test_vm_steps_retry_a_lost_connection_but_never_double_launch():
    driver = (SCRIPT_DIR / "colab_gpu_reference.sh").read_text()
    step = driver[driver.index("remote_step() {") : driver.index("has_session() {")]
    assert "for attempt in 1 2 3; do" in step
    assert 'grep -q "Connection was lost" || break' in step
    assert (
        '[[ "${name}" == "launch" ]] && remote_state'
        ' && [[ "${STATE}" == "RUNNING" || "${STATE}" == "SERVING" ]]' in step
    )


def test_provenance_records_the_client_concurrency_actually_used(tmp_path):
    log = tmp_path / "run_py.log"
    log.write_text(
        "lm_eval --model_args base_url=x,num_concurrent=32,max_retries=0\n" * 2
    )
    assert gpuref.client_concurrency_used(log) == [32]
    assert gpuref.client_concurrency_used(tmp_path / "missing.log") == []


# --- --serve: agentic tasks through Shield's endpoint tests ------------------


def test_task_kind_splits_lm_eval_from_agentic(monkeypatch):
    from reference_config.evals import eval_config
    from reference_config.evals.eval_config import EvalConfig, EvalTask
    from workflows.workflow_types import WorkflowVenvType

    model = "org/agentic-kind"
    tasks = [
        EvalTask(task_name="mmlu_pro"),
        EvalTask(task_name="swe_bench_verified", workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC),
        EvalTask(task_name="terminal_bench_2_1", workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC),
    ]  # fmt: skip
    monkeypatch.setitem(
        eval_config._eval_config_map,
        model,
        EvalConfig(hf_model_repo=model, tasks=tasks),
    )
    assert gpuref.task_kind(model, ["mmlu_pro"]) == "lm-eval"
    assert (
        gpuref.task_kind(model, ["swe_bench_verified", "terminal_bench_2_1"])
        == "agentic"
    )
    for bad in (["mmlu_pro", "swe_bench_verified"], ["nope"], []):
        with pytest.raises(ValueError):
            gpuref.task_kind(model, bad)
    # lm-eval selection still refuses agentic tasks.
    with pytest.raises(ValueError):
        gpuref.task_selection(model, ["swe_bench_verified"])


SHA = "0123456789abcdef0123456789abcdef01234567"


def test_shield_inputs_one_agentic_test_on_gpu_with_the_key_by_name():
    inputs = gpuref.shield_inputs("Qwen/Qwen3.5-27B", "swe_bench_verified",
                                  "https://abc.ngrok.app", SHA)  # fmt: skip
    assert inputs["target"] == "None" and inputs["test_url"] == "https://abc.ngrok.app"
    assert inputs["inference-server-git-ref"] == SHA
    assert all(inputs[box] == "false" for box in gpuref.SHIELD_CHECKBOXES)
    assert json.loads(inputs["extra-tests"]) == [
        "inference-workflow-agentic --dev-mode --device gpu --agentic-benchmark swe_bench_verified"
    ]
    env = inputs["env-variables"].split()
    assert "HARBOR_ENV_TYPE=kubernetes" in env
    assert "API_KEY=SECRET_TEST_GPUREF_ENDPOINT_API_KEY" in env
    assert "OPENAI_API_KEY=SECRET_TEST_GPUREF_ENDPOINT_API_KEY" in env
    for url in ("http://abc.ngrok.app", "https://abc.ngrok.app/v1", "https://a b"):
        with pytest.raises(ValueError):
            gpuref.shield_inputs("org/m", "t", url, SHA)
    with pytest.raises(ValueError):
        gpuref.shield_inputs("org/m", "t; rm -rf /", "https://abc.ngrok.app", SHA)
    with pytest.raises(ValueError):
        gpuref.shield_inputs("org/m", "t", "https://abc.ngrok.app", "main")


def test_shield_find_run_matches_model_url_and_time():
    url = "https://abc.ngrok.app"
    runs = [
        {"databaseId": 1, "createdAt": "2026-10-14T00:00:00Z",
         "displayTitle": f"Run tests | v1-old-stack | Qwen/Qwen3.5-27B | {url}"},
        {"databaseId": 2, "createdAt": "2026-10-14T01:00:00Z",
         "displayTitle": f"Run tests | v1-old-stack | Qwen/Qwen3.5-27B | {url}"},
        {"databaseId": 3, "createdAt": "2026-10-14T02:00:00Z",
         "displayTitle": "Run tests | v1-old-stack | Qwen/Qwen3.5-27B | https://other.ngrok.app"},
        {"databaseId": 4, "createdAt": "2026-10-14T03:00:00Z",
         "displayTitle": f"Run tests | v1-old-stack | google/gemma-4-31B-it | {url}"},
    ]  # fmt: skip
    assert (
        gpuref.shield_find_run(runs, "Qwen/Qwen3.5-27B", url, "2026-10-14T00:30:00Z")
        == 2
    )
    assert (
        gpuref.shield_find_run(runs, "Qwen/Qwen3.5-27B", url, "2026-10-14T05:00:00Z")
        is None
    )


def test_redact_url_hides_the_tunnel_host():
    shown = gpuref.redact_url("https://abcdef123456.ngrok-free.app")
    assert shown == "https://abcd…ngrok-free.app" and "ef123456" not in shown
    assert gpuref.redact_url(None) is None


def _agentic_report(
    path, model, task, score, device="GPU", stamp="2026-10-14T00:00:00Z", check=2
):
    # AgenticEvalParser's schema: Harbor counters (n_trials) live in the
    # block's targets, not its data.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "metadata": {"model_repo": model, "device": device, "generated_at": stamp},
        "sections": [{"kind": "evals", "data": {"task_name": task, "score": score, "accuracy_check": check},
                      "targets": {"task_name": task, "n_trials": 89, "n_errors": 0}}],
    }))  # fmt: skip


def test_summary_rows_keeps_each_tasks_newest_report(tmp_path):
    model = "Qwen/Qwen3.5-27B"
    _agentic_report(
        tmp_path / "terminal_bench_2_1" / "report_a.json",
        model,
        "terminal_bench_2_1",
        40.0,
    )
    _agentic_report(tmp_path / "swe_bench_verified" / "report_b.json", model, "swe_bench_verified", 70.0,
                    stamp="2026-10-15T00:00:00Z")  # fmt: skip
    rows = {r["task"]: r for r in gpuref.summary_rows(tmp_path)}
    assert (
        rows["terminal_bench_2_1"]["score"] == 40.0
        and rows["swe_bench_verified"]["score"] == 70.0
    )
    assert rows["terminal_bench_2_1"]["n_trials"] == 89


def _snapshot(serving, prov, epoch="a1b2c3d4e5f6"):
    """A VM's serving snapshot (serving_provenance.json) under SERVING_DIR."""
    serving.mkdir(parents=True, exist_ok=True)
    (serving / f"{epoch}.json").write_text(json.dumps(dict(prov, status="serving")))
    return epoch


def test_shield_provenance_ok_only_when_every_task_scored(tmp_path):
    model = "Qwen/Qwen3.5-27B"
    shield, serving = tmp_path / "shield", tmp_path / "serving"
    _agentic_report(
        shield / "terminal_bench_2_1" / "x" / "report_1.json",
        model,
        "terminal_bench_2_1",
        40.0,
        check=3,  # complete, but under the accuracy target
    )
    prov = {"model": model, "status": "ok",
            "agentic_expected_trials": {"terminal_bench_2_1": 89, "swe_bench_verified": 89}}  # fmt: skip
    tb = {
        "task": "terminal_bench_2_1",
        "run_id": "7",
        "conclusion": "failure",
        "state": "completed",
        "serving_epochs": _snapshot(serving, prov),
    }
    out = gpuref.shield_provenance(prov, [tb], shield, serving)
    # A completed run that missed its acceptance criteria still measured the task.
    assert out["status"] == "ok" and out["agentic"][0]["score"] == 40.0
    assert out["agentic"][0]["report_files"] == ["terminal_bench_2_1/x/report_1.json"]
    assert out["tasks_selected"] == ["terminal_bench_2_1"]
    swe = {
        "task": "swe_bench_verified",
        "run_id": "8",
        "conclusion": "success",
        "state": "completed",
        "serving_epochs": tb["serving_epochs"],
    }
    assert (
        gpuref.shield_provenance(prov, [tb, swe], shield, serving)["status"] == "failed"
    )  # no report
    cancelled = dict(tb, conclusion="cancelled")
    assert (
        gpuref.shield_provenance(prov, [cancelled], shield, serving)["status"]
        == "failed"
    )
    assert (
        gpuref.shield_provenance(dict(prov, status="failed"), [tb], shield, serving)[
            "status"
        ]
        == "failed"
    )
    # A run with no serving snapshot (no VM it can be bound to) does not count.
    unbound = {k: v for k, v in tb.items() if k != "serving_epochs"}
    assert (
        gpuref.shield_provenance(prov, [unbound], shield, serving)["status"] == "failed"
    )
    # A TT (P300X2) report does not count as this GPU run's score.
    tt = tmp_path / "tt"
    _agentic_report(
        tt / "terminal_bench_2_1" / "r.json",
        model,
        "terminal_bench_2_1",
        40.0,
        device="P300X2",
    )
    assert gpuref.shield_provenance(prov, [tb], tt, serving)["status"] == "failed"


def test_derived_spec_carries_the_rows_frontend_flags():
    import yaml

    row = dict(_spec_row("P300X2", ctx=262144, rev="abc"),
               frontend={"reasoning_parser": "qwen3", "enable_auto_tool_choice": True,
                         "default-chat-template-kwargs": '{"enable_thinking": true}'})  # fmt: skip
    d = gpuref.derive_gpu_spec([row], native_ctx=262144)
    args = gpuref.derived_vllm_args(d)
    assert (
        args["reasoning_parser"] == "qwen3" and args["enable_auto_tool_choice"] is True
    )
    text = gpuref.write_derived_specs(
        "templates:\n- weights:\n  - org/x\n", {"org/m": d}
    )
    entry = next(
        e for e in yaml.safe_load(text)["templates"] if e["weights"] == ["org/m"]
    )
    vargs = entry["device_model_specs"][0]["vllm_args"]
    assert vargs["default-chat-template-kwargs"] == '{"enable_thinking": true}'
    assert (
        vargs["enable_auto_tool_choice"] is True
        and vargs["reasoning_parser"] == "qwen3"
    )
    flags = gpuref.vllm_flags(vargs)
    assert "--enable-auto-tool-choice" in flags
    assert (
        flags[flags.index("--default-chat-template-kwargs") + 1]
        == '{"enable_thinking": true}'
    )
    assert gpuref.frontend_args({"block_size": 64, "tool_call_parser": "x"}) == {
        "tool_call_parser": "x"
    }


def test_kv_bytes_counts_only_attention_layers_in_a_hybrid_model():
    qwen35 = {"num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256,
              "num_hidden_layers": 64,
              "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 16}  # fmt: skip
    assert gpuref.kv_bytes_per_token(qwen35) == 16 * 2 * 4 * 256 * 2  # 64 KiB/token
    dense = {k: v for k, v in qwen35.items() if k != "layer_types"}
    assert gpuref.kv_bytes_per_token(dense) == 64 * 2 * 4 * 256 * 2


def test_gemma4_unified_stack_has_the_model_and_its_config():
    # gemma-4-12B-it is Gemma4Unified, not Gemma4: without its own entry it fell
    # back to vLLM 0.13, which has neither the model nor the template kwargs flag.
    stack = gpuref.serving_stack(
        {"architectures": ["Gemma4UnifiedForConditionalGeneration"]}
    )
    assert (stack["vllm"], stack["transformers"]) == ("0.23.0", "5.10.0")
    gemma = gpuref.serving_stack({"architectures": ["Gemma4ForConditionalGeneration"]})
    assert gemma["vllm"] == "0.20.0"


def test_qwen3_5_stack_is_the_first_vllm_with_it():
    stack = gpuref.serving_stack({"architectures": ["Qwen3_5ForConditionalGeneration"]})
    assert (stack["vllm"], stack["transformers"]) == ("0.17.0", "4.57.5")


def test_parse_status_reports_serving():
    out = "GPUREF_STATE=RUNNING\nGPUREF_SERVING=1\nGPUREF_DONE=0\nGPUREF_PHASE=model 1/1 org/m: serving\n"
    assert gpuref.parse_status(out)[0] == "SERVING"
    assert gpuref.parse_status(out.replace("SERVING=1", "SERVING=0"))[0] == "RUNNING"


def test_serve_mode_keeps_secrets_off_command_lines():
    runner = (SCRIPT_DIR / "remote_runner.sh").read_text()
    driver = (SCRIPT_DIR / "colab_gpu_reference.sh").read_text()
    # The key reaches vLLM and ngrok by environment, curl by a header file.
    assert 'VLLM_API_KEY="$(< "${SECRETS}/endpoint_api_key")"' in runner
    assert 'NGROK_AUTHTOKEN="$(< "${SECRETS}/ngrok_authtoken")"' in runner
    assert '-H "@${SECRETS}/auth_header"' in runner
    assert "--api-key" not in runner
    # The driver uploads 0600 copies and dispatches with the secret's name only.
    assert 'upload_secret "${ENDPOINT_KEY_FILE}" endpoint_api_key.upload' in driver
    assert '--secret "${SHIELD_SECRET}"' in driver
    assert "must be mode 0600" in driver
    # A recorded run is resumed, never dispatched twice on the same endpoint.
    assert '"$(field "${rec}" endpoint_id)" != "${ENDPOINT_ID}"' in driver


def test_agentic_fingerprint_ignores_where_the_sandboxes_run(monkeypatch):
    from dataclasses import replace

    from reference_config.evals import eval_config
    from reference_config.evals.eval_config import (
        EvalConfig,
        EvalTask,
        HarborEvalConfig,
    )
    from workflows.workflow_types import WorkflowVenvType

    model = "org/agentic-fp"
    base = HarborEvalConfig(
        dataset="swebench-verified", agent="mini-swe-agent", n_concurrent_trials=5
    )

    def fp(cfg):
        task = EvalTask(task_name="swe_bench_verified",
                        workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC, agentic_eval_config=cfg)  # fmt: skip
        monkeypatch.setitem(eval_config._eval_config_map, model,
                            EvalConfig(hf_model_repo=model, tasks=[task]))  # fmt: skip
        return gpuref.eval_fingerprint(model, "be23028b", gpuref.serving_stack(None),
                                       venvs=gpuref.AGENTIC_VENVS)  # fmt: skip

    f0 = fp(base)
    assert f0 and fp(replace(base, environment_type="kubernetes")) == f0
    assert fp(replace(base, agent_kwargs={"temperature": 1.0})) != f0
    # The lm-eval fingerprint does not see agentic tasks at all.
    assert gpuref.eval_fingerprint(model, "be23028b", gpuref.serving_stack(None)) != f0


# --- serve mode: Shield-run records (review on #5395) -------------------------


def _serve_helpers(tmp_path, gh_downloads_report):
    """Run the driver's serve-record helpers under `set -euo pipefail`, with
    `field` reading the JSON record, `gpuref driver-status` writing it, and `gh
    run download` either producing a report or failing."""
    driver = (SCRIPT_DIR / "colab_gpu_reference.sh").read_text()
    block = driver.split("# >>> serve-record helpers", 1)[1].split(
        "# <<< serve-record helpers", 1
    )[0]
    block = block.split("\n", 1)[1]  # drop the rest of the marker's comment line
    gh_body = (
        'mkdir -p "$7" && echo \'{}\' > "$7/report_x.json"'
        if gh_downloads_report
        else "return 1"
    )
    stubs = f"""
set -euo pipefail
SHIELD_DIR={tmp_path}/shield SHIELD_REPO=org/shield TTIS_SHA=abc REPORT_RETRY_SECONDS=0
SERVING_DIR={tmp_path}/serving PYRUN=({shlex.quote(sys.executable)})
mkdir -p "$SHIELD_DIR"
log() {{ echo "LOG $*" >&2; }}
field() {{ python3 -c 'import json,sys; print((json.load(open(sys.argv[1])) or {{}}).get(sys.argv[2]) or "")' "$1" "$2" 2>/dev/null || true; }}
gpuref() {{  # driver-status REC k=v ...: write the record
    shift; local rec="$1"; shift
    python3 -c 'import json,sys; json.dump(dict(a.split("=",1) for a in sys.argv[2:]), open(sys.argv[1],"w"))' "$rec" "$@"
}}
gh() {{  # run download RUN -R REPO -D DIR -p PAT
    {gh_body}
}}
"""
    return stubs + "#\n" + block


def _bash(script):
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def test_first_serve_pass_reads_no_run_id_without_exiting(tmp_path):
    # No record exists on the first pass. Under set -e the old
    # `run_id="$([[ -s rec ]] && field ...)"` exited the driver right here.
    script = _serve_helpers(tmp_path, True) + (
        f'run_id="$(record_run_id {tmp_path}/shield/t.json)"\n'
        'echo "RUN_ID=[${run_id}]"\n'
    )
    out = _bash(script)
    assert out.returncode == 0, out.stderr
    assert "RUN_ID=[]" in out.stdout
    rec = tmp_path / "shield" / "t.json"
    rec.write_text(json.dumps({"run_id": "123"}))
    out = _bash(script)
    assert out.returncode == 0 and "RUN_ID=[123]" in out.stdout


def test_failed_report_download_is_recorded_pending_not_completed(tmp_path):
    rec = tmp_path / "shield" / "t.json"
    script = _serve_helpers(tmp_path, False) + (
        f"REPORT_TRIES=2\nretry_report t {rec} 99 success && echo GOT || echo PENDING\n"
    )
    out = _bash(script)
    assert out.returncode == 0, out.stderr
    assert "PENDING" in out.stdout
    assert json.loads(rec.read_text())["state"] == "report_pending"
    assert out.stderr.count("not downloaded") == 2


def test_report_retrieval_is_retried_until_the_report_is_on_disk(tmp_path):
    rec = tmp_path / "shield" / "t.json"
    pending = (
        _serve_helpers(tmp_path, False)
        + f"retrieve_report t {rec} 99 success || true\n"
    )
    assert _bash(pending).returncode == 0
    assert json.loads(rec.read_text())["state"] == "report_pending"
    # A later pass (or re-run) retrieves it and only then records completed.
    ok = (
        _serve_helpers(tmp_path, True)
        + f"REPORT_TRIES=3\nretry_report t {rec} 99 success && echo GOT\n"
    )
    out = _bash(ok)
    assert out.returncode == 0 and "GOT" in out.stdout, out.stderr
    record = json.loads(rec.read_text())
    assert record["state"] == "completed" and record["run_id"] == "99"
    assert (tmp_path / "shield" / "t" / "report_x.json").exists()


def test_rerun_retrieves_a_pending_report_and_never_redispatches():
    driver = (SCRIPT_DIR / "colab_gpu_reference.sh").read_text()
    loop = driver.split('for task in "${SERVE_TASKS[@]}"; do\n        rec=', 1)[1]
    head = loop.split("while :; do", 1)[0]
    # Completed only counts as done when the report is on disk ...
    assert 'report_on_disk "${SHIELD_DIR}/${task}"' in head
    # ... and a completed/report_pending record retries retrieval and moves on,
    # before any dispatch code is reached.
    assert '"${rec_state}" == "report_pending"' in head
    assert "retry_report" in head and "shield_dispatch" not in head
    assert 'run_id="$(record_run_id "${rec}")"' in driver
    assert 'run_id="$( [[ -s "${rec}" ]]' not in driver


# --- report-only resume keeps the original serving provenance (#5395) --------


def _resume_helpers(tmp_path, tasks, gh_downloads_report, prov=None):
    """The serve-record helpers plus status/summary/shield-provenance stubs.
    gpuref shield-provenance echoes the GPU it reads from --prov, so a test
    can see which provenance graded the run."""
    out_dir = tmp_path / "out"
    model_dir = out_dir / "results" / "org__m"
    model_dir.mkdir(parents=True, exist_ok=True)  # a test may run several passes
    if prov is not None:
        (model_dir / "provenance.json").write_text(json.dumps(prov))
    driver = (SCRIPT_DIR / "colab_gpu_reference.sh").read_text()
    block = driver.split("# >>> serve-record helpers", 1)[1].split(
        "# <<< serve-record helpers", 1
    )[0]
    block = block.split("\n", 1)[1]
    gh_body = (
        'mkdir -p "$7" && echo \'{}\' > "$7/report_x.json"'
        if gh_downloads_report
        else "return 1"
    )
    return (
        f"""
set -euo pipefail
OUT_DIR={out_dir} SHIELD_DIR={out_dir}/shield SHIELD_REPO=org/shield TTIS_SHA=abc
REPORT_RETRY_SECONDS=0 REPORT_TRIES=2 ROLLOVERS=0 AGENTIC_TASKS={",".join(tasks)}
SERVE_PROV={model_dir}/provenance.json SERVING_DIR={out_dir}/serving
PYRUN=({shlex.quote(sys.executable)})
SERVE_TASKS=({" ".join(tasks)})
mkdir -p "$SHIELD_DIR"
log() {{ echo "LOG $*" >&2; }}
status() {{ echo "STATUS $*" >&2; }}
start_session() {{ echo "VM_STARTED" >&2; return 1; }}
colab() {{ echo "COLAB $*" >&2; }}  # never the real CLI; no session listed
has_session() {{ grep -qF "[${{SESSION}}]"; }}
SESSION=s1 SESSION_ACTIVE=0
field() {{ python3 -c 'import json,sys; print((json.load(open(sys.argv[1])) or {{}}).get(sys.argv[2]) or "")' "$1" "$2" 2>/dev/null || true; }}
gpuref() {{
    case "$1" in
      driver-status) shift; local rec="$1"; shift
        python3 -c 'import json,sys; json.dump(dict(a.split("=",1) for a in sys.argv[2:]), open(sys.argv[1],"w"))' "$rec" "$@" ;;
      shield-provenance) python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); print("GRADED_ON " + p["gpu"]["name"], file=sys.stderr); print("ok")' "$3" ;;
      summary) echo "SUMMARY" ;;
    esac
}}
gh() {{ {gh_body}; }}
#
"""
        + block
    )


def _record(tmp_path, task, **fields):
    shield = tmp_path / "out" / "shield"
    shield.mkdir(parents=True, exist_ok=True)
    (shield / f"{task}.json").write_text(json.dumps({"task": task, **fields}))


def test_report_pending_resume_retrieves_reports_without_starting_a_vm(tmp_path):
    h100 = {"gpu": {"name": "NVIDIA H100 80GB HBM3"}, "status": "ok"}
    script = _resume_helpers(tmp_path, ["tb2"], True, prov=h100)
    _record(tmp_path, "tb2", state="report_pending", run_id="77", conclusion="success")
    out = _bash(script + 'serve_report_only_resume && echo "HANDLED rc=$FINISH_RC"\n')
    assert out.returncode == 0, out.stderr
    assert "HANDLED rc=0" in out.stdout
    assert "VM_STARTED" not in out.stderr
    assert "GRADED_ON NVIDIA H100 80GB HBM3" in out.stderr
    record = json.loads((tmp_path / "out" / "shield" / "tb2.json").read_text())
    assert record["state"] == "completed"
    # The original serving provenance is untouched.
    prov = tmp_path / "out" / "results" / "org__m" / "provenance.json"
    assert json.loads(prov.read_text()) == h100


def test_resume_needing_a_dispatch_still_uses_a_vm(tmp_path):
    script = _resume_helpers(
        tmp_path, ["tb2", "swe"], True, prov={"gpu": {"name": "x"}}
    )
    _record(tmp_path, "tb2", state="completed", run_id="1", conclusion="success")
    out = _bash(script + "serve_report_only_resume || echo NEEDS_VM\n")
    assert out.returncode == 0 and "NEEDS_VM" in out.stdout, out.stderr


def test_report_only_resume_without_provenance_fails_instead_of_inventing_one(tmp_path):
    script = _resume_helpers(tmp_path, ["tb2"], True, prov=None)
    _record(tmp_path, "tb2", state="report_pending", run_id="77", conclusion="success")
    out = _bash(script + 'serve_report_only_resume && echo "HANDLED rc=$FINISH_RC"\n')
    assert out.returncode == 0 and "HANDLED rc=1" in out.stdout, out.stderr
    assert "VM_STARTED" not in out.stderr and "no serving provenance" in out.stderr


def test_a_later_fetch_never_replaces_the_serving_provenance(tmp_path):
    h100 = {"gpu": {"name": "NVIDIA H100 80GB HBM3"}}
    script = _resume_helpers(tmp_path, ["tb2"], True, prov=h100)
    prov = tmp_path / "out" / "results" / "org__m" / "provenance.json"
    saved = tmp_path / "kept.json"
    saved.write_text(prov.read_text())
    # A fetch from a new VM (A100) overwrote the file.
    prov.write_text(json.dumps({"gpu": {"name": "NVIDIA A100-SXM4-40GB"}}))
    out = _bash(script + f"keep_serving_provenance {saved}\n")
    assert out.returncode == 0, out.stderr
    assert json.loads(prov.read_text()) == h100
    refetched = list(prov.parent.glob("provenance.refetch-*.json"))
    assert len(refetched) == 1 and "A100" in refetched[0].read_text()


def test_a_recovered_report_passes_on_the_unchanged_serving_provenance(tmp_path):
    # Review on #5395 (colab_gpu_reference.sh:475): the first pass finished
    # while the report was missing and wrote status "failed" into
    # provenance.json; the retry that recovered the report then read that
    # result back as "did the VM serve" and stayed failed forever. The real
    # shield-provenance grades both passes here.
    served = {
        "model": "org/m",
        "status": "ok",  # the runner's own status: the VM served
        "agentic_expected_trials": {"tb2": 89},
        "gpu": {
            "name": "NVIDIA H100 80GB HBM3",
            "driver": "570",
            "memory_total": "80 GiB",
        },
        "vllm_serve_command": ["vllm", "serve", "org/m", "--max-model-len", "131072"],
        "packages": {"vllm": "0.23.0"},
        "started_at": "2026-10-14T10:00:00Z",
        "finished_at": "2026-10-14T18:00:00Z",
    }
    report = json.dumps({
        "metadata": {"model_repo": "org/m", "device": "GPU", "generated_at": "2026-10-14T18:00:00Z"},
        "sections": [{"kind": "evals", "data": {"task_name": "tb2", "score": 40.0},
                      "targets": {"task_name": "tb2", "n_trials": 89, "n_errors": 0}}],
    })  # fmt: skip
    prov = tmp_path / "out" / "results" / "org__m" / "provenance.json"

    def resume_pass(gh_downloads_report):
        script = _resume_helpers(tmp_path, ["tb2"], gh_downloads_report)
        gh_body = (
            f'mkdir -p "$7" && printf \'%s\' {shlex.quote(report)} > "$7/report_tb2.json"'
            if gh_downloads_report
            else "return 1"
        )
        # The real shield-provenance and summary; driver-status stays stubbed.
        script += f"""
gh() {{ {gh_body}; }}
gpuref() {{
    case "$1" in
      driver-status) shift; local rec="$1"; shift
        python3 -c 'import json,sys; json.dump(dict(a.split("=",1) for a in sys.argv[2:]), open(sys.argv[1],"w"))' "$rec" "$@" ;;
      *) {shlex.quote(sys.executable)} {shlex.quote(str(GPUREF_PATH))} "$@" ;;
    esac
}}
serve_report_only_resume && echo "HANDLED rc=$FINISH_RC"
"""
        return _bash(script)

    prov.parent.mkdir(parents=True)
    prov.write_text(json.dumps(served))
    epoch = _snapshot(tmp_path / "out" / "serving", served)
    _record(tmp_path, "tb2", state="report_pending", run_id="77", conclusion="success",
            serving_epochs=epoch)  # fmt: skip
    first = resume_pass(gh_downloads_report=False)
    assert first.returncode == 0 and "HANDLED rc=1" in first.stdout, first.stderr
    assert json.loads(prov.read_text())["status"] == "failed"  # report missing

    second = resume_pass(gh_downloads_report=True)
    assert second.returncode == 0 and "HANDLED rc=0" in second.stdout, second.stderr
    assert "VM_STARTED" not in first.stderr + second.stderr
    graded = json.loads(prov.read_text())
    assert graded["status"] == "ok"
    assert graded["serving_status"] == "ok"
    assert graded["agentic"][0]["score"] == 40.0
    # The serving provenance is the one the scores were measured with.
    for key, value in served.items():
        if key != "status":
            assert graded[key] == value, key


def test_shield_provenance_never_rewrites_the_serving_status(tmp_path):
    model = "Qwen/Qwen3.5-27B"
    shield, serving = tmp_path / "shield", tmp_path / "serving"
    vm = {"model": model, "status": "ok", "agentic_expected_trials": {"tb2": 89}}
    tb = {"task": "tb2", "run_id": "7", "conclusion": "success", "state": "completed",
          "serving_epochs": _snapshot(serving, vm)}  # fmt: skip
    pending = gpuref.shield_provenance(vm, [tb], shield, serving)  # no report yet
    assert pending["status"] == "failed"
    _agentic_report(shield / "tb2" / "report_1.json", model, "tb2", 40.0)
    recovered = gpuref.shield_provenance(pending, [tb], shield, serving)
    assert recovered["status"] == "ok"
    assert pending["serving_status"] == recovered["serving_status"] == "ok"
    # A VM that did not serve stays failed whatever reports turn up.
    down = gpuref.shield_provenance(
        {"model": model, "status": "failed"}, [tb], shield, serving
    )
    again = gpuref.shield_provenance(down, [tb], shield, serving)
    assert down["status"] == again["status"] == "failed"
    assert down["serving_status"] == again["serving_status"] == "failed"


def test_report_only_resume_runs_before_any_vm_is_started():
    driver = (SCRIPT_DIR / "colab_gpu_reference.sh").read_text()
    resume = driver.index("serve_report_only_resume; then exit")
    first_start = driver.index("\nstart_session\n")
    assert resume < first_start
    tail = driver.split("# Every task ran: stop serving", 1)[1]
    assert "keep_serving_provenance" in tail.split("finish_serve", 1)[0]


def test_serve_secrets_dir_is_never_chmodded_and_files_stay_0600(tmp_path):
    src = (SCRIPT_DIR / "snippets" / "install_serve_secrets.py").read_text()
    assert "os.chmod(S," not in src  # the Cycode-flagged permissive chmod
    work = tmp_path / "content" / "gpuref"
    work.mkdir(parents=True)
    for name in ("endpoint_api_key", "ngrok_authtoken"):
        (work / f"{name}.upload").write_text("secret-" + name)
    code = src.replace('"/content/gpuref"', repr(str(work)))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "secret" not in out.stdout + out.stderr
    secrets = work / "secrets"
    assert stat.S_IMODE(secrets.stat().st_mode) == 0o700
    for f in secrets.iterdir():
        assert stat.S_IMODE(f.stat().st_mode) == 0o600, f


# Each permission bit the snippet refuses on an existing secrets directory.
GROUP_OTHER_BITS = [
    stat.S_IRGRP, stat.S_IWGRP, stat.S_IXGRP, stat.S_IROTH, stat.S_IWOTH, stat.S_IXOTH,
]  # fmt: skip


@pytest.mark.parametrize("loose_bit", GROUP_OTHER_BITS, ids=oct)
def test_serve_secrets_dir_with_any_group_or_other_bit_is_refused(tmp_path, loose_bit):
    # A pre-existing secrets dir that grants group/other any access is refused,
    # not reused or re-permissioned. The fixture dir is the owner's plus one
    # refused bit, so each bit is shown to be refused on its own.
    src = (SCRIPT_DIR / "snippets" / "install_serve_secrets.py").read_text()
    loose = tmp_path / "gpuref"
    (loose / "secrets").mkdir(parents=True)
    os.chmod(loose / "secrets", stat.S_IRWXU | loose_bit)
    for name in ("endpoint_api_key", "ngrok_authtoken"):
        (loose / f"{name}.upload").write_text("secret-" + name)
    code = src.replace('"/content/gpuref"', repr(str(loose)))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode != 0 and "REFUSED" in out.stderr
    assert not (loose / "secrets" / "endpoint_api_key").exists()


# --- each score stays with the VM that served it, across rollovers (#5395) ---


def test_rollover_keeps_each_tasks_score_with_the_vm_that_served_it(tmp_path):
    # Review on #5395 (colab_gpu_reference.sh:519): task 1 finished on an H100
    # VM, the VM was lost during task 2 and the replacement fell back to A100.
    # The final fetch only brings home the A100 VM's provenance; task 1's score
    # must stay bound to the H100 snapshot, never be filed under the A100.
    def vm(gpu, cap=None):
        return {"model": "org/m", "gpu": {"name": gpu, "memory_total": "80 GiB"},
                "vllm_serve_command": ["vllm", "serve", "org/m"], "max_model_len_cap": cap,
                "packages": {"vllm": "0.23.0"}, "started_at": gpu,
                "agentic_expected_trials": {"tb2": 89, "swe": 89}}  # fmt: skip

    def report(task):
        return json.dumps({
            "metadata": {"model_repo": "org/m", "device": "GPU", "generated_at": "2026-10-16T00:00:00Z"},
            "sections": [{"kind": "evals", "data": {"task_name": task, "score": 40.0},
                          "targets": {"task_name": task, "n_trials": 89, "n_errors": 0}}],
        })  # fmt: skip

    def run(name, first_vm, second_vm):
        out = tmp_path / name
        script = _resume_helpers(out, ["tb2", "swe"], True)
        for vm_name, snap in (("h", first_vm), ("a", second_vm)):
            (out / f"{vm_name}.json").write_text(
                json.dumps(dict(snap, status="serving"))
            )
        for task in ("tb2", "swe"):
            (out / f"report_{task}.json").write_text(report(task))
        script += f"""
gh() {{ mkdir -p "$7" && cp {out}/report_"$(basename "$7")".json "$7/report_1.json"; }}
gpuref() {{
    case "$1" in
      driver-status) shift; local rec="$1"; shift
        python3 -c 'import json,sys; json.dump(dict(a.split("=",1) for a in sys.argv[2:]), open(sys.argv[1],"w"))' "$rec" "$@" ;;
      *) {shlex.quote(sys.executable)} {shlex.quote(str(GPUREF_PATH))} "$@" ;;
    esac
}}
# VM 1 serves: snapshot, tb2 dispatched and polled on it, completes.
SERVE_EPOCH="$(save_serving_snapshot {out}/h.json)"
gpuref driver-status $SHIELD_DIR/tb2.json task=tb2 run_id=1 state=dispatched serving_epochs=$SERVE_EPOCH
add_serving_epoch $SHIELD_DIR/tb2.json $SERVE_EPOCH
retrieve_report tb2 $SHIELD_DIR/tb2.json 1 success
# VM 1 is lost during swe (dispatched on it); the rollover's VM 2 serves the rest.
gpuref driver-status $SHIELD_DIR/swe.json task=swe run_id=2 state=dispatched serving_epochs=$SERVE_EPOCH
SERVE_EPOCH="$(save_serving_snapshot {out}/a.json)"; ROLLOVERS=1
add_serving_epoch $SHIELD_DIR/swe.json $SERVE_EPOCH
add_serving_epoch $SHIELD_DIR/swe.json $SERVE_EPOCH
retrieve_report swe $SHIELD_DIR/swe.json 2 success
# The final fetch brings home only VM 2's provenance.
mkdir -p "$(dirname "$SERVE_PROV")"
{shlex.quote(sys.executable)} -c 'import json,sys; json.dump(dict(json.load(open(sys.argv[1])), status="ok"), open(sys.argv[2],"w"))' {out}/a.json "$SERVE_PROV"
finish_serve && echo FINISH=ok || echo FINISH=failed
"""
        res = _bash(script)
        prov = out / "out" / "results" / "org__m" / "provenance.json"
        return res, json.loads(prov.read_text())

    h100, a100 = vm("NVIDIA H100 80GB HBM3"), vm("NVIDIA A100-SXM4-80GB")
    res, prov = run("fallback", h100, a100)
    assert res.returncode == 0, res.stderr
    by_task = {a["task"]: a for a in prov["agentic"]}
    # tb2 was served only by the H100 VM; swe by both (it ran across the rollover).
    assert [v["gpu"] for v in by_task["tb2"]["served_by"]] == ["NVIDIA H100 80GB HBM3"]
    assert [v["gpu"] for v in by_task["swe"]["served_by"]] == [
        "NVIDIA H100 80GB HBM3", "NVIDIA A100-SXM4-80GB"]  # fmt: skip
    assert all(v["found"] for a in prov["agentic"] for v in a["served_by"])
    # Filed under the A100 provenance, neither score may count.
    assert prov["gpu"]["name"] == "NVIDIA A100-SXM4-80GB"
    assert (
        not by_task["tb2"]["serving_matches"] and not by_task["swe"]["serving_matches"]
    )
    assert prov["status"] == "failed" and "FINISH=failed" in res.stdout
    # A rollover onto the same GPU and serving settings grades ok.
    same = dict(h100, started_at="second VM")
    res, prov = run("same", h100, same)
    assert res.returncode == 0, res.stderr
    assert len({v["epoch"] for a in prov["agentic"] for v in a["served_by"]}) == 2
    assert prov["status"] == "ok" and "FINISH=ok" in res.stdout, res.stderr


def test_serve_loop_snapshots_each_vm_and_binds_runs_to_it():
    driver = (SCRIPT_DIR / "colab_gpu_reference.sh").read_text()
    rollover = driver.split("    rollover() {", 1)[1].split("\n    }", 1)[0]
    assert 'SERVE_EPOCH=""' in rollover  # a new VM is a new epoch
    loop = driver.split("serve_endpoint || {", 1)[1]
    # No dispatch or poll before this VM's snapshot is saved ...
    assert loop.index("snapshot_serving") < loop.index("shield_dispatch")
    # ... and every poll binds the run to the VM serving it.
    assert loop.index('add_serving_epoch "${rec}" "${SERVE_EPOCH}"') < loop.index(
        "gh run view"
    )
    assert '"serving_epochs=${SERVE_EPOCH}"' in driver  # shield_dispatch
    assert (
        "--serving-dir" in driver.split("finish_serve() {", 1)[1].split("\n    }", 1)[0]
    )
    runner = (SCRIPT_DIR / "remote_runner.sh").read_text()
    serve = runner.split("serve_until_stop() {", 1)[1]
    assert serve.index("serving_provenance.json") < serve.index(
        'touch "${WORK_DIR}/SERVING"'
    )


# --- n_trials from the real AgenticEvalParser report (#5395) -----------------


def _parser_report(path, n_trials, n_errors=0, n_total=None, mean=0.4, score=None):
    from llm_module.parsers.agentic import AgenticEvalParser

    evals = {"metrics": [{"mean": mean}], "n_errors": n_errors}
    if n_trials is not None:
        evals["n_trials"] = n_trials
    raw = {"stats": {"evals": {"terminal-bench": evals}}}
    if n_total is not None:
        raw["n_total_trials"] = n_total
    block = AgenticEvalParser(task_name="terminal_bench_2_1", score=score).parse(
        raw, device="gpu"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "metadata": {"model_repo": "org/m", "device": "GPU", "generated_at": "2026-10-16T00:00:00Z"},
        "sections": [block.to_dict()],
    }))  # fmt: skip


def test_n_trials_is_read_from_the_agentic_parsers_report(tmp_path):
    # Review on #5395 (gpuref.py:540): AgenticEvalParser writes n_trials to
    # section.targets; reading section.data gave n_trials null with status ok.
    shield, serving = tmp_path / "shield", tmp_path / "serving"
    _parser_report(shield / "terminal_bench_2_1" / "report_1.json", 89)
    row = gpuref.summary_rows(shield)[0]
    assert row["score"] == 40.0 and row["n_trials"] == 89
    prov = {
        "model": "org/m",
        "status": "ok",
        "agentic_expected_trials": {"terminal_bench_2_1": 89},
    }
    tb = {"task": "terminal_bench_2_1", "run_id": "7", "conclusion": "success",
          "state": "completed", "serving_epochs": _snapshot(serving, prov)}  # fmt: skip
    out = gpuref.shield_provenance(prov, [tb], shield, serving)
    assert out["agentic"][0]["n_trials"] == 89 and out["status"] == "ok"


def test_a_report_without_n_trials_is_not_ok(tmp_path):
    shield, serving = tmp_path / "shield", tmp_path / "serving"
    _parser_report(shield / "terminal_bench_2_1" / "report_1.json", None)
    prov = {
        "model": "org/m",
        "status": "ok",
        "agentic_expected_trials": {"terminal_bench_2_1": 89},
    }
    tb = {"task": "terminal_bench_2_1", "run_id": "7", "conclusion": "success",
          "state": "completed", "serving_epochs": _snapshot(serving, prov)}  # fmt: skip
    out = gpuref.shield_provenance(prov, [tb], shield, serving)
    assert out["agentic"][0]["score"] == 40.0
    assert out["agentic"][0]["n_trials"] is None and out["status"] == "failed"


def test_a_vm_without_a_serving_snapshot_is_stopped_and_never_graded(tmp_path):
    # A runner that never writes serving_provenance.json (e.g. launched before
    # the snapshot existed) must not be waited on forever, nor graded.
    script = _resume_helpers(
        tmp_path, ["tb2"], True, prov={"model": "org/m", "status": "ok"}
    )
    _record(
        tmp_path, "tb2", state="dispatched", run_id="5", serving_epochs="a1b2c3d4e5f6"
    )
    rec = tmp_path / "out" / "shield" / "tb2.json"
    script += f"""
SESSION=s1 SESSION_ACTIVE=1 SNAPSHOT_WAITS=10
colab_cmd() {{ echo "COLAB $*" >&2; }}
gpuref() {{
    case "$1" in
      driver-status) shift; local rec="$1"; shift
        python3 -c 'import json,sys; json.dump(dict(a.split("=",1) for a in sys.argv[2:]), open(sys.argv[1],"w"))' "$rec" "$@" ;;
      *) echo "GRADED" >&2 ;;
    esac
}}
no_serving_snapshot {rec}
echo NOT_REACHED
"""
    out = _bash(script)
    assert out.returncode == 1 and "NOT_REACHED" not in out.stdout, out.stderr
    assert "COLAB stop -s s1" in out.stderr  # the VM is stopped
    assert "GRADED" not in out.stderr  # finish_serve / shield-provenance never run
    assert (
        "STATUS FAILED reason=no serving snapshot (pre-snapshot runner?)" in out.stderr
    )
    marker = json.loads((tmp_path / "out" / "serving" / "no_snapshot.json").read_text())
    assert marker["serving_status"] == "failed"
    assert marker["reason"] == "no serving snapshot (pre-snapshot runner?)"
    # The in-flight run is bound to a VM epoch that never resolves: it can
    # never count, even if a later VM finishes it.
    record = json.loads(rec.read_text())
    assert record["serving_epochs"] == "a1b2c3d4e5f6,unsnapshotted"
    served = gpuref.served_by(record, tmp_path / "out" / "serving")
    assert [v["found"] for v in served] == [False, False]


def test_the_snapshot_wait_is_bounded_and_reset_per_vm():
    driver = (SCRIPT_DIR / "colab_gpu_reference.sh").read_text()
    loop = driver.split("serve_endpoint || {", 1)[1].split("shield_dispatch", 1)[0]
    assert "SNAPSHOT_WAITS=$((SNAPSHOT_WAITS + 1))" in loop
    assert (
        '(( SNAPSHOT_WAITS >= ${SNAPSHOT_TRIES:-10} )) && no_serving_snapshot "${rec}"'
        in loop
    )
    rollover = driver.split("    rollover() {", 1)[1].split("\n    }", 1)[0]
    assert "SNAPSHOT_WAITS=0" in rollover


# --- trial coverage: a partial Harbor run never grades ok (#5395) ------------


def _graded(tmp_path, conclusion="success", expected=89, **report):
    shield, serving = tmp_path / "shield", tmp_path / "serving"
    _parser_report(shield / "terminal_bench_2_1" / "report_1.json", **report)
    prov = {"model": "org/m", "status": "ok",
            "agentic_expected_trials": {"terminal_bench_2_1": expected}}  # fmt: skip
    tb = {"task": "terminal_bench_2_1", "run_id": "7", "conclusion": conclusion,
          "state": "completed", "serving_epochs": _snapshot(serving, prov)}  # fmt: skip
    return gpuref.shield_provenance(prov, [tb], shield, serving)


def test_a_run_whose_trials_mostly_errored_is_not_ok(tmp_path):
    # Review on #5395 (gpuref.py:1147): 1 scored trial + 88 infrastructure
    # errors gave status ok with n_trials 1 and a 1.12% score.
    out = _graded(tmp_path, n_trials=1, n_errors=88, n_total=89, mean=1.0)
    task = out["agentic"][0]
    assert task["n_trials"] == 1 and task["n_errors"] == 88
    assert task["trials_expected"] == 89 and not task["trials_complete"]
    assert out["status"] == "failed"


def test_trial_coverage_needs_the_expected_count_and_the_error_count(tmp_path):
    # The whole dataset (no fixed count in the config): Harbor's own job size.
    full = _graded(tmp_path / "a", expected=None, n_trials=500, n_total=500)
    assert full["agentic"][0]["trials_expected"] == 500 and full["status"] == "ok"
    short = _graded(tmp_path / "b", expected=None, n_trials=499, n_total=500)
    assert short["status"] == "failed"
    # Neither the config nor the report gives a count: not verifiable.
    unknown = _graded(tmp_path / "c", expected=None, n_trials=89)
    assert unknown["status"] == "failed"
    # Config and job size disagree: the job did not run the configured set.
    mismatch = _graded(tmp_path / "d", n_trials=50, n_total=50)
    assert mismatch["status"] == "failed"
    # A report without Harbor's error count (an older parser) does not count.
    no_errors = _graded(tmp_path / "e", n_trials=89, n_errors=None)
    assert no_errors["agentic"][0]["n_errors"] is None
    assert no_errors["status"] == "failed"
    # A provenance without expected counts (an older runner) does not count.
    shield, serving = tmp_path / "f" / "shield", tmp_path / "f" / "serving"
    _parser_report(shield / "terminal_bench_2_1" / "report_1.json", 89)
    prov = {"model": "org/m", "status": "ok"}
    tb = {"task": "terminal_bench_2_1", "run_id": "7", "conclusion": "success",
          "state": "completed", "serving_epochs": _snapshot(serving, prov)}  # fmt: skip
    assert gpuref.shield_provenance(prov, [tb], shield, serving)["status"] == "failed"


def test_a_complete_run_that_missed_its_accuracy_target_still_counts(tmp_path):
    from reference_config.evals.eval_config import EvalTaskScore

    score = EvalTaskScore(published_score=81.0, published_score_ref="x",
                          score_func=len, gpu_reference_score=77.5)  # fmt: skip
    # 89 scored (3 agent timeouts were still verified), 40% < 77.5% reference:
    # Shield fails the job, the score is still a full measurement.
    missed = _graded(tmp_path / "a", conclusion="failure", n_trials=89, n_errors=3,
                     n_total=89, mean=0.4, score=score)  # fmt: skip
    assert missed["agentic"][0]["accuracy_check"] == 3
    assert missed["status"] == "ok"
    # A failed job whose report met its target failed for some other reason.
    other = _graded(tmp_path / "b", conclusion="failure", n_trials=89, n_total=89,
                    mean=0.8, score=score)  # fmt: skip
    assert other["agentic"][0]["accuracy_check"] == 2
    assert other["status"] == "failed"
    # No reference to miss (N/A): a failed job is not an accuracy miss either.
    na = _graded(tmp_path / "c", conclusion="failure", n_trials=89, n_total=89)
    assert na["status"] == "failed"


def test_the_runner_records_each_agentic_tasks_expected_trials():
    expected = gpuref.agentic_expected_trials(
        "Qwen/Qwen3.8-27B", ["terminal_bench_2_1", "swe_bench_verified"]
    )
    # terminal-bench 2.1 fixes 89 tasks x 1 attempt; SWE-bench Verified runs
    # the whole dataset (the job's own n_total_trials is the count).
    assert expected == {"terminal_bench_2_1": 89, "swe_bench_verified": None}


# --- a report-only resume recovers a final provenance that was never fetched --


def _unfetched_resume(tmp_path, vm_up, fetched_snapshot):
    """Every run finished, reports on disk, the serving snapshot saved, but the
    final fetch failed: no provenance.json. ``fetch_results`` (stubbed) brings
    home a provenance.json and the VM's serving snapshot."""
    snap = json.dumps(
        {"gpu": {"name": "NVIDIA H100 80GB HBM3"}, "status": "serving", "t": 1}
    )
    serving = tmp_path / "out" / "serving"
    serving.mkdir(parents=True)
    import hashlib

    epoch = hashlib.sha256(snap.encode()).hexdigest()[:12]
    (serving / f"{epoch}.json").write_text(snap)
    _record(tmp_path, "tb2", state="completed", run_id="77", conclusion="success",
            serving_epochs=epoch)  # fmt: skip
    _agentic_report(
        tmp_path / "out" / "shield" / "tb2" / "report_1.json", "org/m", "tb2", 40.0
    )
    model_dir = tmp_path / "out" / "results" / "org__m"
    final = json.dumps({"gpu": {"name": "NVIDIA H100 80GB HBM3"}, "status": "ok"})
    script = _resume_helpers(tmp_path, ["tb2"], True)
    script += f"""
colab() {{ echo "COLAB $*" >&2; {"echo '[s1]'" if vm_up else "true"}; }}
remote_state() {{ STATE=SERVING; echo "STATE_READ" >&2; }}
remote_step() {{ echo "STEP $1" >&2; }}
STOP_POLL_SECONDS=0
fetch_results() {{
    echo FETCHED >&2; mkdir -p {model_dir}
    printf '%s' {shlex.quote(final)} > {model_dir}/provenance.json
    printf '%s' {shlex.quote(fetched_snapshot or snap)} > {model_dir}/serving_provenance.json
}}
serve_report_only_resume && echo "HANDLED rc=$FINISH_RC"
"""
    return _bash(script), model_dir


def test_report_only_resume_fetches_a_missing_final_provenance_from_the_kept_vm(
    tmp_path,
):
    # Review on #5395 (colab_gpu_reference.sh:512): every run finished, the
    # final fetch failed, and every later resume failed on the missing
    # provenance.json without ever retrying the fetch, though --keep kept the VM.
    out, model_dir = _unfetched_resume(tmp_path, vm_up=True, fetched_snapshot=None)
    assert out.returncode == 0 and "HANDLED rc=0" in out.stdout, out.stderr
    assert (
        "STEP stop" in out.stderr and "FETCHED" in out.stderr
    )  # stopped, then fetched
    assert "GRADED_ON NVIDIA H100 80GB HBM3" in out.stderr
    assert "VM_STARTED" not in out.stderr  # never a new VM


def test_report_only_resume_fails_clearly_when_the_provenance_is_unrecoverable(
    tmp_path,
):
    gone, _ = _unfetched_resume(tmp_path / "gone", vm_up=False, fetched_snapshot=None)
    assert gone.returncode == 0 and "HANDLED rc=1" in gone.stdout, gone.stderr
    assert "FETCHED" not in gone.stderr and "VM_STARTED" not in gone.stderr
    assert "is gone" in gone.stderr and "cannot be recovered" in gone.stderr
    # A session of that name that is not the VM that served: set aside, not graded.
    other = json.dumps({"gpu": {"name": "NVIDIA A100-SXM4-80GB"}, "status": "serving"})
    stray, model_dir = _unfetched_resume(
        tmp_path / "stray", vm_up=True, fetched_snapshot=other
    )
    assert stray.returncode == 0 and "HANDLED rc=1" in stray.stdout, stray.stderr
    assert "GRADED_ON" not in stray.stderr
    assert not (model_dir / "provenance.json").exists()
    assert len(list(model_dir.glob("provenance.unbound-*.json"))) == 1
