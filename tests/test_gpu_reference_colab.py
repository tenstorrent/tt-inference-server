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
    status, lines, done = state()
    assert status == "FAILED" and "  | line2" in lines and done == 2


def test_parse_status_and_balance():
    state, lines, done = gpuref.parse_status(
        "noise\nGPUREF_STATE=RUNNING\nGPUREF_DONE=1\nGPUREF_PHASE=model 1/3 x: evals\nGPUREF_LOG| hello\n"
    )
    assert state == "RUNNING" and done == 1
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
    # Regenerated in full: a rule change reaches every derived spec, once.
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
    }
    results = session / "results"
    assert gpuref.run_covers(prov, results, ["serve", model, "--x", "1"], ["mmlu_pro"])
    assert not gpuref.run_covers(
        prov, results, ["serve", model, "--x", "2"], ["mmlu_pro"]
    )
    assert not gpuref.run_covers(
        prov, results, ["serve", model, "--x", "1"], ["humaneval"]
    )
    # --max-num-seqs only schedules requests: a run at another value still counts.
    prov["vllm_serve_command"] += ["--max-num-seqs", "32"]
    assert gpuref.run_covers(
        prov,
        results,
        ["serve", model, "--x", "1", "--max-num-seqs", "256"],
        ["mmlu_pro"],
    )


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
