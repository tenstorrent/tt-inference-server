# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the Hy4-Preview benchmark harness (scripts/hy4_agentx, reference_config entries): dry-run modes of the scripts, argument parsing, runtime-spec
generation for B=4 / B=8 against golden files, grid cell lists vs the stock benchmark config, thinking switch rendering, AgentX/eval config entries.
Regenerate the golden files after an intended change with ``HY4_UPDATE_GOLDEN=1 pytest tests/test_hy4_harness.py``."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
H = REPO / "scripts" / "hy4_agentx"
FIX = REPO / "tests" / "fixtures" / "hy4"
CATALOG = FIX / "model_spec_hy4_fixture.json"
GOLD = FIX / "golden"
sys.path.insert(0, str(H))

import grid_cells  # noqa: E402
import harness_model  # noqa: E402
from harness_model import HarnessModel, apply_think_payload  # noqa: E402


def sh(*args, env=None, check=False):
    e = {**os.environ, **(env or {})}
    return subprocess.run([str(a) for a in args], capture_output=True, text=True, env=e, check=check)


def golden(name, text):
    p = GOLD / name
    if os.environ.get("HY4_UPDATE_GOLDEN"):
        p.write_text(text)
    assert p.read_text() == text, f"{name} differs from golden (HY4_UPDATE_GOLDEN=1 to regenerate)"


M = HarnessModel.load()


# ---- model constants --------------------------------------------------------------------------------------------
def test_model_constants():
    assert M.model_id == "tencent/Hy4-preview" and M.model_name == "Hy4-preview" and M.max_ctx == 1048576
    assert M.batch_sizes == (4, 8)
    assert (M.impl_id(4), M.impl_name(4)) == ("hy4_preview_b4", "hy4-preview-b4")
    assert M.catalog_model_id(8) == "id_hy4-preview-b8_Hy4-preview_blackhole_galaxy"
    assert M.tokenizer_dirname == "models--tencent--Hy4-preview"
    assert M.device_key == "BLACKHOLE_GALAXY"
    assert (M.temperature, M.top_p, M.seed) == (0.9, 1.0, 42)
    for b in (0, 1, 16):
        with pytest.raises(ValueError):
            M.impl_id(b)


def test_model_env_parser_rejects_garbage(tmp_path):
    p = tmp_path / "bad.env"
    p.write_text("HARNESS_MODEL_ID tencent/x\n")
    with pytest.raises(ValueError):
        HarnessModel.load(p)
    q = tmp_path / "short.env"
    q.write_text("HARNESS_MODEL_ID=a/b\n")
    with pytest.raises(ValueError, match="missing key"):
        HarnessModel.load(q)


def test_cli():
    assert sh("python3", H / "harness_model.py", "key", 8).stdout.strip() == "hy4_preview_b8"
    assert sh("python3", H / "harness_model.py", "impl", 4).stdout.strip() == "hy4-preview-b4"
    assert json.loads(sh("python3", H / "harness_model.py", "think", "on").stdout) == {"reasoning_effort": "high"}
    assert json.loads(sh("python3", H / "harness_model.py", "think", "off").stdout) == {"reasoning_effort": "no_think"}


# ---- thinking switch ---------------------------------------------------------------------------------------------
def test_chat_template_kwargs_and_payload():
    assert M.chat_template_kwargs(True) == {"reasoning_effort": "high"}
    assert M.chat_template_kwargs(False) == {"reasoning_effort": "no_think"}
    p = {"messages": [], "chat_template_kwargs": {"preserve": 1}}
    out = apply_think_payload(p, M, True)
    assert out["chat_template_kwargs"] == {"preserve": 1, "reasoning_effort": "high"}
    assert out["stream_options"] == {"include_usage": True}
    assert "reasoning_effort" not in out and "chat_template_kwargs" in p and "stream_options" not in p  # no top-level field, input untouched


def test_template_accepts_exactly_the_two_values():
    tpl = (H / "hf_tokenizer" / M.tokenizer_dirname / "snapshots" / "local" / "chat_template.jinja").read_text()
    assert "reasoning_effort not in ['high', 'no_think']" in tpl


def test_resolve_request_body_string_switch():
    from llm_module.request_overrides import resolve_request_body

    class T:
        task_name = "t"
        request_body = {}

    t = T()
    t.thinking, t.thinking_kwarg, t.thinking_values = True, "reasoning_effort", ("high", "no_think")
    assert resolve_request_body(t) == {"chat_template_kwargs": {"reasoning_effort": "high"}}
    t.thinking = False
    assert resolve_request_body(t) == {"chat_template_kwargs": {"reasoning_effort": "no_think"}}
    t.request_body = {"chat_template_kwargs": {"reasoning_effort": "high"}}
    with pytest.raises(ValueError, match="conflicts"):
        resolve_request_body(t)
    # default (no thinking_values) keeps the boolean rendering of the DSV4 / Qwen tasks
    u = T()
    u.thinking, u.thinking_kwarg = False, "thinking"
    assert resolve_request_body(u) == {"chat_template_kwargs": {"thinking": False}}


def test_eval_config_hy4_tasks(monkeypatch):
    monkeypatch.delenv("HY4_EVAL_TASKS", raising=False)
    import importlib

    import reference_config.evals.eval_config as ec

    ec = importlib.reload(ec)
    from llm_module.request_overrides import resolve_request_body

    tasks = {t.task_name: t for t in ec.ALL_EVAL_CONFIGS["tencent/Hy4-preview"].tasks}
    assert list(tasks) == ["gpqa_diamond_cot_zeroshot", "dsv41_gsm8k_cot", "tau3_bench_airline"]
    g = tasks["gpqa_diamond_cot_zeroshot"]
    assert resolve_request_body(g) == {"chat_template_kwargs": {"reasoning_effort": "high"}}  # GPQA thinking ON
    assert g.capture_reasoning and g.gen_kwargs["temperature"] == 0.9 and g.gen_kwargs["top_p"] == 1.0
    assert resolve_request_body(tasks["dsv41_gsm8k_cot"]) == {"chat_template_kwargs": {"reasoning_effort": "no_think"}}
    cfg = tasks["tau3_bench_airline"].agentic_eval_config
    assert json.loads(cfg.agent_kwargs["llm_args_json"])["extra_body"] == {"chat_template_kwargs": {"reasoning_effort": "high"}}
    assert cfg.n_tasks == 50 and cfg.n_concurrent_trials == 4 and cfg.extra_docker_compose
    # no gate until the own baseline exists; the gate is injected as a percent
    assert g.score.published_score == 0.0
    monkeypatch.setenv("HY4_GATE_GPQA", "71.5")
    monkeypatch.setenv("HY4_EVAL_TASKS", "gpqa_diamond_cot_zeroshot")
    ec = importlib.reload(ec)
    sel = ec.ALL_EVAL_CONFIGS["tencent/Hy4-preview"].tasks
    assert [t.task_name for t in sel] == ["gpqa_diamond_cot_zeroshot"] and sel[0].score.published_score == 71.5 and sel[0].score.tolerance == 0.0
    monkeypatch.delenv("HY4_GATE_GPQA")
    monkeypatch.delenv("HY4_EVAL_TASKS")
    importlib.reload(ec)


def test_dsv4_eval_tasks_unchanged_by_thinking_values():
    import reference_config.evals.eval_config as ec

    for t in ec.ALL_EVAL_CONFIGS["deepseek-ai/DeepSeek-V4.1-Flash"].tasks:
        assert t.thinking_values is None


# ---- run_agentx.sh -----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("b", [4, 8])
def test_run_agentx_dry_run(b):
    r = sh("bash", H / "run_agentx.sh", b, "--dry-run")
    assert r.returncode == 0, r.stderr
    o = r.stdout
    assert f"impl_id=hy4_preview_b{b} impl_name=hy4-preview-b{b} catalog_model_id=id_hy4-preview-b{b}_Hy4-preview_blackhole_galaxy" in o
    assert f"sessions(AGENTX_CONCURRENCY)={b} duration=3600 mode=full" in o
    assert "--model tencent/Hy4-preview --tt-device blackhole_galaxy --workflow agentic_traces" in o
    assert f"--runtime-model-spec-json /agentx_out/runtime_entry_b{b}.json --agentic-traces-mode full --agentic-traces-duration 3600" in o
    assert "models--tencent--Hy4-preview" in o and "DeepSeek" not in o and "deepseek" not in o


def test_run_agentx_args():
    r = sh("bash", H / "run_agentx.sh", 8, "--dry-run", "--concurrency", 12, "--duration", 900, "--mode", "ci", "--server-url", "http://x:9", "--cache-salt", "s1")
    assert "sessions(AGENTX_CONCURRENCY)=12 duration=900 mode=ci server=http://x:9 cache_salt=s1" in r.stdout
    for bad in ([16], [0], [], ["x"]):
        r = sh("bash", H / "run_agentx.sh", *bad, "--dry-run")
        assert r.returncode == 2 and "must be one of: 4 8" in r.stdout
    assert sh("bash", H / "run_agentx.sh", 4, "--bogus").returncode == 2
    assert sh("bash", H / "run_agentx.sh", 4, "--mode", "huge", "--dry-run").returncode == 2
    r = sh("bash", H / "run_agentx.sh", 4, "--dry-run", env={"HARNESS_MODEL_ENV": "/nonexistent"})
    assert r.returncode == 2 and "not found" in r.stdout
    r = sh("bash", H / "run_agentx.sh", 4, "--dry-run", env={"AGENTX_CONCURRENCY": "6"})
    assert "sessions(AGENTX_CONCURRENCY)=6" in r.stdout


# ---- tokenizer assets (the .gitignore '*.json' trap) --------------------------------------------------------------
def test_tokenizer_assets_present_and_tracked():
    d = H / "hf_tokenizer" / M.tokenizer_dirname
    snap = d / "snapshots" / "local"
    assert (d / "refs" / "main").read_text() == "local"
    names = ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"]
    for n in names:
        assert (snap / n).stat().st_size > 100, n
    import hashlib

    # config.json of the checkpoint = the D0-1 checkpoint id
    assert hashlib.sha256((snap / "config.json").read_bytes()).hexdigest().startswith("a2c0dc5e")
    r = sh("git", "-C", REPO, "ls-files", "--error-unmatch", *[str((snap / n).relative_to(REPO)) for n in names])
    if (REPO / ".git").exists() and "did not match" in r.stderr:
        pytest.fail("baked tokenizer files are not tracked (git add -f: the *.json ignore trap): " + r.stderr)


# ---- runtime spec generation (B=4 / B=8) vs golden -----------------------------------------------------------------
@pytest.mark.parametrize("b", [4, 8])
def test_entry_only_golden(b, tmp_path):
    r = sh("python3", H / "mk_runtime_spec.py", b, "agentic_traces", tmp_path, CATALOG, "--entry-only")
    assert r.returncode == 0, r.stderr + r.stdout
    assert r.stdout.splitlines()[0] == f"catalog entry: hy4_preview_b{b} id_hy4-preview-b{b}_Hy4-preview_blackhole_galaxy max_concurrency {b} max_context 1048576"
    got = json.loads((tmp_path / f"entry_hy4_preview_b{b}.json").read_text())
    golden(f"entry_hy4_preview_b{b}.json", json.dumps(got, indent=1, sort_keys=True) + "\n")


def test_entry_only_errors(tmp_path):
    bad = tmp_path / "c.json"
    d = json.loads(CATALOG.read_text())
    e = d["model_specs"]["tencent/Hy4-preview"]["BLACKHOLE_GALAXY"]["vLLM"]
    e["hy4_preview_b4"]["model_id"] = "id_wrong"
    del e["hy4_preview_b8"]
    bad.write_text(json.dumps(d))
    assert "model_id" in sh("python3", H / "mk_runtime_spec.py", 4, "agentic_traces", tmp_path, bad, "--entry-only").stderr
    assert "no entry" in sh("python3", H / "mk_runtime_spec.py", 8, "agentic_traces", tmp_path, bad, "--entry-only").stderr
    assert sh("python3", H / "mk_runtime_spec.py", 16, "agentic_traces", tmp_path, CATALOG, "--entry-only").returncode != 0


@pytest.mark.parametrize("b", [4, 8])
def test_full_runtime_spec_and_sweep(b, tmp_path):
    """The full mode builds a ModelSpec + RuntimeConfig from the entry (what run.py --runtime-model-spec-json takes) and prints the expanded sweep."""
    r = sh("python3", H / "mk_runtime_spec.py", b, "benchmarks", tmp_path, CATALOG, env={"MKSPEC_WT": str(REPO), "TT_BENCH_NO_STRUCTURED": "1"})
    assert r.returncode == 0, r.stderr[-2000:]
    assert f"max_concurrency {b} max_context 1048576 tok_all {b * 1048576}" in r.stdout
    from workflows.model_spec import ModelSpec

    spec = ModelSpec.from_json(str(tmp_path / f"entry_hy4_preview_b{b}.json"))
    assert spec.model_name == "Hy4-preview" and spec.hf_model_repo == "tencent/Hy4-preview"
    assert spec.model_id == M.catalog_model_id(b) and spec.impl.impl_name == f"hy4-preview-b{b}"
    assert spec.device_model_spec.vllm_args["default-chat-template-kwargs"] == '{"reasoning_effort": "high"}'
    assert spec.device_model_spec.vllm_args["reasoning-parser"] == "hy_v4" and spec.device_model_spec.vllm_args["tool-call-parser"] == "hy_v4"


# ---- grids --------------------------------------------------------------------------------------------------------
def test_grid_pairs():
    a = grid_cells.grid_pairs("A", M.max_ctx)
    assert len(a) == 15 and a[0] == (128, 128) and a[-1] == (1044224, 128) and (10000, 1024) in a and (8192, 1024) in a
    b = grid_cells.grid_pairs("B", M.max_ctx)
    assert len(b) == 11 and all(o == 4096 for _, o in b) and b[-1] == (1044224, 4096)
    assert all(i + o <= M.max_ctx for i, o in a + b)
    assert grid_cells.grid_pairs("A", M.max_ctx, maxisl=131072)[-1] == (131072, 128)
    assert len(grid_cells.cells("A", 4, M.max_ctx)) == 30 and len(grid_cells.cells("B", 4, M.max_ctx)) == 22
    with pytest.raises(ValueError):
        grid_cells.grid_pairs("C", M.max_ctx)


def test_grid_prompts_match_skill_definition():
    cells = {(i, o, c): n for i, o, c, n in grid_cells.cells("A", 4, M.max_ctx) + grid_cells.cells("B", 4, M.max_ctx)}
    # skill: c1 n=8 at 128/128, 4 up to ISL 4096, 2 at 8192-16384, 1 from 32768; c=B n=8B at 128, 4B at 1024-4096, 2B at 8192-16384, B from 32768
    assert cells[(128, 128, 1)] == 8 and cells[(4096, 128, 1)] == 4 and cells[(8192, 128, 1)] == 2 and cells[(16384, 128, 1)] == 2 and cells[(32768, 128, 1)] == 1
    assert cells[(128, 128, 4)] == 32 and cells[(1024, 128, 4)] == 16 and cells[(8192, 128, 4)] == 8 and cells[(32768, 128, 4)] == 4 and cells[(1044224, 4096, 4)] == 4


def test_grid_prompts_equal_stock_benchmark_config():
    from reference_config.benchmarking.benchmark_config import get_num_prompts

    for isl, osl, c, n in grid_cells.cells("A", 8, M.max_ctx) + grid_cells.cells("B", 8, M.max_ctx):
        assert n == get_num_prompts(isl, osl, c)


@pytest.mark.parametrize("b", [4, 8])
@pytest.mark.parametrize("grid", ["A", "B"])
def test_grid_cells_equal_build_benchmark_config(b, grid, tmp_path, monkeypatch):
    """The cells grid_run.sh asks for (TT_BENCH_ISL_OSL_PAIRS + TT_BENCH_CONCURRENCIES=1,B) expand to exactly grid_cells.cells in the stock benchmark config."""
    from workflows.model_spec import ModelSpec

    monkeypatch.setenv("MODEL_SPECS_ENV", "dev")
    sh("python3", H / "mk_runtime_spec.py", b, "benchmarks", tmp_path, CATALOG, "--entry-only", check=True)
    spec = ModelSpec.from_json(str(tmp_path / f"entry_hy4_preview_b{b}.json"))
    monkeypatch.setenv("TT_BENCH_ISL_OSL_PAIRS", ",".join(f"{i}x{o}" for i, o in grid_cells.grid_pairs(grid, M.max_ctx)))
    monkeypatch.setenv("TT_BENCH_CONCURRENCIES", f"1,{b}")
    monkeypatch.setenv("TT_BENCH_NO_STRUCTURED", "1")
    from reference_config.benchmarking import benchmark_config as bc

    cfg = bc.build_benchmark_config(spec)
    got = sorted((p.isl, p.osl, p.max_concurrency, p.num_prompts) for t in cfg.tasks for params in t.param_map.values() for p in params)
    assert got == sorted(grid_cells.cells(grid, b, M.max_ctx))


def test_grid_run_dry_run():
    r = sh("bash", H / "grid_run.sh", "10.0.0.1", 4, "/tmp/out", "/tmp/spec.json", "--dry-run")
    assert r.returncode == 0, r.stderr
    assert r.stdout.count("[dry-run] grid ") == 2 and "30 cells" in r.stdout and "22 cells" in r.stdout
    assert "TT_BENCH_CONCURRENCIES=1,4" in r.stdout and "--model tencent/Hy4-preview --workflow benchmarks --tools vllm" in r.stdout
    assert "1044224x128" in r.stdout and "1044224x4096" in r.stdout
    r = sh("bash", H / "grid_run.sh", "10.0.0.1", 8, "/tmp/out", "/tmp/spec.json", "--dry-run", env={"MAXISL": "131072", "GRIDS": "B"})
    assert r.stdout.count("[dry-run] grid ") == 1 and "TT_BENCH_CONCURRENCIES=1,8" in r.stdout and "262144" not in r.stdout
    assert sh("bash", H / "grid_run.sh", "10.0.0.1", 16, "/tmp/out", "/tmp/spec.json", "--dry-run").returncode == 2
    assert sh("bash", H / "grid_run.sh").returncode == 2


def test_cost_estimate_is_monotonic_and_sane():
    lo = grid_cells.estimate_seconds(4, M.max_ctx, 0, scale=2.0)
    hi = grid_cells.estimate_seconds(4, M.max_ctx, 0, scale=4.0)
    short = grid_cells.estimate_seconds(4, M.max_ctx, 131072, scale=3.0)
    assert lo["total"] < hi["total"] and short["total"] < hi["total"]
    assert 1.5 < lo["total"] / 3600 < 8 and 1.5 < hi["total"] / 3600 < 10
    assert grid_cells.ttft_c1_dsv4(1044224) == 309.0 and grid_cells.ttft_c1_dsv4(2 * 1044224) > 309.0


# ---- eval scripts ---------------------------------------------------------------------------------------------------
def test_gpqa_dry_run():
    r = sh("bash", H / "gpqa_think_remote.sh", "10.0.0.1", "/tmp/o", 4, "--dry-run")
    assert r.returncode == 0, r.stderr
    o = r.stdout
    assert 'chat_template_kwargs {"reasoning_effort": "high"}' in o
    assert "--tasks gpqa_diamond_cot_zeroshot" in o and "model=tencent/Hy4-preview" in o and "tokenizer=/mnt/tt-data/Hy4-Preview" in o
    assert "temperature=0.9,top_p=1.0" in o and "max_gen_toks=64500" in o and "seed=42" in o and "num_concurrent=4" in o
    assert "--apply_chat_template" in o
    r = sh("bash", H / "gpqa_think_remote.sh", "10.0.0.1", "/tmp/o", 8, "--dry-run", env={"PROBE": "8"})
    assert "--limit 8" in r.stdout and "num_concurrent=8" in r.stdout
    assert sh("bash", H / "gpqa_think_remote.sh").returncode == 2


def test_gsm8k_dry_run():
    r = sh("bash", H / "gsm8k_remote.sh", "10.0.0.1", "/tmp/o", "--dry-run")
    assert r.returncode == 0, r.stderr
    assert '"reasoning_effort": "no_think"' in r.stdout and "--tasks dsv41_gsm8k_cot" in r.stdout and "--num_fewshot 8" in r.stdout
    assert "max_gen_toks=2048" in r.stdout and "temperature=0.9,top_p=1.0" in r.stdout and "reference_config/evals/lm_eval_tasks" in r.stdout


# ---- AgentX config -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("b", [4, 8])
def test_agentic_traces_config(b, tmp_path, monkeypatch):
    from workflows.model_spec import ModelSpec

    sh("python3", H / "mk_runtime_spec.py", b, "agentic_traces", tmp_path, CATALOG, "--entry-only", check=True)
    spec = ModelSpec.from_json(str(tmp_path / f"entry_hy4_preview_b{b}.json"))
    monkeypatch.delenv("AGENTX_CONCURRENCY", raising=False)
    import importlib

    import reference_config.agentic_traces.agentic_traces_config as ac

    ac = importlib.reload(ac)
    cfg = ac.get_agentic_traces_config(spec)
    assert cfg is not None and cfg.model_id == M.catalog_model_id(b)
    (run,) = cfg.runs
    assert run.concurrency == b and run.public_dataset == "semianalysis_cc_traces_weka_062126"
    monkeypatch.setenv("AGENTX_CONCURRENCY", "6")
    ac = importlib.reload(ac)
    assert ac.get_agentic_traces_config(spec).runs[0].concurrency == 6
    monkeypatch.delenv("AGENTX_CONCURRENCY")
    importlib.reload(ac)


def test_dsv4_agentx_script_untouched():
    s = (REPO / "scripts" / "dsv41_agentx" / "run_agentx.sh").read_text()
    assert "deepseek-v41-flash-b${B}-1m" in s and "hy4" not in s.lower()


# ---- Harbor: local dataset + extra compose (tau3 airline), default behaviour unchanged ---------------------------------
def _harbor_cfg(tmp_path, **kw):
    from llm_module.agentic import harbor

    return harbor.HarborRunConfig(
        task_name="tau3", dataset=kw.pop("dataset", "sierra-research/tau3-bench"), agent="tau3_llm_agent", model_name="m", jobs_dir=tmp_path / "jobs",
        api_base="http://localhost:8000/v1", n_concurrent_trials=1, n_attempts=1, environment_type="docker", agent_kwargs={}, n_tasks=None, override_cpus=None,
        override_memory_mb=None, timeout_multiplier=None, agent_timeout_sec=None, venv_python=Path("/opt/venvs/evals_agentic/bin/python"), **kw)


def test_harbor_local_dataset_and_extra_compose(tmp_path):
    from llm_module.agentic import harbor

    reg = json.loads(Path(harbor._write_harbor_config(_harbor_cfg(tmp_path))).read_text())
    assert reg["datasets"] == [{"name": "sierra-research/tau3-bench"}]
    assert "extra_docker_compose" not in reg["environment"]
    loc = json.loads(Path(harbor._write_harbor_config(_harbor_cfg(tmp_path, dataset="/data/tau3-airline", extra_docker_compose=["/x/ov.yaml"]))).read_text())
    assert loc["datasets"] == [{"path": "/data/tau3-airline"}]
    assert loc["environment"]["extra_docker_compose"] == ["/x/ov.yaml"]
    assert (REPO / "reference_config/evals/harbor_overlays/tau3_build_network_host.yaml").is_file()
