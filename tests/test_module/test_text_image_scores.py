# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The image eval's text-to-image scores against the GPU run's, on fake scorer results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from report_module.renderers import render_image_compare
from report_module.schema import Block
from test_module.eval_tests import image_compare_eval as eval_
from test_module.eval_tests import text_image_scores as t2i

LONGTEXT_GPU = {
    "cases": {
        "longtext_0": {"matched_words": 9, "expected_words": 10},
        "longtext_1": {"matched_words": 1, "expected_words": 10},
        "longtext_2": {"matched_words": 5, "expected_words": 5},
    }
}
DPG_GPU = {"cases": {"dpg_a": {"score": 0.9}, "dpg_b": {"score": 0.7}}}


def test_longtext_is_compared_word_for_word_over_the_shared_cases():
    tt = {
        "cases": {
            "longtext_0": {"matched_words": 8, "expected_words": 10},
            "longtext_1": {"matched_words": 2, "expected_words": 10},
        }
    }
    # longtext_2 wasn't made: the GPU's score is over the same two images.
    assert t2i.compare_scores("longtext", tt, LONGTEXT_GPU) == {
        "title": "LongText-Bench",
        "images": 2,
        "not_scored": [],
        "tt": 0.5,
        "gpu": 0.5,
        "delta": 0.0,
    }


def test_dpg_is_the_mean_image_score_times_100_and_lists_what_its_scorer_left_out():
    tt = {
        "cases": {"dpg_a": {"score": 0.8}},
        "failed": {"dpg_b": "KeyError: 1"},
    }
    score = t2i.compare_scores("dpg", tt, DPG_GPU)
    assert (score["images"], score["tt"], score["gpu"]) == (1, 80.0, 90.0)
    assert score["delta"] == -10.0 and score["not_scored"] == ["dpg_b"]
    assert "error" in t2i.compare_scores("dpg", {"cases": {}}, DPG_GPU)


def test_only_benchmarks_the_run_made_images_for_are_scored(tmp_path, monkeypatch):
    (tmp_path / "refs" / "r1").mkdir(parents=True)
    (tmp_path / "refs" / "r1" / "dpg.json").write_text(json.dumps(DPG_GPU))
    cases = [
        {"case_id": "dpg_a", "benchmark": "dpg"},
        {"case_id": "dpg_b", "benchmark": "dpg"},
        {"case_id": "longtext_0", "benchmark": "longtext"},
        {"case_id": "gedit_0", "benchmark": "gedit"},
    ]
    index = [{"case_id": c} for c in ("dpg_a", "longtext_0", "gedit_0")]
    ran = []

    def scorer(benchmark, out_dir, data_dir, images):
        ran.append((benchmark.name, images))
        return {"cases": {"dpg_a": {"score": 0.9}}}, None

    monkeypatch.setattr(t2i, "run_scorer", scorer)
    scores = t2i.score_text_images(tmp_path / "out", tmp_path, "refs/r1", index, cases)
    # dpg_b wasn't made; there's no GPU longtext.json to compare with.
    assert ran == [("dpg", 1)]
    assert scores["dpg"]["tt"] == scores["dpg"]["gpu"] == 90.0
    assert scores["longtext"]["error"] == "refs/r1 has no longtext.json"
    assert t2i.score_text_images(tmp_path, tmp_path, "refs/r1", [], cases) == {}


def fake_run(writes: dict | None, code: int = 0):
    calls = []

    def run(command, check=None, cwd=None, timeout=None):
        calls.append((command, timeout))
        if writes is not None:
            Path(command[command.index("--out") + 1]).write_text(json.dumps(writes))
        return SimpleNamespace(returncode=code)

    return run, calls


def test_a_scorer_runs_on_the_cpu_with_cpu_torch_and_a_checked_mplug(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(t2i, "_uv", lambda: Path("/uv"))
    monkeypatch.setattr(t2i, "fetch_mplug", lambda: (Path("/mplug"), None))
    run, calls = fake_run({"cases": {}})
    monkeypatch.setattr(t2i.subprocess, "run", run)
    result, problem = t2i.run_scorer(t2i.BENCHMARKS["dpg"], tmp_path, tmp_path, 100)
    assert (result, problem) == ({"cases": {}}, None)
    [(command, timeout)] = calls
    assert command[:5] == ["/uv", "run", "--managed-python", "--python", "3.10"]
    assert command[command.index("--index") + 1] == t2i.CPU_TORCH_INDEX
    assert command[command.index("--device") + 1] == "cpu"
    assert command[command.index("--vqa-model") + 1] == "/mplug"
    assert command[5:].count("--index-strategy") == 1
    # 100 images at a minute each, over the half hour floor.
    assert timeout == 6000

    monkeypatch.setattr(t2i.subprocess, "run", fake_run(None, code=1)[0])
    assert t2i.run_scorer(t2i.BENCHMARKS["longtext"], tmp_path, tmp_path, 2) == (
        None,
        "the scorer exited with code 1",
    )
    monkeypatch.setattr(t2i, "fetch_mplug", lambda: (None, "no mPLUG"))
    assert t2i.run_scorer(t2i.BENCHMARKS["dpg"], tmp_path, tmp_path, 2) == (
        None,
        "no mPLUG",
    )


def test_mplug_is_used_only_if_its_weights_are_the_gpu_runs(tmp_path, monkeypatch):
    (tmp_path / "pytorch_model.bin").write_bytes(b"weights")
    monkeypatch.setattr(t2i.subprocess, "run", lambda *a, **k: 1 / 0)
    path, problem = t2i.fetch_mplug(tmp_path)
    assert path is None and "not the GPU run's" in problem
    monkeypatch.setattr(t2i, "MPLUG_SHA256", hashlib.sha256(b"weights").hexdigest())
    # Already there and matching: no download.
    assert t2i.fetch_mplug(tmp_path) == (tmp_path, None)


def test_the_gate_checks_a_benchmark_against_the_gpu_score_less_its_drop():
    model, ref = "black-forest-labs/flux.2-klein-4b", "refs/r1"
    config = {
        "models": {
            model: {ref: {"text_image_max_drop": {"longtext": 0.05, "dpg": 2.0}}}
        }
    }
    text_image = {
        "longtext": {"tt": 0.6, "gpu": 0.63, "delta": -0.03, "images": 7},
        "dpg": {"tt": 85.0, "gpu": 88.0, "delta": -3.0, "images": 5},
    }
    gate = eval_.grade({}, model, ref, config, text_image)
    by_check = {c["check"]: c for c in gate["checks"]}
    assert by_check["min_longtext"]["status"] == "PASS"
    assert (by_check["min_dpg"]["status"], by_check["min_dpg"]["threshold"]) == (
        "FAIL",
        86.0,
    )
    # A benchmark the run doesn't have isn't checked; one that wasn't scored fails.
    gate = eval_.grade({}, model, ref, config, {"dpg": {"error": "timed out"}})
    assert [(c["check"], c["status"]) for c in gate["checks"]] == [("min_dpg", "FAIL")]
    unset = {"models": {model: {ref: {"text_image_max_drop": {"dpg": None}}}}}
    assert eval_.grade({}, model, ref, unset, text_image)["verdict"] == "NA"


def test_the_section_renders_the_text_to_image_scores():
    block = Block(
        kind="image_compare",
        title="t",
        data={
            "api": "generation",
            "reference": "refs/r1",
            "subset": "ci",
            "num_cases": 27,
            "num_generated": 27,
            "text_image": {
                "dpg": {
                    "title": "DPG-Bench",
                    "images": 4,
                    "tt": 87.5,
                    "gpu": 88.48,
                    "delta": -0.98,
                    "not_scored": ["dpg_partiprompts97_1024x1024"],
                },
                "longtext": {
                    "title": "LongText-Bench",
                    "error": "the scorer exited with code 1",
                },
            },
        },
    )
    rendered = render_image_compare(block, {})
    assert "#### Text to image" in rendered and "| DPG-Bench" in rendered
    assert "88.48" in rendered and "LongText-Bench: not scored" in rendered
    assert "left out by its scorer, dpg_partiprompts97_1024x1024" in rendered
