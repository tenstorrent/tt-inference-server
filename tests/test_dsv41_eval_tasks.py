# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash opt-in eval tasks (reference_config/evals/eval_config.py): task selection by
DSV41_EVAL_TASKS, the long-context ISL windows, and the CV-Bench scorer on canned model outputs."""
import importlib.util
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REPO_ID = "deepseek-ai/DeepSeek-V4.1-Flash"
DEFAULT_TASKS = [
    "gpqa_diamond_cot_zeroshot", "mmlu_pro", "humaneval_instruct", "longbench2_generate", "dsv41_gsm8k_cot",
    "minerva_math", "chartqa", "docvqa_val", "mmmu_val",
]
OPT_IN = ["dsv41_longbench2_262k", "dsv41_longbench2_1m", "bbh_cot_fewshot", "mmmu_pro_standard", "dsv41_cv_bench"]


def _tasks(monkeypatch, selection):
    import reference_config.evals.eval_config as ec

    if selection is None:
        monkeypatch.delenv("DSV41_EVAL_TASKS", raising=False)
    else:
        monkeypatch.setenv("DSV41_EVAL_TASKS", selection)
    ec = __import__("importlib").reload(ec)
    return ec.ALL_EVAL_CONFIGS[REPO_ID].tasks


def test_default_run_unchanged(monkeypatch):
    assert [t.task_name for t in _tasks(monkeypatch, None)] == DEFAULT_TASKS


def test_opt_in_only_when_named(monkeypatch):
    tasks = _tasks(monkeypatch, ",".join(OPT_IN + ["dsv41_gsm8k_cot"]))
    assert [t.task_name for t in tasks] == OPT_IN + ["dsv41_gsm8k_cot"] or sorted(t.task_name for t in tasks) == sorted(OPT_IN + ["dsv41_gsm8k_cot"])
    by = {t.task_name: t for t in tasks}
    for name, score in [("dsv41_longbench2_262k", 45.2), ("dsv41_longbench2_1m", 45.2), ("bbh_cot_fewshot", 86.1),
                        ("mmmu_pro_standard", 56.5), ("dsv41_cv_bench", 77.9)]:
        assert by[name].score.published_score == score
    for name, ctx, conc in [("dsv41_longbench2_262k", 262144, 16), ("dsv41_longbench2_1m", 1048576, 4)]:
        t = by[name]
        assert t.min_context_required == ctx and t.max_concurrent == conc
        # the longest kept prompt plus the generation budget plus the template fits the served context
        assert t.custom_dataset_kwargs["maximum_isl"] + t.gen_kwargs["max_gen_toks"] + 1024 <= ctx
    assert by["dsv41_cv_bench"].include_path.endswith("lmms_eval_tasks")
    assert Path(by["dsv41_cv_bench"].include_path, "dsv41_cv_bench", "dsv41_cv_bench.yaml").is_file()
    for n in ("262k", "1m"):
        assert (REPO / "reference_config/evals/lm_eval_tasks/dsv41_longbench2" / f"dsv41_longbench2_{n}.yaml").is_file()


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cvb():
    return _load(REPO / "reference_config/evals/lmms_eval_tasks/dsv41_cv_bench/utils.py", "dsv41_cvb_utils")


@pytest.mark.parametrize(
    "pred,want",
    [("(B)", "B"), ("B", "B"), ("B. table", "B"), ("The answer is (C).", "C"), ("answer: d", "D"), ("", ""),
     ("I cannot tell", "")],
)
def test_cv_bench_extract_choice(cvb, pred, want):
    assert cvb.extract_choice(pred) == want


def test_cv_bench_prompt_and_score(cvb):
    doc = {"prompt": "How many organs are in the image? Select from the following choices.\n(A) 3\n(B) 2\n(C) 1\n(D) 0",
           "answer": "(C)", "source": "ADE20K", "type": "2D"}
    text = cvb.doc_to_text(doc, {"post_prompt": "\nAnswer with the option letter from the given choices directly."})
    assert text.endswith("(D) 0\nAnswer with the option letter from the given choices directly.")
    assert cvb.process_results(doc, ["(C)"])["cvbench_acc"]["correct"] == 1
    assert cvb.process_results(doc, ["A"])["cvbench_acc"]["correct"] == 0


def test_cv_bench_official_aggregate(cvb):
    rows = (
        [{"type": "2D", "source": "ADE20K", "correct": c} for c in (1, 1, 0, 0)]        # 0.5
        + [{"type": "2D", "source": "COCO", "correct": c} for c in (1, 1, 1, 0)]        # 0.75 -> 2D = 0.625
        + [{"type": "3D", "source": "Omni3D", "correct": c} for c in (1, 1, 1, 1, 0)]   # 0.8
    )
    assert cvb.aggregate_2d(rows) == pytest.approx(0.625)
    assert cvb.aggregate_3d(rows) == pytest.approx(0.8)
    assert cvb.aggregate_official(rows) == pytest.approx((0.625 + 0.8) / 2)
    assert cvb.aggregate_official(rows[:8]) == pytest.approx(0.625)  # 2D-only (--limit) run
