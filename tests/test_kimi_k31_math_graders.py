#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Kimi-K3.1's math graders: the fork's verdict, plus the right-hand side of a boxed equation."""

import importlib.util
from pathlib import Path

import pytest

official = pytest.importorskip("lm_eval.tasks.aime.utils")

TASKS = Path(__file__).resolve().parents[1] / "reference_config/evals/lm_eval_tasks/kimi_k31"
_spec = importlib.util.spec_from_file_location("kimi_k31_utils", TASKS / "utils.py")
k31 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(k31)


def grade(response, target):
    doc = {"answer": target}
    return (
        official.process_results(doc, [response])["exact_match"],
        k31.process_results_aime(doc, [response]),
    )


@pytest.mark.parametrize("response", [
    "... \\boxed{21+49=70}.",               # an equation whose right-hand side is the answer
    "so the answer is \\boxed{70}.",         # what the fork already accepts
    "\\boxed{x=70}",
    "\\boxed{70\\approx 70.0}",           # the exact value before an approximation
])
def test_a_correct_answer_scores_one(response):
    assert grade(response, "70")[1] == {"exact_match": 1}


@pytest.mark.parametrize("response", [
    "\\boxed{69}",
    "\\boxed{21+49}",                         # nothing is computed
    "\\boxed{21+49=71}",
    "\\boxed{x=70, y=5}",                     # several values are never reduced to one
    "\\boxed{x=3, y=70}",
    "\\boxed{n\\le 70}",                   # nor is an inequality
    "\\boxed{70=x}",                          # only the right-hand side is taken
    "no box at all, 70",
    "<think>\\boxed{1=70}</think> final \\boxed{12}",  # a box inside the reasoning is ignored
])
def test_a_wrong_answer_stays_wrong(response):
    assert grade(response, "70") == (0, {"exact_match": 0})


def test_the_fork_s_correct_verdict_is_never_lowered():
    for response in ("\\boxed{70}", "\\boxed{x=70}", "$70$", "\\boxed{070}", "\\boxed{7{,}0}"):
        fork, ours = grade(response, "70")
        assert ours["exact_match"] >= fork


r1 = pytest.importorskip("lm_eval.tasks.r1_evals.utils")


def grade_r1(response, target):
    doc = {"answer": target}
    return (
        r1.process_results_math(doc, [response])["exact_match"],
        k31.process_results_math(doc, [response])["exact_match"],
    )


@pytest.mark.parametrize("response, target", [
    ("\\boxed{xy=25}", "025"),                  # the four r1_aime24 answers of run 45895077698
    ("\\boxed{404+405=809}", "809"),
    ("\\boxed{AB\\cdot AC=468}", "468"),
    ("\\boxed{Q+R=699}", "699"),
    ("\\boxed{\\frac{1}{3}\\approx 0.333}", "\\frac{1}{3}"),
    ("\\boxed{3\\pm 2}", "1, 5"),
    ("\\boxed{1, 5}", "3\\pm 2"),
    (r"\boxed{\begin{pmatrix}1\\[2pt]2\end{pmatrix}}", r"\begin{pmatrix}1\\2\end{pmatrix}"),
])
def test_r1_a_correct_answer_the_fork_rejects_scores_one(response, target):
    assert grade_r1(response, target) == (0, 1)


@pytest.mark.parametrize("response, target", [
    ("\\boxed{808}", "809"),
    ("\\boxed{404+405=808}", "809"),
    ("\\boxed{x=809, y=3}", "809"),
    ("\\boxed{x=3, y=809}", "809"),
    ("\\boxed{n\\le 809}", "809"),
    ("\\boxed{809=x}", "809"),
    ("<think>\\boxed{1=809}</think> final \\boxed{12}", "809"),
    ("\\boxed{\\frac{1}{4}\\approx 0.333}", "\\frac{1}{3}"),
    ("\\boxed{3\\pm 2}", "1, 6"),
    (r"\boxed{\begin{pmatrix}1\\[2pt]3\end{pmatrix}}", r"\begin{pmatrix}1\\2\end{pmatrix}"),
])
def test_r1_a_wrong_answer_stays_wrong(response, target):
    assert grade_r1(response, target) == (0, 0)


@pytest.mark.parametrize("response", ["\\boxed{809}", "\\boxed{404+405}", "no box 809", "$809$"])
def test_r1_the_fork_s_verdict_is_never_lowered(response):
    fork, ours = grade_r1(response, "809")
    assert ours >= fork


def test_avg_is_the_mean_over_the_sampled_answers():
    answers = ["\\boxed{809}", "\\boxed{404+405=809}", "\\boxed{808}", "\\boxed{1}"]
    assert k31.process_results_math_avg({"answer": "809"}, [answers]) == {"exact_match": 0.5}


class _Loader(__import__("yaml").SafeLoader):
    pass


_Loader.add_constructor("!function", lambda loader, node: loader.construct_scalar(node))


def _task(path):
    import yaml

    return yaml.load(Path(path).read_text(), _Loader)


def test_aime25_is_the_fork_s_task_except_for_the_grader():
    ours = _task(TASKS / "aime25.yaml")
    fork = _task(Path(official.__file__).parent / "aime25.yaml")
    assert ours.pop("process_results") == "utils.process_results_aime"
    assert fork.pop("process_results") == "utils.process_results"
    assert ours == fork


@pytest.mark.parametrize("ours, theirs", [
    ("r1_aime24.yaml", Path(r1.__file__).parent / "aime24.yaml"),
    ("r1_math500.yaml", Path(r1.__file__).parent / "math_500.yaml"),
    ("hmmt_feb_2026_avg4.yaml", TASKS.parent / "k2_horizon" / "hmmt_feb_2026_avg4.yaml"),
])
def test_the_r1_tasks_are_the_originals_with_this_folder_s_grader(ours, theirs):
    """Same fields, including the grader's name: only the utils.py it resolves in differs."""
    assert _task(TASKS / ours) == _task(theirs)
