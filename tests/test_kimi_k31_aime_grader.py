#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Kimi-K3.1's aime25 grader: the fork's verdict, plus the right-hand side of a boxed equation."""

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


def test_the_task_is_the_fork_s_except_for_the_grader():
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    Loader.add_constructor("!function", lambda loader, node: loader.construct_scalar(node))
    ours = yaml.load((TASKS / "aime25.yaml").read_text(), Loader)
    fork = yaml.load((Path(official.__file__).parent / "aime25.yaml").read_text(), Loader)
    assert ours.pop("process_results") == "utils.process_results_aime"
    assert fork.pop("process_results") == "utils.process_results"
    assert ours == fork
