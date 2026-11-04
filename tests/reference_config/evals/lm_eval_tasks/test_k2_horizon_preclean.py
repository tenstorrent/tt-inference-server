# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""The k2_horizon math pre-clean: readings of a boxed answer the r1_evals normaliser
rejects, tried only after the fork's grader said no (HMMT Feb 2026, job 169463: 17 of
132 correct answers rejected, 117/132 with the pre-clean, nothing lost)."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_UTILS = (
    Path(__file__).resolve().parents[4]
    / "reference_config"
    / "evals"
    / "lm_eval_tasks"
    / "k2_horizon"
    / "utils.py"
)


@pytest.fixture
def k2(monkeypatch):
    """k2_horizon.utils with a stand-in for the pinned fork's r1_evals grader: the
    normaliser strips spaces, equality is string equality after it."""
    calls = []

    def strip_answer_string(s):
        return s.replace(" ", "")

    def math_equal(a, b, **kw):
        calls.append((a, b))
        return a == b

    def process_results_math(doc, results):
        return {"exact_match": int(math_equal(results[0], strip_answer_string(doc["answer"])))}

    fake = types.ModuleType("lm_eval.tasks.r1_evals.utils")
    fake.math_equal = math_equal
    fake.strip_answer_string = strip_answer_string
    fake.postprocess_target = strip_answer_string
    fake.process_results_math = process_results_math
    for name in ("lm_eval", "lm_eval.tasks", "lm_eval.tasks.r1_evals"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "lm_eval.tasks.r1_evals.utils", fake)
    spec = importlib.util.spec_from_file_location("k2_horizon_utils_under_test", _UTILS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._calls = calls
    return mod


def test_last_boxed_balances_braces(k2):
    assert k2.last_boxed(r"so \boxed{\frac{7}{\sqrt{3}}} done") == r"\frac{7}{\sqrt{3}}"
    assert k2.last_boxed(r"\boxed{1} then \boxed{2}") == "2"
    assert k2.last_boxed("no box") is None
    assert k2.last_boxed(r"\boxed{unclosed") is None


@pytest.mark.parametrize(
    "boxed, expected",
    [
        (r"\displaystyle S=50-\frac{50}{2^{101}-1}", ["S", r"50-\frac{50}{2^{101}-1}"]),
        (r"2-\frac{\pi}{2}=\frac{4-\pi}{2}\approx 0.43", [r"2-\frac{\pi}{2}", r"\frac{4-\pi}{2}"]),
        (r"\ell=\frac{7}{\sqrt{3}}", [r"\ell", r"\frac{7}{\sqrt{3}}"]),
        (r"\mathbb E[\text{score}]=\frac{63}{256}", [r"\mathbbE[]", r"\frac{63}{256}"]),
        (r"40\%", [r"\frac{40}{100}"]),
        (r"\displaystyle x_{\min}=\sqrt[3]{\frac{3-\sqrt6}{3}}.", [r"x_{\min}", r"\sqrt[3]{\frac{3-\sqrt6}{3}}"]),
        (r"\quad 7\,", ["7"]),
    ],
)
def test_preclean_candidates(k2, boxed, expected):
    assert k2.preclean_candidates(boxed) == expected


def test_preclean_runs_only_after_the_fork_rejects(k2):
    doc = {"answer": "42"}
    # the fork accepts: no pre-clean reading is tried
    assert k2.process_results_math_preclean(doc, ["42"]) == {"exact_match": 1}
    assert k2._calls == [("42", "42")]
    # the fork rejects a labelled chain, the pre-clean's value segment matches
    k2._calls.clear()
    reply = r"think</think> hence \boxed{\displaystyle n=6\cdot 7=42}"
    assert k2.process_results_math_preclean(doc, [reply]) == {"exact_match": 1}
    assert k2._calls[0] == (reply, "42")          # the fork's verdict first
    assert ("42", "42") in k2._calls             # then the pre-cleaned readings
    # a wrong answer stays wrong; a reply without a box is left to the fork
    assert k2.process_results_math_preclean(doc, [r"\boxed{x=41}"]) == {"exact_match": 0}
    assert k2.process_results_math_preclean(doc, ["no box at all"]) == {"exact_match": 0}


def test_avg_uses_the_preclean(k2):
    doc = {"answer": "42"}
    out = k2.process_results_math_avg(doc, [[r"\boxed{n=42}", "41", "42", r"\boxed{42\%}"]])
    assert out == {"exact_match": 0.5}
