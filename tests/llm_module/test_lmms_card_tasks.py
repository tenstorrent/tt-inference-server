# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Scoring helpers of the lmms-eval card tasks in llm_module/lmms_plugins, and
the TT client's Wikipedia search tool.

The task utils run inside the EVALS_VISION venv; the tests skip when its
dependencies (scipy, loguru) are missing from the test environment.
"""

import importlib.util
from pathlib import Path

import pytest

TASKS = (
    Path(__file__).resolve().parents[2]
    / "llm_module/lmms_plugins/tt_lmms_plugins/tasks"
)


def _load(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, TASKS / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def odb():
    pytest.importorskip("scipy")
    return _load("odb_utils", "omnidocbench/utils.py")


@pytest.fixture(scope="module")
def hle():
    pytest.importorskip("loguru")
    return _load("hle_utils", "hle/utils.py")


def _page():
    return {
        "page_info": {"image_path": "p.jpg"},
        "extra": {
            "relation": [
                {"source_anno_id": 1, "target_anno_id": 2, "relation_type": "truncated"}
            ]
        },
        "layout_dets": [
            {"anno_id": 0, "category_type": "title", "order": 0, "text": "Results"},
            {
                "anno_id": 1,
                "category_type": "text_block",
                "order": 1,
                "text": "The first half",
            },
            {
                "anno_id": 2,
                "category_type": "text_block",
                "order": 2,
                "text": " and the second half.",
            },
            {
                "anno_id": 3,
                "category_type": "equation_isolated",
                "order": 3,
                "latex": "$$E = m c^{2}$$",
            },
            {
                "anno_id": 4,
                "category_type": "table",
                "order": 4,
                "html": "<table><tr><th>a</th></tr><tr><td>1</td></tr></table>",
            },
            {"anno_id": 5, "category_type": "page_number", "order": None, "text": "7"},
            {"anno_id": 6, "category_type": "figure", "order": 5},
        ],
    }


PERFECT = (
    "# Results\n\nThe first half and the second half.\n\n"
    "\\[ E = mc^2 \\]\n\n"
    "<table><thead><tr><td>a</td></tr></thead><tbody><tr><td>1</td></tr></tbody></table>"
)


def test_omnidocbench_perfect_page_scores_zero(odb):
    scores = odb.score_page(_page(), PERFECT)
    assert scores == {"text": 0.0, "formula": 0.0, "table": 0.0, "reading_order": 0.0}


def test_omnidocbench_empty_page_scores_one(odb):
    scores = odb.score_page(_page(), "")
    assert scores == {"text": 1.0, "formula": 1.0, "table": 1.0, "reading_order": 1.0}


def test_omnidocbench_ignored_categories_cost_nothing(odb):
    # The page number matches an ignored category, so transcribing it is free.
    assert odb.score_page(_page(), PERFECT + "\n\n7")["text"] == 0.0


def test_omnidocbench_reading_order(odb):
    swapped = (
        "The first half and the second half.\n\n# Results\n\n"
        + PERFECT.split("\n\n", 2)[2]
    )
    scores = odb.score_page(_page(), swapped)
    assert scores["text"] == 0.0
    assert scores["reading_order"] == 1.0  # both blocks out of place


def test_omnidocbench_aggregates(odb):
    results = [
        {"text": 0.2, "formula": None, "table": None, "reading_order": 0.0},
        {"text": 0.4, "formula": 0.5, "table": None, "reading_order": 0.2},
    ]
    assert odb.aggregate_text_edit(results) == pytest.approx(0.3)
    assert odb.aggregate_formula_edit(results) == pytest.approx(0.5)
    assert odb.aggregate_reading_order_edit(results) == pytest.approx(0.1)
    # The table average is NaN (no tables) and drops out of the overall mean.
    assert odb.aggregate_overall_edit(results) == pytest.approx(0.3)
    assert odb.aggregate_overall_edit_score(results) == pytest.approx(70.0)


@pytest.mark.parametrize(
    "reply, verdict",
    [
        (
            "extracted_final_answer: 5\nreasoning: correct. No issue.\ncorrect: yes\nconfidence: 90",
            ["yes"],
        ),
        ("**extracted_final_answer:** 4\n**correct:** no\n**confidence:** 100", ["no"]),
        ("The response is correct: yes, it is", []),
    ],
)
def test_hle_verdict_line(hle, reply, verdict):
    assert [v.lower() for v in hle._VERDICT_RE.findall(reply)] == verdict


def test_hle_prompt_and_empty_response(hle, monkeypatch):
    doc = {
        "id": "x",
        "question": "What is 2+2?",
        "answer": "4",
        "answer_type": "exactMatch",
        "category": "Math",
        "image": "",
    }
    assert hle.doc_to_text(doc).startswith(hle.SYSTEM_EXACT_ANSWER)
    assert hle.doc_to_visual(doc) == []
    monkeypatch.setattr(hle, "judge", lambda *a: pytest.fail("judged an empty reply"))
    out = hle.process_results(doc, [""])["hle_accuracy"]
    assert out["correct"] is False and out["judge_failed"] is False
    monkeypatch.setattr(hle, "judge", lambda *a: None)
    out = hle.process_results(doc, ["Exact Answer: 4"])["hle_accuracy"]
    assert out["correct"] is False and out["judge_failed"] is True
    assert hle.aggregate_accuracy([{"correct": True}, {"correct": False}]) == 50.0


@pytest.fixture(scope="module")
def search():
    return _load("tt_search", "../search.py")


@pytest.mark.parametrize(
    "reply, request_",
    [
        ("SEARCH: boiling point of tungsten", ("SEARCH", "boiling point of tungsten")),
        ("Let me look.\n**OPEN:** Tungsten  ", ("OPEN", "Tungsten")),
        ("`SEARCH: x`", ("SEARCH", "x")),
        ("Explanation: ...\nExact Answer: 5555 K\nConfidence: 80%", None),
        ("SEARCH: x\nExact Answer: 4", None),  # a final answer wins
        ("SEARCH:", None),
        ("", None),
    ],
)
def test_search_request_line(search, reply, request_):
    assert search.find_request(reply) == request_


def test_search_results_formatting(search, monkeypatch):
    pages = [
        {"index": 2, "title": "B", "extract": "second  page\n text"},
        {"index": 1, "title": "A", "extract": "x" * 2000},
    ]
    monkeypatch.setattr(search, "_api", lambda params: {"query": {"pages": pages}})
    out = search.run_request("SEARCH", "q")
    assert out.startswith("Wikipedia results for: q\n\n1. A\n")
    assert "2. B\nsecond page text" in out
    assert "x" * (search.SUMMARY_CHARS + 1) not in out
    monkeypatch.setattr(search, "_api", lambda params: {"query": {"pages": []}})
    assert search.run_request("SEARCH", "q") == "No Wikipedia results for: q"


def test_search_open_and_failure(search, monkeypatch):
    page = {"title": "Tungsten", "extract": "y" * (search.ARTICLE_CHARS + 10)}
    monkeypatch.setattr(search, "_api", lambda params: {"query": {"pages": [page]}})
    out = search.run_request("OPEN", "tungsten")
    assert out.startswith("Wikipedia article: Tungsten\n\n")
    assert out.endswith("[... truncated]")
    monkeypatch.setattr(
        search, "_api", lambda params: {"query": {"pages": [{"missing": True}]}}
    )
    assert search.run_request("OPEN", "nope") == "No Wikipedia article titled: nope"

    def boom(params):
        raise OSError("network down")

    monkeypatch.setattr(search, "_api", boom)
    assert (
        search.run_request("SEARCH", "q") == "The search request failed: network down"
    )


def test_hle_search_prompt(hle, monkeypatch):
    monkeypatch.syspath_prepend(str(TASKS.parents[1]))
    doc = {"question": "Q?", "answer_type": "multipleChoice"}
    text = hle.doc_to_text_search(doc, {"search_rounds": 3})
    assert text.startswith(hle.SYSTEM_MC)
    assert "up to 3 times" in text and text.endswith("Q?")
