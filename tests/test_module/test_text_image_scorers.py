# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""LongText-Bench and DPG-Bench scorers, and the multi-reference similarity, on fakes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from test_module._test_common import dpg_score as dpg
from test_module._test_common import longtext_score as longtext


def test_longtext_keeps_the_originals_cleanup_quirk():
    # Only the last keyword is removed: the loop rebuilds from the raw text.
    assert (
        longtext.clean_and_remove_hallucinations("addCriterion HELLO")
        == "addCriterion HELLO"
    )
    assert longtext.clean_and_remove_hallucinations("No text recognized.") == ""


def test_longtext_matches_words_as_a_multiset():
    result = longtext.score_case(
        ["Fresh Brews,", "every MORNING!"], "fresh brews every every"
    )
    assert (result["matched_words"], result["expected_words"]) == (3, 4)
    assert result["accuracy"] == 0.75


def write_manifest(root: Path, cases: list[dict]) -> None:
    (root / "cases").mkdir(parents=True, exist_ok=True)
    (root / "cases" / "manifest.json").write_text(json.dumps(cases))


def test_longtext_score_is_micro_averaged_over_the_images_present(tmp_path):
    cases = [
        {
            "case_id": f"longtext_{i}",
            "benchmark": "longtext",
            "category": "sign",
            "length": "short",
            "width": 1024,
            "height": 1024,
            "text": [text],
        }
        for i, text in enumerate(["one two", "three four five six"])
    ] + [{"case_id": "dpg_1", "benchmark": "dpg"}]
    write_manifest(tmp_path, cases)
    images = tmp_path / "images"
    images.mkdir()
    for i in (0, 1):
        (images / f"longtext_{i}.png").write_bytes(b"png")
    reads = {"longtext_0.png": "one two", "longtext_1.png": "three"}
    result = longtext.score(images, tmp_path, ocr=lambda path: reads[path.name])
    # 3 of 6 words, not the mean of 100% and 25%.
    assert result["scores"]["all"] == {"images": 2, "score": 0.5}
    assert result["scores"]["size:1024x1024"]["images"] == 2
    assert result["missing"] == []


CSV = """item_id,text,keywords,proposition_id,dependency,category_broad,category_detailed,tuple,question_natural_language
a,prompt a,,1,0,entity,whole,entity - whole (cat),Is there a cat?
b,prompt b,,1,0,entity,whole,entity - whole (dog),Is there a dog?
b,prompt b,,2,1,attribute,color,attribute - color (dog brown),Is the dog brown?
b,prompt b,,3,2,attribute,size,attribute - size (dog big),Is the dog big?
c,prompt c,,1,0,entity,whole,entity - whole (car),Is there a car?
c,prompt c,,2,1,attribute,color,attribute - color (car red),Is the car red?
"""


@pytest.fixture
def csv_bytes(monkeypatch):
    data = CSV.encode()
    monkeypatch.setattr(dpg, "DPG_CSV_SHA256", hashlib.sha256(data).hexdigest())
    return data


def test_dpg_skips_the_first_data_row_as_the_original(csv_bytes):
    questions = dpg.prepare_questions(csv_bytes)
    assert "a" not in questions
    assert questions["b"]["qid2dependency"] == {1: [0], 2: [1], 3: [2]}


def test_dpg_zeroes_children_in_order_and_categories_use_the_raw_answers(csv_bytes):
    item = dpg.prepare_questions(csv_bytes)["b"]
    # No dog: brown and big are zeroed, big through brown's zero.
    answers = {
        "Is there a dog?": "no",
        "Is the dog brown?": "yes",
        "Is the dog big?": "yes",
    }
    result = dpg.score_item(item, answers.get)
    assert result["score"] == 0.0
    assert result["original_scores"] == {1: 0.0, 2: 1.0, 3: 1.0}
    # Only an exact "yes" counts.
    result = dpg.score_item(item, lambda q: "Yes")
    assert result["score"] == 0.0


def test_dpg_score_leaves_out_an_image_with_a_missing_parent(tmp_path, csv_bytes, monkeypatch):
    pytest.importorskip("PIL")
    from PIL import Image

    # Item c's second question points at question 1, which is there; drop it to
    # make the missing-parent case of the skipped first row.
    broken = csv_bytes.replace(b"c,prompt c,,1,0", b"c,prompt c,,9,0")
    dpg_sha = hashlib.sha256(broken).hexdigest()
    cases = [
        {
            "case_id": f"dpg_{item}_1024x1024",
            "benchmark": "dpg",
            "source_id": item,
            "width": 1024,
            "height": 1024,
        }
        for item in ("b", "c")
    ]
    write_manifest(tmp_path, cases)
    images = tmp_path / "images"
    images.mkdir()
    for case in cases:
        Image.new("RGB", (8, 8)).save(images / f"{case['case_id']}.png")
    (tmp_path / "dpg.csv").write_bytes(broken)
    monkeypatch.setattr(dpg, "DPG_CSV_SHA256", dpg_sha)
    result = dpg.score(
        images, tmp_path, vqa=lambda image, q: "yes", csv_path=tmp_path / "dpg.csv"
    )
    assert set(result["failed"]) == {"dpg_c_1024x1024"}
    assert result["scores"]["all"] == {"images": 1, "score": 100.0}
    assert result["scores"]["L1:attribute"]["questions"] == 2
    assert result["scores"]["L2:entity - whole"]["score"] == 100.0


def test_dpg_refuses_a_csv_that_is_not_the_pinned_one(tmp_path):
    path = tmp_path / "dpg.csv"
    path.write_bytes(b"not it")
    with pytest.raises(ValueError, match="sha256"):
        dpg.load_csv(path)


def test_a_multi_reference_case_reports_both_outputs_closeness_to_its_references(
    tmp_path,
):
    np = pytest.importorskip("numpy")
    pytest.importorskip("skimage")
    from PIL import Image

    from test_module._test_common import image_compare as scorer

    refs = []
    for i in range(2):
        path = tmp_path / f"ref{i}.png"
        Image.fromarray(np.full((32, 32, 3), 40 * (i + 1), np.uint8)).save(path)
        refs.append(path.name)
    tt, gpu = tmp_path / "tt.png", tmp_path / "gpu.png"
    Image.fromarray(np.full((32, 32, 3), 40, np.uint8)).save(tt)
    Image.fromarray(np.full((32, 32, 3), 200, np.uint8)).save(gpu)

    def dino(a, b):
        return 1.0

    # Closeness is 1 / (1 + mean absolute pixel difference), enough to rank.
    dino.to_references = lambda image, references: float(
        np.mean([1 / (1 + np.abs(image.astype(float) - r).mean()) for r in references])
    )
    case = {"case_id": "m", "kind": "multi_ref", "images": refs}
    result = scorer.compare_case(case, tt, gpu, tmp_path, {"dino": dino})
    assert result["reference_similarity"]["tt"] > result["reference_similarity"]["gpu"]
    summary = scorer.summarize({"m": case}, {"m": {**result, "group": "multi_ref_2"}})
    assert "reference_similarity_tt" in summary["multi_ref_2"]
