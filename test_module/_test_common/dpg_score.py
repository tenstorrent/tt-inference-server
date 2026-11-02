#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# /// script
# requires-python = ">=3.10,<3.11"
# dependencies = [
#     "modelscope==1.9.5",
#     "torch==2.1.2",
#     "torchvision==0.16.2",
#     "torchaudio==2.1.2",
#     "transformers==4.33.3",
#     "datasets==2.13.0",
#     "pyarrow<15",
#     "huggingface_hub<0.20",
#     "setuptools<70",
#     "numpy<2",
#     "fairseq==0.12.2",
#     "librosa==0.10.1",
#     "decord>=0.6.0",
#     "soundfile",
#     "rapidfuzz",
#     "tiktoken",
#     "transformers_stream_generator",
#     "unicodedata2",
#     "zhconv",
#     "ftfy>=6.0.3",
#     "cloudpickle",
#     "rouge_score<=0.0.4",
#     "taming-transformers-rom1504",
#     "oss2",
#     "addict",
#     "simplejson",
#     "sortedcontainers",
#     "opencv-python-headless",
#     "timm",
#     "pillow",
# ]
# ///

"""Score DPG-Bench images: ask mPLUG-large each prompt's questions, then the share answered yes.

A port of DPG-Bench's own scorer (ELLA ``dpg_bench/compute_dpg_bench.py`` at
``3c228f1d``), kept as it is, quirks included, so scores stay comparable with
published ones:

- the questions come from ``dpg_bench.csv`` (pinned by sha256), whose first
  data row the reader skips (``if i == 0: continue``); an image whose
  questions then point at a missing parent fails and is left out of the score,
  as the original's ``except`` does (``partiprompts97``, which this set holds);
- each question is put to mPLUG-large (modelscope
  ``damo/mplug_visual-question-answering_coco_large_en``) and scores 1 if the
  answer is exactly ``"yes"``;
- a question whose parent was answered no is zeroed, in question order, the
  zeroing seeing earlier zeroes;
- an image scores the mean over its questions; the benchmark score is the mean
  over images, times 100;
- category scores use the answers before zeroing (``qid2scores_orig``).

The validation set has one image per prompt (``--pic-num 1`` in the original),
at 1024x1024 or 2048x2048: the whole image is the crop. Images are a folder of
``<case_id>.png`` and the cases are the eval data's ``cases/manifest.json``,
whose DPG cases carry the prompt's ``source_id`` (DPG's ``item_id``):

    uv run test_module/_test_common/dpg_score.py --images-dir refs/r1 --data-dir .

Writes ``<images-dir>/dpg.json``: per case each question's answer and the
score; overall, per size and per category (DPG's L1 and L2).

The environment is modelscope 1.9.5's, as DPG-Bench's unpinned requirements
resolved then (its VQA pipeline imports OFA, so fairseq too). Linux only:
``decord`` has no macOS wheels. mPLUG answers in about 1 s on a CPU, so the
110 DPG images (10 to 20 questions each) take about half an hour without a GPU
(``--device cpu``). Its weights come from ModelScope's hub (modelscope.cn).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import urllib.request
from collections import defaultdict
from pathlib import Path

ELLA_COMMIT = "3c228f1dc6c4d3cad0a47493816151a419f14db3"
DPG_CSV_URL = f"https://raw.githubusercontent.com/TencentQQGYLab/ELLA/{ELLA_COMMIT}/dpg_bench/dpg_bench.csv"
DPG_CSV_SHA256 = "68f507652a49aaa4f103f17bbf25e2322a9a318dc61380e6534b749fbf9317aa"
VQA_MODEL = "damo/mplug_visual-question-answering_coco_large_en"


def load_csv(path: Path | None = None) -> bytes:
    if path is not None:
        data = path.read_bytes()
    else:
        with urllib.request.urlopen(DPG_CSV_URL, timeout=60) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != DPG_CSV_SHA256:
        raise ValueError("dpg_bench.csv does not match its pinned sha256")
    return data


def prepare_questions(data: bytes) -> dict:
    """ELLA's prepare_dpg_data: per item, each question, its tuple and its parents."""
    questions: dict = {}
    previous_id = ""
    for i, line in enumerate(csv.DictReader(io.StringIO(data.decode()))):
        # As the original: the first data row is skipped.
        if i == 0:
            continue
        current_id = line["item_id"]
        qid = int(line["proposition_id"])
        parents = [int(d.strip()) for d in line["dependency"].split(",")]
        if current_id == previous_id:
            questions[current_id]["qid2tuple"][qid] = line["tuple"]
            questions[current_id]["qid2dependency"][qid] = parents
            questions[current_id]["qid2question"][qid] = line[
                "question_natural_language"
            ]
        else:
            questions[current_id] = {
                "qid2tuple": {qid: line["tuple"]},
                "qid2dependency": {qid: parents},
                "qid2question": {qid: line["question_natural_language"]},
            }
        previous_id = current_id
    return questions


def score_item(item: dict, answer) -> dict:
    """One image (one crop): each question answered, parents applied, and the score.

    ``answer(question) -> str`` asks the VQA model about this image.
    """
    answers = {qid: answer(question) for qid, question in item["qid2question"].items()}
    scores = {qid: float(a == "yes") for qid, a in answers.items()}
    original = dict(scores)
    for qid, parents in item["qid2dependency"].items():
        # In place, in question order: a zeroed parent zeroes its children.
        if any(parent != 0 and scores[parent] == 0 for parent in parents):
            scores[qid] = 0
    return {
        "answers": answers,
        "score": sum(scores.values()) / len(scores),
        "original_scores": original,
    }


def summarize(cases: dict, items: dict, results: dict) -> dict:
    """The DPG-Bench score (x100) overall and per size, and per L1 and L2 category."""
    by_size: dict = defaultdict(list)
    categories: dict = defaultdict(list)
    for case_id, result in results.items():
        case = cases[case_id]
        by_size["all"].append(result["score"])
        by_size[f"{case['width']}x{case['height']}"].append(result["score"])
        for qid, value in result["original_scores"].items():
            category = items[case["source_id"]]["qid2tuple"][qid].split("(")[0].strip()
            categories[f"L2:{category}"].append(value)
            categories[f"L1:{category.split('-')[0].strip()}"].append(value)
    mean = lambda values: round(100 * sum(values) / len(values), 2)  # noqa: E731
    return {
        **{
            name: {"images": len(v), "score": mean(v)}
            for name, v in sorted(by_size.items())
        },
        **{
            name: {"questions": len(v), "score": mean(v)}
            for name, v in sorted(categories.items())
        },
    }


class MPLUG:
    """mPLUG-large VQA, as DPG-Bench's scorer loads it."""

    def __init__(self, device: str = "gpu"):
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        self.pipeline = pipeline(
            Tasks.visual_question_answering, model=VQA_MODEL, device=device
        )

    def __call__(self, image, question: str) -> str:
        return self.pipeline({"image": image, "question": question})["text"]


def score(
    images_dir: Path, data_dir: Path, vqa=None, csv_path: Path | None = None
) -> dict:
    """Every DPG case with an image in ``images_dir``."""
    from PIL import Image

    manifest = json.loads((data_dir / "cases" / "manifest.json").read_text())
    cases = {c["case_id"]: c for c in manifest if c.get("benchmark") == "dpg"}
    items = prepare_questions(load_csv(csv_path))
    vqa = vqa or MPLUG()
    results, missing, failed = {}, [], {}
    for case_id, case in sorted(cases.items()):
        path = images_dir / f"{case_id}.png"
        if not path.is_file():
            missing.append(case_id)
            continue
        with Image.open(path) as image:
            # pic_num 1: the crop is the whole image.
            image = image.convert("RGB")
        try:
            result = score_item(items[case["source_id"]], lambda q: vqa(image, q))
        except Exception as e:  # noqa: BLE001 - as the original, the image is left out
            failed[case_id] = f"{type(e).__name__}: {e}"
            print(f"{case_id}: failed, {failed[case_id]}", flush=True)
            continue
        results[case_id] = result
        print(f"{case_id}: {result['score']:.3f}", flush=True)
    return {
        "scorer": f"DPG-Bench (ELLA {ELLA_COMMIT[:8]}), VQA {VQA_MODEL}",
        "images": len(results),
        "missing": missing,
        "failed": failed,
        "scores": summarize(cases, items, results),
        "cases": {
            cid: {
                "score": round(r["score"], 4),
                "answers": {str(q): a for q, a in r["answers"].items()},
            }
            for cid, r in results.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--images-dir", type=Path, required=True, help="<case_id>.png images"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="the eval data: cases/manifest.json",
    )
    parser.add_argument(
        "--csv", type=Path, help="a local dpg_bench.csv (checked against its sha256)"
    )
    parser.add_argument("--out", type=Path, help="default: <images-dir>/dpg.json")
    parser.add_argument(
        "--device", default="gpu", help="modelscope's device: gpu or cpu"
    )
    args = parser.parse_args(argv)
    result = score(args.images_dir, args.data_dir, MPLUG(args.device), args.csv)
    (args.out or args.images_dir / "dpg.json").write_text(
        json.dumps(result, indent=1) + "\n"
    )
    print(json.dumps(result["scores"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
