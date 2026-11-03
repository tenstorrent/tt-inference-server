#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "torch>=2.5",
#     "torchvision",
#     "transformers==4.51.3",
#     "accelerate",
#     "qwen-vl-utils==0.0.11",
#     "pillow",
# ]
# ///

"""Score LongText-Bench images: OCR with Qwen2.5-VL-7B, then the share of expected words found.

A port of LongText-Bench's own scorer (X-Omni, ``textbench/`` at
``2b8237bb``): ``evaluate_text_reward.py`` reads each image with
Qwen2.5-VL-7B-Instruct (bf16, its prompt, ``max_new_tokens`` 1024; here at
the revision the GPU references were scored with) and
``summary_scores.py`` matches the words it read against the prompt's text.
Both are kept as they are, quirks included, so scores stay comparable with
published ones:

- the hallucination clean-up only drops ``"No text recognized."``: its loop
  rebuilds the string from the raw text each time, so only the last keyword
  sticks;
- text is normalized to lowercase letters, digits and spaces, and words are
  matched as a multiset;
- the score is micro-averaged: all words matched over all words expected.

Images are a folder of ``<case_id>.png`` (an eval's or a GPU run's) and the
cases are the eval data's ``cases/manifest.json``: its LongText cases carry
the prompt's ``text``. Needs a GPU in practice (7B, about 2 s an image on an
H100):

    uv run test_module/_test_common/longtext_score.py --images-dir refs/r1 --data-dir .

Writes ``<images-dir>/longtext.json``: per case the OCR text, words matched and
expected and the accuracy; overall and per category, length and size.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

X_OMNI_COMMIT = "2b8237bb3789638c290eeda3e83ed81bd3652c3b"
OCR_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
OCR_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
OCR_PROMPT = (
    "Recognize the text in the image, only reply with the text content, "
    "but avoid repeating previously mentioned content. "
    "If no text is recognized, please reply with 'No text recognized'."
)
MAX_NEW_TOKENS = 1024


def clean_and_remove_hallucinations(text: str) -> str:
    """X-Omni's clean-up, as written: only the last keyword is removed."""
    keywords_list = ["addCriterion", "No text recognized."]
    for keyword in keywords_list:
        s = (
            text.replace(keyword, "")
            .replace(f"\n{keyword}", "")
            .replace(f"{keyword}\n", "")
        )
    return s


def preprocess_string(s: str) -> str:
    """X-Omni's English normalization: letters, digits and spaces, lowercase."""
    cleaned = re.sub(r"[^一-龥a-zA-Z0-9\sàâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]", "", s)
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def word_match(text_gt: str, ocr: str) -> tuple[int, int]:
    """Words of ``text_gt`` found in ``ocr`` (as multisets), and words in ``text_gt``."""
    gt = Counter(text_gt.split())
    return sum((gt & Counter(ocr.split())).values()), sum(gt.values())


def score_case(text: list[str], ocr_raw: str) -> dict:
    """One image: what was read, and how many expected words it holds."""
    ocr = preprocess_string(clean_and_remove_hallucinations(ocr_raw))
    gt = preprocess_string(" ".join(text))
    matched, expected = word_match(gt, ocr)
    return {
        "ocr": ocr_raw,
        "matched_words": matched,
        "expected_words": expected,
        "accuracy": round(matched / expected, 4) if expected else None,
    }


def summarize(cases: dict[str, dict], results: dict[str, dict]) -> dict:
    """Micro-averaged accuracy overall and per category, length and size."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for case_id, result in results.items():
        case = cases[case_id]
        groups["all"].append(result)
        groups[f"category:{case['category']}"].append(result)
        groups[f"length:{case.get('length')}"].append(result)
        groups[f"size:{case['width']}x{case['height']}"].append(result)
    return {
        name: {
            "images": len(items),
            "score": round(
                sum(r["matched_words"] for r in items)
                / max(1, sum(r["expected_words"] for r in items)),
                4,
            ),
        }
        for name, items in sorted(groups.items())
    }


class QwenOCR:
    """Qwen2.5-VL-7B reading an image, as LongText-Bench's evaluator runs it."""

    def __init__(self, device: str = "cuda"):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.device = device
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            OCR_MODEL,
            revision=OCR_REVISION,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        self.processor = AutoProcessor.from_pretrained(OCR_MODEL, revision=OCR_REVISION)

    def __call__(self, image_path: Path) -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": OCR_PROMPT},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        images, _ = process_vision_info(message)
        inputs = self.processor(
            text=text, images=images, padding=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]


def score(images_dir: Path, data_dir: Path, ocr=None) -> dict:
    """Every LongText case with an image in ``images_dir``."""
    manifest = json.loads((data_dir / "cases" / "manifest.json").read_text())
    cases = {c["case_id"]: c for c in manifest if c.get("benchmark") == "longtext"}
    present = {
        cid: images_dir / f"{cid}.png"
        for cid in cases
        if (images_dir / f"{cid}.png").is_file()
    }
    ocr = ocr or QwenOCR()
    results = {}
    for case_id, path in sorted(present.items()):
        results[case_id] = score_case(cases[case_id]["text"], ocr(path))
        print(f"{case_id}: {results[case_id]['accuracy']}", flush=True)
    return {
        "scorer": f"LongText-Bench (X-Omni {X_OMNI_COMMIT[:8]}), OCR {OCR_MODEL} at {OCR_REVISION[:8]}",
        "images": len(results),
        "missing": sorted(set(cases) - set(present)),
        "scores": summarize(cases, results),
        "cases": results,
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
    parser.add_argument("--out", type=Path, help="default: <images-dir>/longtext.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    result = score(args.images_dir, args.data_dir, QwenOCR(args.device))
    (args.out or args.images_dir / "longtext.json").write_text(
        json.dumps(result, indent=1, ensure_ascii=False) + "\n"
    )
    print(json.dumps(result["scores"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
