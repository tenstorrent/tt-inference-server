# SPDX-License-Identifier: Apache-2.0
"""CV-Bench prompt formatting, answer extraction and the official aggregate."""
import re

_LETTER = re.compile(r"\(?\b([A-D])\b\)?")


def doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kw = lmms_eval_specific_kwargs or {}
    # doc["prompt"] is the question followed by the lettered choices ("(A) ...")
    return f"{kw.get('pre_prompt', '')}{doc['prompt']}{kw.get('post_prompt', '')}"


def extract_choice(pred):
    """First standalone option letter of the reply ('(B)', 'B', 'B. text', 'The answer is (B)'), '' if none."""
    if not pred:
        return ""
    text = pred.strip()
    m = re.match(r"^\(?([A-D])\)?(?:[\s.:)]|$)", text)
    if m:
        return m.group(1)
    m = re.search(r"answer\s*(?:is|:)?\s*\(?([A-D])\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = _LETTER.search(text)
    return m.group(1) if m else ""


def process_results(doc, results):
    pred = extract_choice(results[0] if results else "")
    gold = str(doc["answer"]).strip().strip("()").upper()
    rec = {"source": doc["source"], "type": doc["type"], "correct": int(pred == gold)}
    return {"cvbench_acc": rec, "cvbench_2d": rec, "cvbench_3d": rec}


def _acc(rows):
    return sum(r["correct"] for r in rows) / len(rows) if rows else 0.0


def aggregate_2d(results):
    by_source = {}
    for r in results:
        if r["type"] == "2D":
            by_source.setdefault(r["source"], []).append(r)
    accs = [_acc(v) for v in by_source.values()]
    return sum(accs) / len(accs) if accs else 0.0


def aggregate_3d(results):
    return _acc([r for r in results if r["type"] == "3D"])


def aggregate_official(results):
    """Mean of the 2D accuracy (mean over ADE20K / COCO) and the 3D accuracy; a missing half (--limit
    runs that only cover the 2D questions) falls back to the other one."""
    halves = [aggregate_2d(results) if any(r["type"] == "2D" for r in results) else None,
              aggregate_3d(results) if any(r["type"] == "3D" for r in results) else None]
    halves = [h for h in halves if h is not None]
    return sum(halves) / len(halves) if halves else 0.0
