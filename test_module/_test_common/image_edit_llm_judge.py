# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Grade instruction edits with a vision LLM.

The model sees what a person reviewing the edit would: the instruction, the
input image and the edited image, and scores it against a fixed rubric. CLIP
can say the output looks like "a snowy mountain"; it cannot say whether the
road, the trees and the framing of the input survived the edit, which is half
of what makes an edit good. Informational: nothing gates on it.

Runs as its own CI step after the tests, over the edits the image eval saved
(``<run output>/images/evals/index.json``), through the same gateway and
request shape as the video judge (``video_llm_judge``):

    python -m test_module._test_common.image_edit_llm_judge --edits-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
from pathlib import Path

import requests

from .video_llm_judge import (
    DEFAULT_MODEL,
    REQUEST_TIMEOUT_SECONDS,
    _jpeg_data_url,
    gateway_url,
)

logger = logging.getLogger(__name__)

IMAGE_WIDTH = 768
CRITERIA = {
    "instruction_applied": "The edited image shows the change the instruction asks for.",
    "preservation": (
        "Everything the instruction did not ask to change is kept: composition, "
        "subjects and their identity, background and colors."
    ),
    "edit_integration": (
        "The edit blends in: no seams, smears, duplicated or malformed objects, and "
        "lighting and perspective match the rest of the image."
    ),
    "visual_quality": "Sharp and well exposed, without artifacts or distortions.",
}
INSTRUCTIONS = (
    "You are reviewing an image produced by an instruction-based image editing "
    "model. You get the instruction, the input image and the edited image. Score "
    "each criterion from 1 (fails) to 5 (no problem):\n"
    + "\n".join(f"- {name}: {text}" for name, text in CRITERIA.items())
    + "\n\nReply with only a JSON object, no other text: "
    '{"scores": {"<criterion>": <1-5>, ...}, "issues": ["<short concrete '
    'problem>", ...], "summary": "<one sentence>"}. Use an empty issues list '
    "when there are none."
)


def build_messages(instruction: str, input_image, edited_image) -> list[dict]:
    """One user message: rubric, instruction, input image, edited image."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": INSTRUCTIONS},
                {"type": "text", "text": f"Instruction: {instruction}"},
                {"type": "text", "text": "Input image:"},
                {
                    "type": "image_url",
                    "image_url": {"url": _jpeg_data_url(input_image, IMAGE_WIDTH)},
                },
                {"type": "text", "text": "Edited image:"},
                {
                    "type": "image_url",
                    "image_url": {"url": _jpeg_data_url(edited_image, IMAGE_WIDTH)},
                },
            ],
        }
    ]


def parse_verdict(text: str) -> dict:
    """The model's JSON verdict, checked against the edit rubric."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"no JSON object in reply: {text[:200]!r}")
    verdict = json.loads(text[start : end + 1])
    scores = verdict.get("scores", {})
    missing = [name for name in CRITERIA if name not in scores]
    if missing:
        raise ValueError(f"reply is missing scores for {missing}")
    for name in CRITERIA:
        if not isinstance(scores[name], (int, float)) or not 1 <= scores[name] <= 5:
            raise ValueError(f"score for {name} is not 1-5: {scores[name]!r}")
    return {
        "scores": {name: scores[name] for name in CRITERIA},
        "overall": round(statistics.mean(scores[name] for name in CRITERIA), 2),
        "issues": [str(issue) for issue in verdict.get("issues", [])],
        "summary": str(verdict.get("summary", "")),
    }


def judge_edit(
    edited_path: Path,
    input_path: Path,
    instruction: str,
    *,
    api_key: str,
    base_url: str,
    model: str = DEFAULT_MODEL,
) -> dict:
    """Ask the model for a verdict on one edit; raises on any failure."""
    from PIL import Image

    response = requests.post(
        gateway_url(base_url),
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": build_messages(
                instruction, Image.open(input_path), Image.open(edited_path)
            ),
            "temperature": 0,
            "max_tokens": 1500,
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"gateway returned {response.status_code}: {response.text[:300]}"
        )
    choices = response.json().get("choices") or []
    if not choices:
        raise RuntimeError(f"gateway returned no choices: {response.text[:300]}")
    return parse_verdict(choices[0]["message"]["content"])


def judge_directory(
    edits_dir: Path, *, api_key: str, base_url: str, model: str = DEFAULT_MODEL
) -> dict:
    """Judge every edit in a saved eval folder's index.json; one failure skips one edit."""
    edits = []
    for entry in json.loads((edits_dir / "index.json").read_text()):
        record = {
            "file": entry["file"],
            "instruction": entry["instruction"],
            "expected": entry.get("expected"),
        }
        try:
            record.update(
                judge_edit(
                    edits_dir / entry["file"],
                    edits_dir / entry["input_image"],
                    entry["instruction"],
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                )
            )
        except Exception as e:  # noqa: BLE001 - recorded per edit; the rest still run
            logger.warning(f"Could not judge {entry['file']}: {e}")
            record["error"] = str(e)
        edits.append(record)
    judged = [edit["overall"] for edit in edits if "overall" in edit]
    return {
        "model": model,
        "criteria": CRITERIA,
        "mean_overall": round(statistics.mean(judged), 2) if judged else None,
        "num_judged": len(judged),
        "edits": edits,
    }


def to_markdown(result: dict, title: str) -> str:
    lines = [
        f"### Image edit LLM judge: {title}",
        "",
        f"Model `{result['model']}`; {result['num_judged']}/{len(result['edits'])} "
        f"edits judged; mean overall **{result['mean_overall']}** / 5. "
        "Informational, not gated.",
        "",
        "| Edit | Instruction | "
        + " | ".join(name.replace("_", " ") for name in CRITERIA)
        + " | Overall | Issues |",
        "|" + "---|" * (len(CRITERIA) + 4),
    ]
    for edit in result["edits"]:
        if "error" in edit:
            cells = ["—"] * len(CRITERIA) + ["—", f"not judged: {edit['error'][:80]}"]
        else:
            cells = [str(edit["scores"][name]) for name in CRITERIA] + [
                str(edit["overall"]),
                "; ".join(edit["issues"]) or "none",
            ]
        lines.append(
            f"| {edit['file'][:2]} | {edit['instruction']} | "
            + " | ".join(cells)
            + " |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--edits-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="default: <edits-dir>/llm_judge.json")
    parser.add_argument("--summary-md", type=Path, help="append a markdown table here")
    args = parser.parse_args(argv)

    api_key, base_url = os.getenv("TT_CHAT_API_KEY"), os.getenv("TT_CHAT_URL")
    if not api_key or not base_url:
        print("Image edit LLM judge skipped: no gateway credentials.")
        return 0
    result = judge_directory(
        args.edits_dir,
        api_key=api_key,
        base_url=base_url,
        model=os.getenv("TT_CHAT_MODEL") or DEFAULT_MODEL,
    )
    out = args.out or args.edits_dir / "llm_judge.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    markdown = to_markdown(result, args.edits_dir.parent.parent.name)
    print(markdown)
    if args.summary_md:
        with open(args.summary_md, "a") as f:
            f.write(markdown + "\n")
    return 0 if result["num_judged"] else 1


if __name__ == "__main__":
    sys.exit(main())
