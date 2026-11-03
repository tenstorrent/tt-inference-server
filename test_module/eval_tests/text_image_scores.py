# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The image eval's text-to-image benchmarks: TT's images scored as the GPU run's were.

LongText-Bench and DPG-Bench (``longtext_score.py``, ``dpg_score.py``) score
the run's images of their cases. Each scorer is a uv script that pins its own
environment; on the CPU it runs with torch's CPU build. TT's score is compared
with the GPU run's (``<reference>/<benchmark>.json`` in the eval data), over
the cases both have.

The eval host has no GPU, and LongText-Bench's Qwen2.5-VL-7B is too slow on a
CPU (minutes an image), so the eval scores DPG-Bench only and reports LongText
as not scored. Its images are saved with the run (``images/evals``), so
``longtext_score.py`` can score them on a GPU host as it scored the GPU run's,
or the eval can score them there itself (``IMAGE_EVAL_SCORER_DEVICE=cuda``).

The scorers' models are the ones the GPU run was scored with: Qwen2.5-VL at
the revision ``longtext_score.py`` pins, and mPLUG-large, checked by sha256.
mPLUG is downloaded with a current modelscope, because the 1.9.5 that
DPG-Bench's scorer pins refuses files the hub sends without a
``Content-Length``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCORERS_DIR = _PROJECT_ROOT / "test_module" / "_test_common"
CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
MPLUG_MODEL = "damo/mplug_visual-question-answering_coco_large_en"
MPLUG_SHA256 = "702b1fcb760b1a2bb4ab7f7b0d6f540aad438716f9fdb602f85e121d5a371332"
MPLUG_DIR = Path.home() / ".cache" / "modelscope" / "tt-image-eval" / "mplug_large"
MODELSCOPE_DOWNLOADER = "modelscope==1.40.2"
DOWNLOAD_TIMEOUT_SECONDS = 60 * 60
MIN_TIMEOUT_SECONDS = 30 * 60


@dataclass(frozen=True)
class Benchmark:
    name: str
    title: str
    script: str
    python: str
    # The scorer's time limit, per image, on a CPU.
    seconds_per_image: int
    # Too slow on a CPU to score there.
    gpu_only: bool = False


BENCHMARKS = {
    b.name: b
    for b in (
        Benchmark(
            "longtext",
            "LongText-Bench",
            "longtext_score.py",
            "3.12",
            300,
            gpu_only=True,
        ),
        Benchmark("dpg", "DPG-Bench", "dpg_score.py", "3.10", 60),
    )
}


def _uv() -> Path:
    from workflows.bootstrap_uv import UV_EXEC, bootstrap_uv

    if not UV_EXEC.exists():
        bootstrap_uv()
    return UV_EXEC


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_mplug(dest: Path = MPLUG_DIR) -> tuple[Path | None, str | None]:
    """mPLUG-large in ``dest``, downloaded if it isn't there, or why it isn't."""
    weights = dest / "pytorch_model.bin"
    if not weights.is_file():
        command = [
            str(_uv()),
            "tool",
            "run",
            "--managed-python",
            "--python",
            "3.12",
            "--from",
            MODELSCOPE_DOWNLOADER,
            "modelscope",
            "download",
            "--model",
            MPLUG_MODEL,
            "--local_dir",
            str(dest),
        ]
        logger.info("Downloading mPLUG-large: %s", " ".join(command))
        try:
            code = subprocess.run(
                command, check=False, timeout=DOWNLOAD_TIMEOUT_SECONDS
            ).returncode
        except (subprocess.TimeoutExpired, OSError) as e:
            return None, f"mPLUG-large did not download: {type(e).__name__}: {e}"
        if code or not weights.is_file():
            return None, f"mPLUG-large did not download: exit code {code}"
    found = sha256(weights)
    if found != MPLUG_SHA256:
        return None, f"mPLUG-large's weights are not the GPU run's: sha256 {found}"
    return dest, None


def run_scorer(
    benchmark: Benchmark,
    out_dir: Path,
    data_dir: Path,
    images: int,
    device: str = "cpu",
) -> tuple[dict | None, str | None]:
    """The scorer's result for the images in ``out_dir`` on ``device``, or why it has none."""
    extra = []
    if benchmark.name == "dpg":
        mplug, problem = fetch_mplug()
        if problem:
            return None, problem
        extra = ["--vqa-model", str(mplug)]
    result_path = out_dir / f"{benchmark.name}.json"
    # On the CPU, torch from its CPU index; what that index lacks at the pinned version, from PyPI.
    cpu_torch = (
        ["--index", CPU_TORCH_INDEX, "--index-strategy", "unsafe-first-match"]
        if device == "cpu"
        else []
    )
    command = [
        str(_uv()),
        "run",
        "--managed-python",
        "--python",
        benchmark.python,
        *cpu_torch,
        str(SCORERS_DIR / benchmark.script),
        "--images-dir",
        str(out_dir),
        "--data-dir",
        str(data_dir),
        "--out",
        str(result_path),
        "--device",
        device,
        *extra,
    ]
    timeout = max(MIN_TIMEOUT_SECONDS, benchmark.seconds_per_image * images)
    logger.info("Scoring %s: %s", benchmark.title, " ".join(command))
    try:
        code = subprocess.run(
            command, check=False, cwd=_PROJECT_ROOT, timeout=timeout
        ).returncode
    except subprocess.TimeoutExpired:
        return None, f"the scorer stopped at its {timeout // 60} min limit"
    except Exception as e:  # noqa: BLE001 - reported in the section
        return None, f"the scorer did not run: {type(e).__name__}: {e}"
    if code or not result_path.is_file():
        return None, f"the scorer exited with code {code}"
    return json.loads(result_path.read_text()), None


def overall(name: str, cases: dict) -> float:
    """The benchmark's score over ``cases``, as its scorer computes ``all``."""
    if name == "longtext":
        matched = sum(c["matched_words"] for c in cases.values())
        expected = sum(c["expected_words"] for c in cases.values())
        return round(matched / max(1, expected), 4)
    return round(100 * sum(c["score"] for c in cases.values()) / len(cases), 2)


def compare_scores(
    name: str, tt: dict, gpu: dict, made: set[str] | None = None
) -> dict:
    """TT's score and the GPU run's over the cases both scored, of those the run
    ``made`` (any other image the scorer found is not this run's)."""
    scored, failed = set(tt["cases"]), set(tt.get("failed") or {})
    if made is not None:
        scored, failed = scored & made, failed & made
    shared = sorted(scored & set(gpu["cases"]))
    entry = {
        "title": BENCHMARKS[name].title,
        "images": len(shared),
        "not_scored": sorted(failed),
    }
    if not shared:
        return {**entry, "error": "no case scored in both runs"}
    tt_score = overall(name, {c: tt["cases"][c] for c in shared})
    gpu_score = overall(name, {c: gpu["cases"][c] for c in shared})
    return {
        **entry,
        "tt": tt_score,
        "gpu": gpu_score,
        "delta": round(tt_score - gpu_score, 4),
    }


def score_text_images(
    out_dir: Path,
    data_dir: Path,
    reference: str,
    index: list[dict],
    cases: list[dict],
    device: str = "cpu",
) -> dict[str, dict]:
    """Per benchmark with images in the run: TT's score against the GPU run's, or
    why there is none (``skipped`` for a benchmark ``device`` can't score)."""
    made = {entry["case_id"] for entry in index}
    counts: dict[str, int] = {}
    for case in cases:
        if case["case_id"] in made and case.get("benchmark") in BENCHMARKS:
            counts[case["benchmark"]] = counts.get(case["benchmark"], 0) + 1
    scores = {}
    for name, images in counts.items():
        benchmark = BENCHMARKS[name]
        if benchmark.gpu_only and device == "cpu":
            scores[name] = {
                "title": benchmark.title,
                "images": images,
                "skipped": "too slow on a CPU; the run's images can be scored on a GPU",
            }
            continue
        gpu_path = data_dir / reference / f"{name}.json"
        if not gpu_path.is_file():
            scores[name] = {
                "title": benchmark.title,
                "error": f"{reference} has no {name}.json",
            }
            continue
        tt, problem = run_scorer(benchmark, out_dir, data_dir, images, device)
        if problem:
            scores[name] = {"title": benchmark.title, "error": problem}
            continue
        scores[name] = compare_scores(name, tt, json.loads(gpu_path.read_text()), made)
    return scores
