# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Compare an image model's TT outputs with its GPU reference images, case by case.

For each case both sides ran with the same inputs, seed and saved noise (the
eval data: ``image_eval_data``), so the outputs should agree up to numerics.
Per case, TT's image against the GPU's:

- PSNR, SSIM and LPIPS (AlexNet, via pyiqa, as the Wan2.2 I2V scorer);
- DINOv2 embedding similarity (cosine of the CLS tokens);
- for an edit: whether TT changes its input where the GPU does: the
  correlation of the two change maps, and the overlap (IoU) of the regions
  each changed;
- for an edit: spatial drift, how far the parts that should not change moved
  against the input (masked phase correlation), for TT and for the GPU;
- for a multi-reference case: how close each output is to its references
  (mean DINOv2 similarity to them), TT's and the GPU's, so a subject TT drops
  shows as a lower number than the GPU's. Faces are not checked separately yet.

The edit checks need the output in about the input's ratio (the API's closest);
a case that asks for another ratio recomposes the scene, so they are left out
for it. Results are
summed up per group (kind, reference count, size): means and the worst case.

Runs in the VBENCH_I2V venv (torch, pyiqa, scikit-image, transformers), which
it sets up and re-runs itself in, like ``vbench_i2v``:

    python3 -m test_module._test_common.image_compare --outputs-dir <run>/images/evals \\
        --refs-dir <eval data>/refs/r1 --data-dir <eval data>
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import statistics
import sys
from pathlib import Path

logger = logging.getLogger("image_compare")

REPO_ROOT = Path(__file__).resolve().parents[2]
PSNR_IDENTICAL_DB = 100.0
DINO_MODEL = "facebook/dinov2-small"
# Edit checks run on the images scaled to this long side, by area.
EDIT_LONG_SIDE = 256
# A pixel whose mean change from the input exceeds this (0-255) has changed.
CHANGE_THRESHOLD = 20.0
# Within this ratio difference (16%) the output keeps the input's frame. An
# edit's output is snapped to the API ratio closest to its input's; both
# outputs are compared with the same resized input, so the stretch is shared.
# This covers 178 of FLUX.2 klein's 180 edits and 175 of Qwen-Image-Edit's;
# the rest have inputs far wider or taller than any API ratio, and a case asked
# for another ratio recomposes the scene, so neither is checked.
SAME_RATIO_TOLERANCE = 0.15


def load_rgb(path: Path):
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def psnr(image, reference) -> float:
    import numpy as np

    error = np.mean((image.astype(np.float64) - reference.astype(np.float64)) ** 2)
    if error == 0:
        return PSNR_IDENTICAL_DB
    return min(PSNR_IDENTICAL_DB, float(10 * np.log10(255.0**2 / error)))


def ssim(image, reference) -> float:
    from skimage.metrics import structural_similarity

    return float(
        structural_similarity(image, reference, channel_axis=-1, data_range=255)
    )


def lpips_scorer(device: str):
    """LPIPS (AlexNet) between two uint8 RGB images; lower is closer."""
    import pyiqa
    import torch

    metric = pyiqa.create_metric("lpips", device=torch.device(device))

    def to_tensor(image):
        return (
            torch.from_numpy(image.copy())
            .permute(2, 0, 1)[None]
            .float()
            .div(255)
            .to(device)
        )

    def score(image, reference) -> float:
        with torch.no_grad():
            return float(metric(to_tensor(image), to_tensor(reference)))

    return score


def dino_scorer(device: str):
    """Cosine similarity of two images' DINOv2 CLS embeddings; 1 is the same."""
    import torch
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(DINO_MODEL)
    model = AutoModel.from_pretrained(DINO_MODEL).to(device).eval()

    def embed(image):
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            return model(**inputs).last_hidden_state[:, 0]

    def score(image, reference) -> float:
        return float(
            torch.nn.functional.cosine_similarity(embed(image), embed(reference)).item()
        )

    def to_references(image, references) -> float:
        """Mean similarity of ``image`` to each of ``references``."""
        own = embed(image)
        sims = [
            float(torch.nn.functional.cosine_similarity(own, embed(r)).item())
            for r in references
        ]
        return sum(sims) / len(sims)

    score.to_references = to_references
    return score


def resize(image, size: tuple[int, int]):
    import numpy as np
    from PIL import Image

    return np.asarray(Image.fromarray(image).resize(size, Image.Resampling.LANCZOS))


def same_ratio(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(math.log((a[0] / a[1]) / (b[0] / b[1]))) <= SAME_RATIO_TOLERANCE


def change_map(output, source):
    """Per pixel, the mean absolute change from ``source`` (0-255), at EDIT_LONG_SIDE."""
    import numpy as np

    h, w = output.shape[:2]
    scale = EDIT_LONG_SIDE / max(h, w)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    small = (
        resize(output, size).astype(np.float64),
        resize(source, size).astype(np.float64),
    )
    return np.abs(small[0] - small[1]).mean(axis=-1)


def drift(output, source, keep) -> float:
    """How far ``output`` moved against ``source`` where ``keep`` (pixels), by phase correlation."""
    import numpy as np
    from skimage.registration import phase_cross_correlation

    def gray(image):
        return image.astype(np.float64).mean(axis=-1)

    h, w = output.shape[:2]
    scale = EDIT_LONG_SIDE / max(h, w)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    a, b = gray(resize(source, size)), gray(resize(output, size))
    shift = phase_cross_correlation(a, b, reference_mask=keep, moving_mask=keep)
    shift = shift[0] if isinstance(shift, tuple) else shift
    # Back to the output's own pixels.
    return round(float(np.hypot(*shift)) / scale, 2)


def edit_metrics(tt, gpu, source) -> dict:
    """Whether TT changes ``source`` where the GPU does, and how far each drifted."""
    import numpy as np

    tt_change, gpu_change = change_map(tt, source), change_map(gpu, source)
    tt_mask, gpu_mask = tt_change > CHANGE_THRESHOLD, gpu_change > CHANGE_THRESHOLD
    union = np.logical_or(tt_mask, gpu_mask).sum()
    if tt_change.std() == 0 or gpu_change.std() == 0:
        correlation = 1.0 if np.allclose(tt_change, gpu_change) else 0.0
    else:
        correlation = float(np.corrcoef(tt_change.ravel(), gpu_change.ravel())[0, 1])
    metrics = {
        "change_correlation": round(correlation, 4),
        "change_iou": round(
            float(np.logical_and(tt_mask, gpu_mask).sum() / union) if union else 1.0, 4
        ),
        "changed_fraction": {
            "tt": round(float(tt_mask.mean()), 4),
            "gpu": round(float(gpu_mask.mean()), 4),
        },
    }
    # What should not change is what the GPU left as it was; a TT image that
    # drifted changed it, so its own change map cannot say where to look.
    keep = ~gpu_mask
    if keep.mean() >= 0.1:
        metrics["drift_px"] = {
            "tt": drift(tt, source, keep),
            "gpu": drift(gpu, source, keep),
        }
    return metrics


def group_of(case: dict) -> str:
    """The case's group: its kind, reference count for multi-reference, and size past 1K."""
    kind = case.get("kind") or case.get("task") or "case"
    if case.get("benchmark") and case.get("task") == "t2i":
        kind = case["benchmark"]
    if kind == "multi_ref":
        kind += f"_{len(case.get('images') or [])}"
    if case.get("size") not in (None, "1K"):
        kind += f"_{case['size']}"
    return kind


def compare_case(
    case: dict,
    tt_path: Path,
    gpu_path: Path,
    data_dir: Path,
    scorers: dict,
) -> dict:
    """TT's image against the GPU's."""
    tt, gpu = load_rgb(tt_path), load_rgb(gpu_path)
    if tt.shape != gpu.shape:
        return {
            "error": f"TT is {tt.shape[1]}x{tt.shape[0]}, GPU is {gpu.shape[1]}x{gpu.shape[0]}"
        }
    result = {"psnr": round(psnr(tt, gpu), 2), "ssim": round(ssim(tt, gpu), 4)}
    for name, scorer in scorers.items():
        result[name] = round(scorer(tt, gpu), 4)
    # Single-image edits only: a multi-reference output is a new scene, not an edit.
    kind = case.get("kind") or case.get("task")
    images = case.get("images") or []
    if kind == "multi_ref" and "dino" in scorers:
        references = [load_rgb(data_dir / path) for path in images]
        to_references = scorers["dino"].to_references
        result["reference_similarity"] = {
            "tt": round(to_references(tt, references), 4),
            "gpu": round(to_references(gpu, references), 4),
        }
    if len(images) == 1 and kind != "multi_ref":
        given = load_rgb(data_dir / images[0])
        size = (tt.shape[1], tt.shape[0])
        if same_ratio((given.shape[1], given.shape[0]), size):
            result.update(edit_metrics(tt, gpu, resize(given, size)))
    return result


def summarize(cases: dict, results: dict) -> dict:
    """Per group: how many, the means, and the worst case by LPIPS (else PSNR)."""
    groups: dict = {}
    for case_id, result in results.items():
        if "error" in result:
            continue
        groups.setdefault(group_of(cases[case_id]), []).append((case_id, result))
    summary = {}
    for group, items in sorted(groups.items()):
        entry = {"cases": len(items)}
        for key in (
            "psnr",
            "ssim",
            "lpips",
            "dino",
            "change_correlation",
            "change_iou",
        ):
            values = [r[key] for _, r in items if key in r]
            if values:
                entry[key] = round(statistics.fmean(values), 4)
        for side in ("tt", "gpu"):
            values = [r["drift_px"][side] for _, r in items if "drift_px" in r]
            if values:
                entry[f"drift_px_{side}"] = round(statistics.fmean(values), 2)
            values = [
                r["reference_similarity"][side]
                for _, r in items
                if "reference_similarity" in r
            ]
            if values:
                entry[f"reference_similarity_{side}"] = round(
                    statistics.fmean(values), 4
                )
        if any("lpips" in r for _, r in items):
            worst = max(items, key=lambda item: item[1].get("lpips", -1))
            entry["worst"] = {"case_id": worst[0], "lpips": worst[1]["lpips"]}
        else:
            worst = min(items, key=lambda item: item[1]["psnr"])
            entry["worst"] = {"case_id": worst[0], "psnr": worst[1]["psnr"]}
        summary[group] = entry
    return summary


def compare(
    outputs_dir: Path,
    refs_dir: Path,
    data_dir: Path,
    device: str = "cpu",
    dino: bool = True,
) -> dict:
    """Every case in ``outputs_dir/index.json`` against its GPU image in ``refs_dir``."""
    manifest = {
        c["case_id"]: c
        for c in json.loads((data_dir / "cases" / "manifest.json").read_text())
    }
    index = json.loads((outputs_dir / "index.json").read_text())
    scorers, notes = {}, {}
    try:
        scorers["lpips"] = lpips_scorer(device)
    except Exception as e:  # noqa: BLE001 - a missing scorer is reported, not fatal
        notes["lpips_error"] = f"{type(e).__name__}: {e}"[:500]
    if dino:
        try:
            scorers["dino"] = dino_scorer(device)
        except Exception as e:  # noqa: BLE001
            notes["dino_error"] = f"{type(e).__name__}: {e}"[:500]
    results = {}
    for entry in index:
        case_id = entry["case_id"]
        case = manifest.get(case_id)
        gpu = refs_dir / f"{case_id}.png"
        if case is None:
            results[case_id] = {"error": "not in the eval data's manifest"}
        elif not gpu.is_file():
            results[case_id] = {
                "error": f"no GPU reference {gpu.name} in {refs_dir.name}"
            }
        else:
            try:
                results[case_id] = compare_case(
                    case, outputs_dir / entry["file"], gpu, data_dir, scorers
                )
                results[case_id]["group"] = group_of(case)
            except Exception as e:  # noqa: BLE001 - one bad case does not stop the rest
                results[case_id] = {"error": f"{type(e).__name__}: {e}"[:500]}
        logger.info(f"{case_id}: {results[case_id]}")
    return {
        "reference": str(refs_dir.relative_to(data_dir))
        if refs_dir.is_relative_to(data_dir)
        else str(refs_dir),
        "device": device,
        "cases": results,
        "groups": summarize(manifest, results),
        "failed": sorted(c for c, r in results.items() if "error" in r),
        **notes,
    }


def _provision_and_reexec(argv: list[str]) -> int:
    """Set up the VBENCH_I2V venv, then run this file under its python."""
    sys.path.insert(0, str(REPO_ROOT))
    from workflow_module.engine_types import WorkflowVenvType
    from workflows.bootstrap_uv import UV_EXEC, bootstrap_uv
    from workflows.workflow_venvs import VENV_CONFIGS

    if not UV_EXEC.exists():
        bootstrap_uv()
    venv = VENV_CONFIGS[WorkflowVenvType.VBENCH_I2V]
    venv.setup(model_spec=None)
    # Run the file, not the module: the venv lacks test_module's package imports.
    command = [str(venv.venv_python), str(Path(__file__).resolve()), *argv, "--in-venv"]
    os.execv(command[0], command)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--outputs-dir", type=Path, required=True, help="TT's images and index.json"
    )
    parser.add_argument(
        "--refs-dir", type=Path, required=True, help="the GPU run: <case_id>.png"
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="the eval data: cases/ and inputs/"
    )
    parser.add_argument("--out", type=Path, help="default: <outputs-dir>/compare.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--no-dino", action="store_true", help="skip the DINOv2 similarity"
    )
    parser.add_argument("--in-venv", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not args.in_venv:
        return _provision_and_reexec(argv)
    result = compare(
        args.outputs_dir, args.refs_dir, args.data_dir, args.device, not args.no_dino
    )
    out = args.out or args.outputs_dir / "compare.json"
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(
        json.dumps({"groups": result["groups"], "failed": result["failed"]}, indent=1)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
