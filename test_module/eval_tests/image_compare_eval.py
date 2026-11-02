# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The image models' eval: TT's images against GPU reference images, case by case.

For a model with eval data on Hugging Face (``image_eval_data``): each case of
the run's subset (dev, ci or all, by its eval scope) is sent to the deployment
with its prompt, inputs, shape and seed; the image is saved with the run
(``images/evals``) and compared with the GPU image made from the same inputs
and noise (``image_compare``, in its own venv). The comparison is a report
section; the eval is graded by the compare gate
(``image_compare_gate.json``), NA until a calibration run sets its thresholds.
Text-to-image cases are also scored on their benchmarks (``text_image_scores``)
and compared with the GPU run's scores.

Until tt-media-server serves the generation API, Qwen-Image-Edit is asked
through today's ``/v1/images/edits``, which takes one input on a 1024 square:
its cases with one input are sent already padded to that square, and compared
with the GPU run made the same way (``refs/r1_square``). FLUX.2 klein is asked
through ``/v1/images/generations`` with the case's references, size and seed,
and compared with ``refs/r1``. Set ``IMAGE_EVAL_API=generation`` to use the
generation API instead.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from report_module.schema import Block

from .._test_common import ReportCheckTypes, block_id
from .._test_common.image_clients import (
    Flux2KleinImagesClient,
    GenerationAPIClient,
    GenerationError,
    LegacyEditsClient,
)
from .._test_common.image_eval_data import EVAL_DATA, eval_subset, letterbox, load_cases
from .._test_common.image_generation_api import ImageModel, model_for
from ..context import MediaContext
from .text_image_scores import score_text_images

logger = logging.getLogger(__name__)

TASK_NAME = "image_compare"
GATE_CONFIG = (
    _PROJECT_ROOT
    / "reference_config"
    / "evals"
    / "eval_targets"
    / "image_compare_gate.json"
)
COMPARE_TIMEOUT_SECONDS = 2 * 60 * 60
API_ENV = "IMAGE_EVAL_API"
# Models whose deployments still serve only today's API, and the client for it.
LEGACY_CLIENTS = {
    "qwen/qwen-image-edit-2511": LegacyEditsClient,
    "black-forest-labs/flux.2-klein-4b": Flux2KleinImagesClient,
}


def eval_model(hf_model_repo: str) -> ImageModel | None:
    """The image model the deployment serves, if it has eval data."""
    try:
        model = model_for(hf_model_repo)
    except KeyError:
        return None
    return model if model.model_id in EVAL_DATA else None


def make_client(model: ImageModel, base_url: str):
    """Today's API for a model that only has that, unless ``IMAGE_EVAL_API=generation``."""
    legacy = LEGACY_CLIENTS.get(model.model_id)
    if legacy and os.environ.get(API_ENV, "").lower() != "generation":
        return legacy(base_url, model)
    return GenerationAPIClient(base_url, model)


async def generate_all(
    client, cases: list[dict], data_dir: Path, out_dir: Path
) -> tuple[list[dict], list[dict]]:
    """Each case's image from the deployment, saved in ``out_dir``; the index and the failures."""
    import aiohttp
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    index, failures = [], []
    async with aiohttp.ClientSession() as session:
        for case in cases:
            inputs = [
                (data_dir / path).read_bytes() for path in case.get("images") or []
            ]
            entry: dict[str, Any] = {
                "case_id": case["case_id"],
                "file": f"{case['case_id']}.png",
            }
            if isinstance(client, LegacyEditsClient):
                # Sent already on the square, so TT and the GPU run get the same pixels.
                with Image.open(io.BytesIO(inputs[0])) as image:
                    square = letterbox(image)
                buf = io.BytesIO()
                square.save(buf, format="PNG")
                inputs = [buf.getvalue()]
                entry["source"] = f"inputs/{case['case_id']}.png"
                (out_dir / "inputs").mkdir(exist_ok=True)
                (out_dir / entry["source"]).write_bytes(inputs[0])
            started = time.monotonic()
            try:
                image, sent = await client.generate(session, case, inputs)
            except (
                GenerationError,
                aiohttp.ClientError,
                asyncio.TimeoutError,
                OSError,
                ValueError,
            ) as e:
                logger.error(f"❌ {case['case_id']}: {e}")
                failures.append({"case_id": case["case_id"], "problem": str(e)[:300]})
                continue
            (out_dir / entry["file"]).write_bytes(image)
            entry.update(seconds=round(time.monotonic() - started, 2), **sent)
            index.append(entry)
            logger.info(f"✅ {case['case_id']}: {entry['seconds']} s")
    (out_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    return index, failures


def run_compare(
    out_dir: Path, data_dir: Path, reference: str
) -> tuple[dict | None, str | None]:
    """The scorer's result for ``out_dir`` against ``reference``, or the reason it has none."""
    command = [
        sys.executable,
        "-m",
        "test_module._test_common.image_compare",
        "--outputs-dir",
        str(out_dir),
        "--refs-dir",
        str(data_dir / reference),
        "--data-dir",
        str(data_dir),
    ]
    logger.info("Comparing with the GPU images: %s", " ".join(command))
    try:
        code = subprocess.run(
            command, cwd=_PROJECT_ROOT, timeout=COMPARE_TIMEOUT_SECONDS
        ).returncode
    except subprocess.TimeoutExpired:
        return (
            None,
            f"the comparison stopped at its {COMPARE_TIMEOUT_SECONDS // 3600} h limit",
        )
    except Exception as e:  # noqa: BLE001 - reported in the section
        return None, f"the comparison did not run: {type(e).__name__}: {e}"
    result_path = out_dir / "compare.json"
    if code or not result_path.is_file():
        return None, f"the comparison exited with code {code}"
    return json.loads(result_path.read_text()), None


def grade(
    result: dict,
    model_id: str,
    reference: str,
    config: dict | None = None,
    text_image: dict | None = None,
) -> dict:
    """The compare gate's checks per group, and the verdict as a ReportCheckTypes name.

    A threshold the config leaves null is not checked (NA); a gate with nothing
    set is NA. Calibration means are per group, over the cases both runs have.
    A text-to-image benchmark the run has fails if TT's score is more than
    ``text_image_max_drop`` below the GPU run's, or if it could not be scored.
    """
    config = config if config is not None else json.loads(GATE_CONFIG.read_text())
    limits = (config.get("models", {}).get(model_id) or {}).get(reference) or {}
    cases = result.get("cases") or {}
    checks = []
    for group, summary in (result.get("groups") or {}).items():
        group_cases = {
            c: r
            for c, r in cases.items()
            if "error" not in r and r.get("group") == group
        }
        cap = limits.get("max_image_lpips")
        if cap is not None and "lpips" in summary:
            over = sorted(c for c, r in group_cases.items() if r.get("lpips", 0) > cap)
            checks.append(
                {
                    "group": group,
                    "check": "max_image_lpips",
                    "value": summary["worst"]["lpips"],
                    "threshold": cap,
                    "status": "FAIL" if over else "PASS",
                    "detail": ", ".join(over),
                }
            )
        floor = limits.get("min_mean_dino")
        if floor is not None and "dino" in summary:
            checks.append(
                {
                    "group": group,
                    "check": "min_mean_dino",
                    "value": summary["dino"],
                    "threshold": floor,
                    "status": "PASS" if summary["dino"] >= floor else "FAIL",
                }
            )
        margins = limits.get("mean_margin") or {}
        calibration = limits.get("calibration") or {}
        shared = [c for c in group_cases if c in calibration]
        for key, margin in margins.items():
            if not shared:
                break
            now = statistics.fmean(cases[c][key] for c in shared)
            then = statistics.fmean(calibration[c][key] for c in shared)
            worse = (now - then) if key == "lpips" else (then - now)
            checks.append(
                {
                    "group": group,
                    "check": f"mean_{key}",
                    "value": round(now, 4),
                    "threshold": round(
                        then + margin if key == "lpips" else then - margin, 4
                    ),
                    "status": "FAIL" if worse > margin else "PASS",
                    "clips": len(shared),
                }
            )
    for name, drop in (limits.get("text_image_max_drop") or {}).items():
        score = (text_image or {}).get(name)
        if drop is None or score is None:
            continue
        check = {"group": "t2i", "check": f"min_{name}"}
        if "error" in score:
            checks.append({**check, "status": "FAIL", "detail": score["error"]})
            continue
        floor = round(score["gpu"] - drop, 4)
        checks.append(
            {
                **check,
                "value": score["tt"],
                "threshold": floor,
                "status": "PASS" if score["tt"] >= floor else "FAIL",
                "detail": f"GPU {score['gpu']} over the same {score['images']} images",
            }
        )
    statuses = {c["status"] for c in checks}
    verdict = "FAIL" if "FAIL" in statuses else "PASS" if statuses else "NA"
    return {"verdict": verdict, "checks": checks, "reference": reference}


def run_image_compare_eval(ctx: MediaContext, model: ImageModel) -> Block:
    """Generate the subset's images, compare them with the GPU's, and grade them."""
    from workflow_module import accept_blocks

    from .._test_common import sweep_envelope

    client = make_client(model, ctx.base_url)
    runtime_config = getattr(ctx, "runtime_config", None)
    subset = eval_subset(getattr(runtime_config, "limit_samples_mode", None))
    data_dir, cases = load_cases(model.model_id, subset, client.reference)
    usable = [
        c
        for c in cases
        if client.max_images is None or len(c.get("images") or []) == client.max_images
    ]
    skipped = [c["case_id"] for c in cases if c not in usable]
    logger.info(
        f"{model.model_id}: {len(usable)} of {len(cases)} {subset} cases through {client.name}, "
        f"against {client.reference}"
    )
    out_dir = Path(ctx.output_path) / "images" / "evals"
    index, failures = asyncio.run(generate_all(client, usable, data_dir, out_dir))

    result, problem = (
        (None, "no image was made")
        if not index
        else run_compare(out_dir, data_dir, client.reference)
    )
    text_image = score_text_images(out_dir, data_dir, client.reference, index, usable)
    gate = (
        grade(result or {}, model.model_id, client.reference, text_image=text_image)
        if result or text_image
        else None
    )
    data: dict[str, Any] = {
        "heading": f"{model.model_id} on {ctx.device.name}",
        "model": model.model_id,
        "api": client.name,
        "reference": client.reference,
        "subset": subset,
        "num_cases": len(cases),
        "num_generated": len(index),
        "generation_failed": failures,
        "not_served": skipped,
    }
    if result:
        data.update(
            groups=result["groups"],
            compare_failed={
                c: r["error"] for c, r in result["cases"].items() if "error" in r
            },
            **{k: result[k] for k in ("lpips_error", "dino_error") if k in result},
        )
    if text_image:
        data["text_image"] = text_image
    if problem:
        data["error"] = problem
    if gate:
        data["gate"] = gate
    section = Block(
        kind=TASK_NAME,
        task_type="image",
        title="Image compare",
        id=block_id(ctx) or None,
        targets={},
        data=data,
    )
    accept_blocks([section], envelope=sweep_envelope(ctx))

    accuracy_check = ReportCheckTypes[gate["verdict"]] if gate else ReportCheckTypes.NA
    if failures:
        # A run that could not make every case it asked for is not a valid pass.
        accuracy_check = ReportCheckTypes.FAIL
    summary = {
        "task_name": TASK_NAME,
        "reference": client.reference,
        "num_cases": len(usable),
        "num_generated": len(index),
        "accuracy_check": accuracy_check,
    }
    if gate:
        failed = [c for c in gate["checks"] if c["status"] == "FAIL"]
        summary.update(num_checks=len(gate["checks"]), num_failed=len(failed))
    return Block(
        kind="evals",
        task_type="image",
        title="Image Eval",
        id=block_id(ctx) or None,
        targets={"task_name": TASK_NAME, "reference": client.reference},
        data=summary,
    )
