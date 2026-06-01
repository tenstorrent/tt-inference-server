# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Shared helpers for Z-Image-Turbo PCC eval variants."""

from __future__ import annotations

import base64
import io
import json
import logging
import time
from pathlib import Path
from typing import Optional

import aiohttp
import numpy as np
from PIL import Image

from ...context import MediaContext
from ..image_generation_eval_test import AccuracyResult

logger = logging.getLogger(__name__)


# Runner forces these internally; mirroring them
Z_IMAGE_TURBO_INFERENCE_STEPS = 9
Z_IMAGE_TURBO_GUIDANCE_SCALE = 0.0

DATASETS_AND_PAYLOADS_DIR = (
    Path(__file__).resolve().parent.parent.parent / "datasets_and_payloads"
)

# Shared prompts payload
Z_IMAGE_TURBO_PROMPTS_PAYLOAD = "tt-z-image-turbo_payload.json"

# HTTP request format constants
_IMAGE_FORMAT = "PNG"
_IMAGE_QUALITY = 100


def load_prompts() -> list[str]:
    """Load the shared Z-Image-Turbo prompts payload."""
    path = DATASETS_AND_PAYLOADS_DIR / Z_IMAGE_TURBO_PROMPTS_PAYLOAD
    with open(path, "r") as f:
        data = json.load(f)
    prompts = data.get("prompts") or []
    if not prompts:
        raise RuntimeError(f"No prompts found in {path}")
    return prompts


def pcc(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation over flattened tensors. Same formula as
    ``models/demos/z_image_turbo/tests/test_dit.py`` in tt-metal."""
    a_flat = a.astype(np.float64).flatten()
    b_flat = b.astype(np.float64).flatten()
    a_centered = a_flat - a_flat.mean()
    b_centered = b_flat - b_flat.mean()
    num = float((a_centered * b_centered).sum())
    den = float(np.linalg.norm(a_centered) * np.linalg.norm(b_centered))
    return num / den if den > 0 else 0.0


def decode_base64_to_image(base64image: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(base64image))).convert("RGB")


def load_pcc_thresholds(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_pcc_thresholds(path: Path, thresholds: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(thresholds, f, indent=2, sort_keys=True)


def evaluate_pcc(
    key: str,
    new_image: Image.Image,
    thresholds: dict[str, float],
    golden_dir: Path,
    tolerance: float,
) -> tuple[AccuracyResult, dict]:
    """Compare a generated image to the locked golden for ``key``.

    Lock-on-first-measurement state machine:
      1. No golden file → save the image as the golden, return BASELINE.
      2. Golden exists, no threshold yet → measure PCC, lock it as the
         threshold, return BASELINE.
      3. Golden + threshold exist → require ``pcc >= threshold - tolerance``,
         else FAIL.

    Mutates ``thresholds`` in place when a new threshold gets locked.
    """
    golden_path = golden_dir / f"{key}.png"

    if not golden_path.exists():
        golden_dir.mkdir(parents=True, exist_ok=True)
        new_image.save(golden_path, format="PNG")
        logger.info(f"📌 Saved new golden reference: {golden_path}")
        return AccuracyResult.BASELINE, {
            "key": key,
            "stage": "GOLDEN_SAVED",
            "pcc": None,
            "threshold": None,
        }

    golden_image = Image.open(golden_path).convert("RGB")
    if golden_image.size != new_image.size:
        logger.error(
            f"❌ Golden size mismatch for {key}: "
            f"golden={golden_image.size}, new={new_image.size}"
        )
        return AccuracyResult.FAIL, {
            "key": key,
            "stage": "SIZE_MISMATCH",
            "pcc": None,
            "threshold": thresholds.get(key),
        }

    measured = pcc(np.asarray(golden_image), np.asarray(new_image))
    logger.info(f"🧮 {key}: PCC={measured:.6f}")

    if key not in thresholds:
        thresholds[key] = measured
        logger.info(
            f"🔒 Locked PCC threshold for {key}: {measured:.6f} "
            f"(tolerance ±{tolerance})"
        )
        return AccuracyResult.BASELINE, {
            "key": key,
            "stage": "THRESHOLD_LOCKED",
            "pcc": measured,
            "threshold": measured,
        }

    threshold = thresholds[key]
    min_acceptable = threshold - tolerance
    if measured >= min_acceptable:
        logger.info(
            f"✅ {key}: PCC {measured:.6f} ≥ {min_acceptable:.6f} "
            f"(locked={threshold:.6f})"
        )
        return AccuracyResult.PASS, {
            "key": key,
            "stage": "PASS",
            "pcc": measured,
            "threshold": threshold,
        }
    logger.error(
        f"❌ {key}: PCC {measured:.6f} < {min_acceptable:.6f} (locked={threshold:.6f})"
    )
    return AccuracyResult.FAIL, {
        "key": key,
        "stage": "FAIL",
        "pcc": measured,
        "threshold": threshold,
    }


async def generate_image_async(
    ctx: MediaContext,
    session: aiohttp.ClientSession,
    prompt: str,
    seed: int,
    num_inference_steps: int = Z_IMAGE_TURBO_INFERENCE_STEPS,
    guidance_scale: float = Z_IMAGE_TURBO_GUIDANCE_SCALE,
) -> tuple[bool, float, Optional[str]]:
    """Single Z-Image-Turbo HTTP call. Returns (success, elapsed_s, base64_png_or_None)."""
    logger.info(f"🌅 Generating Z-Image-Turbo image (seed={seed}): {prompt[:60]}...")
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer your-secret-key",
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": prompt,
        "seed": seed,
        "guidance_scale": guidance_scale,
        "image_return_format": _IMAGE_FORMAT,
        "image_quality": _IMAGE_QUALITY,
        "number_of_images": 1,
        "num_inference_steps": num_inference_steps,
    }
    start = time.time()
    try:
        async with session.post(
            f"{ctx.base_url}/v1/images/generations",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=25000),
        ) as response:
            elapsed = time.time() - start
            if response.status != 200:
                logger.error(
                    f"❌ Z-Image-Turbo generation failed with status: {response.status}"
                )
                return False, elapsed, None
            response_data = await response.json()
            images = response_data.get("images", [])
            base64image = images[0] if images else None
            logger.info(f"✅ Z-Image-Turbo generation succeeded in {elapsed:.2f}s")
            return True, elapsed, base64image
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"❌ Z-Image-Turbo generation failed: {e}")
        return False, elapsed, None
