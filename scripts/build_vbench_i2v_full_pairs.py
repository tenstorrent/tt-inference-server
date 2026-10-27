#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow"]
# ///

"""Build the full VBench-I2V suite as an I2V pair set: 1118 prompts over 355 images.

Writes test_fixtures/datasets/vbench_i2v_full (not committed: the images are
fetched) in the layout of the fifteen-pair vbench_i2v set, so
scripts/wan_i2v_gpu_reference.py --pairs vbench_i2v_full and the VBench scorer
take it unchanged:

- prompts and their order from VBench's vbench2_i2v_full_info.json: the 355
  base prompts first (one per image), then the 763 camera-motion prompts (each
  of the 109 background images with all seven moves), so a run cut short
  still covers every image
- images as the fifteen-pair set makes them: each Pexels original, checked
  against the size VBench recorded for it, cropped with VBench's own 16:9 box
  and resized to 1280x720. With ``--crop-dir`` (VBench's crop.zip, unpacked)
  the 16:9 crops come from there instead, which also covers the five Pixabay
  images that cannot be fetched directly

Prompts whose image could not be fetched are left out and listed in
missing.json. VBench's metadata comes from its repo (Apache-2.0); the photos
are under the Pexels and Pixabay licenses.

    uv run scripts/build_vbench_i2v_full_pairs.py --vbench-dir <VBench checkout>
"""

from __future__ import annotations

import argparse
import io
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "test_fixtures" / "datasets" / "vbench_i2v_full"
DEFAULT_VBENCH_DIR = REPO_ROOT / ".workflow_venvs" / ".venv_vbench_i2v" / "VBench"
# VBench's download_data.sh fetches this from Google Drive with gdown.
BENCH_INFO_URL = "https://drive.google.com/uc?id=1zmWs_m_A4q6YgTZwIZ230jW0ttknlGJA"
SIZE = (1280, 720)
CAMERA_MOVES = {
    "camera pans left": "pan_left",
    "camera pans right": "pan_right",
    "camera tilts up": "tilt_up",
    "camera tilts down": "tilt_down",
    "camera zooms in": "zoom_in",
    "camera zooms out": "zoom_out",
    "camera static": "static",
}
ATTEMPTS = 4
HEADERS = {"User-Agent": "Mozilla/5.0 (tt-inference-server VBench-I2V pair builder)"}


def fetch(url: str) -> bytes:
    for attempt in range(1, ATTEMPTS + 1):
        try:
            request = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except Exception:
            if attempt == ATTEMPTS:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def pexels_id(url: str) -> str | None:
    """The photo id at the end of a Pexels page URL, if it is one."""
    if "pexels.com/photo/" not in url:
        return None
    match = re.search(r"(\d+)/?$", url)
    return match.group(1) if match else None


def image_name(entry: dict) -> str:
    """The file the pair set stores an image as: pexels_<id>.jpg, as in vbench_i2v."""
    pid = pexels_id(entry["url"])
    if pid:
        return f"pexels_{pid}.jpg"
    stem = re.sub(r"[^a-z0-9]+", "_", Path(entry["file_name"]).stem.lower()).strip("_")
    return f"vbench_{stem}.jpg"


def crop_from_original(entry: dict):
    """VBench's 16:9 crop of the Pexels original, checked against its recorded size."""
    from PIL import Image, ImageOps

    pid = pexels_id(entry["url"])
    if not pid:
        raise ValueError(f"not a Pexels page: {entry['url']}")
    data = fetch(f"https://images.pexels.com/photos/{pid}/pexels-photo-{pid}.jpeg")
    original = Image.open(io.BytesIO(data))
    expected = (entry["origin_width"], entry["origin_height"])
    # VBench's boxes are on the image as it recorded it; take whichever
    # orientation matches that size.
    for candidate in (original, ImageOps.exif_transpose(original)):
        if candidate.size == expected:
            break
    else:
        raise ValueError(f"original is {original.size}, VBench recorded {expected}")
    x, y, w, h = entry["diff_ratio_crop"]["16-9"]
    return candidate.convert("RGB").crop((x, y, x + w, y + h))


def crop_from_archive(entry: dict, crop_dir: Path):
    from PIL import Image

    return Image.open(crop_dir / "16-9" / entry["file_name"]).convert("RGB")


def build_image(entry: dict, out: Path, crop_dir: Path | None) -> str:
    """Write one image; returns where it came from."""
    from PIL import Image

    target = out / image_name(entry)
    if target.is_file():
        return "existing"
    if crop_dir and (crop_dir / "16-9" / entry["file_name"]).is_file():
        image, source = crop_from_archive(entry, crop_dir), "crop.zip"
    else:
        image, source = crop_from_original(entry), "original"
    image.resize(SIZE, Image.Resampling.LANCZOS).save(target, quality=95)
    return source


def manifest_entries(full_info: list[dict], bench_info: dict[str, dict]) -> list[dict]:
    """One manifest entry per VBench prompt: base prompts first, then camera moves."""
    base, camera = [], []
    for record in full_info:
        info = bench_info[record["image_name"]]
        entry = {
            "image": image_name(info),
            "caption": record["prompt_en"],
            "vbench_caption": info["caption"],
            "vbench_category": record.get("image_type") or info["type"],
            "vbench_dimensions": record["dimension"],
            "source_url": "https://" + re.sub(r"^https?://", "", info["url"]),
            "crop": "VBench-I2V 16:9 crop, resized",
            "size": list(SIZE),
        }
        if "camera_motion" in record["dimension"]:
            move = record["prompt_en"].rsplit(", ", 1)[-1]
            entry["camera_motion"] = CAMERA_MOVES[move]
            camera.append(entry)
        else:
            base.append(entry)
    return base + camera


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--vbench-dir",
        type=Path,
        default=DEFAULT_VBENCH_DIR,
        help="a VBench checkout, for vbench2_beta_i2v/vbench2_i2v_full_info.json",
    )
    parser.add_argument(
        "--bench-info",
        type=Path,
        help="VBench's i2v-bench-info.json (default: fetch it from Google Drive)",
    )
    parser.add_argument(
        "--crop-dir",
        type=Path,
        help="VBench's crop.zip unpacked (holds 16-9/); default: crop the originals",
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    full_info = json.loads(
        (
            args.vbench_dir / "vbench2_beta_i2v" / "vbench2_i2v_full_info.json"
        ).read_text()
    )
    raw = args.bench_info.read_bytes() if args.bench_info else fetch(BENCH_INFO_URL)
    bench_info = {e["file_name"]: e for e in json.loads(raw)}
    args.out.mkdir(parents=True, exist_ok=True)

    def build(entry: dict) -> tuple[str, str | None]:
        try:
            return entry["file_name"], build_image(entry, args.out, args.crop_dir)
        except Exception as e:
            return entry["file_name"], f"error: {type(e).__name__}: {e}"

    with ThreadPoolExecutor(args.workers) as pool:
        outcomes = dict(pool.map(build, bench_info.values()))
    missing = {
        name: outcome
        for name, outcome in outcomes.items()
        if outcome.startswith("error")
    }
    for name, outcome in sorted(missing.items()):
        print(f"missing {name}: {outcome}")

    entries = [
        e
        for e in manifest_entries(full_info, bench_info)
        if (args.out / e["image"]).is_file()
    ]
    (args.out / "manifest.json").write_text(json.dumps(entries, indent=2) + "\n")
    (args.out / "missing.json").write_text(json.dumps(missing, indent=2) + "\n")
    sources = {}
    for outcome in outcomes.values():
        key = "missing" if outcome.startswith("error") else outcome
        sources[key] = sources.get(key, 0) + 1
    print(
        f"{len(entries)} of {len(full_info)} prompts over "
        f"{len(bench_info) - len(missing)} of {len(bench_info)} images: {sources}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
