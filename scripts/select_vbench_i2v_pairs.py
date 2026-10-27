#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow"]
# ///

"""Select the Wan2.2 I2V validation pairs from VBench-I2V and prepare their first frames.

Validation plan v2: 50 of VBench-I2V's 1,118 image-prompt pairs, stratified by
VBench's image category, at four output shapes. 65 videos in all:

    480p  832x480 (16:9) 40 pairs   480x832 (9:16) 10 pairs
    720p 1280x720 (16:9) 11 pairs   720x1280 (9:16) 4 pairs   (same pairs as 480p)

The dev set is 5 videos (one per shape plus one more at 480p 16:9). The CI set
is 20 videos and holds the dev set. Each pair keeps VBench's own prompt, the
dimensions VBench scores it on, and a seed. Camera-motion pairs cover all
seven of VBench's moves.

First frames come from VBench's official crops (the image suite's ``crop/``
folder): 16:9 shapes from the annotation-guided 16-9 crop, 9:16 shapes from the
1-1 crop, since VBench crops nothing taller than square. Each is center-cropped
to the exact aspect of its output shape and resized with Lanczos to it, so the
GPU reference and TT receive the same pixels at the size they generate.

Writes, under ``--out``, the folder that goes to the Hugging Face dataset:

    pairs/manifest.json      one entry per video
    pairs/provenance.json    selection seed, source hashes, Pillow version
    images/<W>x<H>/<pair_id>.jpg
    review/portrait.jpg      the 9:16 first frames side by side, to check by eye

and copies the manifest to test_fixtures/datasets/vbench_i2v_50/manifest.json.

    uv run scripts/select_vbench_i2v_pairs.py --crop-dir ~/Downloads/crop --out vbench_i2v_50
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import urllib.request
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    REPO_ROOT / "test_fixtures" / "datasets" / "vbench_i2v_50" / "manifest.json"
)

VBENCH_COMMIT = "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"
FULL_INFO_URL = (
    f"https://raw.githubusercontent.com/Vchitect/VBench/{VBENCH_COMMIT}"
    "/vbench2_beta_i2v/vbench2_i2v_full_info.json"
)
FULL_INFO_SHA256 = "8cf4d34dc11779a336c216a1a71d482927cfa24a1bf736ab5c92425609f129f8"
# VBench ships the image annotations on Google Drive only; this mirror holds
# them unmodified, which the official hash checks.
BENCH_INFO_URL = (
    "https://huggingface.co/datasets/xcpan/VBench-I2V-832x480"
    "/resolve/main/metadata/i2v-bench-info.json"
)
BENCH_INFO_SHA256 = "7dbe47684eadb2b4ace3fe452a4d8e8ceffa9c788d9c1ff048af3bb2585f96bf"

SELECTION_SEED = 20261027
NUM_PAIRS = 50
NUM_CAMERA_PAIRS = 10
# Output shapes (width, height) and how many pairs each gets.
LANDSCAPE_480P = (832, 480)
PORTRAIT_480P = (480, 832)
LANDSCAPE_720P = (1280, 720)
PORTRAIT_720P = (720, 1280)
NUM_PORTRAIT = 10
NUM_720P_LANDSCAPE = 11
NUM_720P_PORTRAIT = 4
# CI videos per shape; the dev set is one per shape plus one at 480p 16:9.
CI_COUNTS = {LANDSCAPE_480P: 10, PORTRAIT_480P: 4, LANDSCAPE_720P: 4, PORTRAIT_720P: 2}
SHAPE_ORDER = (LANDSCAPE_480P, PORTRAIT_480P, LANDSCAPE_720P, PORTRAIT_720P)
# Images whose 9:16 center crop loses the subject, found by looking at
# review/portrait.jpg; the selection draws another portrait pair instead.
PORTRAIT_EXCLUDE: frozenset[str] = frozenset()

# VBench's camera-motion prompts end in one of these; the key is the name the
# VBench scorer and the eval use (test_module/_test_common/vbench_i2v.py).
CAMERA_MOVES = {
    "camera pans left": "pan_left",
    "camera pans right": "pan_right",
    "camera tilts up": "tilt_up",
    "camera tilts down": "tilt_down",
    "camera zooms in": "zoom_in",
    "camera zooms out": "zoom_out",
    "camera static": "static",
}
JPEG_OPTIONS = {"quality": 100, "subsampling": 0}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_verified(url: str, expected_sha256: str, path: Path | None) -> list:
    """A JSON source, read from ``path`` or downloaded, after checking its hash."""
    if path is not None:
        data = path.read_bytes()
    else:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
    if sha256(data) != expected_sha256:
        raise ValueError(
            f"{path or url} has sha256 {sha256(data)}, expected {expected_sha256}"
        )
    return json.loads(data)


def camera_move(prompt: str) -> str | None:
    """The VBench camera move a prompt ends in, if it asks for one."""
    for phrase, move in CAMERA_MOVES.items():
        if prompt.endswith(f", {phrase}"):
            return move
    return None


def category_quotas(categories: Counter, total: int) -> dict[str, int]:
    """``total`` split over categories by their share, at least one each.

    Largest remainder; ties go to the alphabetically first category.
    """
    n = sum(categories.values())
    exact = {c: total * k / n for c, k in categories.items()}
    quotas = {c: max(1, int(x)) for c, x in exact.items()}
    by_remainder = sorted(exact, key=lambda c: (-(exact[c] - int(exact[c])), c))
    i = 0
    while sum(quotas.values()) < total:
        quotas[by_remainder[i % len(by_remainder)]] += 1
        i += 1
    return quotas


def spread(items: list, k: int) -> list:
    """``k`` items evenly spaced through ``items``, starting with the first."""
    if k > len(items):
        raise ValueError(f"want {k} of {len(items)} items")
    return [items[i * len(items) // k] for i in range(k)]


def select_pairs(
    full_info: list[dict],
    seed: int = SELECTION_SEED,
    portrait_exclude: frozenset[str] = PORTRAIT_EXCLUDE,
) -> list[dict]:
    """The 50 pairs, each with the output shapes it runs at and its dev and CI shapes.

    Every pair is a different image. Camera-motion pairs are drawn from the
    images VBench gives camera prompts (its background categories), and their
    moves cover all seven before any repeats.
    """
    rng = random.Random(seed)
    base = {}  # image name -> (index, entry) of its one non-camera prompt
    camera = {}  # image name -> {move: (index, entry)}
    for index, entry in enumerate(full_info):
        move = camera_move(entry["prompt_en"])
        if move is None:
            base[entry["image_name"]] = (index, entry)
        else:
            camera.setdefault(entry["image_name"], {})[move] = (index, entry)

    categories = Counter(entry["image_type"] for _, entry in base.values())
    quotas = category_quotas(categories, NUM_PAIRS)
    camera_categories = sorted({base[name][1]["image_type"] for name in camera})

    # Which background-category slots are camera pairs, and which move each gets.
    slots = [c for c in camera_categories for _ in range(quotas[c])]
    rng.shuffle(slots)
    camera_slots = Counter(slots[:NUM_CAMERA_PAIRS])
    moves = list(CAMERA_MOVES.values())
    rng.shuffle(moves)
    moves += rng.sample(moves, NUM_CAMERA_PAIRS - len(moves))

    pairs = []
    for category in sorted(quotas):
        images = sorted(n for n, (_, e) in base.items() if e["image_type"] == category)
        for i, name in enumerate(rng.sample(images, quotas[category])):
            if i < camera_slots[category]:
                index, entry = camera[name][moves.pop()]
            else:
                index, entry = base[name]
            pairs.append(
                {
                    "pair_id": f"vbi2v_{index:04d}",
                    "seed": index,
                    "caption": entry["prompt_en"],
                    "vbench_image": name,
                    "vbench_category": category,
                    "vbench_dimensions": entry["dimension"],
                    "camera_motion": camera_move(entry["prompt_en"]),
                }
            )
    assign_shapes(pairs, rng, portrait_exclude)
    return pairs


def assign_shapes(
    pairs: list[dict], rng: random.Random, portrait_exclude: frozenset[str]
) -> None:
    """Give each pair its output shapes and mark its dev and CI videos, in place.

    Pairs are spread by category order, so each shape and subset draws from
    across the categories. Every 720p video has a 480p twin of the same aspect.
    """
    order = sorted(pairs, key=lambda p: (p["vbench_category"], rng.random()))
    portrait = spread(
        [p for p in order if p["vbench_image"] not in portrait_exclude], NUM_PORTRAIT
    )
    portrait_ids = {p["pair_id"] for p in portrait}
    landscape = [p for p in order if p["pair_id"] not in portrait_ids]
    for p in landscape:
        p["shapes"] = [LANDSCAPE_480P]
    for p in portrait:
        p["shapes"] = [PORTRAIT_480P]
    for p in spread(landscape, NUM_720P_LANDSCAPE):
        p["shapes"].append(LANDSCAPE_720P)
    for p in spread(portrait, NUM_720P_PORTRAIT):
        p["shapes"].append(PORTRAIT_720P)
    for p in pairs:
        p["dev"], p["ci"] = [], []

    def with_shape(group, shape):
        return [p for p in group if shape in p["shapes"]]

    # Dev: one pair per aspect at both resolutions, and one 480p-only pair.
    # The extra is a camera pair when one is free, so the dev set has a move.
    dev_landscape = with_shape(landscape, LANDSCAPE_720P)[0]
    dev_portrait = with_shape(portrait, PORTRAIT_720P)[0]
    dev_landscape["dev"] = [LANDSCAPE_480P, LANDSCAPE_720P]
    dev_portrait["dev"] = [PORTRAIT_480P, PORTRAIT_720P]
    single = [p for p in landscape if len(p["shapes"]) == 1]
    extra = next((p for p in single if p["camera_motion"]), single[0])
    extra["dev"] = [LANDSCAPE_480P]

    # CI: the dev videos, then 720p videos and their 480p twins, then 480p only.
    for p in pairs:
        p["ci"] = list(p["dev"])
    for group, big, small in (
        (landscape, LANDSCAPE_720P, LANDSCAPE_480P),
        (portrait, PORTRAIT_720P, PORTRAIT_480P),
    ):
        need = CI_COUNTS[big] - sum(big in p["ci"] for p in group)
        rest = [p for p in with_shape(group, big) if big not in p["ci"]]
        for p in spread(rest, need) if need else []:
            p["ci"] += [small, big]
        need = CI_COUNTS[small] - sum(small in p["ci"] for p in group)
        rest = [p for p in group if small not in p["ci"]]
        for p in spread(rest, need) if need else []:
            p["ci"].append(small)


def center_crop_box(size: tuple[int, int], shape: tuple[int, int]) -> tuple:
    """The largest box of ``shape``'s aspect centered in an image of ``size``."""
    src_w, src_h = size
    width, height = shape
    if src_w * height > src_h * width:  # source wider: trim the sides
        crop_w, crop_h = round(src_h * width / height), src_h
    else:  # source taller: trim top and bottom
        crop_w, crop_h = src_w, round(src_w * height / width)
    left, top = (src_w - crop_w) // 2, (src_h - crop_h) // 2
    return (left, top, left + crop_w, top + crop_h)


def source_ratio(shape: tuple[int, int]) -> str:
    """The VBench crop a shape starts from: 16-9 for landscape, 1-1 for portrait."""
    width, height = shape
    return "16-9" if width >= height else "1-1"


def prepare_first_frame(crop_dir: Path, image_name: str, shape: tuple[int, int]):
    """The first frame for ``shape``: VBench's crop, center-cropped to the aspect, resized."""
    from PIL import Image

    with Image.open(crop_dir / source_ratio(shape) / image_name) as source:
        image = source.convert("RGB")
    image = image.crop(center_crop_box(image.size, shape))
    return image.resize(shape, Image.Resampling.LANCZOS)


def videos(pairs: list[dict]) -> list[dict]:
    """One manifest entry per pair and shape, ordered by shape then pair."""
    entries = []
    for shape in SHAPE_ORDER:
        for p in sorted(pairs, key=lambda p: p["pair_id"]):
            if shape not in p["shapes"]:
                continue
            width, height = shape
            entries.append(
                {
                    "video_id": f"{p['pair_id']}_{width}x{height}",
                    "pair_id": p["pair_id"],
                    "image": f"images/{width}x{height}/{p['pair_id']}.jpg",
                    "caption": p["caption"],
                    "width": width,
                    "height": height,
                    "seed": p["seed"],
                    "camera_motion": p["camera_motion"],
                    "vbench_image": p["vbench_image"],
                    "vbench_category": p["vbench_category"],
                    "vbench_dimensions": p["vbench_dimensions"],
                    "dev": shape in p["dev"],
                    "ci": shape in p["ci"],
                }
            )
    return entries


def contact_sheet(paths: list[Path], out: Path, columns: int = 5) -> None:
    """``paths`` side by side at a quarter of their size, to look over at once."""
    from PIL import Image

    tiles = []
    for path in paths:
        with Image.open(path) as image:
            tiles.append(image.resize((image.width // 4, image.height // 4)))
    w, h = tiles[0].size
    rows = -(-len(tiles) // columns)
    sheet = Image.new("RGB", (w * columns, h * rows), "white")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % columns) * w, (i // columns) * h))
    sheet.save(out, quality=90)


def main(argv: list[str] | None = None) -> int:
    import PIL

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--crop-dir",
        type=Path,
        required=True,
        help="VBench-I2V's crop/ folder, with 16-9/ and 1-1/ in it",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="folder to write; it goes to the Hugging Face dataset",
    )
    parser.add_argument(
        "--full-info",
        type=Path,
        help="local vbench2_i2v_full_info.json (default: download)",
    )
    parser.add_argument(
        "--bench-info", type=Path, help="local i2v-bench-info.json (default: download)"
    )
    parser.add_argument("--seed", type=int, default=SELECTION_SEED)
    args = parser.parse_args(argv)

    full_info = fetch_verified(FULL_INFO_URL, FULL_INFO_SHA256, args.full_info)
    # Checked so the provenance can name it; the crops in --crop-dir come from it.
    fetch_verified(BENCH_INFO_URL, BENCH_INFO_SHA256, args.bench_info)

    entries = videos(select_pairs(full_info, args.seed))
    for entry in entries:
        path = args.out / entry["image"]
        path.parent.mkdir(parents=True, exist_ok=True)
        prepare_first_frame(
            args.crop_dir, entry["vbench_image"], (entry["width"], entry["height"])
        ).save(path, **JPEG_OPTIONS)
        entry["image_sha256"] = sha256(path.read_bytes())

    manifest = json.dumps(entries, indent=2) + "\n"
    (args.out / "pairs").mkdir(parents=True, exist_ok=True)
    (args.out / "pairs" / "manifest.json").write_text(manifest)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(manifest)
    provenance = {
        "selection_seed": args.seed,
        "vbench_commit": VBENCH_COMMIT,
        "vbench2_i2v_full_info_sha256": FULL_INFO_SHA256,
        "i2v_bench_info_sha256": BENCH_INFO_SHA256,
        "first_frames": "VBench crop (16-9 for landscape, 1-1 for portrait), "
        "center-cropped to the output aspect, Lanczos resize, JPEG quality 100 4:4:4",
        "pillow": PIL.__version__,
        "portrait_exclude": sorted(PORTRAIT_EXCLUDE),
    }
    (args.out / "pairs" / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )

    (args.out / "review").mkdir(parents=True, exist_ok=True)
    contact_sheet(
        [
            args.out / e["image"]
            for e in entries
            if (e["width"], e["height"]) == PORTRAIT_480P
        ],
        args.out / "review" / "portrait.jpg",
    )

    shapes = Counter(f"{e['width']}x{e['height']}" for e in entries)
    print(
        f"{len(entries)} videos over {len({e['pair_id'] for e in entries})} pairs: {dict(shapes)}"
    )
    print(f"dev {sum(e['dev'] for e in entries)}, ci {sum(e['ci'] for e in entries)}")
    print(f"wrote {args.out} and {MANIFEST_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
