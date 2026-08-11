# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Generate compact MP4 fixtures for every supported output combination."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from minimax_mock.media_fixtures import (
    FIXTURE_VIDEO_FPS,
    OUTPUT_DIMENSIONS,
    OUTPUT_RATIOS,
    RATIO_ASSET_KEYS,
)

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "fixtures" / "media"
_COLORS = {
    "21x9": "0x1d4ed8",
    "16x9": "0x0f766e",
    "4x3": "0x4d7c0f",
    "1x1": "0xa16207",
    "3x4": "0xb45309",
    "9x16": "0x9f1239",
}


def generate_all(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    force: bool = False,
    workers: int = 4,
) -> int:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to generate mock media fixtures")

    jobs = []
    for resolution, dimensions_by_ratio in OUTPUT_DIMENSIONS.items():
        for ratio in OUTPUT_RATIOS:
            width, height = dimensions_by_ratio[ratio]
            ratio_key = RATIO_ASSET_KEYS[ratio]
            for duration in range(4, 16):
                output_path = (
                    output_root
                    / resolution.value.lower()
                    / ratio_key
                    / f"{duration}.mp4"
                )
                if force or not output_path.is_file():
                    jobs.append(
                        (
                            ffmpeg,
                            output_path,
                            width,
                            height,
                            duration,
                            _COLORS[ratio_key],
                        )
                    )

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        list(executor.map(_generate_one, jobs))
    return len(jobs)


def _generate_one(job) -> None:
    ffmpeg, output_path, width, height, duration, color = job
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.mp4")
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        (f"color=c={color}:s={width}x{height}:r={FIXTURE_VIDEO_FPS}:d={duration}"),
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-t",
        str(duration),
        "-shortest",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "38",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "32k",
        "-movflags",
        "+faststart",
        str(temporary_path),
    ]
    try:
        subprocess.run(command, check=True)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
    )
    args = parser.parse_args()
    generated = generate_all(
        args.output_root,
        force=args.force,
        workers=args.workers,
    )
    print(f"Generated {generated} MiniMax media fixtures")


if __name__ == "__main__":
    main()
