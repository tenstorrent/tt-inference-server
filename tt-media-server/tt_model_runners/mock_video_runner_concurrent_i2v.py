# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Concurrent I2V mock runner — validates the SHM I2V contract end-to-end.

Drop-in replacement for the runner peer that exercises every link of the
I2V SHM bridge without spinning up a real DiT pipeline. Each in-flight
request goes through the same producer/consumer split as the post-Phase-1
``video_runner.py``; the only difference vs. the T2V mock
(``mock_video_runner_concurrent.py``) is the per-request validator that
runs INSIDE the worker, BEFORE the simulated inference sleep.

The validator pins the cross-process contract that the runner peer
relies on:

  1. ``image_path`` must be non-empty (T2V requests are rejected)
  2. side-file must open + parse as JSON
  3. top-level must be a non-empty list
  4. each entry must be a dict with both ``image`` and ``frame_pos`` keys
  5. each ``image`` must base64-decode to a real PIL image
     (catches the historical padding bug in
     ``ImageManager.base64_to_pil_image`` and any future regression in
     the SHM string pack/unpack of nested base64)

Any violation raises ``ValueError`` from inside the worker; the bridge
catches it and surfaces an ERROR response on SHM output. SP_RUNNER then
raises ``RuntimeError`` to the API. End-to-end, this exercises the same
fail-fast machinery the production runner peer uses.

The mock does NOT unlink the side-file — that is SP_RUNNER's job in its
``run()`` ``finally`` block. Reading without unlinking matches the real
runner peer's contract.

Env knobs are inherited from :mod:`mock_video_runner_base`. One I2V-only
addition::

    MOCK_I2V_VALIDATE_B64    (default 1)   when 0, skips base64 → PIL
                                           decoding to keep validator
                                           cheap for perf testing

Usage::

    TT_VIDEO_SHM_INPUT=tt_video_in TT_VIDEO_SHM_OUTPUT=tt_video_out \\
    MOCK_CONCURRENCY=8 MOCK_LATENCY_S=2.0 MOCK_ENCODE_S=1.0 \\
        python -m tt_model_runners.mock_video_runner_concurrent_i2v
"""

from __future__ import annotations

import json
import os
import signal

from tt_model_runners.mock_video_runner_base import handleSignal, runMockBridge

LABEL = "[I2V-MOCK]"
_VALIDATE_B64_ENV = "MOCK_I2V_VALIDATE_B64"


def _shouldValidateB64() -> bool:
    return os.environ.get(_VALIDATE_B64_ENV, "1") not in ("0", "false", "False")


def _validateEntry(idx: int, entry) -> None:
    if not isinstance(entry, dict):
        raise ValueError(
            f"image_prompts[{idx}] is not a dict (got {type(entry).__name__})"
        )
    if "image" not in entry:
        raise ValueError(f"image_prompts[{idx}] missing 'image' field")
    if "frame_pos" not in entry:
        raise ValueError(f"image_prompts[{idx}] missing 'frame_pos' field")
    if not isinstance(entry["image"], str) or not entry["image"]:
        raise ValueError(f"image_prompts[{idx}].image must be a non-empty string")
    if not isinstance(entry["frame_pos"], int):
        raise ValueError(
            f"image_prompts[{idx}].frame_pos must be int "
            f"(got {type(entry['frame_pos']).__name__})"
        )


def validateI2VRequest(req) -> None:
    """Validate the I2V side-file matches the cross-process contract.

    Raises ``ValueError`` on any violation; the bridge's worker catches
    it and writes an ERROR response on SHM output. The structural checks
    are cheap; the per-image base64 decode is gated by
    ``MOCK_I2V_VALIDATE_B64`` so callers running pure-perf tests can opt
    out of the (small) decode cost.
    """
    if not req.image_path:
        raise ValueError(
            f"I2V mock: image_path is empty (T2V request rejected) "
            f"for task {req.task_id}"
        )
    try:
        with open(req.image_path, "r") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        raise ValueError(
            f"I2V mock: side-file {req.image_path!r} unreadable: {e}"
        ) from e
    if not isinstance(data, list):
        raise ValueError(
            f"I2V mock: side-file is not a JSON list (got {type(data).__name__})"
        )
    if not data:
        raise ValueError("I2V mock: side-file contains an empty image_prompts list")

    for idx, entry in enumerate(data):
        _validateEntry(idx, entry)

    if _shouldValidateB64():
        # Lazy-import so perf-testing variants that opt out via
        # MOCK_I2V_VALIDATE_B64=0 don't pay the PIL/ImageManager import cost.
        from utils.image_manager import ImageManager

        imageManager = ImageManager()
        for idx, entry in enumerate(data):
            try:
                imageManager.base64_to_pil_image(entry["image"])
            except Exception as e:
                raise ValueError(
                    f"I2V mock: image_prompts[{idx}].image failed to decode: {e}"
                ) from e


def main() -> None:
    signal.signal(signal.SIGTERM, handleSignal)
    signal.signal(signal.SIGINT, handleSignal)
    runMockBridge(label=LABEL, perRequestValidator=validateI2VRequest)


if __name__ == "__main__":
    main()
