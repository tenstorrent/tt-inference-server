# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Structured logs for the **delivery phase**: full frame tensor in RAM → MP4 on disk → optional SHM.

This is separate from model inference (device / diffusion). Use for TTFT-style and throughput
metrics on the encode + IPC tail that matches production rank-0 / external runner behavior.
"""

from __future__ import annotations

import os
from typing import Any, Mapping


def num_frames_from_video_tensor(frames) -> int:
    """Best-effort frame count from (B,T,H,W,C) or (T,H,W,C) ndarray-like."""
    shape = getattr(frames, "shape", ())
    if len(shape) == 5:
        return int(shape[1])
    if len(shape) == 4:
        return int(shape[0])
    return 0


def log_video_delivery_phase(
    logger,
    *,
    task_id: str,
    t_tensor_ready_monotonic: float,
    export_timing: Mapping[str, Any] | None,
    t_after_export_monotonic: float,
    mp4_path: str,
    num_frames: int,
    shm_write_s: float | None = None,
) -> None:
    """
    One grep-friendly line plus a short glossary for operators.

    **Legacy field ``ttft_to_first_frame_appended_s``:** for an **incremental** encoder (imageio
    ``append_data``), this is time until the first frame is handed to the writer. For any **batch**
    path (diffusers one-shot, or FFmpeg stdin ``communicate``), there is no “first frame” moment in
    the middle of encode — we set this to **full** ``export_to_mp4`` wall time and
    ``encoder_incremental`` is false. Do not read it as LLM-style TTFT in that case.

    **``encode_after_first_frame_s`` (batch):** means **entire batch encode phase** (e.g. FFmpeg
    subprocess), not “time after frame 1 of N” in a streaming sense.

    **tensor_ready_to_mp4_return_s:** wall time from when the full tensor exists (caller’s
    ``perf_counter`` right before ``export_to_mp4``) until that call returns — your main
    “encode + write file” budget.
    """
    tensor_to_return = t_after_export_monotonic - t_tensor_ready_monotonic
    size_bytes = 0
    try:
        size_bytes = os.path.getsize(mp4_path)
    except OSError:
        pass

    ext = export_timing or {}
    export_wall = ext.get("export_wall_s")
    ttft = ext.get("ttft_to_first_frame_appended_s")
    prep = ext.get("prep_before_first_frame_s")
    tail = ext.get("encode_after_first_frame_s")
    incremental = ext.get("encoder_incremental")

    def _fmt_num(key: str, val: Any) -> str:
        if val is None:
            return f"{key}=n/a"
        if isinstance(val, (int, float)):
            return f"{key}={val:.4f}"
        return f"{key}={val}"

    mb = size_bytes / (1024 * 1024) if size_bytes else 0.0
    mb_per_s = mb / tensor_to_return if tensor_to_return > 0 else 0.0
    eff_fps = (
        num_frames / tensor_to_return
        if tensor_to_return > 0 and num_frames > 0
        else 0.0
    )

    shm_part = (
        f" shm_write_response_s={shm_write_s:.4f}" if shm_write_s is not None else ""
    )
    inc_part = f" encoder_incremental={incremental}" if incremental is not None else ""

    logger.info(
        f"[VIDEO_DELIVERY] task_id={task_id} "
        f"tensor_ready_to_mp4_return_s={tensor_to_return:.4f} "
        f"{_fmt_num('export_wall_s', export_wall)} "
        f"{_fmt_num('ttft_to_first_frame_appended_s', ttft)} "
        f"{_fmt_num('prep_before_first_frame_s', prep)} "
        f"{_fmt_num('encode_after_first_frame_s', tail)}{inc_part} "
        f"mp4_bytes={size_bytes} mp4_mib={mb:.2f} "
        f"throughput_mib_per_s={mb_per_s:.2f} effective_fps_during_encode={eff_fps:.1f}"
        f"{shm_part}"
    )
    logger.info(
        "[VIDEO_DELIVERY_GLOSSARY] "
        "tensor_ready_to_mp4_return_s: RAM tensor exists → export_to_mp4 returned; "
        "ttft_to_first_frame_appended_s: incremental encoder = time to first append_data; "
        "batch encoder (encoder_incremental=false) = full export_to_mp4 wall (name is historical); "
        "prep_before_first_frame_s: _process_frames_for_export CPU time (batch) or pre-append wait; "
        "encode_after_first_frame_s: incremental = after first append until close; "
        "batch = entire encode phase (e.g. ffmpeg); "
        "shm_write_response_s: VideoResponse written to output SHM (external runner only)."
    )
