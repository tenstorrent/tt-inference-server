# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""MiniMax-H3 adapter (TTMiniMaxH3*Runner): t2va, fl2va (first_image /
last_image), ref2va (references) and the FastH3 / LightX2V variants, which
differ only in the create() options the C++ model sends."""

from __future__ import annotations

import os
import tempfile

import numpy as np

from . import register
from ._common import Adapter, decode_base64, decode_image


def _write_temp(b64: str, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(fd, decode_base64(b64))
    finally:
        os.close(fd)
    return path


def _references(refs: dict) -> list:
    """MiniMaxH3Reference list, packed images, then videos, then audios."""
    from models.tt_dit.pipelines.minimax_h3.packing_ref2va import (
        MiniMaxH3Reference,
        decode_reference_audio,
        reference_from_video_file,
    )

    out = [MiniMaxH3Reference(image=decode_image(b)) for b in refs.get("images", [])]
    for b64 in refs.get("videos", []):
        path = _write_temp(b64, ".mp4")
        try:
            out.append(reference_from_video_file(path))
        finally:
            os.unlink(path)
    for b64 in refs.get("audios", []):
        path = _write_temp(b64, ".wav")
        try:
            waveform, sample_rate = decode_reference_audio(path)
        finally:
            os.unlink(path)
        out.append(MiniMaxH3Reference(audio=waveform, sample_rate=sample_rate))
    return out


def _av_result(output) -> dict:
    if getattr(output, "video_format", "rgb_float") == "yuv420":
        # Planar (F, H*3//2, W) uint8 straight from the device.
        frames, pixel_format = np.ascontiguousarray(output.video), "yuv420p"
    else:
        # (1, 3, F, H, W) in [0, 1] -> (F, H, W, 3).
        video = output.video[0].permute(1, 2, 3, 0).contiguous().numpy()
        frames, pixel_format = np.ascontiguousarray(video, dtype=np.float32), "rgb24"
    return {
        "kind": "av",
        "frames": frames,
        "audio": np.ascontiguousarray(output.audio[0].numpy(), dtype=np.float32),
        "sampling_rate": int(output.sampling_rate),
        "fps": float(output.fps),
        "pixel_format": pixel_format,
    }


@register("minimax")
def minimax(mesh, o: dict) -> Adapter:
    if o["pipeline"] == "turbo":
        from models.tt_dit.pipelines.minimax_h3.pipeline_minimax_h3_turbo import (
            MiniMaxH3TurboPipeline as Pipeline,
        )
    else:
        from models.tt_dit.pipelines.minimax_h3.pipeline_minimax_h3 import (
            MiniMaxH3Pipeline as Pipeline,
        )
    extra = {}
    for key in ("lora_path", "video_shift", "audio_shift"):
        if key in o:
            extra[key] = o[key]
    if "vsa_sparsity" in o:
        from models.tt_dit.models.transformers.minimax_h3.vsa_stages_minimax_h3 import (
            MiniMaxH3VSAConfig,
        )

        extra["vsa_config"] = MiniMaxH3VSAConfig(sparsity=o["vsa_sparsity"])
    # No topology: the pipeline resolves it from its per-mesh preset.
    pipeline = Pipeline.create_pipeline(
        mesh_device=mesh,
        weights_dir=o["weights_dir"],
        task=o["task"],
        dit_fsdp=o["dit_fsdp"],
        trace_denoise=o["trace_denoise"],
        bucket_denoise=o["bucket_denoise"],
        **extra,
    )

    def run(args: dict) -> dict:
        kwargs = dict(args)
        prompt = kwargs.pop("prompt")
        kwargs["aspect_ratio"] = tuple(kwargs["aspect_ratio"])
        first, last = kwargs.pop("first_image", None), kwargs.pop("last_image", None)
        if first or last:
            kwargs["image"] = decode_image(first) if first else None
            kwargs["last_image"] = decode_image(last) if last else None
        refs = kwargs.pop("references", None)
        if refs is not None:
            kwargs["references"] = _references(refs)
        return _av_result(pipeline(prompt, **kwargs))

    return Adapter(run, pipeline)
