# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Mock video runner for testing the SHM IPC path without TTNN devices.

**Two phases (for metrics):**

1. **Bench / synthetic generation** — SHM read + ``MockVideoPipeline`` (sleeps, RNG, ``np.stack``).
   This is *not* device inference; use only for IPC and scheduling tests.

2. **Delivery phase** (production-relevant) — immediately **after** the ``[MOCK_BENCH] … full tensor in RAM``
   log: ``VideoManager.export_to_mp4``, then ``output_shm.write_response`` (SHM bridge). A summary line
   ``[MOCK_AFTER_BENCH] bench_to_handoff_s=…`` measures that slice. ``[VIDEO_DELIVERY]`` adds TTFT-style
   timing (see ``utils/video_delivery_metrics``).

When run as a standalone script, reads ``VideoRequest`` from input SHM, runs the
synthetic pipeline, encodes to mp4 (same as ``video_runner`` rank-0), and sends the path on
output SHM.

Env (optional)::

    TT_MOCK_VIDEO_TARGET_SECONDS   Spread ~uniformly across frames (±5% jitter per frame).
    TT_MOCK_OUTPUT_NUM_FRAMES      Override frame count (and divisor for target-seconds).
    TT_MOCK_OUTPUT_HEIGHT / WIDTH  Per-axis overrides.
    TT_MOCK_SIMULATE_30S_1080P     1 → 1920×1080 × 480 frames (heavy).
    TT_MOCK_FRAME_DELAY_MIN / MAX  Legacy per-frame bounds when target seconds unset.
    TT_MOCK_WARN_NON_SLEEP_PIPELINE_S  If set to a float (seconds), log WARNING when
        pipeline wall time minus accumulated mock sleep exceeds it (detects heavy
        RNG/stack cost vs. export_to_mp4).
    TT_MOCK_FLOAT32_TENSOR  If 1/true: stacked tensor is float32 [0,1) per pixel
        (like many DiT outputs). Size is capped by config.constants.MAX_VIDEO_SIZE
        (~1 GiB; WAN22 quad 720p×81 fits).

Usage::

    TT_VIDEO_SHM_INPUT=video_in TT_VIDEO_SHM_OUTPUT=video_out \\
        python -m tt_model_runners.mock_video_runner
"""

from __future__ import annotations

import os
import random
import signal
import time
from typing import Generator

import numpy as np
from config.constants import MAX_VIDEO_SIZE
from tt_model_runners.base_device_runner import BaseDeviceRunner
from utils.video_delivery_metrics import (
    log_video_delivery_phase,
    num_frames_from_video_tensor,
)

# Match TTWan22Runner.run() when mesh_device.shape == (4, 8) (quad / large mesh).
# Smaller mesh in dit_runners uses 480×832×81; mock defaults follow max supported res.
WAN22_QUAD_MESH_HEIGHT = 720
WAN22_QUAD_MESH_WIDTH = 1280
WAN22_QUAD_MESH_NUM_FRAMES = 81

DEFAULT_HEIGHT = WAN22_QUAD_MESH_HEIGHT
DEFAULT_WIDTH = WAN22_QUAD_MESH_WIDTH
DEFAULT_NUM_FRAMES = WAN22_QUAD_MESH_NUM_FRAMES
DEFAULT_CHANNELS = 3
MOCK_FRAME_DELAY_MIN = 0.05
MOCK_FRAME_DELAY_MAX = 0.15

_MOCK_PRESET_30S_1080P_HEIGHT = 1080
_MOCK_PRESET_30S_1080P_WIDTH = 1920
_MOCK_PRESET_30S_1080P_NUM_FRAMES = 480


def _parse_dim_override(env_key: str, fallback: int) -> int:
    raw = os.environ.get(env_key)
    if raw is None or not str(raw).strip():
        return fallback
    return max(1, int(raw))


def _simulate_30s_1080p_enabled() -> bool:
    v = os.environ.get("TT_MOCK_SIMULATE_30S_1080P", "").strip().lower()
    return v in ("1", "true", "yes")


def _mock_float32_tensor_enabled() -> bool:
    v = os.environ.get("TT_MOCK_FLOAT32_TENSOR", "").strip().lower()
    return v in ("1", "true", "yes")


def _mock_tensor_nbytes(
    height: int, width: int, num_frames: int, item_size: int
) -> int:
    return height * width * num_frames * DEFAULT_CHANNELS * item_size


def _mock_tensor_size_log_fragment(frames: np.ndarray) -> str:
    if frames.dtype != np.float32:
        return f"dtype={frames.dtype}"
    return (
        f"dtype={frames.dtype} nbytes={frames.nbytes} "
        f"max_video_size_cap={MAX_VIDEO_SIZE}"
    )


def _ensure_mock_tensor_within_cap(height: int, width: int, num_frames: int) -> None:
    """Reject float32 mock tensors that would exceed MAX_VIDEO_SIZE (uint8 is uncapped)."""
    if not _mock_float32_tensor_enabled():
        return
    n = _mock_tensor_nbytes(height, width, num_frames, np.dtype(np.float32).itemsize)
    if n > MAX_VIDEO_SIZE:
        raise ValueError(
            f"Mock float32 video tensor would be {n} bytes (MAX_VIDEO_SIZE={MAX_VIDEO_SIZE}); "
            "reduce TT_MOCK_OUTPUT_* / request dims or unset TT_MOCK_FLOAT32_TENSOR."
        )


def effective_output_dims(
    height: int, width: int, num_frames: int
) -> tuple[int, int, int]:
    """SHM request dimensions, optionally replaced by preset or per-axis env overrides."""
    if _simulate_30s_1080p_enabled():
        height = _MOCK_PRESET_30S_1080P_HEIGHT
        width = _MOCK_PRESET_30S_1080P_WIDTH
        num_frames = _MOCK_PRESET_30S_1080P_NUM_FRAMES
    height = _parse_dim_override("TT_MOCK_OUTPUT_HEIGHT", height)
    width = _parse_dim_override("TT_MOCK_OUTPUT_WIDTH", width)
    num_frames = _parse_dim_override("TT_MOCK_OUTPUT_NUM_FRAMES", num_frames)
    return height, width, num_frames


def _mock_frame_delay_bounds(num_frames: int) -> tuple[float, float]:
    """Per-frame sleep bounds; ``TT_MOCK_VIDEO_TARGET_SECONDS`` spreads across ``num_frames``."""
    target = os.environ.get("TT_MOCK_VIDEO_TARGET_SECONDS")
    if target is not None and str(target).strip():
        sec = float(target)
        n = max(int(num_frames), 1)
        per = sec / n
        return (per * 0.95, per * 1.05)
    lo = os.environ.get("TT_MOCK_FRAME_DELAY_MIN")
    hi = os.environ.get("TT_MOCK_FRAME_DELAY_MAX")
    if lo is not None and hi is not None and str(lo).strip() and str(hi).strip():
        return (float(lo), float(hi))
    return (MOCK_FRAME_DELAY_MIN, MOCK_FRAME_DELAY_MAX)


def _request_int(request, name: str, default: int) -> int:
    """Avoid MagicMock non-int attributes in unit tests."""
    v = getattr(request, name, None)
    return v if isinstance(v, int) else default


def _task_id_for_delivery_log(request, fallback: str) -> str:
    tid = getattr(request, "task_id", None)
    if isinstance(tid, str) and tid.strip():
        return tid
    return fallback


def _log_mock_pipeline_timing(pipeline, log) -> None:
    """Log mock pipeline wall vs sleep vs work excluding sleep (still excludes export_to_mp4)."""
    wall = getattr(pipeline, "_mock_pipeline_wall_s", None)
    if wall is None:
        return
    sleep_s = float(getattr(pipeline, "_mock_sleep_accum_s", 0.0))
    excl = getattr(pipeline, "_mock_pipeline_excluding_sleep_s", None)
    stack_s = getattr(pipeline, "_mock_stack_s", None)
    log.info(
        f"[MOCK] non_sleep_pipeline: pipeline_wall_s={wall:.4f} mock_sleep_s={sleep_s:.4f} "
        f"excluding_sleep_s={excl:.4f} stack_s={stack_s:.4f} "
        f"(RNG+list+np.stack; excludes export_to_mp4)"
    )
    warn_raw = os.environ.get("TT_MOCK_WARN_NON_SLEEP_PIPELINE_S", "").strip()
    if not warn_raw or excl is None:
        return
    try:
        limit = float(warn_raw)
    except ValueError:
        return
    if excl > limit:
        log.warning(
            f"[MOCK] excluding_sleep_s={excl:.4f}s exceeds "
            f"TT_MOCK_WARN_NON_SLEEP_PIPELINE_S={limit}"
        )


class MockVideoPipeline:
    """Fake pipeline matching the WanPipeline / MochiPipeline callable interface."""

    def generate_frames(
        self,
        height: int,
        width: int,
        num_frames: int,
        seed: int = 0,
    ) -> Generator[np.ndarray, None, None]:
        rng = np.random.RandomState(seed)
        lo, hi = _mock_frame_delay_bounds(num_frames)
        use_f32 = _mock_float32_tensor_enabled()
        for _ in range(num_frames):
            sleep_started = time.perf_counter()
            time.sleep(random.uniform(lo, hi))
            self._mock_sleep_accum_s += time.perf_counter() - sleep_started
            if use_f32:
                yield rng.rand(height, width, DEFAULT_CHANNELS).astype(np.float32)
            else:
                yield rng.randint(
                    0, 256, (height, width, DEFAULT_CHANNELS), dtype=np.uint8
                )

    def __call__(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
        num_frames: int = DEFAULT_NUM_FRAMES,
        num_inference_steps: int = 20,
        guidance_scale: float = 3.0,
        guidance_scale_2: float = 4.0,
        seed: int = 0,
        **kwargs,
    ) -> np.ndarray:
        _ensure_mock_tensor_within_cap(height, width, num_frames)
        wall_started = time.perf_counter()
        self._mock_sleep_accum_s = 0.0
        frames = list(self.generate_frames(height, width, num_frames, seed))
        stack_started = time.perf_counter()
        stacked = np.stack(frames)[np.newaxis]
        stack_s = time.perf_counter() - stack_started
        wall_s = time.perf_counter() - wall_started
        self._mock_pipeline_wall_s = wall_s
        self._mock_stack_s = stack_s
        self._mock_pipeline_excluding_sleep_s = wall_s - self._mock_sleep_accum_s
        return stacked


class MockVideoRunner(BaseDeviceRunner):
    """Mock video runner that simulates Wan/Mochi inference without devices."""

    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.pipeline: MockVideoPipeline | None = None
        self.logger.info(f"MockVideoRunner initialized for device {self.device_id}")

    def set_device(self):
        self.logger.info("MockVideoRunner set_device (no-op, no TTNN)")
        return {}

    def close_device(self):
        self.logger.info("MockVideoRunner close_device (no-op)")
        return True

    async def warmup(self) -> bool:
        self.logger.info(f"MockVideoRunner warmup for device {self.device_id}")
        self.pipeline = MockVideoPipeline()
        self.logger.info(
            f"MockVideoRunner warmup completed for device {self.device_id}"
        )
        return True

    def run(self, requests):
        request = requests[0]
        self.logger.info(
            f"MockVideoRunner running inference for prompt: {request.prompt!r}"
        )
        h, w, nf = effective_output_dims(
            _request_int(request, "height", DEFAULT_HEIGHT),
            _request_int(request, "width", DEFAULT_WIDTH),
            _request_int(request, "num_frames", DEFAULT_NUM_FRAMES),
        )
        self.logger.info(
            f"[MOCK_BENCH] device={self.device_id} synthetic generation starting "
            "(not device inference)"
        )
        frames = self.pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            height=h,
            width=w,
            num_frames=nf,
            num_inference_steps=request.num_inference_steps,
            seed=int(request.seed or 0),
        )
        _log_mock_pipeline_timing(self.pipeline, self.logger)
        tb = _mock_tensor_size_log_fragment(frames)
        self.logger.info(
            f"[MOCK_BENCH] device={self.device_id} full tensor in RAM shape={frames.shape} {tb}; "
            f"delivery phase (encode only — no SHM in in-proc runner)"
        )
        t_after_mock_bench = time.perf_counter()
        from utils.video_manager import VideoManager

        export_timing = {}
        mp4_path = VideoManager().export_to_mp4(frames, timing_out=export_timing)
        t_after_export = time.perf_counter()
        self.logger.info(f"MockVideoRunner encoded mp4: {mp4_path}")
        log_video_delivery_phase(
            self.logger,
            task_id=_task_id_for_delivery_log(request, self.device_id),
            t_tensor_ready_monotonic=t_after_mock_bench,
            export_timing=export_timing,
            t_after_export_monotonic=t_after_export,
            mp4_path=mp4_path,
            num_frames=num_frames_from_video_tensor(frames),
            shm_write_s=None,
        )
        bench_to_handoff_s = time.perf_counter() - t_after_mock_bench
        self.logger.info(
            f"[MOCK_AFTER_BENCH] task_id={_task_id_for_delivery_log(request, self.device_id)} "
            f"bench_to_handoff_s={bench_to_handoff_s:.4f} "
            f"(in-proc: after MOCK_BENCH full-tensor line → VIDEO_DELIVERY logged; no SHM)"
        )
        return [mp4_path]


# ---------------------------------------------------------------------------
# Standalone SHM bridge
# ---------------------------------------------------------------------------

_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    _shutdown = True


def _run_shm_bridge() -> None:
    import os

    from ipc.video_shm import (
        VideoResponse,
        VideoShm,
        VideoStatus,
        cleanup_orphaned_video_files,
    )
    from utils.logger import TTLogger
    from utils.video_manager import VideoManager

    logger = TTLogger()

    input_name = os.environ.get("TT_VIDEO_SHM_INPUT", "tt_video_in")
    output_name = os.environ.get("TT_VIDEO_SHM_OUTPUT", "tt_video_out")

    def is_shutdown() -> bool:
        return _shutdown

    input_shm = VideoShm(input_name, mode="input", is_shutdown=is_shutdown)
    output_shm = VideoShm(output_name, mode="output", is_shutdown=is_shutdown)
    input_shm.open(create=True)
    output_shm.open(create=True)

    pipeline = MockVideoPipeline()
    logger.info("Mock video SHM runner ready, waiting for requests...")

    try:
        while not _shutdown:
            req = input_shm.read_request()
            if req is None:
                break

            logger.info(
                f"Received request task_id={req.task_id} "
                f"prompt={req.prompt!r} frames={req.num_frames}"
            )

            try:
                h, w, nf = effective_output_dims(req.height, req.width, req.num_frames)
                if (h, w, nf) != (req.height, req.width, req.num_frames):
                    logger.info(
                        f"[MOCK] Effective output dims (env overrides): "
                        f"{nf}×{h}×{w} (was request {req.num_frames}×{req.height}×{req.width})"
                    )
                logger.info(
                    f"[MOCK_BENCH] task_id={req.task_id} synthetic generation starting "
                    f"(sleep/RNG/stack only — not device inference)"
                )
                frames = pipeline(
                    prompt=req.prompt,
                    negative_prompt=req.negative_prompt,
                    height=h,
                    width=w,
                    num_frames=nf,
                    num_inference_steps=req.num_inference_steps,
                    guidance_scale=req.guidance_scale,
                    guidance_scale_2=req.guidance_scale_2,
                    seed=req.seed,
                )
                _log_mock_pipeline_timing(pipeline, logger)
                tb = _mock_tensor_size_log_fragment(frames)
                logger.info(
                    f"[MOCK_BENCH] task_id={req.task_id} full tensor in RAM "
                    f"shape={frames.shape} {tb}; "
                    f"delivery phase next (encode+SHM — same tail as prod rank-0)"
                )
                t_after_mock_bench = time.perf_counter()

                export_timing = {}
                mp4_path = VideoManager().export_to_mp4(
                    frames, timing_out=export_timing
                )
                t_after_export = time.perf_counter()
                logger.info(
                    f"[MOCK] Encoded mp4 at {mp4_path} "
                    f"({os.path.getsize(mp4_path):,} bytes)"
                )

                t_shm0 = time.perf_counter()
                output_shm.write_response(
                    VideoResponse(
                        task_id=req.task_id,
                        status=VideoStatus.SUCCESS,
                        file_path=mp4_path,
                        error_message="",
                    )
                )
                shm_write_s = time.perf_counter() - t_shm0
                bench_to_handoff_s = time.perf_counter() - t_after_mock_bench
                log_video_delivery_phase(
                    logger,
                    task_id=req.task_id,
                    t_tensor_ready_monotonic=t_after_mock_bench,
                    export_timing=export_timing,
                    t_after_export_monotonic=t_after_export,
                    mp4_path=mp4_path,
                    num_frames=num_frames_from_video_tensor(frames),
                    shm_write_s=shm_write_s,
                )
                logger.info(
                    f"[MOCK_AFTER_BENCH] task_id={req.task_id} bench_to_handoff_s={bench_to_handoff_s:.4f} "
                    f"(wall from after MOCK_BENCH full-tensor line → SHM success response written; "
                    f"then server + client still follow)"
                )
            except Exception as e:
                logger.error(f"Error generating frames for {req.task_id}: {e}")
                output_shm.write_response(
                    VideoResponse(
                        task_id=req.task_id,
                        status=VideoStatus.ERROR,
                        file_path="",
                        error_message=str(e)[:256],
                    )
                )
                continue

            logger.info(f"Request {req.task_id} completed")
    finally:
        input_shm.close()
        output_shm.close()
        removed = cleanup_orphaned_video_files()
        if removed:
            logger.info(f"Cleaned up {removed} orphaned video file(s)")
        logger.info("Mock video SHM runner shut down")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    _run_shm_bridge()
