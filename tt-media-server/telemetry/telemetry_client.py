# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import time
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from queue import Queue
from threading import Thread
from typing import Optional

from config.settings import get_settings
from prometheus_client import Counter, Gauge, Histogram
from utils.logger import TTLogger


class TelemetryEvent(Enum):
    DEVICE_WARMUP = "device_warmup"
    MODEL_INFERENCE = "model_inference"
    PRE_PROCESSING = "pre_processing"
    POST_PROCESSING = "post_processing"
    TOTAL_PROCESSING = "total_processing"
    BASE_TOTAL_PROCESSING = "base_total_processing"
    BASE_SINGLE_PROCESSING = "base_single_processing"
    DOWNLOAD_RESULT = "download_result"
    VIDEO_GENERATION = "video_generation"


# top metric for total requests from inherited service
request_counter = Counter(
    "tt_media_server_requests_total", "Total number of requests", ["model_type"]
)

request_duration = Histogram(
    "tt_media_server_request_duration_seconds",
    "Request duration in seconds",
    ["model_type"],
)

# Service request counter from process method in BaseService
base_request_counter = Counter(
    "tt_media_server_requests_base_counter",
    "Total number of base requests",
    ["model_type"],
)

base_request_duration = Histogram(
    "tt_media_server_requests_base_duration_seconds",
    "Base request duration in seconds",
    ["model_type"],
)

# Total time for base service method process_request
total_base_request_counter = Counter(
    "tt_media_server_requests_base_total",
    "Total number of base requests",
    ["model_type"],
)

total_base_request_duration = Histogram(
    "tt_media_server_requests_base_duration_seconds_total",
    "Base request duration in seconds",
    ["model_type"],
)

# Subprocessing metrics
pre_processing_duration = Histogram(
    "tt_media_server_pre_processing_duration_seconds",
    "Pre processing duration in seconds",
    ["model_type", "preprocessing_enabled"],
)

post_processing_duration = Histogram(
    "tt_media_server_post_processing_duration_seconds",
    "Post processing processing duration in seconds",
    ["model_type", "post_processing_enabled"],
)

# Model inference metrics
model_inference_duration = Histogram(
    "tt_media_server_model_inference_duration_seconds",
    "Model inference duration in seconds",
    ["model_type", "device_id"],
)

device_warmup_duration = Histogram(
    "tt_media_server_device_warmup_duration_seconds",
    "Model inference duration in seconds",
    ["model_type", "device_id"],
)

model_inference_counter = Counter(
    "tt_media_server_model_inference_total",
    "Total number of model inferences",
    ["model_type", "device_id", "status"],
)

device_warmup_counter = Counter(
    "tt_media_server_device_warmup_total",
    "Total number of device warmup operations",
    ["model_type", "device_id", "status"],
)

model_load_counter = Counter(
    "tt_media_server_model_load_total",
    "Total number of model load operations",
    ["model_type", "device_id", "status"],
)

download_result_duration = Histogram(
    "tt_media_server_download_result_duration_seconds",
    "Download result duration in seconds",
    ["model_type"],
)

download_result_counter = Counter(
    "tt_media_server_download_result_total",
    "Total number of download result operations",
    ["model_type", "status"],
)

# Live concurrency gauge: jobs that have left the HTTP layer and are
# actively being processed by the model service (post-queue, pre-response).
# multiprocess_mode='livesum' so the parent's /metrics scrape sums the
# live values across all uvicorn worker PIDs.
jobs_in_progress = Gauge(
    "tt_media_server_jobs_in_progress",
    "Generation jobs currently being processed by the model service",
    ["model_type"],
    multiprocess_mode="livesum",
)

# --- Canary monitor metrics (OUTPUT ONLY) ------------------------------------
# Emitted by CanaryMonitor (health_monitoring/). Prometheus is never the
# trigger for a probe; these only export the monitor's observations.
canary_state = Gauge(
    "tt_canary_state",
    "Current canary-monitor state (1 for the active state, 0 otherwise)",
    ["model_type", "state"],
    multiprocess_mode="livesum",
)

canary_failures_total = Counter(
    "tt_canary_failures_total",
    "Total number of canary-monitor probe misses (timeout / error / falsy)",
    ["model_type", "depth"],
)

canary_probe_latency_seconds = Histogram(
    "tt_canary_probe_latency_seconds",
    "Round-trip latency of a canary-monitor probe in seconds",
    ["model_type", "depth"],
)

canary_last_success_timestamp = Gauge(
    "tt_canary_last_success_timestamp",
    "Unix timestamp of the last successful canary-monitor probe",
    ["model_type"],
    multiprocess_mode="max",
)


# --- Video generation metrics -----------------------------------------------
# Video generation is orders of magnitude slower than image/LLM work: one
# request runs for tens of seconds to tens of minutes. prometheus_client's
# default histogram buckets top out at 10s, so every video observation would
# land in +Inf and quantiles would be meaningless — hence the explicit buckets
# below. The generic per-request metrics above stay as-is; these add the
# content-aware dimensions (frames, resolution, denoise steps) that make a
# video number comparable across requests and deployments.

# "t2v" / "i2v" — bounded by design, see VideoService._classify_request.
VIDEO_REQUEST_TYPE_T2V = "t2v"
VIDEO_REQUEST_TYPE_I2V = "i2v"
# Used for the resolution label when the mp4 could not be probed (no PyAV,
# unreadable file). Keeps the label set bounded instead of dropping the series.
VIDEO_UNKNOWN_LABEL = "unknown"

# Generation outcomes. "cancelled" is deliberately distinct from "failure":
# a client calling POST /generations/{id}/cancel is not a pipeline fault, and
# folding the two together would make /cancel traffic read as unreliability.
VIDEO_STATUS_SUCCESS = "success"
VIDEO_STATUS_FAILURE = "failure"
VIDEO_STATUS_CANCELLED = "cancelled"

VIDEO_GENERATION_DURATION_BUCKETS = (
    1.0,
    2.5,
    5.0,
    10.0,
    20.0,
    30.0,
    60.0,
    90.0,
    120.0,
    180.0,
    300.0,
    450.0,
    600.0,
    900.0,
    1800.0,
    3600.0,
    float("inf"),
)
# Per-denoise-step latency: sub-second on distilled/mock pipelines, minutes on
# a cold 4x32 mesh.
VIDEO_STEP_DURATION_BUCKETS = (
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    30.0,
    60.0,
    120.0,
    300.0,
    float("inf"),
)
# Generated frames per wall-clock second. Well below 1 on real hardware today.
VIDEO_FRAMES_PER_SECOND_BUCKETS = (
    0.01,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    25.0,
    50.0,
    100.0,
    float("inf"),
)
# Wall seconds spent per second of playable video (1.0 == realtime).
VIDEO_REALTIME_FACTOR_BUCKETS = (
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    25.0,
    50.0,
    100.0,
    250.0,
    500.0,
    1000.0,
    2500.0,
    float("inf"),
)
# Resolution-independent throughput, so 480p and 720p runs are comparable.
VIDEO_PIXELS_PER_SECOND_BUCKETS = (
    1e4,
    5e4,
    1e5,
    5e5,
    1e6,
    5e6,
    1e7,
    5e7,
    1e8,
    5e8,
    1e9,
    float("inf"),
)
VIDEO_OUTPUT_SIZE_BUCKETS = (
    64 * 1024,
    256 * 1024,
    1024 * 1024,
    4 * 1024 * 1024,
    16 * 1024 * 1024,
    64 * 1024 * 1024,
    256 * 1024 * 1024,
    float("inf"),
)
# Client-facing bounds are MIN/MAX_VIDEO_INFERENCE_STEPS (4..50).
VIDEO_INFERENCE_STEPS_BUCKETS = (
    4.0,
    8.0,
    12.0,
    16.0,
    20.0,
    25.0,
    30.0,
    40.0,
    50.0,
    float("inf"),
)
# Frame counts that the shipped pipelines actually emit (Wan2.2 = 81,
# LTX-2.3 = 145), plus room either side.
VIDEO_FRAME_COUNT_BUCKETS = (
    16.0,
    33.0,
    49.0,
    81.0,
    121.0,
    145.0,
    241.0,
    481.0,
    float("inf"),
)
VIDEO_CONDITIONING_IMAGE_BUCKETS = (1.0, 2.0, 3.0, 4.0, 8.0, float("inf"))
VIDEO_ENCODE_DURATION_BUCKETS = (
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    20.0,
    30.0,
    60.0,
    120.0,
    float("inf"),
)

# Outcome accounting. Split from tt_media_server_requests_total so a video
# failure is distinguishable from any other model_type's failure.
video_generation_counter = Counter(
    "tt_media_server_video_generation_total",
    "Total number of finished video generations",
    ["model_type", "request_type", "status"],
)

# Carries ``status`` so latency panels can filter to status="success" while
# time-to-failure (usually a timeout) stays observable on its own series.
video_generation_duration = Histogram(
    "tt_media_server_video_generation_duration_seconds",
    "Wall-clock duration of a full video generation (queue + inference + encode)",
    ["model_type", "request_type", "resolution", "status"],
    buckets=VIDEO_GENERATION_DURATION_BUCKETS,
)

# Lifetime totals: the numerators for the rate() panels on the dashboard.
video_frames_counter = Counter(
    "tt_media_server_video_frames_generated_total",
    "Total number of video frames generated",
    ["model_type", "request_type"],
)

# Steps actually carried out — incremented on success only, so it stays a
# usable denominator for steps/sec. The requested count (including failures)
# is tt_media_server_video_requested_inference_steps.
video_denoise_steps_counter = Counter(
    "tt_media_server_video_denoise_steps_total",
    "Total number of diffusion denoise steps executed across generations",
    ["model_type", "request_type"],
)

video_content_seconds_counter = Counter(
    "tt_media_server_video_content_seconds_total",
    "Total playable seconds of video produced",
    ["model_type", "request_type"],
)

video_output_bytes_counter = Counter(
    "tt_media_server_video_output_bytes_total",
    "Total bytes of encoded mp4 produced",
    ["model_type", "request_type"],
)

# Throughput distributions. Recorded per generation, so p50/p95 answer
# "how fast is one video" rather than "how busy is the server".
video_frames_per_second = Histogram(
    "tt_media_server_video_frames_per_second",
    "Generated frames per wall-clock second of a video generation",
    ["model_type", "request_type", "resolution"],
    buckets=VIDEO_FRAMES_PER_SECOND_BUCKETS,
)

video_step_duration = Histogram(
    "tt_media_server_video_step_duration_seconds",
    "Mean wall-clock seconds per denoise step of a video generation",
    ["model_type", "request_type", "resolution"],
    buckets=VIDEO_STEP_DURATION_BUCKETS,
)

video_realtime_factor = Histogram(
    "tt_media_server_video_realtime_factor",
    "Wall-clock seconds spent per second of playable video (1.0 = realtime)",
    ["model_type", "request_type", "resolution"],
    buckets=VIDEO_REALTIME_FACTOR_BUCKETS,
)

video_pixels_per_second = Histogram(
    "tt_media_server_video_pixels_per_second",
    "Generated pixels (width x height x frames) per wall-clock second",
    ["model_type", "request_type", "resolution"],
    buckets=VIDEO_PIXELS_PER_SECOND_BUCKETS,
)

# Shape / payload distributions of what clients ask for and what they get back.
video_output_size_bytes = Histogram(
    "tt_media_server_video_output_size_bytes",
    "Size of the encoded mp4 returned to the client, in bytes",
    ["model_type", "request_type"],
    buckets=VIDEO_OUTPUT_SIZE_BUCKETS,
)

video_requested_inference_steps = Histogram(
    "tt_media_server_video_requested_inference_steps",
    "Distribution of num_inference_steps requested per video generation",
    ["model_type", "request_type"],
    buckets=VIDEO_INFERENCE_STEPS_BUCKETS,
)

video_output_frames = Histogram(
    "tt_media_server_video_output_frames",
    "Distribution of frame counts of the videos produced",
    ["model_type", "request_type"],
    buckets=VIDEO_FRAME_COUNT_BUCKETS,
)

video_conditioning_images = Histogram(
    "tt_media_server_video_conditioning_images",
    "Number of I2V conditioning images supplied per generation",
    ["model_type"],
    buckets=VIDEO_CONDITIONING_IMAGE_BUCKETS,
)

# Live concurrency, split by request_type. Narrower than
# tt_media_server_jobs_in_progress (which is per model_type only) so a stuck
# I2V request is visible while T2V traffic keeps flowing.
video_generations_in_progress = Gauge(
    "tt_media_server_video_generations_in_progress",
    "Video generations currently being processed",
    ["model_type", "request_type"],
    multiprocess_mode="livesum",
)

# Freshness signal: alert on time() - this > threshold to catch a pipeline
# that stopped producing without erroring.
video_last_generation_timestamp = Gauge(
    "tt_media_server_video_last_generation_timestamp",
    "Unix timestamp of the last finished video generation",
    ["model_type", "request_type", "status"],
    multiprocess_mode="max",
)

# MP4 encode (ffmpeg) cost, recorded where the encode actually runs. Emitted
# by in-process runners and the CPU postprocessing workers; a multihost
# SP_RUNNER deployment encodes inside the external peer process, which does
# not serve /metrics, so these series are absent there by design.
video_encode_duration = Histogram(
    "tt_media_server_video_encode_duration_seconds",
    "ffmpeg mp4 encode duration in seconds",
    ["model_type", "resolution", "status"],
    buckets=VIDEO_ENCODE_DURATION_BUCKETS,
)

video_encode_frames_per_second = Histogram(
    "tt_media_server_video_encode_frames_per_second",
    "Frames encoded to mp4 per wall-clock second",
    ["model_type", "resolution"],
    buckets=VIDEO_FRAMES_PER_SECOND_BUCKETS,
)

video_encode_counter = Counter(
    "tt_media_server_video_encode_total",
    "Total number of mp4 encode operations",
    ["model_type", "status"],
)


@dataclass(frozen=True)
class VideoGenerationStats:
    """Content-aware facts about one finished video generation.

    Everything except ``request_type``, ``duration_seconds`` and ``status`` is
    optional: the SP_RUNNER path only learns frame count and resolution by
    probing the produced mp4, which can fail (no PyAV, truncated file). Unknown
    fields are skipped rather than guessed, so a bogus 0 never lands in a
    throughput histogram.

    ``status`` is one of ``VIDEO_STATUS_SUCCESS`` / ``_FAILURE`` / ``_CANCELLED``.
    """

    request_type: str
    duration_seconds: float
    status: str = VIDEO_STATUS_SUCCESS
    num_inference_steps: Optional[int] = None
    conditioning_images: int = 0
    width: Optional[int] = None
    height: Optional[int] = None
    num_frames: Optional[int] = None
    content_seconds: Optional[float] = None
    output_bytes: Optional[int] = None

    @property
    def resolution(self) -> str:
        """``"<width>x<height>"``, or ``"unknown"`` when the probe failed."""
        if self.width and self.height:
            return f"{self.width}x{self.height}"
        return VIDEO_UNKNOWN_LABEL


class TelemetryClient:
    """Telemetry client to record events"""

    def __init__(self):
        self.logger = TTLogger()
        self.settings = get_settings()

        # Only start background processing if telemetry is enabled
        if self.settings.enable_telemetry:
            self.queue = Queue()
            self.worker_thread = Thread(target=self._process_telemetry, daemon=True)
            self.worker_thread.start()
            self.logger.info("Telemetry client started")
        else:
            self.queue = None
            self.worker_thread = None
            self.logger.info("Telemetry client disabled")

    def record_telemetry_event_async(
        self, event_name, device_id=None, duration=None, status=True
    ):
        """Non-blocking telemetry recording"""
        if not self.settings.enable_telemetry or self.queue is None:
            return  # Do nothing if telemetry is disabled

        self.queue.put(
            {
                "event_name": event_name,
                "device_id": device_id,
                "duration": duration,
                "status": status,
            }
        )

    def record_video_generation_async(self, stats: "VideoGenerationStats"):
        """Non-blocking recording of one finished video generation.

        Rides the same background queue as ``record_telemetry_event_async`` so a
        slow scrape or a label explosion can never add latency to a request. The
        richer payload does not fit the (event, device_id, duration, status)
        shape, so it travels as its own queue entry keyed by ``video_stats``.
        """
        if not self.settings.enable_telemetry or self.queue is None:
            return  # Do nothing if telemetry is disabled

        self.queue.put(
            {
                "event_name": TelemetryEvent.VIDEO_GENERATION,
                "video_stats": stats,
            }
        )

    def _process_telemetry(self):
        """Background worker to process telemetry"""
        while True:
            try:
                event = self.queue.get(timeout=1)
                if event:
                    # Process telemetry in background
                    video_stats = event.get("video_stats")
                    if video_stats is not None:
                        self.record_video_generation(video_stats)
                    else:
                        self.record_telemetry_event(
                            event["event_name"],
                            event["device_id"],
                            event["duration"],
                            event["status"],
                        )
                    self.queue.task_done()
            except Exception:
                continue  # Keep worker alive

    def record_telemetry_event(
        self,
        event_name: TelemetryEvent,
        device_id: str = None,
        duration: float = None,
        status: bool = True,
    ):
        status_str = "success" if status else "failure"

        if event_name == TelemetryEvent.PRE_PROCESSING:
            self._record_pre_processing(duration, preprocessing_enabled=True)
        elif event_name == TelemetryEvent.POST_PROCESSING:
            self._record_post_processing(duration, post_processing_enabled=True)
        elif event_name == TelemetryEvent.MODEL_INFERENCE:
            self._record_model_inference(device_id, duration, status=status_str)
        elif event_name == TelemetryEvent.DEVICE_WARMUP:
            self._record_device_warmup(device_id, duration, status=status_str)
        elif event_name == TelemetryEvent.TOTAL_PROCESSING:
            self._record_request_duration(duration)
        elif event_name == TelemetryEvent.BASE_TOTAL_PROCESSING:
            self._record_base_total_request_duration(duration)
        elif event_name == TelemetryEvent.BASE_SINGLE_PROCESSING:
            self._record_single_base_request_duration(duration)
        elif event_name == TelemetryEvent.DOWNLOAD_RESULT:
            self._record_download_result(duration, status=status_str)
        elif event_name == TelemetryEvent.VIDEO_GENERATION:
            # Video generations carry a VideoGenerationStats payload and are
            # dispatched straight to record_video_generation by
            # _process_telemetry. Landing here means a caller used the generic
            # entry point, where the payload would be silently dropped.
            self.logger.warning(
                "VIDEO_GENERATION needs VideoGenerationStats; "
                "call record_video_generation_async instead"
            )
        else:
            self.logger.warning(f"Unknown telemetry event: {event_name}")

    # Utility functions for recording metrics
    def _record_pre_processing(self, duration: float, preprocessing_enabled: bool):
        pre_processing_duration.labels(
            model_type=self.settings.model_runner,
            preprocessing_enabled=str(preprocessing_enabled),
        ).observe(duration)

    def _record_post_processing(self, duration: float, post_processing_enabled: bool):
        post_processing_duration.labels(
            model_type=self.settings.model_runner,
            post_processing_enabled=str(post_processing_enabled),
        ).observe(duration)

    def _record_model_inference(
        self, device_id: str, duration: float, status: str = "success"
    ):
        model_inference_duration.labels(
            model_type=self.settings.model_runner, device_id=device_id or "unknown"
        ).observe(duration)

        model_inference_counter.labels(
            model_type=self.settings.model_runner,
            device_id=device_id or "unknown",
            status=status,
        ).inc()

    def _record_device_warmup(
        self, device_id: str, duration: float, status: str = "success"
    ):
        device_warmup_duration.labels(
            model_type=self.settings.model_runner, device_id=device_id or "unknown"
        ).observe(duration)

        # Create a separate counter for device warmup
        device_warmup_counter.labels(
            model_type=self.settings.model_runner,
            device_id=device_id or "unknown",
            status=status,
        ).inc()

    def _record_request_duration(self, duration: float):
        request_duration.labels(model_type=self.settings.model_runner).observe(duration)

        request_counter.labels(model_type=self.settings.model_runner).inc()

    def _record_base_total_request_duration(self, duration: float):
        total_base_request_duration.labels(
            model_type=self.settings.model_runner
        ).observe(duration)

        total_base_request_counter.labels(model_type=self.settings.model_runner).inc()

    def _record_single_base_request_duration(self, duration: float):
        base_request_duration.labels(model_type=self.settings.model_runner).observe(
            duration
        )

        base_request_counter.labels(model_type=self.settings.model_runner).inc()

    def record_video_generation(self, stats: "VideoGenerationStats") -> None:
        """Fan one generation out across the video metric family.

        Derived throughputs are recorded only when both operands are known and
        positive: a failed generation has no frames, and the SP_RUNNER path can
        fail to probe the mp4. Observing 0 there would drag every quantile down
        and make a broken probe look like a slow pipeline.
        """
        model_type = self.settings.model_runner
        request_type = stats.request_type
        resolution = stats.resolution
        status_str = stats.status
        succeeded = stats.status == VIDEO_STATUS_SUCCESS
        duration = stats.duration_seconds

        video_generation_counter.labels(
            model_type=model_type, request_type=request_type, status=status_str
        ).inc()

        video_last_generation_timestamp.labels(
            model_type=model_type, request_type=request_type, status=status_str
        ).set(time.time())

        if duration is not None and duration > 0:
            video_generation_duration.labels(
                model_type=model_type,
                request_type=request_type,
                resolution=resolution,
                status=status_str,
            ).observe(duration)

        if stats.num_inference_steps:
            video_requested_inference_steps.labels(
                model_type=model_type, request_type=request_type
            ).observe(stats.num_inference_steps)
            # Executed steps and per-step latency are success-only: a request
            # that timed out (or was cancelled) after 6s of a 300s budget did
            # not run 16 steps in 0.4s each, and letting that land here would
            # drag every step-time quantile toward zero.
            if succeeded:
                video_denoise_steps_counter.labels(
                    model_type=model_type, request_type=request_type
                ).inc(stats.num_inference_steps)
                if duration and duration > 0:
                    video_step_duration.labels(
                        model_type=model_type,
                        request_type=request_type,
                        resolution=resolution,
                    ).observe(duration / stats.num_inference_steps)

        if stats.conditioning_images:
            video_conditioning_images.labels(model_type=model_type).observe(
                stats.conditioning_images
            )

        # Everything below describes the produced mp4, so it is skipped
        # wholesale on the failure path (nothing was produced) and on an
        # unsuccessful probe.
        if stats.num_frames:
            video_frames_counter.labels(
                model_type=model_type, request_type=request_type
            ).inc(stats.num_frames)
            video_output_frames.labels(
                model_type=model_type, request_type=request_type
            ).observe(stats.num_frames)
            if duration and duration > 0:
                video_frames_per_second.labels(
                    model_type=model_type,
                    request_type=request_type,
                    resolution=resolution,
                ).observe(stats.num_frames / duration)
                if stats.width and stats.height:
                    video_pixels_per_second.labels(
                        model_type=model_type,
                        request_type=request_type,
                        resolution=resolution,
                    ).observe(stats.width * stats.height * stats.num_frames / duration)

        if stats.content_seconds:
            video_content_seconds_counter.labels(
                model_type=model_type, request_type=request_type
            ).inc(stats.content_seconds)
            if duration and duration > 0:
                video_realtime_factor.labels(
                    model_type=model_type,
                    request_type=request_type,
                    resolution=resolution,
                ).observe(duration / stats.content_seconds)

        if stats.output_bytes:
            video_output_bytes_counter.labels(
                model_type=model_type, request_type=request_type
            ).inc(stats.output_bytes)
            video_output_size_bytes.labels(
                model_type=model_type, request_type=request_type
            ).observe(stats.output_bytes)

    def record_video_encode(
        self,
        duration: float,
        num_frames: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        status: bool = True,
    ) -> None:
        """Record one ffmpeg mp4 encode.

        Called synchronously from the encode path (``VideoManager``) rather than
        through the queue: the encode already runs off the event loop, in a CPU
        worker or runner process, so three cheap observe() calls there cost
        nothing and avoid needing a telemetry worker thread in that process.
        """
        model_type = self.settings.model_runner
        resolution = f"{width}x{height}" if width and height else VIDEO_UNKNOWN_LABEL
        status_str = "success" if status else "failure"

        video_encode_counter.labels(model_type=model_type, status=status_str).inc()
        video_encode_duration.labels(
            model_type=model_type, resolution=resolution, status=status_str
        ).observe(duration)

        if num_frames and duration > 0:
            video_encode_frames_per_second.labels(
                model_type=model_type, resolution=resolution
            ).observe(num_frames / duration)

    def _record_download_result(self, duration: float, status: str = "success"):
        download_result_duration.labels(model_type=self.settings.model_runner).observe(
            duration
        )

        download_result_counter.labels(
            model_type=self.settings.model_runner,
            status=status,
        ).inc()


@lru_cache(maxsize=1)
def get_telemetry_client() -> TelemetryClient:
    return TelemetryClient()
