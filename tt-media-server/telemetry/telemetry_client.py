# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from enum import Enum
from functools import lru_cache
from queue import Queue
from threading import Thread

from config.settings import get_settings
from prometheus_client import Counter, Histogram
from utils.logger import TTLogger


class TelemetryEvent(Enum):
    DEVICE_WARMUP = "device_warmup"
    MODEL_INFERENCE = "model_inference"
    PRE_PROCESSING = "pre_processing"
    POST_PROCESSING = "post_processing"
    TOTAL_PROCESSING = "total_processing"
    BASE_TOTAL_PROCESSING = "base_total_processing"
    BASE_SINGLE_PROCESSING = "base_single_processing"


# top metric for total requests from intherited service
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

    def _process_telemetry(self):
        """Background worker to process telemetry"""
        while True:
            try:
                event = self.queue.get(timeout=1)
                if event:
                    # Process telemetry in background
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
        """Record a telemetry event"""

        status_str = "success" if status else "failure"

        if event_name == TelemetryEvent.PRE_PROCESSING:
            self._record_pre_processing(
                duration, preprocessing_enabled=True, status=status_str
            )
        elif event_name == TelemetryEvent.POST_PROCESSING:
            self._record_post_processing(
                duration, preprocessing_enabled=True, status=status_str
            )
        elif event_name == TelemetryEvent.MODEL_INFERENCE:
            self._record_model_inference(device_id, duration, status=status_str)
        elif event_name == TelemetryEvent.DEVICE_WARMUP:
            self._record_device_warmup(device_id, duration, status=status_str)
        elif event_name == TelemetryEvent.TOTAL_PROCESSING:
            self._record_request_duration(device_id, duration, status=status_str)
        elif event_name == TelemetryEvent.BASE_TOTAL_PROCESSING:
            self._record_base_total_request_duration(
                device_id, duration, status=status_str
            )
        elif event_name == TelemetryEvent.BASE_SINGLE_PROCESSING:
            self._record_single_base_request_duration(
                device_id, duration, status=status_str
            )
        else:
            self.logger.warning(f"Unknown telemetry event: {event_name}")

    # Utility functions for recording metrics
    def _record_pre_processing(
        self, duration: float, preprocessing_enabled: bool, status: str = "success"
    ):
        """Record audio processing metrics"""
        pre_processing_duration.labels(
            model_type=self.settings.model_runner,
            preprocessing_enabled=str(preprocessing_enabled),
        ).observe(duration)

    def _record_post_processing(
        self, duration: float, preprocessing_enabled: bool, status: str = "success"
    ):
        """Record audio processing metrics"""
        post_processing_duration.labels(
            model_type=self.settings.model_runner,
            preprocessing_enabled=str(preprocessing_enabled),
        ).observe(duration)

    def _record_model_inference(
        self, device_id: str, duration: float, status: str = "success"
    ):
        """Record model inference metrics"""

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
        """Record model warmup metrics"""

        device_warmup_duration.labels(
            model_type=self.settings.model_runner, device_id=device_id or "unknown"
        ).observe(duration)

        # Create a separate counter for device warmup
        device_warmup_counter.labels(
            model_type=self.settings.model_runner,
            device_id=device_id or "unknown",
            status=status,
        ).inc()

    def _record_request_duration(
        self, device_id: str, duration: float, status: str = "success"
    ):
        """Record end to end duration"""
        request_duration.labels(model_type=self.settings.model_runner).observe(duration)

        request_counter.labels(model_type=self.settings.model_runner).inc()

    def _record_base_total_request_duration(
        self, device_id: str, duration: float, status: str = "success"
    ):
        """Record end to end duration"""
        total_base_request_duration.labels(
            model_type=self.settings.model_runner
        ).observe(duration)

        total_base_request_counter.labels(model_type=self.settings.model_runner).inc()

    def _record_single_base_request_duration(
        self, device_id: str, duration: float, status: str = "success"
    ):
        """Record end to end duration"""
        base_request_duration.labels(model_type=self.settings.model_runner).observe(
            duration
        )

        base_request_counter.labels(model_type=self.settings.model_runner).inc()


@lru_cache(maxsize=1)
def get_telemetry_client() -> TelemetryClient:
    return TelemetryClient()
