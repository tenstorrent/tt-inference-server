# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os

from config.settings import get_settings
from fastapi import FastAPI, Response
import torch
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    Gauge,
    Info,
    generate_latest,
)
from prometheus_client.gc_collector import GCCollector
from prometheus_client.multiprocess import MultiProcessCollector
from prometheus_client.platform_collector import PlatformCollector
from prometheus_client.process_collector import ProcessCollector
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from utils.logger import TTLogger


# Set up multiprocess directory FIRST, before any other imports or class definitions
def setup_multiprocess_dir():
    """Setup multiprocess directory for Prometheus"""
    multiproc_dir = "/tmp/prometheus_multiproc"
    try:
        os.makedirs(multiproc_dir, exist_ok=True)
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = multiproc_dir
        return multiproc_dir
    except Exception:
        # Remove the env var if directory creation fails
        os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
        return None


# Call this immediately
MULTIPROC_DIR = setup_multiprocess_dir()

# System info
system_info = Info("tt_media_server_info", "System information")

# PyTorch metrics
pytorch_info = Info("pytorch", "PyTorch build information")
pytorch_num_threads = Gauge("pytorch_num_threads", "PyTorch intra-op parallelism threads")
pytorch_num_interop_threads = Gauge(
    "pytorch_num_interop_threads", "PyTorch inter-op parallelism threads"
)


class PrometheusMetrics:
    def __init__(self, app: FastAPI):
        self.app = app
        self.settings = get_settings()
        self.instrumentator = None
        self.logger = TTLogger()
        self.multiproc_dir = MULTIPROC_DIR
        self._python_collectors_registered = False

    def _register_python_collectors(self, registry: CollectorRegistry = REGISTRY):
        """Register Python internal metrics collectors (GC, process, platform).

        These provide insight into:
        - Process: CPU time, memory (RSS/VMS), open file descriptors
        - GC: Garbage collection counts per generation
        - Platform: Python version and implementation
        """
        if self._python_collectors_registered:
            return

        try:
            # Process metrics: cpu_seconds, memory bytes, open_fds, etc.
            ProcessCollector(registry=registry)
            # GC metrics: collected objects per generation
            GCCollector(registry=registry)
            # Platform info: python version, implementation
            PlatformCollector(registry=registry)
            self._python_collectors_registered = True
            self.logger.info("Python internal metrics collectors registered")
        except Exception as e:
            self.logger.warning(f"Failed to register Python collectors: {e}")

    def _set_pytorch_metrics(self):
        """Set PyTorch CPU metrics (thread config and build info)."""
        try:
            # Thread configuration
            pytorch_num_threads.set(torch.get_num_threads())
            pytorch_num_interop_threads.set(torch.get_num_interop_threads())

            # Build info
            pytorch_info.info(
                {
                    "version": torch.__version__,
                    "cuda_available": str(torch.cuda.is_available()),
                    "cuda_version": torch.version.cuda or "N/A",
                    "cudnn_version": str(torch.backends.cudnn.version() or "N/A"),
                }
            )
            self.logger.info("PyTorch metrics registered")
        except Exception as e:
            self.logger.warning(f"Failed to set PyTorch metrics: {e}")

    def setup_metrics(self):
        """Setup Prometheus metrics collection"""
        if not self.settings.enable_telemetry:
            self.logger.info("Telemetry disabled, skipping Prometheus setup")
            return

        self.logger.info("Setting up Prometheus metrics...")

        # Ensure multiprocess directory exists
        if self.multiproc_dir:
            try:
                os.makedirs(self.multiproc_dir, exist_ok=True)
                self.logger.info(f"Multiprocess directory ready: {self.multiproc_dir}")
            except Exception as e:
                self.logger.error(f"Failed to create multiprocess directory: {e}")
                self.multiproc_dir = None

        # Register Python internal metrics
        self._register_python_collectors()

        # Set PyTorch metrics
        self._set_pytorch_metrics()

        # Initialize instrumentator
        self.instrumentator = Instrumentator(
            should_group_status_codes=True,
            should_ignore_untemplated=True,
            should_respect_env_var=True,
            should_instrument_requests_inprogress=True,
            excluded_handlers=["/health", "/metrics"],
            env_var_name="ENABLE_METRICS",
            inprogress_name="tt_media_server_requests_inprogress",
            inprogress_labels=True,
        )

        # Add default metrics
        self.instrumentator.add(
            metrics.request_size(
                should_include_handler=True,
                should_include_method=True,
                should_include_status=True,
                metric_namespace="tt_media_server",
                metric_subsystem="requests",
            )
        ).add(
            metrics.response_size(
                should_include_handler=True,
                should_include_method=True,
                should_include_status=True,
                metric_namespace="tt_media_server",
                metric_subsystem="responses",
            )
        ).add(
            metrics.latency(
                should_include_handler=True,
                should_include_method=True,
                should_include_status=True,
                metric_namespace="tt_media_server",
                metric_subsystem="requests",
            )
        )

        # Instrument the app
        self.instrumentator.instrument(self.app)

        # Add custom metrics endpoint
        @self.app.get(self.settings.prometheus_endpoint)
        async def get_metrics():
            metrics_data = None
            try:
                if self.multiproc_dir and os.path.exists(self.multiproc_dir):
                    # Get multiprocess metrics
                    mp_registry = CollectorRegistry()
                    MultiProcessCollector(mp_registry)
                    mp_metrics = generate_latest(mp_registry).decode("utf-8")

                    # Get default registry metrics
                    default_metrics = generate_latest(REGISTRY).decode("utf-8")

                    # Combine both (simple concatenation)
                    combined_metrics = default_metrics + "\n" + mp_metrics

                    metrics_data = combined_metrics.encode("utf-8")
            except Exception as e:
                self.logger.error(f"Error generating metrics: {e}")
                # Fallback to default registry
                metrics_data = generate_latest(REGISTRY)

            return Response(metrics_data, media_type=CONTENT_TYPE_LATEST)

        # Set system info
        system_info.info(
            {
                "version": "1.0.0",
                "environment": self.settings.environment,
                "device_mesh_shape": str(self.settings.device_mesh_shape),
                "model_runner": self.settings.model_runner,
            }
        )

        self.logger.info(
            f"Prometheus metrics available at {self.settings.prometheus_endpoint}"
        )
