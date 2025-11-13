from enum import Enum
from functools import lru_cache
from queue import Queue
from threading import Thread
import time
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST, REGISTRY
from prometheus_client import multiprocess, CollectorRegistry
from prometheus_client.multiprocess import MultiProcessCollector
from config.settings import get_settings
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from fastapi import FastAPI, Response
from utils.logger import TTLogger
import os

# Set up multiprocess directory FIRST, before any other imports or class definitions
def setup_multiprocess_dir():
    """Setup multiprocess directory for Prometheus"""
    multiproc_dir = '/tmp/prometheus_multiproc'
    try:
        os.makedirs(multiproc_dir, exist_ok=True)
        os.environ['PROMETHEUS_MULTIPROC_DIR'] = multiproc_dir
        return multiproc_dir
    except Exception as e:
        # Remove the env var if directory creation fails
        if 'PROMETHEUS_MULTIPROC_DIR' in os.environ:
            del os.environ['PROMETHEUS_MULTIPROC_DIR']
        return None

# Call this immediately
MULTIPROC_DIR = setup_multiprocess_dir()

class TelmetryEvent(Enum):
    DEVICE_WARMUP = "device_warmup"
    MODEL_INFERENCE = "model_inference"
    PRE_PROCESSING = "pre_processing"
    POST_PROCESSING = "post_processing"
    TOTAL_PROCESSING = "total_processing"
    

# Custom metrics for your application
request_counter = Counter(
    'tt_media_server_requests_total',
    'Total number of requests',
    ['model_type']
)

request_duration = Histogram(
    'tt_media_server_request_duration_seconds',
    'Request duration in seconds',
    ['model_type']
)

# Subprocessing metrics
pre_processing_duration = Histogram(
    'tt_media_server_pre_processing_duration_seconds',
    'Pre processing duration in seconds',
    ['model_type', 'preprocessing_enabled']
)

post_processing_duration = Histogram(
    'tt_media_server_post_processing_duration_seconds',
    'Post processing processing duration in seconds',
    ['model_type', 'post_processing_enabled']
)

# Model inference metrics
model_inference_duration = Histogram(
    'tt_media_server_model_inference_duration_seconds',
    'Model inference duration in seconds',
    ['model_type', 'device_id']
)

device_warmup_duration = Histogram(
    'tt_media_server_device_warmup_duration_seconds',
    'Model inference duration in seconds',
    ['model_type', 'device_id']
)

model_inference_counter = Counter(
    'tt_media_server_model_inference_total',
    'Total number of model inferences',
    ['model_type', 'device_id', 'status']
)

device_warmup_counter = Counter(
    'tt_media_server_device_warmup_total',
    'Total number of device warmup operations',
    ['model_type', 'device_id', 'status']
)

model_load_counter = Counter(
    'tt_media_server_model_load_total',
    'Total number of model load operations',
    ['model_type', 'device_id', 'status']
)

# System info
system_info = Info(
    'tt_media_server_info',
    'System information'
)

class PrometheusMetrics:
    def __init__(self, app: FastAPI):
        self.app = app
        self.settings = get_settings()
        self.instrumentator = None
        self.logger = TTLogger()
        self.multiproc_dir = MULTIPROC_DIR
        
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
        
        # Initialize instrumentator
        self.instrumentator = Instrumentator(
            should_group_status_codes=True,
            should_ignore_untemplated=True,
            should_respect_env_var=True,
            should_instrument_requests_inprogress=True,
            excluded_handlers=["/health", "/metrics"],
            env_var_name="ENABLE_METRICS",
            inprogress_name="tt_media_server_requests_inprogress",
            inprogress_labels=True
        )
        
        # Add default metrics
        self.instrumentator.add(
            metrics.request_size(
                should_include_handler=True,
                should_include_method=True,
                should_include_status=True,
                metric_namespace="tt_media_server",
                metric_subsystem="requests"
            )
        ).add(
            metrics.response_size(
                should_include_handler=True,
                should_include_method=True,
                should_include_status=True,
                metric_namespace="tt_media_server",
                metric_subsystem="responses"
            )
        ).add(
            metrics.latency(
                should_include_handler=True,
                should_include_method=True,
                should_include_status=True,
                metric_namespace="tt_media_server",
                metric_subsystem="requests"
            )
        )
        
        # Instrument the app
        self.instrumentator.instrument(self.app)
        
        # Add custom metrics endpoint
        @self.app.get(self.settings.prometheus_endpoint)
        async def get_metrics():
            try:
                if self.multiproc_dir and os.path.exists(self.multiproc_dir):
                    # Multi-process mode - combine both registries
                    from prometheus_client import CollectorRegistry, generate_latest
                    
                    # Get multiprocess metrics
                    mp_registry = CollectorRegistry()
                    MultiProcessCollector(mp_registry)
                    mp_metrics = generate_latest(mp_registry).decode('utf-8')
                    
                    # Get default registry metrics
                    default_metrics = generate_latest(REGISTRY).decode('utf-8')
                    
                    # Combine both (simple concatenation)
                    combined_metrics = default_metrics + "\n" + mp_metrics
                    
                    return Response(
                        combined_metrics.encode('utf-8'),
                        media_type=CONTENT_TYPE_LATEST
                    )
                else:
                    # Single process mode
                    return Response(
                        generate_latest(REGISTRY),
                        media_type=CONTENT_TYPE_LATEST
                    )
            except Exception as e:
                self.logger.error(f"Error generating metrics: {e}")
                # Fallback to default registry
                return Response(
                    generate_latest(REGISTRY),
                    media_type=CONTENT_TYPE_LATEST
                )
        
        # Set system info
        system_info.info({
            'version': '1.0.0',
            'environment': self.settings.environment,
            'device_mesh_shape': str(self.settings.device_mesh_shape),
            'model_runner': self.settings.model_runner
        })
        
        self.logger.info(f"Prometheus metrics available at {self.settings.prometheus_endpoint}")

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

    def record_telemetry_event_async(self, event_name, device_id=None, duration=None, status=True):
        """Non-blocking telemetry recording"""
        if not self.settings.enable_telemetry or self.queue is None:
            return  # Do nothing if telemetry is disabled

        self.queue.put({
            'event_name': event_name,
            'device_id': device_id,
            'duration': duration,
            'status': status,
            'timestamp': time.time()  # Now this will work correctly
        })
    
    def _process_telemetry(self):
        """Background worker to process telemetry"""
        while True:
            try:
                event = self.queue.get(timeout=1)
                if event:
                    # Process telemetry in background
                    self.record_telemetry_event(
                        event['event_name'],
                        event['device_id'],
                        event['duration'],
                        event['status']
                    )
                    self.queue.task_done()
            except:
                continue  # Keep worker alive

    def record_telemetry_event(self, event_name: TelmetryEvent, device_id: str = None, duration: float = None, status: bool = True):
        """Record a telemetry event"""
        if event_name == TelmetryEvent.PRE_PROCESSING:
            self._record_pre_processing(duration, preprocessing_enabled=True, status="success" if status else "failure")
        elif event_name == TelmetryEvent.POST_PROCESSING:
            self._record_post_processing(duration, preprocessing_enabled=True, status="success" if status else "failure")
        elif event_name == TelmetryEvent.MODEL_INFERENCE:
            self._record_model_inference(device_id, duration, status="success" if status else "failure")
        elif event_name == TelmetryEvent.DEVICE_WARMUP:
            self._record_device_warmup(device_id, duration, status="success" if status else "failure")
        elif event_name == TelmetryEvent.TOTAL_PROCESSING:
            self._record_request_duration(device_id, duration, status="success" if status else "failure")
        else:
            self.logger.warning(f"Unknown telemetry event: {event_name}")
    
        
    # Utility functions for recording metrics
    def _record_pre_processing(self, duration: float, preprocessing_enabled: bool, status: str = "success"):
        """Record audio processing metrics"""
        pre_processing_duration.labels(
            model_type=get_settings().model_runner,
            preprocessing_enabled=str(preprocessing_enabled)
        ).observe(duration)

    def _record_post_processing(self, duration: float, preprocessing_enabled: bool, status: str = "success"):
        """Record audio processing metrics"""
        post_processing_duration.labels(
            model_type=get_settings().model_runner,
            preprocessing_enabled=str(preprocessing_enabled)
        ).observe(duration)


    def _record_model_inference(self, device_id: str, duration: float, status: str = "success"):
        """Record model inference metrics"""

        model_inference_duration.labels(
            model_type=get_settings().model_runner,
            device_id=device_id or "unknown"
        ).observe(duration)
        
        model_inference_counter.labels(
            model_type=get_settings().model_runner,
            device_id=device_id or "unknown",
            status=status
        ).inc()

    def _record_device_warmup(self, device_id: str, duration: float, status: str = "success"):
        """Record model warmup metrics"""
        
        device_warmup_duration.labels(
            model_type=get_settings().model_runner,
            device_id=device_id or "unknown"
        ).observe(duration)
        
        # Create a separate counter for device warmup
        device_warmup_counter.labels(
            model_type=get_settings().model_runner,
            device_id=device_id or "unknown",
            status=status
        ).inc()

    def _record_request_duration(self, device_id: str, duration: float, status: str = "success"):
        """Record end to end duration"""
        request_duration.labels(
            model_type=get_settings().model_runner
        ).observe(duration)
        
        request_counter.labels(
            model_type=get_settings().model_runner
        ).inc()

@lru_cache(maxsize=1)
def get_telemetry_client() -> TelemetryClient:
    return TelemetryClient()