import os
from collections import namedtuple
from pprint import pprint


def get_env_var(var, msg):
    """Get an environment variable or raise an exception with helpful message."""
    value = os.getenv(var)
    if value is None:
        raise ValueError(f"Environment variable is required: {var}. {msg}")
    return value


InferenceConfig = namedtuple(
    "InferenceConfig",
    [
        "cache_root",
        "hf_cache",
        "log_cache",
        "max_input_qsize",
        "input_timeout",
        "max_inactive_seconds",
        "backend_server_port",
        "keepalive_input_period_seconds",
        "max_seconds_healthy_no_response",
        "backend_debug_mode",
        "frontend_debug_mode",
        "mock_model",
        "model_weights_id",
        "model_weights_path",
        "n_devices",
        "inference_route_name",
        "end_of_sequence_str",
        "model_config",
    ],
)

ModelConfig = namedtuple(
    "ModelConfig",
    [
        "model_name",
        "model_version",
        "device_type",
        "batch_size",
        "num_layers",
        "max_seq_len",
        "default_top_p",
        "default_top_k",
        "default_temperature",
        "chat",
        "llama_version",
    ],
)

# Do as much environment variable termination here as possible.
# The exception is secrets, which are used directly as os.getenv() calls.
# get_env_var() is used to add helpful documentation for environment variables
CACHE_ROOT = get_env_var("CACHE_ROOT", msg="Base path for all data caches.")
MODEL_NAME = get_env_var("MODEL_NAME", msg="Model name must be configured.")
SERVICE_PORT = int(os.getenv("SERVICE_PORT", 7000))
MOCK_MODEL = bool(int(os.getenv("MOCK_MODEL", 0)))
BACKEND_DEBUG_MODE = bool(int(os.getenv("BACKEND_DEBUG_MODE", 0)))
FRONTEND_DEBUG_MODE = bool(int(os.getenv("FRONTEND_DEBUG_MODE", 0)))
MODEL_WEIGHTS_ID = os.getenv("MODEL_WEIGHTS_ID")
MODEL_WEIGHTS_PATH = os.getenv("MODEL_WEIGHTS_PATH")
LLAMA_VERSION = os.getenv("LLAMA_VERSION", "llama3")
MODEL_VERSION = os.getenv("MODEL_VERSION", "0.0.0")
DEVICE_TYPE = os.getenv("DEVICE_TYPE")
route_map = {
    "llama-3.1-70b-instruct": "llama3-1-70b-instruct",
    "llama-3.1-70b": "llama3-1-70b",
    "llama-3.1-8b-instruct": "llama3-1-8b-instruct",
    "llama-3.1-8b": "llama3-1-8b",
    "llama-3-70b-instruct": "llama3-70b-instruct",
    "llama-3-70b": "llama3-70b",
    "llama-3-8b-instruct": "llama3-8b-instruct",
    "llama-3-8b": "llama3-8b",
}
if MODEL_NAME not in route_map:
    raise ValueError(
        f"MODEL_NAME:={MODEL_NAME} is not supported. Must be one of: {route_map.keys()}"
    )
inference_route_name = route_map[MODEL_NAME]

inference_config = InferenceConfig(
    cache_root=CACHE_ROOT,
    hf_cache=f"{CACHE_ROOT}/hf_cache",
    log_cache=f"{CACHE_ROOT}/logs",
    max_input_qsize=32,  # last in queue can get response before request timeout
    input_timeout=30,  # input q backpressure, timeout in seconds
    max_inactive_seconds=60.0,  # maximum time between decode reads to be active
    backend_server_port=SERVICE_PORT,
    keepalive_input_period_seconds=120,
    max_seconds_healthy_no_response=600,
    backend_debug_mode=BACKEND_DEBUG_MODE,
    frontend_debug_mode=FRONTEND_DEBUG_MODE,
    mock_model=MOCK_MODEL,
    model_weights_id=MODEL_WEIGHTS_ID,
    model_weights_path=MODEL_WEIGHTS_PATH,
    n_devices=8,
    inference_route_name=inference_route_name,
    end_of_sequence_str="<|endoftext|>",
    model_config=ModelConfig(
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        device_type=DEVICE_TYPE,
        batch_size=1,
        num_layers=32,
        max_seq_len=2048,
        default_top_p=0.9,
        default_top_k=40,
        default_temperature=1.0,
        chat=True,
        llama_version=LLAMA_VERSION,
    ),
)

print("using inference_config:\n")
pprint(inference_config._asdict())
print("\n")
