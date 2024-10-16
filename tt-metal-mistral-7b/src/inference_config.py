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
        "end_of_sequence_str",
        "inference_route", 
        "model_config",
    ],
)

ModelConfig = namedtuple(
    "ModelConfig",
    [
        "model_version",
        "model_cache_name",
        "batch_size",
        "num_layers",
        "max_seq_len",
        "default_top_p",
        "default_top_k",
        "default_temperature",
    ],
)

# Do as much environment variable termination here as possible.
# The exception is secrets, which are used directly as os.getenv() calls.
# get_env_var() is used to add helpful documentation for environment variables
CACHE_ROOT = get_env_var("CACHE_ROOT", msg="Base path for all data caches.")
SERVICE_PORT = int(os.getenv("SERVICE_PORT", 7000))
MOCK_MODEL = bool(int(os.getenv("MOCK_MODEL", 0)))
BACKEND_DEBUG_MODE = bool(int(os.getenv("BACKEND_DEBUG_MODE", 0)))
FRONTEND_DEBUG_MODE = bool(int(os.getenv("FRONTEND_DEBUG_MODE", 0)))
MODEL_WEIGHTS_ID = os.getenv("MODEL_WEIGHTS_ID")
MODEL_WEIGHTS_PATH = os.getenv("MODEL_WEIGHTS_PATH")

inference_config = InferenceConfig(
    cache_root=CACHE_ROOT,
    hf_cache=f"{CACHE_ROOT}/hf_cache",
    log_cache=f"{CACHE_ROOT}/logs",
    max_input_qsize=4,  # last in queue can get response before request timeout
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
    end_of_sequence_str="<|endoftext|>",
    inference_route="/inference/mistral7b", 
    model_config=ModelConfig(
        model_version="mistralai/Mistral-7B-Instruct-v0.2",
        model_cache_name="mistral-7b-instruct",
        batch_size=8,
        num_layers=32,
        max_seq_len=4096,
        default_top_p=0.9,
        default_top_k=40,
        default_temperature=1.0,
    ),
)

print("using inference_config:\n")
pprint(inference_config._asdict())
print("\n")
