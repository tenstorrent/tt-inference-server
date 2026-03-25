import os
from pathlib import Path


def faces_dir() -> Path:
    p = Path(os.environ.get("FACES_DIR", "/app/registered_faces"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def device_id() -> int:
    return int(os.environ.get("DEVICE_ID", "0"))


def recognition_threshold() -> float:
    return float(os.environ.get("RECOGNITION_THRESHOLD", "0.5"))


def service_port() -> int:
    return int(os.environ.get("SERVICE_PORT", "7070"))
