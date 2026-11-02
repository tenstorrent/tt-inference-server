# SPDX-License-Identifier: Apache-2.0
"""Retain Qwen source-side serving counters in the native workflow artifact tree."""
import json
from pathlib import Path
import re

ENV_KEY = "QWEN38_SERVING_EVENTS"
CONTAINER_DIR = "/home/container_app_user/qwen38_serving_events"
CONTAINER_FILE = CONTAINER_DIR + "/events.jsonl"


def events_mount(model_spec, artifact_root, container_name):
    if (getattr(model_spec, "hf_model_repo", None) != "Qwen/Qwen3.8-27B"
            or getattr(getattr(model_spec, "impl", None), "impl_id", None) != "qwen38_autoport"):
        return [], {}
    if model_spec.env_vars.get(ENV_KEY) != CONTAINER_FILE:
        raise RuntimeError("Autoport runtime spec must select the collected event path")
    if not re.fullmatch(r"tt-inference-server-[A-Za-z0-9_-]+", container_name):
        raise ValueError("Invalid owned container name")
    # This is the exact native upload-artifact tree, independent of cache volumes.
    # A fresh directory is required: never chmod, reuse or truncate a peer's logs.
    parent = Path(artifact_root).resolve() / "serving_events"
    parent.mkdir(parents=True, exist_ok=True)
    host_dir = parent / container_name
    host_dir.mkdir(exist_ok=False)
    host_dir.chmod(0o777)  # Only this newly owned dir; image user may differ from host.
    receipt = {
        "schema_version": 1,
        "container_name": container_name,
        "container_path": CONTAINER_FILE,
        "artifact_path": f"serving_events/{container_name}/events.jsonl",
        "source": "generated autoport QWEN38_SERVING_EVENTS",
        "runtime_verified": False,
    }
    (host_dir / "transport.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return ["--mount", f"type=bind,src={host_dir},dst={CONTAINER_DIR}"], {ENV_KEY: CONTAINER_FILE}
