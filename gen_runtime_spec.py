#!/usr/bin/env python3
"""Generate a fresh runtime_model_spec JSON for a given model+device.

Usage: python3 gen_runtime_spec.py <model> <device> --device-id <N> [--service-port <PORT>]
Prints the absolute path of the written JSON file to stdout.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflows.model_spec import get_runtime_model_spec
from workflows.runtime_config import RuntimeConfig
from workflows.utils import ensure_readwriteable_dir, get_default_workflow_root_log_dir


def _populate_cli_args(model_spec, runtime_config):
    # Backfill cli_args so tt-media-server can still read them (see run.py:455, TODO #1767)
    if not hasattr(model_spec, "cli_args") or model_spec.cli_args is None:
        return
    cli_args = runtime_config.to_dict()
    cli_args["tt_device"] = runtime_config.device
    model_spec.cli_args.clear()
    model_spec.cli_args.update(cli_args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("device")
    parser.add_argument("--device-id", type=int, required=True)
    parser.add_argument("--service-port", default="8000")  # container-internal port
    args = parser.parse_args()

    model_spec, impl, engine = get_runtime_model_spec(model=args.model, device=args.device)
    runtime_config = RuntimeConfig(
        model=args.model,
        workflow="server",
        device=args.device,
        impl=impl,
        engine=engine,
        docker_server=True,
        dev_mode=True,
        service_port=args.service_port,
        device_id=[args.device_id],
        disable_trace_capture=True,
        skip_system_sw_validation=True,
    )
    _populate_cli_args(model_spec, runtime_config)

    spec_dir = get_default_workflow_root_log_dir() / "runtime_model_specs"
    ensure_readwriteable_dir(spec_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    json_fpath = runtime_config.to_json(model_spec, timestamp, model_spec.model_id, spec_dir)
    print(json_fpath)


if __name__ == "__main__":
    main()
