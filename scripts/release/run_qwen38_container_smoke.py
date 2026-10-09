# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Bounded hardware/API smoke of a pinned image after an owned hardware queue."""

import argparse
import asyncio
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time


def predecessor_ready(properties, receipt, invocation):
    if properties.get("InvocationID") not in ("", invocation):
        raise ValueError("Predecessor invocation changed")
    if properties.get("MainPID") != "0" or properties.get("ActiveState") not in (
        "inactive",
        "failed",
    ):
        return False
    if properties.get("LoadState") not in ("loaded", "not-found"):
        return False
    if (
        properties.get("LoadState") == "loaded"
        and properties.get("Result") != "success"
    ):
        raise ValueError("Predecessor did not exit successfully")
    if (
        not receipt
        or receipt.get("state") != "completed"
        or receipt.get("cleanup_completed") is not True
    ):
        raise ValueError("Predecessor lacks a complete clean terminal receipt")
    return True


def evaluation_command(path):
    command = json.loads(path.read_text())
    if (
        not isinstance(command, list)
        or not command
        or not all(
            isinstance(value, str) and value and "\0" not in value for value in command
        )
    ):
        raise ValueError("Evaluation command must be a nonempty JSON argv list")
    if not Path(command[0]).is_absolute():
        raise ValueError("Evaluation executable must be an absolute path")
    return command


def run(args):
    import httpx

    sys.path.insert(0, str(args.model_source))
    from models.demos.qwen38_27b_qb2.demo.galaxy_serving import (
        MODEL_NAME,
        qualified_groups,
        verify_worker_bindings,
        verify_worker_precision,
    )
    from models.demos.qwen38_27b_qb2.tests.galaxy_api import validate_api

    root = args.output
    root.mkdir(exist_ok=False)
    state = dict(
        state="waiting",
        hardware_opened=False,
        image=args.image,
        predecessor_unit=args.predecessor_unit,
        predecessor_invocation=args.predecessor_invocation,
        survives_disconnect=True,
        resumes_after_reboot=False,
        started_at=time.time(),
    )
    name = args.container_name
    lock = None
    container = None

    def save():
        state["updated_at"] = time.time()
        temporary = root / "status.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(root / "status.json")

    def terminate(signum, frame):
        raise InterruptedError(f"Container supervisor received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    save()
    try:
        deadline = time.monotonic() + args.wait_timeout
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("Predecessor wait expired")
            try:
                raw = subprocess.check_output(
                    [
                        "systemctl",
                        "--user",
                        "show",
                        args.predecessor_unit,
                        "-p",
                        "MainPID",
                        "-p",
                        "ActiveState",
                        "-p",
                        "LoadState",
                        "-p",
                        "Result",
                        "-p",
                        "InvocationID",
                    ],
                    text=True,
                    timeout=15,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                state["observation_error"] = type(error).__name__
                save()
                time.sleep(30)
                continue
            properties = dict(
                line.split("=", 1) for line in raw.splitlines() if "=" in line
            )
            receipt = (
                json.loads(args.predecessor_receipt.read_text())
                if args.predecessor_receipt.exists()
                else None
            )
            state["predecessor"] = properties
            save()
            if predecessor_ready(properties, receipt, args.predecessor_invocation):
                break
            time.sleep(30)
        probe = json.loads(args.startup_receipt.read_text())
        if (
            probe.get("passed") is not True
            or probe.get("image_manifest_digest") != args.image
        ):
            raise ValueError("Image has no passing startup handoff receipt")
        for relative, expected in json.loads(args.source_manifest.read_text()).items():
            if (
                hashlib.sha256((args.model_source / relative).read_bytes()).hexdigest()
                != expected
            ):
                raise ValueError("Frozen model/API source changed: " + relative)
        qualification = json.loads(args.qualification.read_text())
        groups = qualified_groups(qualification)
        if shutil.disk_usage(root).free < 16 * 1024**3:
            raise RuntimeError("Insufficient task disk for container caches")
        with socket.socket() as port_probe:
            port_probe.bind(("127.0.0.1", args.port))
        lock = open("/tmp/tt-device.lock", "a")
        state["state"] = "waiting_for_device_lock"
        save()
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Hardware lock wait expired")
                time.sleep(10)
        # Reset once under the lock before moving from native to container
        # workers. Leave the dirty marker for the next authorized runner.
        state["state"] = "resetting"
        save()
        with (root / "reset.log").open("w") as log:
            subprocess.run(
                ["/usr/local/bin/tt-smi", "-glx_reset"],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=600,
            )
        Path("/tmp/tt-device.dirty").touch()
        cache = root / "cache"
        cache.mkdir()
        command = [
            "docker",
            "create",
            "--name",
            name,
            "--label",
            "qwen38.task=" + name,
            "--read-only",
            "--memory=256g",
            "--memory-swap=256g",
            "--cpus=16",
            "--pids-limit=2048",
            "--log-opt=max-size=16m",
            "--log-opt=max-file=2",
            "--device=/dev/tenstorrent",
            "--ulimit=memlock=-1",
            "--cap-add=IPC_LOCK",
            "--shm-size=32g",
            "--publish",
            f"127.0.0.1:{args.port}:8000",
            "--tmpfs",
            "/tmp:rw,size=2g",
            "--mount",
            "type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G",
            "--mount",
            f"type=bind,src={args.weights},dst=/mnt/hf-cache,readonly",
            "--mount",
            f"type=bind,src={cache},dst=/home/container_app_user/cache_root",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "TT_METAL_INSPECTOR_RPC=0",
            "--env",
            "TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0",
            args.image,
            "--model",
            "Qwen3.8-27B",
            "--tt-device",
            "blackhole_galaxy",
        ]
        state.update(state="starting", command=command, hardware_opened=None)
        save()
        container = subprocess.check_output(command, text=True, timeout=60).strip()
        state["container_id"] = container
        save()
        subprocess.run(["docker", "start", container], check=True, timeout=60)
        ready_deadline = time.monotonic() + 5400
        base_url = f"http://127.0.0.1:{args.port}"
        with httpx.Client(timeout=5) as client:
            while True:
                runtime = json.loads(
                    subprocess.check_output(
                        ["docker", "inspect", "--format", "{{json .State}}", container],
                        text=True,
                        timeout=15,
                    )
                )
                if runtime.get("Running") is not True:
                    raise RuntimeError(
                        f"Container exited before readiness: {runtime.get('ExitCode')}"
                    )
                if shutil.disk_usage(root).free < 8 * 1024**3:
                    raise RuntimeError("Container cache exhausted task disk reserve")
                try:
                    if client.get(base_url + "/health").is_success:
                        break
                except httpx.HTTPError:
                    pass
                if time.monotonic() >= ready_deadline:
                    raise TimeoutError("Container readiness exceeded 90 minutes")
                time.sleep(10)
        logs = subprocess.run(
            ["docker", "logs", container],
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        )
        text = logs.stdout + logs.stderr
        (root / "startup.log").write_text(text)
        state.update(
            state="api_checks",
            hardware_opened=True,
            worker_bindings=verify_worker_bindings(text, groups),
            worker_precision=verify_worker_precision(text, qualification["precision"]),
        )
        save()
        asyncio.run(
            asyncio.wait_for(
                validate_api(base_url, root / "api.json", concurrency=128), timeout=1800
            )
        )
        tool = {
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "description": "Read current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
        messages = [
            {
                "role": "user",
                "content": "Use lookup_weather to get the weather for Paris. Do not invent a weather reading.",
            }
        ]
        payload = dict(
            model=MODEL_NAME,
            messages=messages,
            tools=[tool],
            tool_choice="auto",
            temperature=0,
            top_k=1,
            max_tokens=256,
            chat_template_kwargs={"enable_thinking": False},
        )
        with httpx.Client(timeout=300) as client:
            response = client.post(base_url + "/v1/chat/completions", json=payload)
            response.raise_for_status()
            first = response.json()
            message = first["choices"][0]["message"]
            calls = message.get("tool_calls") or []
            assert len(calls) == 1 and calls[0]["function"]["name"] == "lookup_weather"
            assert (
                json.loads(calls[0]["function"]["arguments"])["city"].lower() == "paris"
            )
            messages.extend(
                [
                    dict(
                        role="assistant",
                        content=message.get("content"),
                        tool_calls=calls,
                    ),
                    dict(
                        role="tool",
                        tool_call_id=calls[0]["id"],
                        content='{"city":"Paris","temperature_c":21}',
                    ),
                ]
            )
            response = client.post(base_url + "/v1/chat/completions", json=payload)
            response.raise_for_status()
            second = response.json()
            answer = second["choices"][0]
            assert answer["finish_reason"] == "stop" and "21" in answer["message"].get(
                "content", ""
            )
            assert not answer["message"].get("tool_calls")
            (root / "synthetic-tool-roundtrip.json").write_text(
                json.dumps(dict(passed=True, first=first, second=second), indent=2)
                + "\n"
            )
        state["api_and_tool_smoke_passed"] = True
        save()
        if args.evaluation_command:
            command = evaluation_command(args.evaluation_command)
            state.update(state="evaluating", evaluation_command=command)
            save()
            with (root / "evaluation.log").open("w") as log:
                result = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.evaluation_timeout,
                )
            state["evaluation_exit_code"] = result.returncode
            save()
            result.check_returncode()
        state.update(
            state="completed",
            passed=True,
            finished_at=time.time(),
            scope=(
                "Container startup and short API/tool smoke with bounded evaluation; inspect separate evaluation receipt"
                if args.evaluation_command
                else "Container startup and short API/tool smoke; not full eval or Helm qualification"
            ),
        )
    except BaseException as error:
        state.update(
            state="failed",
            error=type(error).__name__,
            detail=str(error)[:2000],
            finished_at=time.time(),
        )
        raise
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            if container:
                logs = subprocess.run(
                    ["docker", "logs", container],
                    text=True,
                    capture_output=True,
                    timeout=30,
                )
                (root / "server.log").write_text(logs.stdout + logs.stderr)
                subprocess.run(
                    ["docker", "stop", "--time", "60", container],
                    check=True,
                    timeout=90,
                )
                subprocess.run(["docker", "rm", container], check=True, timeout=30)
                state["owned_container_removed"] = True
        finally:
            if lock:
                lock.close()
            save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "output",
        "model-source",
        "weights",
        "source-manifest",
        "qualification",
        "startup-receipt",
        "predecessor-receipt",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in (
        "image",
        "container-name",
        "predecessor-unit",
        "predecessor-invocation",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--port", type=int, default=8079)
    parser.add_argument("--wait-timeout", type=int, default=32400)
    parser.add_argument("--evaluation-command", type=Path)
    parser.add_argument("--evaluation-timeout", type=int, default=9000)
    run(parser.parse_args())
