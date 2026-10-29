# SPDX-License-Identifier: Apache-2.0
"""Run closed native capacity trials before the serving process opens devices."""

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def load_planner(path):
    spec = importlib.util.spec_from_file_location("qwen38_capacity_planner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_sources(root, manifest):
    for directory, key in (("tt", "sha256"), ("tests", "diagnostics_sha256")):
        entries = manifest[key]
        if not entries:
            raise RuntimeError(f"Empty source manifest: {key}")
        for name, expected in entries.items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Invalid source manifest path")
            actual = hashlib.sha256(
                (root / directory / relative).read_bytes()
            ).hexdigest()
            if actual != expected:
                raise RuntimeError(f"Source hash mismatch: {directory}/{name}")


def run_trial(command, env, cwd, log, timeout_s, stem):
    process = subprocess.Popen(
        command, env=env, cwd=cwd, stdout=log, stderr=subprocess.STDOUT
    )
    try:
        return process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        stem.with_suffix(".failure.json").write_text(
            json.dumps(
                {
                    "status": "timeout",
                    "partial_receipt": str(stem.with_suffix(".json")),
                    "no_further_device_trials": True,
                }
            )
            + "\n"
        )
        # Preserve live failure evidence before terminating the timed-out child.
        triage = Path(cwd) / "tools/tt-triage.py"
        if triage.is_file():
            try:
                subprocess.run(
                    [
                        sys.executable,
                        str(triage),
                        "--llm-output",
                        "--llm-output-path",
                        str(stem.with_suffix(".triage.txt")),
                    ],
                    env=env,
                    cwd=cwd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=120,
                    check=False,
                )
            except (subprocess.TimeoutExpired, OSError) as error:
                log.write(f"\nTriage could not finish: {error}\n")
        try:
            process.terminate()
        except ProcessLookupError:
            pass  # The child can finish while triage is collecting evidence.
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=20)
        raise


def screen(
    root,
    output,
    image,
    checkpoint,
    *,
    budget_s=14400,
    manifest_path=Path("/opt/qwen38-source-manifest.json"),
):
    """Never skip unknown failures or retry them as if the first run succeeded."""
    if importlib.metadata.version("vllm").split("+")[0] != "0.30.0":
        raise RuntimeError("Capacity campaign requires the actual vLLM0.30 runtime")
    manifest = json.loads(manifest_path.read_text())
    verify_sources(root, manifest)
    output.mkdir(parents=True, exist_ok=False)
    (output / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    planner = load_planner(root / "tests/plan_capacity_trials.py")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root.parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
    env.update(
        QWEN38_GROUPED_RECURRENCE="1",
        QWEN38_MASKED_RECURRENCE="1",
        QWEN38_FLAT_PREFILL="1",
        QWEN38_SHARED_RECURRENT_SNAPSHOTS="1",
        QWEN38_CACHE_BLOCK_SIZE="512",
        QWEN38_SHARED_MLP_READERS="1",
        QWEN38_SHARED_MLP_CORES="80",
        QWEN38_MEMORY_ATTESTATION="1",
        TT_METAL_TRACE_ALLOC_TRACKING="1",
        TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0",
        QWEN38_CAPACITY_IMAGE_DIGEST=image,
        QWEN38_CAPACITY_STUDY_ID="qwen38-maxkv-512-v1",
    )
    # Allocator diagnostics must not inherit watcher/profiler modes.
    env.pop("TT_METAL_WATCHER", None)
    env.pop("TT_METAL_DEVICE_PROFILER", None)
    rows = []
    started = time.monotonic()
    selection = output / "selection.json"
    while True:
        arms = planner.next_trials(rows)
        selection.write_text(json.dumps({"arms": arms}, indent=2) + "\n")
        pending = [
            (key, value)
            for key, value in arms.items()
            if value.get("status") == "next_native_trial"
        ]
        if not pending:
            if set(arms) != {"0", "1"} or any(
                value.get("status") != "native_max_safe_verified_at_one_ID_boundary"
                for value in arms.values()
            ):
                raise RuntimeError(f"Capacity campaign incomplete; inspect {selection}")
            return
        # Alternate modes; both current-pool anchors precede larger pools.
        shared, trial = min(
            pending, key=lambda pair: sum(r["shared"] == pair[0] for r in rows)
        )
        remaining = budget_s - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Capacity campaign exhausted its four-hour budget")
        stem = (
            output
            / f"trial-{len(rows):02d}-shared{shared}-n{trial['ids']}-c{trial['cycles']}"
        )
        receipt = stem.with_suffix(".json")
        env["QWEN38_SHARED_MLP_WEIGHTS"] = shared
        command = [
            sys.executable,
            str(root / "tests/probe_shared_mlp_memory.py"),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(receipt),
            *trial["argv"],
        ]
        (stem.with_suffix(".command.json")).write_text(
            json.dumps(command, indent=2) + "\n"
        )
        print(f"CAPACITY_TRIAL_START {stem.name}", flush=True)
        with stem.with_suffix(".log").open("w") as log:
            try:
                returncode = run_trial(
                    command, env, root.parents[2], log, min(1800, remaining), stem
                )
            except subprocess.TimeoutExpired:
                (stem.with_suffix(".failure.json")).write_text(
                    json.dumps(
                        {
                            "status": "timeout",
                            "partial_receipt": str(receipt),
                            "no_further_device_trials": True,
                        }
                    )
                    + "\n"
                )
                # No more devices are opened after a timed-out trial.
                raise
        if not receipt.is_file():
            raise RuntimeError(
                f"Trial exited {returncode} without receipt; inspect {stem}"
            )
        row = json.loads(receipt.read_text())
        if returncode and not planner.known_guard_failure(row):
            raise RuntimeError(f"Unknown native failure; inspect {receipt}")
        if not row.get("close_mesh_device_returned"):
            raise RuntimeError(f"Native mesh did not close; inspect {receipt}")
        rows.append(row)
        print(f"CAPACITY_TRIAL_CLOSED {stem.name} {row['status']}", flush=True)


def maybe_run(model_spec):
    if os.environ.get("QWEN38_RUN_CAPACITY_SCREEN") != "1":
        return
    if model_spec.get("impl", {}).get("impl_id") != "qwen38_autoport":
        raise ValueError("Capacity screen is restricted to Qwen38 autoport")
    root = (
        Path(os.environ.get("TT_METAL_HOME", "/home/container_app_user/tt-metal"))
        / "models/autoports/qwen_qwen3_8_27b"
    )
    screen(
        root,
        Path(os.environ["QWEN38_CAPACITY_OUTPUT"]),
        os.environ["QWEN38_CAPACITY_IMAGE_DIGEST"],
        Path(os.environ["MODEL_WEIGHTS_DIR"]),
    )
