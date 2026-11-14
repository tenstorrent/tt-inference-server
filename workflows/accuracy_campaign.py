# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Run full accuracy tasks serially inside one guarded device-queue job.

No task sampling, token-budget overrides or score-driven retries are supported.
Every invocation has a fresh artifact root. Exit codes and accuracy verdicts
are independent of exact coverage. The caller must submit through its device
queue; this module must never be wrapped in an outer timeout.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import subprocess
import time
import traceback
import urllib.request
import uuid

from workflows.isolated_eval_docker import IsolatedDocker


def read_hf_token():
    """Read an explicitly supplied token or token file without a host-specific default."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token and os.environ.get("HF_TOKEN_PATH"):
        token = Path(os.environ["HF_TOKEN_PATH"]).expanduser().read_text().strip()
    if not token:
        raise RuntimeError("Set HF_TOKEN or HF_TOKEN_PATH for the evaluation campaign")
    return token


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def byte_count(path):
    total = 0
    for parent, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(parent) / name).stat().st_size
            except FileNotFoundError:
                pass
    return total


def available_memory():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is missing")


def git(repo, *args):
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, timeout=60
    ).strip()


def verify_sources(root, manifest):
    for name, revision in manifest["commits"].items():
        repo = root / name
        if git(repo, "rev-parse", "HEAD") != revision:
            raise RuntimeError(f"Source revision changed: {name}")
        if git(repo, "status", "--porcelain", "--untracked-files=all"):
            raise RuntimeError(f"Dirty source tree: {name}")


def owned_pids(token):
    marker = b"INFERENCE_EVAL_OWNER=" + token.encode()
    found = []
    for process in Path("/proc").glob("[0-9]*"):
        try:
            if marker in (process / "environ").read_bytes().split(b"\0"):
                found.append(int(process.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return found


def cleanup_owned(token, child):
    for sig, seconds in ((signal.SIGTERM, 45), (signal.SIGKILL, 20)):
        deadline = time.monotonic() + seconds
        while True:
            child.poll()
            pids = owned_pids(token)
            if not pids:
                return
            for pid in pids:
                try:
                    if pid in owned_pids(token):
                        os.kill(pid, sig)
                except ProcessLookupError:
                    pass
            if time.monotonic() >= deadline:
                break
            time.sleep(0.5)
    raise RuntimeError(f"Owned worker processes survived cleanup: {owned_pids(token)}")


def reset_free_devices(env):
    devices = sorted(map(str, Path("/dev/tenstorrent").glob("*")))
    if (
        not devices
        or subprocess.run(["fuser", "-s", *devices], env=env, timeout=30).returncode
        != 1
    ):
        raise RuntimeError(
            "Reset refused: devices missing, busy or ownership check failed"
        )
    subprocess.run(["tt-smi", "-glx_reset"], env=env, check=True, timeout=300)


def dictionaries(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from dictionaries(child)
    elif isinstance(value, list):
        for child in value:
            yield from dictionaries(child)


def runtime_inventory(tis):
    inventory = {}
    for metadata in (tis / ".workflow_venvs").glob(
        "*/lib/python*/site-packages/*.dist-info/METADATA"
    ):
        fields = {}
        for line in metadata.read_text(errors="replace").splitlines():
            if line.startswith(("Name: ", "Version: ")):
                key, value = line.split(": ", 1)
                fields[key] = value
        direct = metadata.with_name("direct_url.json")
        if direct.is_file():
            fields["source"] = json.loads(direct.read_text())
        inventory[str(metadata.relative_to(tis))] = fields
    return inventory


def assess_task(task, directory, workflow_exit):
    name = task["name"]
    problems, scores = [], []
    reports = sorted(
        directory.glob(
            f"workflow_logs/reports_output/{task['workflow']}/data/report_data_*.json"
        )
    )
    if len(reports) != 1:
        problems.append(f"Expected one fresh report, found {len(reports)}")
    for report in reports:
        for record in dictionaries(json.loads(report.read_text())):
            if record.get("task_name") == name and isinstance(
                record.get("score"), (int, float)
            ):
                score = float(record["score"])
                reference, tolerance = (
                    record.get("gpu_reference_score"),
                    record.get("tolerance"),
                )
                if (
                    math.isfinite(score)
                    and isinstance(reference, (int, float))
                    and isinstance(tolerance, (int, float))
                ):
                    item = (score, float(reference), float(tolerance))
                    if item not in scores:
                        scores.append(item)
    if len(scores) != 1:
        problems.append(f"Expected one unambiguous official score, found {scores}")
    observed, errors, distinct_tasks = 0, [], None
    if task["workflow"] == "evals":
        identities = set()
        for sample_file in directory.glob(
            "workflow_logs/reports_output/**/samples_*.jsonl"
        ):
            # The timestamp suffix is irrelevant; each invocation runs one task.
            leaf = sample_file.name.rsplit("_20", 1)[0]
            for line in sample_file.open():
                sample = json.loads(line)
                key = (leaf, sample["doc_id"])
                if key in identities:
                    problems.append(f"Duplicate sample: {key}")
                identities.add(key)
                observed += 1
                if sample.get("error"):
                    errors.append(str(sample["error"]))
        if errors:
            problems.append(f"Inference errors in {len(errors)} samples")
    else:
        summaries = list(
            directory.glob(f"workflow_logs/reports_output/**/{name}/result.json")
        )
        if len(summaries) != 1:
            problems.append(f"Expected one Harbor summary, found {len(summaries)}")
        else:
            parent = summaries[0].parent
            trials = []
            for result in parent.glob("*/result.json"):
                trial = json.loads(result.read_text())
                if not trial.get("finished_at"):
                    problems.append(f"Unfinished Harbor trial: {result.parent.name}")
                trials.append(trial)
                if trial.get("exception_info"):
                    errors.append(trial["exception_info"])
            observed = len(trials)
            counts = Counter(trial.get("task_name") for trial in trials)
            distinct_tasks = len(counts)
            repeats = 3 if name == "terminal_bench_2_1" else 1
            if any(count != repeats for count in counts.values()):
                problems.append(
                    "Harbor task repeat coverage differs from the reference"
                )
            summary = json.loads(summaries[0].read_text())
            if summary.get("n_total_trials") != task["samples"]:
                problems.append(
                    "Harbor summary trial count differs from the full reference"
                )
    if observed != task["samples"]:
        problems.append(f"Coverage: {observed} of {task['samples']}")
    if workflow_exit not in (0, 1):
        problems.append(f"Worker or cleanup exit: {workflow_exit}")
    complete = not problems
    score, reference, tolerance = scores[0] if len(scores) == 1 else (None, None, None)
    passed = score >= reference * (1 - tolerance) if score is not None else None
    return {
        "task": name,
        "operationally_complete": complete,
        "samples": observed,
        "expected_samples": task["samples"],
        "distinct_tasks": distinct_tasks,
        "score": score,
        "reference": reference,
        "tolerance": tolerance,
        "accuracy_pass": passed if complete else None,
        "workflow_exit": workflow_exit,
        "problems": problems,
        "harness_errors": errors,
    }


def build_command(root, task, profile, port):
    return [
        "python3",
        "run.py",
        "--model",
        profile["model"],
        "--tt-device",
        profile["device"],
        "--workflow",
        task["workflow"],
        "--eval-tasks",
        task["name"],
        "--dev-mode",
        "--local-server",
        "--service-port",
        str(port),
        "--host-hf-cache",
        "/data/huggingface",
        "--tt-metal-home",
        str(root / "tt-metal"),
        "--tt-metal-python-venv-dir",
        str(root / "tt-metal/python_env"),
        "--vllm-override-args",
        json.dumps(profile["vllm_args"]),
    ]


def run_task(args, task, profile, environment, output, deadline, stopped):
    tis = args.root / "tt-inference-server"
    directory = output / task["name"]
    directory.mkdir(exist_ok=False)
    workflow = tis / "workflow_logs"
    if workflow.exists():
        prior = tis / ("workflow_logs-before-" + output.name + "-" + task["name"])
        if prior.exists():
            raise RuntimeError(f"Refusing to overwrite prior artifacts: {prior}")
        workflow.rename(prior)
    token = uuid.uuid4().hex
    child_env = dict(environment, INFERENCE_EVAL_OWNER=token)
    command = build_command(args.root, task, profile, args.port)
    atomic_json(directory / "command.json", {"argv": command, "cwd": str(tis)})
    with (directory / "worker.log").open("xb", buffering=0) as log:
        child = subprocess.Popen(
            command,
            cwd=tis,
            env=child_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        bound_reason = None
        next_scan, artifacts, low_memory = 0, 0, 0
        try:
            while child.poll() is None:
                if time.monotonic() >= next_scan:
                    artifacts = byte_count(output) + byte_count(workflow)
                    next_scan = time.monotonic() + 30
                memory = available_memory()
                low_memory = low_memory + 1 if memory < 24 * 1024**3 else 0
                if stopped:
                    bound_reason = f"Campaign interrupted by signal {stopped[0]}"
                elif time.time() >= deadline:
                    bound_reason = "Campaign time bound"
                elif artifacts >= args.max_artifact_gib * 1024**3:
                    bound_reason = "Campaign artifact bound"
                elif low_memory >= 2:
                    bound_reason = "Host memory safety floor"
                elif any(
                    shutil.disk_usage(p).free < 40 * 1024**3
                    for p in ("/", "/data", "/data2")
                ):
                    bound_reason = "Filesystem free-space safety floor"
                health = None
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{args.port}/health", timeout=3
                    ) as response:
                        health = response.status
                except Exception:
                    pass
                atomic_json(
                    output / "progress.json",
                    {
                        "state": "running",
                        "task": task["name"],
                        "updated_at": time.time(),
                        "deadline": deadline,
                        "artifact_bytes": artifacts,
                        "available_memory_bytes": memory,
                        "server_health": health,
                        "stop_reason": bound_reason,
                    },
                )
                if bound_reason:
                    break
                time.sleep(10)
            rc = child.returncode if child.returncode is not None else 124
        finally:
            cleanup_owned(token, child)
            reset_free_devices(environment)
    if workflow.exists():
        # Source checkouts and large evaluation artifacts can be on different
        # filesystems. The worker and all its descendants have exited here.
        shutil.move(str(workflow), str(directory / "workflow_logs"))
    atomic_json(directory / "runtime-packages.json", runtime_inventory(tis))
    result = assess_task(task, directory, rc)
    if bound_reason:
        result["problems"].append(bound_reason)
        result["operationally_complete"] = False
        result["accuracy_pass"] = None
    atomic_json(directory / "result.json", result)
    return result, bound_reason


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--docker-storage", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8014)
    parser.add_argument("--max-hours", type=float, default=72)
    parser.add_argument("--max-artifact-gib", type=float, default=100)
    args = parser.parse_args()
    args.root = args.root.resolve()
    if not 0 < args.max_hours <= 72 or not 0 < args.max_artifact_gib <= 100:
        parser.error("Campaign bounds must not exceed 72 hours / 100 GiB")
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.chmod(args.output, 0o700)
    os.umask(0o077)
    with (args.output / "campaign.log").open("x", buffering=1) as record:
        os.dup2(record.fileno(), 1)
        os.dup2(record.fileno(), 2)
        return campaign(args)


def campaign(args):
    output, root = args.output, args.root
    deadline = time.time() + args.max_hours * 3600
    stopped, results = [], []
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda value, frame: stopped.append(value))
    docker = None
    rc, failure = 2, None
    try:
        manifest = json.loads(args.manifest.read_text())
        verify_sources(root, manifest)
        profile_path = (
            root
            / "tt-inference-server/reference_config/evals/deepseek_v41_accuracy.json"
        )
        if digest(profile_path) != manifest["profile_sha256"]:
            raise RuntimeError("Accuracy profile changed")
        profile = json.loads(profile_path.read_text())
        atomic_json(output / "manifest.json", manifest)
        atomic_json(output / "profile.json", profile)
        atomic_json(
            output / "limits.json",
            {
                "deadline": deadline,
                "max_hours": args.max_hours,
                "max_artifact_gib": args.max_artifact_gib,
            },
        )
        with socket.socket() as probe:
            probe.bind(("0.0.0.0", args.port))
        env = {
            key: value
            for key, value in os.environ.items()
            if key
            in {
                "HOME",
                "USER",
                "LOGNAME",
                "LANG",
                "LC_ALL",
                "PATH",
                "SHELL",
                "http_proxy",
                "https_proxy",
                "no_proxy",
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "NO_PROXY",
                "SSL_CERT_FILE",
                "SSL_CERT_DIR",
                "REQUESTS_CA_BUNDLE",
            }
        }
        env.update(
            {
                "TT_METAL_HOME": str(root / "tt-metal"),
                "TT_METAL_CACHE": os.environ.get(
                    "TT_METAL_CACHE", str(output / "kernel-cache")
                ),
                "HF_HOME": "/data/huggingface",
                "HF_HUB_OFFLINE": "0",
                "PYTHONNOUSERSITE": "1",
                "PYTHONUNBUFFERED": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "LM_EVAL_PRESERVE_REASONING": "1",
                "OMP_NUM_THREADS": "16",
                "TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES": "0",
                "UV_CACHE_DIR": str(root / ".uv-cache"),
                "UV_PYTHON_INSTALL_DIR": str(root / ".uv-python"),
                "PIP_CACHE_DIR": str(root / ".pip-cache"),
                "GIT_TERMINAL_PROMPT": "0",
                "AUTOMATIC_HOST_SETUP": "1",
                "JWT_SECRET": secrets.token_hex(32),
            }
        )
        env["HF_TOKEN"] = read_hf_token()
        # run.py's dotenv loader deliberately overwrites environment values.
        # Use an owned private file, not another workflow's credentials or knobs.
        private_env = output / "private.env"
        private_env.write_text(
            "JWT_SECRET=" + env["JWT_SECRET"] + "\nHF_TOKEN=" + env["HF_TOKEN"] + "\n"
        )
        os.chmod(private_env, 0o600)
        env["INFERENCE_SERVER_DOTENV_PATH"] = str(private_env)
        for task in profile["tasks"]:
            if (
                stopped
                or time.time() >= deadline
                or byte_count(output) >= args.max_artifact_gib * 1024**3
            ):
                break
            verify_sources(root, manifest)
            task_env = dict(env)
            if task["workflow"] == "agentic":
                if docker is None:
                    docker = IsolatedDocker(
                        args.docker_storage, output / "private-docker"
                    )
                    docker.start()
                task_env["DOCKER_HOST"] = "unix://" + str(docker.socket)
                task_env["OPENAI_BASE_URL"] = f"http://{docker.gateway}:{args.port}/v1"
                task_env["OPENAI_API_BASE"] = task_env["OPENAI_BASE_URL"]
            result, bound = run_task(
                args, task, profile, task_env, output, deadline, stopped
            )
            results.append(result)
            if docker is not None:
                docker.clear_containers()
            verify_sources(root, manifest)
            atomic_json(output / "results.json", results)
            print("task finished:", json.dumps(result), flush=True)
            if bound:
                failure = bound
                break
            if not result["operationally_complete"]:
                failure = f"Infrastructure or coverage failure in {task['name']}; preserve this attempt for diagnosis"
                break
        full = len(results) == len(profile["tasks"]) and all(
            r["operationally_complete"] for r in results
        )
        rc = (
            0 if full and all(r["accuracy_pass"] for r in results) else 1 if full else 2
        )
    except Exception:
        failure = traceback.format_exc()
        print(failure, flush=True)
    finally:
        if docker is not None:
            try:
                docker.close()
            except Exception:
                rc = 2
                failure = (
                    (failure or "")
                    + "\nDocker cleanup failed:\n"
                    + traceback.format_exc()
                )
        summary = {
            "exit": rc,
            "results": results,
            "failure": failure,
            "complete": len(results) == 5
            and all(r["operationally_complete"] for r in results),
            "finished_at": time.time(),
            "deadline": deadline,
        }
        atomic_json(output / "done.json", summary)
        atomic_json(output / "progress.json", {"state": "finished", **summary})
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
