# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Supply the missing tau2 evaluator dependency in Banking task images."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import sys


def prepare_docker_path(directory: Path, interpreter: Path, path: str) -> str:
    """Scope the Docker adapter and its build records to one Harbor subprocess."""
    docker = shutil.which("docker", path=path)
    if docker is None:
        raise RuntimeError("Docker is required for Tau3 Docker evaluations")
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        str(interpreter),
        str(Path(__file__).resolve()),
        "--docker",
        str(Path(docker).absolute()),
        "--overlay-dir",
        str(directory / "overlays"),
        "--",
    ]
    wrapper = directory / "docker"
    wrapper.write_text(f'#!/bin/sh\nexec {shlex.join(command)} "$@"\n')
    wrapper.chmod(0o755)
    return str(directory) + os.pathsep + path


def docker_command(args: list[str], docker: str, overlay_dir: Path) -> list[str]:
    if not args or args[0] != "compose" or "--project-directory" not in args:
        return [docker, *args]

    directory = Path(args[args.index("--project-directory") + 1]).resolve()
    task_file = directory.parent / "task.toml"
    if not task_file.is_file():
        return [docker, *args]

    # This entry point runs with Harbor's Python 3.12 interpreter, even when
    # the workflow orchestrator importing this module uses Python 3.10.
    import tomllib

    task = tomllib.loads(task_file.read_text())
    name = task.get("task", {}).get("name", "")
    if not name.startswith("sierra-research/tau3-bench__tau3-banking_knowledge-task-"):
        return [docker, *args]

    original = (directory / "Dockerfile").read_text()
    if "ENV TAU2_BENCH_ROOT=/opt/tau2-bench" not in original:
        raise ValueError("Unrecognized Banking Dockerfile; inspect before continuing")
    dockerfile = original + (
        "\nRUN python3 -m pip install --no-cache-dir websockets==17.1"
        " && python3 -c 'import tau2.evaluator.evaluator'\n"
    )
    digest = hashlib.sha256(dockerfile.encode()).hexdigest()
    overlay_dir.mkdir(parents=True, exist_ok=True)
    overlay = overlay_dir / f"{digest}.json"
    contents = (
        json.dumps(
            {
                "services": {
                    "main": {
                        "build": {"dockerfile_inline": dockerfile.replace("$", "$$")}
                    }
                }
            }
        )
        + "\n"
    )
    # Concurrent trials may use the same Dockerfile. Publish complete JSON only.
    candidate = overlay_dir / f".{digest}.{os.getpid()}.json"
    candidate.write_text(contents)
    candidate.replace(overlay)

    # Harbor puts its contiguous -f options before the Compose subcommand.
    # A later `exec main rm -f ...` must not be mistaken for a Compose option.
    insert_at = args.index("-f")
    while insert_at < len(args) and args[insert_at] == "-f":
        insert_at += 2
    return [docker, *args[:insert_at], "-f", str(overlay), *args[insert_at:]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker", required=True)
    parser.add_argument("--overlay-dir", required=True, type=Path)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    args = options.args[1:] if options.args[:1] == ["--"] else options.args
    try:
        command = docker_command(args, options.docker, options.overlay_dir)
    except (OSError, ValueError, IndexError) as error:
        sys.exit(f"Banking container setup failed: {error}")
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
