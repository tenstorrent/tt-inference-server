# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Pin Banking task images to the tau2 revision before the voice import regression."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import sys

TAU2_REVISION = "b351ed5f9281d4bdfa5629262f54c8781da0d5be"


def _pinned_dockerfile(path: Path) -> str:
    original = path.read_text()
    clone = 'git clone --depth=1 "${TAU2_BENCH_REPO}" "${TAU2_BENCH_ROOT}"'
    if (
        "ENV TAU2_BENCH_ROOT=/opt/tau2-bench" not in original
        or original.count(clone) != 1
    ):
        raise ValueError(
            f"Unrecognized Banking Dockerfile: {path}; inspect before continuing"
        )
    checkout = (
        'git init "${TAU2_BENCH_ROOT}"'
        ' && git -C "${TAU2_BENCH_ROOT}" fetch --depth=1'
        f' "${{TAU2_BENCH_REPO}}" {TAU2_REVISION}'
        ' && git -C "${TAU2_BENCH_ROOT}" checkout --detach FETCH_HEAD'
    )
    return original.replace(clone, checkout) + (
        "\nRUN python3 -c 'import tau2.evaluator.evaluator'\n"
    )


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

    contents = (
        json.dumps(
            {
                "services": {
                    service: {
                        "build": {
                            "dockerfile_inline": _pinned_dockerfile(path).replace(
                                "$", "$$"
                            )
                        }
                    }
                    for service, path in (
                        ("main", directory / "Dockerfile"),
                        ("tau3-runtime", directory / "runtime-server/Dockerfile"),
                    )
                }
            }
        )
        + "\n"
    )
    digest = hashlib.sha256(contents.encode()).hexdigest()
    overlay_dir.mkdir(parents=True, exist_ok=True)
    overlay = overlay_dir / f"{digest}.json"
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
