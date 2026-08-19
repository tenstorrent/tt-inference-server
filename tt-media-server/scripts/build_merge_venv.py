# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Build the dedicated CPU venv used to merge LoRA adapters.

The merge (utils/adapter_merge_utils.py) runs in its own venv so it can pin the
two versions that must match the rest of the stack, resolved from their sources
of truth so the merge venv never drifts:
  - transformers <- tt-vllm-plugin/pyproject.toml (the version vLLM serves with)
  - peft         <- the version installed in the forge venv (the env that WROTE
                    the adapter, so the merge reads its config back the same)

The venv then installs adapter_merge_requirements.txt together with those two
pins in a single resolve.

Usage:
    build_merge_venv.py <venv_dir> <tt-vllm-plugin pyproject.toml> \
        [--forge-python PATH] [--requirements PATH]
"""

import argparse
import os
import subprocess
import sys
import tomllib

# This script lives in <tt-media-server>/scripts, so its parent is the app root.
SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_REQUIREMENTS = os.path.join(SERVER_DIR, "adapter_merge_requirements.txt")


def transformers_spec(pyproject_path: str) -> str:
    """Return the `transformers` dependency spec from a pyproject's
    ``[project.dependencies]`` (e.g. ``transformers==4.55.0``)."""
    with open(pyproject_path, "rb") as fh:
        deps = tomllib.load(fh).get("project", {}).get("dependencies", [])
    for dep in deps:
        if dep.strip().startswith("transformers"):
            return dep.strip()
    raise ValueError(
        f"No 'transformers' entry in [project.dependencies] of {pyproject_path}"
    )


def peft_spec(forge_python: str) -> str:
    """Return the peft spec pinned to the version installed in the forge venv
    (the env that wrote the adapter), e.g. ``peft==0.20.0``."""
    try:
        result = subprocess.run(
            [
                forge_python,
                "-c",
                "import importlib.metadata as m; print(m.version('peft'))",
            ],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Interpreter not found: {forge_python}") from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not read 'peft' version from {forge_python}:\n"
            f"{result.stderr.strip()}"
        )
    return f"peft=={result.stdout.strip()}"


def default_forge_python() -> str:
    """Forge venv interpreter (transformers + the peft that trained the adapter).

    Prefers the path the Dockerfile exports via ``PYTHON_ENV_DIR`` and falls back
    to the conventional ``venv-worker`` location next to this script.
    """
    env_dir = os.getenv("PYTHON_ENV_DIR", os.path.join(SERVER_DIR, "venv-worker"))
    return os.path.join(env_dir, "bin", "python")


def build_venv(venv_dir: str, specs, requirements: str = DEFAULT_REQUIREMENTS) -> None:
    """Create `venv_dir` and install `requirements` pinned to `specs`."""
    subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
    pip = os.path.join(venv_dir, "bin", "pip")
    subprocess.run([pip, "install", "--no-cache-dir", "--upgrade", "pip"], check=True)
    subprocess.run(
        [pip, "install", "--no-cache-dir", "-r", requirements, *specs],
        check=True,
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("venv_dir", help="Directory to create the merge venv in")
    parser.add_argument("pyproject", help="Path to tt-vllm-plugin's pyproject.toml")
    parser.add_argument(
        "--forge-python",
        default=None,
        help="Forge venv interpreter (defaults to $PYTHON_ENV_DIR/bin/python)",
    )
    parser.add_argument("--requirements", default=DEFAULT_REQUIREMENTS)
    args = parser.parse_args(argv)

    forge_python = args.forge_python or default_forge_python()
    specs = [transformers_spec(args.pyproject), peft_spec(forge_python)]
    print(f"Merge venv versions -> {specs[0]} | {specs[1]}")
    build_venv(args.venv_dir, specs, args.requirements)


if __name__ == "__main__":
    main()
