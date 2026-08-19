# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Derive the package versions the LoRA adapter-merge venv must install.

Both versions come from their sources of truth so the merge venv never drifts
from the versions that matter (see scripts/build_merge_venv.sh):
  - transformers <- tt-vllm-plugin/pyproject.toml (the version vLLM serves with)
  - peft         <- the version installed in the forge venv (the env that WROTE
                    the adapter), so the merge reads its config back with the
                    same peft.

Used as a CLI by build_merge_venv.sh:
    derive_merge_versions.py transformers <pyproject.toml>   -> "transformers==X"
    derive_merge_versions.py peft         <forge_python>     -> "peft==Y"
"""

import argparse
import subprocess
import sys
import tomllib


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


def installed_version(python_executable: str, package: str) -> str:
    """Return the version of `package` installed in `python_executable`'s env."""
    try:
        result = subprocess.run(
            [
                python_executable,
                "-c",
                f"import importlib.metadata as m; print(m.version({package!r}))",
            ],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Interpreter not found: {python_executable}") from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not read '{package}' version from {python_executable}:\n"
            f"{result.stderr.strip()}"
        )
    return result.stdout.strip()


def peft_spec(forge_python: str) -> str:
    """Return the peft spec pinned to the version installed in the forge venv."""
    return f"peft=={installed_version(forge_python, 'peft')}"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("which", choices=["transformers", "peft"])
    parser.add_argument(
        "source",
        help="pyproject.toml path (transformers) or forge python (peft)",
    )
    args = parser.parse_args(argv)

    if args.which == "transformers":
        print(transformers_spec(args.source))
    else:
        print(peft_spec(args.source))


if __name__ == "__main__":
    main()
